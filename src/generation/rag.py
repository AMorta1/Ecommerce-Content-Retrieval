"""轻量 RAG 的商品事实构建、质量过滤与确定性选择。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_TASKS = ("title", "selling_points", "short_description")
IDENTITY_FIELDS = ("category_l1", "category_l2", "品牌", "型号")


def load_fact_policy(path: Path) -> dict[str, Any]:
    """读取事实策略并检查本模块依赖的最小配置。"""
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "category_field_policy",
        "global_excluded_fields",
        "global_value_gates",
        "initial_retrieval_proposal",
        "selection_policy",
    )
    missing = [key for key in required if key not in policy]
    if missing:
        raise ValueError(f"RAG 事实策略缺少字段：{missing}")
    return policy


def load_quality_audit(path: Path, scope: str) -> dict[str, dict[str, str]]:
    """读取指定 split 的只读审计记录，并拒绝重复商品。"""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = [row for row in csv.DictReader(input_file) if row["scope"] == scope]
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        product_id = str(row["product_id"])
        if product_id in result:
            raise ValueError(f"审计表中商品重复：{product_id}")
        result[product_id] = row
    return result


def normalize_fact_value(value: Any) -> str:
    """只做不改变语义的 Unicode 和空白规范化。"""
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", normalized).strip()


def is_placeholder_value(
    value: str,
    placeholders: set[str],
    *,
    composite_separator_pattern: str | None = None,
) -> bool:
    """识别单一占位值，以及完全由占位词组成的多语种组合值。"""
    normalized = normalize_fact_value(value).casefold()
    if normalized in placeholders:
        return True
    if not composite_separator_pattern:
        return False
    parts = [part.strip() for part in re.split(composite_separator_pattern, normalized)]
    parts = [part for part in parts if part]
    return len(parts) > 1 and all(part in placeholders for part in parts)


def _canonical_identity_field(field_name: str) -> str | None:
    if field_name == "品牌":
        return "品牌"
    if field_name == "型号" or field_name.endswith("型号"):
        return "型号"
    return None


def _split_blocked_fields(audit_row: dict[str, str]) -> set[str]:
    return {item.strip() for item in audit_row.get("blocked_fields", "").split("|") if item.strip()}


def _identity_conflict(audit_row: dict[str, str], blocked_fields: set[str]) -> bool:
    if audit_row.get("status") != "CONFLICT":
        return False
    canonical_blocked = {
        _canonical_identity_field(field) or field for field in blocked_fields
    }
    return bool(canonical_blocked.intersection(IDENTITY_FIELDS))


def _quality_status(
    audit_row: dict[str, str],
    blocked_fields: set[str],
    raw_field: str,
    canonical_field: str,
    block_entire_product: bool,
    has_distinct_multiple_values: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if block_entire_product:
        return "blocked_conflict", ["identity_or_category_conflict"]
    if raw_field in blocked_fields or canonical_field in blocked_fields:
        status = audit_row.get("status")
        if status == "CONFLICT":
            return "blocked_conflict", ["field_conflict"]
        if status == "REVIEW":
            return "withhold_review", ["field_requires_review"]
    if has_distinct_multiple_values:
        reasons.append("distinct_multi_value_requires_sku_resolution")
        return "withhold_review", reasons
    return "eligible", reasons


def _fact_id(
    dataset_version: str,
    split: str,
    product_id: str,
    canonical_field: str,
    normalized_values: list[str],
) -> str:
    value_payload = json.dumps(normalized_values, ensure_ascii=False, separators=(",", ":"))
    value_hash = hashlib.sha256(value_payload.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_version}:{split}:{product_id}:{canonical_field}:{value_hash}"


def _make_fact(
    *,
    record: dict[str, Any],
    audit_row: dict[str, str],
    dataset_version: str,
    source_path: str,
    source_sha256: str,
    field_name: str,
    canonical_field: str,
    raw_values: list[Any],
    normalized_values: list[str],
    knowledge_level: str,
    fact_role: str,
    field_group: str,
    field_order: int,
    quality_status: str,
    quality_reasons: list[str],
    is_core_attribute: bool = False,
    mandatory_status: str = "not_core",
    mandatory_reasons: list[str] | None = None,
) -> dict[str, Any]:
    audit_status = audit_row.get("status", "")
    consistency = "not_flagged" if audit_status == "PASS" else "see_audit_record"
    product_id = str(record["product_id"])
    split = str(record["split"])
    return {
        "fact_id": _fact_id(
            dataset_version, split, product_id, canonical_field, normalized_values
        ),
        "product_id": product_id,
        "dataset_version": dataset_version,
        "split": split,
        "category_l1": record["category_l1"],
        "category_l2": record["category_l2"],
        "knowledge_level": knowledge_level,
        "fact_role": fact_role,
        "field_group": field_group,
        "field_order": field_order,
        "field_name": field_name,
        "canonical_field": canonical_field,
        "raw_values": raw_values,
        "normalized_values": normalized_values,
        "task_tags": list(SUPPORTED_TASKS),
        "quality_status": quality_status,
        "quality_reasons": quality_reasons,
        "is_core_attribute": is_core_attribute,
        "mandatory_status": mandatory_status,
        "mandatory_reasons": list(mandatory_reasons or []),
        "evidence": {
            "structured_source": "standardized_category"
            if field_name.startswith("category_")
            else "structured_attributes",
            "title_consistency": consistency,
            "image_consistency": consistency,
            "audit_version": audit_row.get("audit_version", ""),
            "audit_status": audit_status,
            "audit_issue_code": audit_row.get("issue_code", ""),
            "audit_note": audit_row.get("review_note", ""),
        },
        "source_path": source_path,
        "source_sha256": source_sha256,
    }


def _is_negative_values(values: Iterable[Any], policy: dict[str, Any]) -> bool:
    negative_policy = policy.get("negative_constraint_policy", {})
    exact_values = {
        normalize_fact_value(value).casefold()
        for value in negative_policy.get("exact_values", [])
    }
    contains_values = tuple(
        normalize_fact_value(value).casefold()
        for value in negative_policy.get("contains_values", [])
    )
    normalized = [normalize_fact_value(value).casefold() for value in values]
    return bool(normalized) and all(
        value in exact_values
        or any(token and token in value for token in contains_values)
        for value in normalized
    )


def _matches_supplemental_only_rule(
    category_l2: str,
    canonical_field: str,
    normalized_values: list[str],
    policy: dict[str, Any],
) -> str | None:
    mandatory_policy = policy.get("mandatory_core_policy", {})
    for rule in mandatory_policy.get("supplemental_only_rules", []):
        if rule.get("category_l2") != category_l2 or rule.get("field") != canonical_field:
            continue
        exact_values = {
            normalize_fact_value(value).casefold()
            for value in rule.get("exact_values", [])
        }
        regex_values = [re.compile(pattern) for pattern in rule.get("regex_values", [])]
        if all(
            value.casefold() in exact_values
            or any(pattern.fullmatch(value) for pattern in regex_values)
            for value in normalized_values
        ):
            return str(rule.get("reason", "supplemental_only_rule"))
    return None


def _invalid_value_shape_reason(
    category_l2: str,
    canonical_field: str,
    normalized_values: list[str],
    policy: dict[str, Any],
) -> str | None:
    mandatory_policy = policy.get("mandatory_core_policy", {})
    for rule in mandatory_policy.get("value_shape_rules", []):
        if rule.get("category_l2") != category_l2 or rule.get("field") != canonical_field:
            continue
        pattern = re.compile(str(rule["regex"]))
        if not all(pattern.fullmatch(value) for value in normalized_values):
            return "invalid_value_shape"
        if rule.get("must_be_positive"):
            for value in normalized_values:
                match = re.match(r"[0-9]+(?:\.[0-9]+)?", value)
                if match is None or float(match.group()) <= 0:
                    return "nonpositive_value"
    return None


def _mandatory_metadata(
    *,
    category_l2: str,
    canonical_field: str,
    normalized_values: list[str],
    fact_role: str,
    quality_status: str,
    policy: dict[str, Any],
) -> tuple[bool, str, list[str]]:
    mandatory_fields = set(policy.get("mandatory_core_fields", {}).get(category_l2, []))
    is_core = canonical_field in mandatory_fields
    if not is_core:
        return False, "not_core", []
    if quality_status != "eligible":
        return True, "withheld_quality", [quality_status]
    if fact_role == "identity":
        return True, "identity_coverage", ["render_once_in_identity"]
    if _is_negative_values(normalized_values, policy):
        return True, "negative_constraint", ["negative_core_not_forced"]
    supplemental_reason = _matches_supplemental_only_rule(
        category_l2, canonical_field, normalized_values, policy
    )
    if supplemental_reason:
        return True, "supplemental_only", [supplemental_reason]
    return True, "mandatory", ["eligible_positive_or_neutral_core_fact"]


def build_fact_units(
    record: dict[str, Any],
    audit_row: dict[str, str],
    policy: dict[str, Any],
    *,
    dataset_version: str,
    source_path: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """从一条标准化商品记录构建带质量状态的白名单事实单元。"""
    product_id = str(record.get("product_id", ""))
    if not product_id or product_id != str(audit_row.get("product_id", "")):
        raise ValueError("商品记录与质量审计记录的 product_id 不一致。")
    if record.get("split") != audit_row.get("scope"):
        raise ValueError(f"商品 {product_id} 的 split 与审计 scope 不一致。")
    if audit_row.get("status") not in {"PASS", "REVIEW", "CONFLICT"}:
        raise ValueError(f"商品 {product_id} 缺少有效的质量审计状态。")
    category_l2 = record.get("category_l2")
    category_policy = policy["category_field_policy"].get(category_l2)
    if category_policy is None:
        raise ValueError(f"商品 {product_id} 使用未配置的二级品类：{category_l2}")

    blocked_fields = _split_blocked_fields(audit_row)
    block_entire_product = _identity_conflict(audit_row, blocked_fields)
    placeholders = {
        normalize_fact_value(value).casefold()
        for value in policy["global_value_gates"]["exclude_placeholders"]
    }
    composite_placeholder_separator = policy["global_value_gates"].get(
        "composite_placeholder_separator_pattern"
    )
    excluded_fields = set(policy["global_excluded_fields"]["fields"])
    excluded_patterns = policy["global_excluded_fields"]["patterns"]
    conditional_enabled = bool(policy["selection_policy"]["conditional_fields_enabled"])
    mandatory_fields = set(policy.get("mandatory_core_fields", {}).get(category_l2, []))
    excluded_values_by_attribute = {
        field: {
            normalize_fact_value(value).casefold()
            for value in values
        }
        for field, values in policy["global_value_gates"].get(
            "excluded_values_by_attribute", {}
        ).items()
    }
    identity_absent_values = {
        normalize_fact_value(value).casefold()
        for value in policy["global_value_gates"].get("identity_absent_values", [])
    }
    facts: list[dict[str, Any]] = []

    for order, field_name in enumerate(("category_l1", "category_l2")):
        normalized_values = [normalize_fact_value(record[field_name])]
        status, reasons = _quality_status(
            audit_row,
            blocked_fields,
            field_name,
            field_name,
            block_entire_product,
            False,
        )
        facts.append(
            _make_fact(
                record=record,
                audit_row=audit_row,
                dataset_version=dataset_version,
                source_path=source_path,
                source_sha256=source_sha256,
                field_name=field_name,
                canonical_field=field_name,
                raw_values=[record[field_name]],
                normalized_values=normalized_values,
                knowledge_level="category",
                fact_role="identity",
                field_group="identity",
                field_order=order,
                quality_status=status,
                quality_reasons=reasons,
            )
        )

    identity_allowlist = list(category_policy["identity"])
    group_fields = {
        group: {field: index for index, field in enumerate(category_policy[group])}
        for group in ("function", "parameter", "specification", "conditional")
    }
    attributes = record.get("attributes", {})
    if not isinstance(attributes, dict):
        raise TypeError(f"商品 {product_id} 的 attributes 必须是对象。")
    preferred_identity_fields: dict[str, str] = {}
    for raw_field in attributes:
        canonical_identity = _canonical_identity_field(raw_field)
        if canonical_identity not in identity_allowlist:
            continue
        current = preferred_identity_fields.get(canonical_identity)
        if current is None or raw_field == canonical_identity:
            preferred_identity_fields[canonical_identity] = raw_field

    for raw_field, raw_value in attributes.items():
        canonical_identity = _canonical_identity_field(raw_field)
        canonical_candidate = canonical_identity or raw_field
        is_mandatory_candidate = canonical_candidate in mandatory_fields
        if (
            raw_field in excluded_fields
            or any(pattern in raw_field for pattern in excluded_patterns)
        ) and not is_mandatory_candidate:
            continue
        fact_role = "task"
        field_group = ""
        knowledge_level = ""
        field_order = 0
        canonical_field = raw_field
        if canonical_identity and canonical_identity in identity_allowlist:
            if preferred_identity_fields[canonical_identity] != raw_field:
                continue
            fact_role = "identity"
            field_group = "identity"
            knowledge_level = "specification"
            canonical_field = canonical_identity
            field_order = 2 + identity_allowlist.index(canonical_identity)
        else:
            for candidate_group in ("function", "parameter", "specification", "conditional"):
                if raw_field in group_fields[candidate_group]:
                    field_group = candidate_group
                    knowledge_level = (
                        "specification" if candidate_group == "conditional" else candidate_group
                    )
                    field_order = group_fields[candidate_group][raw_field]
                    break
            if not field_group or (field_group == "conditional" and not conditional_enabled):
                continue

        values = raw_value if isinstance(raw_value, list) else [raw_value]
        normalized_all = [normalize_fact_value(value) for value in values]
        normalized_values = []
        for value in normalized_all:
            if (
                value
                and not is_placeholder_value(
                    value,
                    placeholders,
                    composite_separator_pattern=composite_placeholder_separator,
                )
                and value.casefold()
                not in excluded_values_by_attribute.get(canonical_field, set())
                and not (
                    fact_role == "identity"
                    and value.casefold() in identity_absent_values
                )
                and value not in normalized_values
            ):
                normalized_values.append(value)
        if not normalized_values:
            continue
        distinct_multiple = len(normalized_values) > 1
        status, reasons = _quality_status(
            audit_row,
            blocked_fields,
            raw_field,
            canonical_field,
            block_entire_product,
            distinct_multiple,
        )
        invalid_shape = _invalid_value_shape_reason(
            category_l2, canonical_field, normalized_values, policy
        )
        if status == "eligible" and invalid_shape:
            status = "withhold_review"
            reasons = [*reasons, invalid_shape]
        removed_placeholders = len(normalized_all) - len(normalized_values)
        if removed_placeholders:
            reasons = [*reasons, "placeholder_value_removed"]
        is_core_attribute, mandatory_status, mandatory_reasons = _mandatory_metadata(
            category_l2=category_l2,
            canonical_field=canonical_field,
            normalized_values=normalized_values,
            fact_role=fact_role,
            quality_status=status,
            policy=policy,
        )
        facts.append(
            _make_fact(
                record=record,
                audit_row=audit_row,
                dataset_version=dataset_version,
                source_path=source_path,
                source_sha256=source_sha256,
                field_name=raw_field,
                canonical_field=canonical_field,
                raw_values=values,
                normalized_values=normalized_values,
                knowledge_level=knowledge_level,
                fact_role=fact_role,
                field_group=field_group,
                field_order=field_order,
                quality_status=status,
                quality_reasons=reasons,
                is_core_attribute=is_core_attribute,
                mandatory_status=mandatory_status,
                mandatory_reasons=mandatory_reasons,
            )
        )

    return facts


def select_top_k_facts(
    fact_units: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    *,
    task: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """按任务规则选择已通过质量门控的身份信息和 Top-K 事实。"""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的生成任务：{task}")
    if top_k is None:
        top_k = int(policy["initial_retrieval_proposal"]["top_k"])
    if top_k < 1:
        raise ValueError("top_k 必须大于0。")

    units = list(fact_units)
    product_ids = {str(unit["product_id"]) for unit in units}
    if len(product_ids) > 1:
        raise ValueError("一次事实选择只能处理同一商品，禁止跨商品补充事实。")
    product_id = next(iter(product_ids), "")
    identity_order = {
        field: index
        for index, field in enumerate(policy["selection_policy"]["identity_order"])
    }
    identity_facts = sorted(
        (
            unit
            for unit in units
            if unit["fact_role"] == "identity" and unit["quality_status"] == "eligible"
        ),
        key=lambda unit: (
            identity_order.get(unit["canonical_field"], len(identity_order)),
            unit["field_order"],
            unit["fact_id"],
        ),
    )

    group_order = {
        group: index
        for index, group in enumerate(
            policy["selection_policy"]["task_group_priority"][task]
        )
    }
    candidates = [
        unit
        for unit in units
        if unit["fact_role"] == "task"
        and unit["quality_status"] == "eligible"
        and task in unit["task_tags"]
        and unit["field_group"] in group_order
    ]
    candidates.sort(
        key=lambda unit: (
            group_order[unit["field_group"]],
            unit["field_order"],
            unit["canonical_field"],
            unit["fact_id"],
        )
    )
    selected = []
    for rank, unit in enumerate(candidates[:top_k], start=1):
        selected.append(
            {
                **unit,
                "selection_rank": rank,
                "selection_reason": (
                    f"task={task}; group={unit['field_group']}; "
                    f"group_priority={group_order[unit['field_group']]}; "
                    f"field_order={unit['field_order']}"
                ),
            }
        )

    quality_counts = Counter(unit["quality_status"] for unit in units)
    return {
        "product_id": product_id,
        "task": task,
        "top_k": top_k,
        "identity_facts": identity_facts,
        "selected_facts": selected,
        "eligible_candidate_count": len(candidates),
        "quality_status_counts": dict(sorted(quality_counts.items())),
    }


def select_global_top_k_facts(
    fact_units: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    *,
    top_k: int | None = None,
    separate_negative_constraints: bool = False,
) -> dict[str, Any]:
    """融合三个生成任务的完整候选排名，返回一组全局共享事实。"""
    units = list(fact_units)
    if top_k is None:
        top_k = int(policy["initial_retrieval_proposal"]["top_k"])
    if top_k < 1:
        raise ValueError("top_k 必须大于0。")

    joint_policy = policy["selection_policy"]["global_joint_ranking"]
    if joint_policy["method"] != "reciprocal_rank_fusion":
        raise ValueError(f"不支持的联合排序方法：{joint_policy['method']}")
    tasks = tuple(joint_policy["tasks"])
    if set(tasks) != set(SUPPORTED_TASKS) or len(tasks) != len(SUPPORTED_TASKS):
        raise ValueError("全局联合排序必须且只能覆盖三个生成任务。")
    rrf_k = int(joint_policy["rrf_k"])
    if rrf_k < 1:
        raise ValueError("RRF k 必须大于0。")

    negative_policy = policy.get("negative_constraint_policy", {})
    exact_negative_values = {
        str(value).strip() for value in negative_policy.get("exact_values", [])
    }
    contains_negative_values = tuple(
        str(value).strip() for value in negative_policy.get("contains_values", [])
    )

    def is_negative_constraint(unit: dict[str, Any]) -> bool:
        if unit.get("fact_role") != "task" or unit.get("quality_status") != "eligible":
            return False
        values = [str(value).strip() for value in unit.get("normalized_values", [])]
        return any(
            value in exact_negative_values
            or any(token and token in value for token in contains_negative_values)
            for value in values
        )

    negative_constraint_facts = []
    ranking_units = units
    if separate_negative_constraints:
        negative_constraint_facts = sorted(
            (
                {
                    **unit,
                    "constraint_reason": "eligible_negative_fact_outside_content_top_k",
                }
                for unit in units
                if is_negative_constraint(unit)
            ),
            key=lambda unit: (
                unit["field_order"],
                unit["canonical_field"],
                unit["fact_id"],
            ),
        )
        negative_ids = {fact["fact_id"] for fact in negative_constraint_facts}
        ranking_units = [unit for unit in units if unit["fact_id"] not in negative_ids]

    task_rankings = {
        task: select_top_k_facts(
            ranking_units, policy, task=task, top_k=max(1, len(ranking_units))
        )
        for task in tasks
    }
    identity_facts = task_rankings[tasks[0]]["identity_facts"]
    candidates: dict[str, dict[str, Any]] = {}
    task_ranks: dict[str, dict[str, int]] = {}
    for task, ranking in task_rankings.items():
        for rank, fact in enumerate(ranking["selected_facts"], start=1):
            fact_id = fact["fact_id"]
            candidates[fact_id] = {
                key: value
                for key, value in fact.items()
                if key not in {"selection_rank", "selection_reason"}
            }
            task_ranks.setdefault(fact_id, {})[task] = rank

    scored = []
    for fact_id, fact in candidates.items():
        ranks = task_ranks[fact_id]
        if set(ranks) != set(tasks):
            raise ValueError(f"事实 {fact_id} 没有覆盖全部任务排名。")
        contributions = {task: 1.0 / (rrf_k + ranks[task]) for task in tasks}
        scored.append(
            {
                **fact,
                "joint_score": round(sum(contributions.values()), 12),
                "task_ranks": ranks,
                "task_score_contributions": {
                    task: round(score, 12) for task, score in contributions.items()
                },
                "selection_reason": "reciprocal_rank_fusion_across_all_generation_tasks",
            }
        )
    scored.sort(
        key=lambda fact: (
            -fact["joint_score"],
            sum(fact["task_ranks"].values()),
            fact["field_order"],
            fact["canonical_field"],
            fact["fact_id"],
        )
    )
    selected = []
    for rank, fact in enumerate(scored[:top_k], start=1):
        selected.append({**fact, "selection_rank": rank})

    quality_counts = Counter(unit["quality_status"] for unit in units)
    product_id = next(iter({str(unit["product_id"]) for unit in units}), "")
    return {
        "product_id": product_id,
        "task": "global_all_content",
        "top_k": top_k,
        "identity_facts": identity_facts,
        "selected_facts": selected,
        "negative_constraint_facts": negative_constraint_facts,
        "eligible_candidate_count": len(scored),
        "joint_ranking": {
            "method": joint_policy["method"],
            "tasks": list(tasks),
            "rrf_k": rrf_k,
        },
        "quality_status_counts": dict(sorted(quality_counts.items())),
    }


def select_rag_v2_context(
    fact_units: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    *,
    top_k: int | None = None,
) -> dict[str, Any]:
    """构造 Identity、Mandatory、Supplemental 与 Negative 四个互斥事实区。"""
    units = list(fact_units)
    product_ids = {str(unit["product_id"]) for unit in units}
    if len(product_ids) > 1:
        raise ValueError("一次事实选择只能处理同一商品，禁止跨商品补充事实。")
    product_id = next(iter(product_ids), "")

    identity_order = {
        field: index
        for index, field in enumerate(policy["selection_policy"]["identity_order"])
    }
    identity_facts = sorted(
        (
            unit
            for unit in units
            if unit["fact_role"] == "identity" and unit["quality_status"] == "eligible"
        ),
        key=lambda unit: (
            identity_order.get(unit["canonical_field"], len(identity_order)),
            unit["field_order"],
            unit["fact_id"],
        ),
    )
    mandatory_core_facts = sorted(
        (
            unit
            for unit in units
            if unit.get("mandatory_status") == "mandatory"
            and unit["fact_role"] == "task"
            and unit["quality_status"] == "eligible"
        ),
        key=lambda unit: (
            unit["field_order"],
            unit["canonical_field"],
            unit["fact_id"],
        ),
    )
    negative_constraint_facts = sorted(
        (
            {
                **unit,
                "constraint_reason": "eligible_negative_fact_outside_active_content",
            }
            for unit in units
            if unit["fact_role"] == "task"
            and unit["quality_status"] == "eligible"
            and _is_negative_values(unit.get("normalized_values", []), policy)
        ),
        key=lambda unit: (
            unit["field_order"],
            unit["canonical_field"],
            unit["fact_id"],
        ),
    )

    excluded_ids = {
        fact["fact_id"]
        for fact in [*mandatory_core_facts, *negative_constraint_facts]
    }
    supplemental_pool = [
        unit for unit in units if unit["fact_id"] not in excluded_ids
    ]
    supplemental_selection = select_global_top_k_facts(
        supplemental_pool,
        policy,
        top_k=top_k,
        separate_negative_constraints=False,
    )
    supplemental_facts = supplemental_selection["selected_facts"]

    zones = {
        "identity": identity_facts,
        "mandatory": mandatory_core_facts,
        "supplemental": supplemental_facts,
        "negative": negative_constraint_facts,
    }
    zone_ids = {
        name: {fact["fact_id"] for fact in facts}
        for name, facts in zones.items()
    }
    zone_names = tuple(zone_ids)
    for index, left in enumerate(zone_names):
        for right in zone_names[index + 1 :]:
            overlap = zone_ids[left] & zone_ids[right]
            if overlap:
                raise ValueError(f"RAG v2 事实区重复：{left}/{right}={sorted(overlap)}")
    injected = [fact for facts in zones.values() for fact in facts]
    if any(fact["quality_status"] != "eligible" for fact in injected):
        raise ValueError("RAG v2 只能注入 quality_status=eligible 的事实。")
    if any(str(fact["product_id"]) != product_id for fact in injected):
        raise ValueError("RAG v2 上下文中存在跨商品事实。")

    identity_coverage_facts = [
        fact
        for fact in identity_facts
        if fact.get("mandatory_status") == "identity_coverage"
    ]
    core_status_counts = Counter(
        unit.get("mandatory_status", "not_core")
        for unit in units
        if unit.get("is_core_attribute", False)
    )
    quality_counts = Counter(unit["quality_status"] for unit in units)
    return {
        "product_id": product_id,
        "task": "rag_v2_four_zone_context",
        "top_k": supplemental_selection["top_k"],
        "identity_facts": identity_facts,
        "mandatory_core_facts": mandatory_core_facts,
        "supplemental_facts": supplemental_facts,
        # 兼容现有报告与人工评测展示；与 supplemental_facts 是同一事实集合。
        "selected_facts": supplemental_facts,
        "negative_constraint_facts": negative_constraint_facts,
        "identity_mandatory_coverage_fact_ids": [
            fact["fact_id"] for fact in identity_coverage_facts
        ],
        "mandatory_injection_expected_fact_ids": [
            fact["fact_id"] for fact in mandatory_core_facts
        ],
        "existing_core_fact_count": sum(
            1 for unit in units if unit.get("is_core_attribute", False)
        ),
        "mandatory_status_counts": dict(sorted(core_status_counts.items())),
        "eligible_supplemental_candidate_count": supplemental_selection[
            "eligible_candidate_count"
        ],
        "joint_ranking": supplemental_selection["joint_ranking"],
        "quality_status_counts": dict(sorted(quality_counts.items())),
    }
