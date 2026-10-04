"""Deterministic, fact-grounded LoRA instruction-data construction."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK_TYPES = ("title", "selling_points", "short_description")
IDENTITY_FIELDS = ("品牌", "型号")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", normalized).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def load_consistency_review(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        product_id = str(row["product_id"])
        if product_id in result:
            raise ValueError(f"一致性审查包含重复商品：{product_id}")
        result[product_id] = row
    return result


def _casefold_set(values: Iterable[str]) -> set[str]:
    return {normalize_text(value).casefold() for value in values}


def select_reliable_core_attributes(
    record: dict[str, Any],
    core_fields: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], list[str]]:
    """Select source-present core fields using conservative value gates."""
    gates = config["quality_gates"]
    placeholders = _casefold_set(gates["exclude_placeholders"])
    composite_separator = gates.get("composite_placeholder_separator_pattern")
    identity_absent = _casefold_set(gates.get("identity_absent_values", []))
    excluded_by_field = {
        field: _casefold_set(values)
        for field, values in gates.get("excluded_values_by_field", {}).items()
    }
    model_reject_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in gates.get("model_reject_patterns", [])
    ]
    numeric_fields = set(gates.get("numeric_core_fields_requiring_digit", []))
    unsafe_terms = tuple(gates["unsafe_attribute_value_terms"])
    multivalue_fields = set(gates["allowed_multivalue_set_fields"])
    max_set_values = int(gates["maximum_values_per_set_field"])
    source_attributes = record.get("attributes") or {}
    selected: dict[str, list[str]] = {}
    actions: list[str] = []

    for field in core_fields:
        raw_values = source_attributes.get(field) or []
        values: list[str] = []
        seen: set[str] = set()
        for raw_value in raw_values:
            value = normalize_text(raw_value)
            folded = value.casefold()
            placeholder_parts = (
                [
                    part.strip()
                    for part in re.split(composite_separator, folded)
                    if part.strip()
                ]
                if composite_separator
                else []
            )
            is_composite_placeholder = (
                len(placeholder_parts) > 1
                and all(part in placeholders for part in placeholder_parts)
            )
            if not value or folded in placeholders or is_composite_placeholder:
                actions.append(f"drop_placeholder:{field}={value}")
                continue
            if field in IDENTITY_FIELDS and folded in identity_absent:
                actions.append(f"drop_absent_identity:{field}={value}")
                continue
            if folded in excluded_by_field.get(field, set()):
                actions.append(f"drop_excluded_field_value:{field}={value}")
                continue
            if field == "型号" and any(
                pattern.search(value) for pattern in model_reject_patterns
            ):
                actions.append(f"drop_noninformative_model:{field}={value}")
                continue
            if field in numeric_fields and not re.search(r"\d", value):
                actions.append(f"drop_non_numeric_value:{field}={value}")
                continue
            if any(term in value for term in unsafe_terms):
                actions.append(f"drop_unsafe_value:{field}={value}")
                continue
            if folded not in seen:
                seen.add(folded)
                values.append(value)
        if not values:
            continue
        if len(values) > 1 and field not in multivalue_fields:
            actions.append(f"withhold_multivalue_field:{field}")
            continue
        if len(values) > max_set_values:
            actions.append(f"withhold_oversized_set_field:{field}")
            continue
        selected[field] = values
    return selected, actions


def active_copy_attributes(
    attributes: dict[str, list[str]], config: dict[str, Any]
) -> dict[str, list[str]]:
    negative_values = _casefold_set(
        config["quality_gates"]["negative_values_not_used_as_active_copy_points"]
    )
    return {
        field: values
        for field, values in attributes.items()
        if any(value.casefold() not in negative_values for value in values)
    }


def _value_text(values: list[str]) -> str:
    return "、".join(values)


def _native_position(source_title: str, value: str) -> int:
    position = source_title.casefold().find(value.casefold())
    return position if position >= 0 else 10**9


def build_title_target(
    record: dict[str, Any],
    reliable_attributes: dict[str, list[str]],
    config: dict[str, Any],
) -> tuple[str, list[str]]:
    """Compress the native title to components supported by model input facts."""
    source_title = normalize_text(record.get("title", ""))
    if not source_title or "�" in source_title:
        return "", ["reject_missing_or_garbled_source_title"]

    gates = config["quality_gates"]
    actions = ["discard_untrusted_source_title_free_text"]
    for term in gates["title_always_remove_terms"]:
        if term in source_title:
            actions.append(f"remove_marketing_or_evaluative_term:{term}")

    reliable_value_text = " ".join(
        value for values in reliable_attributes.values() for value in values
    )
    for term in gates["title_conditionally_supported_terms"]:
        if term in source_title and term not in reliable_value_text:
            actions.append(f"remove_unsupported_scope_or_scenario:{term}")

    category = normalize_text(record["category_l2"])
    active = active_copy_attributes(reliable_attributes, config)
    identity_candidates: list[tuple[str, str]] = []
    detail_candidates: list[tuple[str, str]] = []
    for field, values in active.items():
        value = _value_text(values)
        if field in IDENTITY_FIELDS:
            identity_candidates.append((field, value))
        elif len(values) == 1:
            detail_candidates.append((field, value))

    detail_candidates.sort(
        key=lambda item: (_native_position(source_title, item[1]), list(active).index(item[0]))
    )
    chosen = identity_candidates + detail_candidates[:2] + [("category_l2", category)]

    components: list[str] = []
    for field, value in chosen:
        folded = value.casefold()
        if any(folded == existing.casefold() for existing in components):
            continue
        if field == "category_l2" and any(folded in item.casefold() for item in components):
            continue
        components.append(value)
        if _native_position(source_title, value) == 10**9:
            actions.append(f"add_reliable_component:{field}")

    target = " ".join(components)
    max_chars = int(gates["maximum_title_characters"])
    while len(target) > max_chars and len(components) > 2:
        removable = next(
            (
                index
                for index in range(len(components) - 1, -1, -1)
                if components[index] != category
                and components[index]
                not in [value for _, value in identity_candidates]
            ),
            None,
        )
        if removable is None:
            break
        actions.append(f"remove_for_title_length:{components[removable]}")
        components.pop(removable)
        target = " ".join(components)
    if len(target) > max_chars:
        return "", actions + ["reject_title_over_length_limit"]
    return target.strip(), actions


def _copy_point_priority(
    attributes: dict[str, list[str]], config: dict[str, Any]
) -> list[tuple[str, list[str]]]:
    active = active_copy_attributes(attributes, config)
    non_identity = [(field, values) for field, values in active.items() if field not in IDENTITY_FIELDS]
    identity = [(field, values) for field, values in active.items() if field in IDENTITY_FIELDS]
    return non_identity + identity


def build_selling_points_target(
    reliable_attributes: dict[str, list[str]], config: dict[str, Any]
) -> list[str]:
    count = int(config["targets"]["selling_point_count"])
    selected = _copy_point_priority(reliable_attributes, config)[:count]
    return [f"{field}：{_value_text(values)}" for field, values in selected]


def build_short_description_target(
    category_l2: str,
    reliable_attributes: dict[str, list[str]],
    config: dict[str, Any],
) -> str:
    maximum = int(config["quality_gates"]["maximum_short_description_attributes"])
    ordered = list(active_copy_attributes(reliable_attributes, config).items())[:maximum]
    clauses = [_attribute_clause(field, values) for field, values in ordered]
    if not clauses:
        return ""
    return f"这款{category_l2}的" + "，".join(clauses) + "。"


def _attribute_clause(field: str, values: list[str]) -> str:
    value = _value_text(values)
    if field == "是否无线" and value == "有线":
        return "连接方式为有线"
    if field == "是否无线" and value == "无线":
        return "连接方式为无线"
    if field == "是否机械键盘" and value in {"是", "有", "支持"}:
        return "键盘类型为机械键盘"
    if field == "是否有多媒体功能键" and value in {"是", "有", "支持"}:
        return "配有多媒体功能键"
    if field == "有无手托" and value in {"是", "有", "支持"}:
        return "配有手托"
    return f"{field}为{value}"


def source_title_quality_issue(
    record: dict[str, Any], config: dict[str, Any]
) -> tuple[str | None, list[str]]:
    """Use the native title only as quality evidence, never as model input."""
    title = normalize_text(record.get("title", ""))
    category = str(record["category_l2"])
    gates = config["quality_gates"]
    aliases = gates["category_title_aliases"][category]
    if not any(alias in title for alias in aliases):
        return "title_category_not_supported", [f"missing_category_alias:{category}"]
    boundary_terms = gates["category_excluded_title_terms"].get(category, [])
    matched_boundary_terms = [term for term in boundary_terms if term in title]
    if matched_boundary_terms:
        return "accessory_or_product_boundary_title", [
            f"boundary_term:{term}" for term in matched_boundary_terms
        ]
    if category == "移动电源":
        pattern = re.compile(gates["mobile_power_capacity_token_pattern"])
        capacity_tokens = {match.casefold() for match in pattern.findall(title)}
        maximum = int(gates["maximum_distinct_mobile_power_capacity_tokens"])
        if len(capacity_tokens) > maximum:
            return "ambiguous_multiple_capacity_tokens", [
                "capacity_tokens:" + "|".join(sorted(capacity_tokens))
            ]
    return None, []


def build_instruction_text(
    category_l1: str,
    category_l2: str,
    task_type: str,
    reliable_attributes: dict[str, list[str]],
) -> str:
    if task_type not in TASK_TYPES:
        raise ValueError(f"不支持的 task_type：{task_type}")
    requirements = {
        "title": "生成一个简洁商品标题，只组合输入中的品类和可靠核心属性。",
        "selling_points": "生成恰好三个卖点，每条只直述一个输入事实，不添加效果、评价或场景。",
        "short_description": "生成一段自然短详情，只串联输入事实，不增加解释、效果、评价或场景。",
    }
    payload = json.dumps(reliable_attributes, ensure_ascii=False, sort_keys=False)
    return (
        f"任务类型：{task_type}\n"
        f"一级品类：{category_l1}\n"
        f"二级品类：{category_l2}\n"
        f"可靠核心属性：{payload}\n"
        f"输出要求：{requirements[task_type]}"
    )


def _target_text(task_type: str, target: str | list[str]) -> str:
    if task_type == "selling_points":
        return json.dumps(target, ensure_ascii=False)
    if not isinstance(target, str):
        raise TypeError(f"{task_type} target 必须是字符串。")
    return target


def build_instruction_record(
    product: dict[str, Any], task_type: str
) -> dict[str, Any]:
    target = product["targets"][task_type]
    instruction = build_instruction_text(
        product["category_l1"],
        product["category_l2"],
        task_type,
        product["used_attributes"],
    )
    target_text = _target_text(task_type, target)
    return {
        "data_version": product["data_version"],
        "template_version": product["template_version"],
        "split": product["split"],
        "product_id": product["product_id"],
        "category_l1": product["category_l1"],
        "category_l2": product["category_l2"],
        "task_type": task_type,
        "input": {
            "category_l1": product["category_l1"],
            "category_l2": product["category_l2"],
            "core_attributes": product["used_attributes"],
            "task_type": task_type,
        },
        "instruction": instruction,
        "target": target,
        "target_text": target_text,
        "messages": [
            {
                "role": "system",
                "content": "你是中文电商文案编辑，只能使用输入中的品类和可靠核心属性。",
            },
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": target_text},
        ],
        "traceability": {
            "source_title": product["source_title"],
            "source_attributes": product["source_attributes"],
            "used_attributes": product["used_attributes"],
            "cleaning_actions": product["cleaning_actions"],
            "quality_status": product["quality_status"],
        },
    }


def _rank(namespace: str, seed: int, category: str, product_id: str) -> str:
    value = f"{namespace}:{seed}:{category}:{product_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_product_splits(
    products: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, str]:
    split_config = config["split"]
    namespace = str(split_config["namespace"])
    seed = int(split_config["seed"])
    fraction = float(split_config["validation_fraction"])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        by_category[product["category_l2"]].append(product)

    assignments: dict[str, str] = {}
    for category, category_products in sorted(by_category.items()):
        ranked = sorted(
            category_products,
            key=lambda product: _rank(
                namespace, seed, category, product["product_id"]
            ),
        )
        validation_count = max(1, int(len(ranked) * fraction + 0.5))
        for index, product in enumerate(ranked):
            assignments[product["product_id"]] = (
                "validation" if index < validation_count else "train"
            )
    return assignments


def build_dataset(
    records: list[dict[str, Any]],
    review: dict[str, dict[str, str]],
    core_attributes: dict[str, list[str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data_version = str(config["version"])
    template_version = str(config["template_version"])
    minimum_active = int(config["quality_gates"]["minimum_active_core_attributes"])
    selling_point_count = int(config["targets"]["selling_point_count"])
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for record in records:
        product_id = str(record["product_id"])
        audit = review.get(product_id)
        if audit is not None:
            review_status = audit["review_status"]
            reason = (
                "confirmed_conflict"
                if review_status == config["consistency_review"]["confirmed_status"]
                else "review_pending"
            )
            excluded.append(
                {
                    "product_id": product_id,
                    "category_l2": record["category_l2"],
                    "reason": reason,
                    "issue_code": audit.get("issue_code", ""),
                    "review_note": audit.get("review_note", ""),
                }
            )
            continue

        title_issue, title_evidence = source_title_quality_issue(record, config)
        if title_issue is not None:
            excluded.append(
                {
                    "product_id": product_id,
                    "category_l2": record["category_l2"],
                    "reason": title_issue,
                    "cleaning_actions": title_evidence,
                }
            )
            continue

        category_l2 = str(record["category_l2"])
        reliable, value_actions = select_reliable_core_attributes(
            record, core_attributes[category_l2], config
        )
        active = active_copy_attributes(reliable, config)
        if len(active) < minimum_active:
            excluded.append(
                {
                    "product_id": product_id,
                    "category_l2": category_l2,
                    "reason": "insufficient_reliable_core_attributes",
                    "active_core_attribute_count": len(active),
                    "cleaning_actions": value_actions,
                }
            )
            continue

        title_target, title_actions = build_title_target(record, reliable, config)
        selling_points = build_selling_points_target(reliable, config)
        short_description = build_short_description_target(
            category_l2, reliable, config
        )
        if (
            not title_target
            or len(selling_points) != selling_point_count
            or not short_description
        ):
            excluded.append(
                {
                    "product_id": product_id,
                    "category_l2": category_l2,
                    "reason": "incomplete_three_task_targets",
                    "cleaning_actions": value_actions + title_actions,
                }
            )
            continue

        signature_payload = json.dumps(
            [category_l2, title_target, reliable],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
        if signature in seen_signatures:
            excluded.append(
                {
                    "product_id": product_id,
                    "category_l2": category_l2,
                    "reason": "duplicate_final_product_signature",
                }
            )
            continue
        seen_signatures.add(signature)

        accepted.append(
            {
                "data_version": data_version,
                "template_version": template_version,
                "split": "",
                "product_id": product_id,
                "category_l1": record["category_l1"],
                "category_l2": category_l2,
                "source_title": record["title"],
                "source_attributes": record.get("attributes") or {},
                "used_attributes": reliable,
                "cleaning_actions": value_actions + title_actions,
                "targets": {
                    "title": title_target,
                    "selling_points": selling_points,
                    "short_description": short_description,
                },
                "quality_status": "PASS_RULE_BASED_SPOT_CHECK_PENDING",
                "product_signature_sha256": signature,
            }
        )

    assignments = assign_product_splits(accepted, config)
    for product in accepted:
        product["split"] = assignments[product["product_id"]]
    accepted.sort(
        key=lambda product: (
            product["split"],
            product["category_l2"],
            product["product_id"],
        )
    )
    instructions = [
        build_instruction_record(product, task_type)
        for product in accepted
        for task_type in TASK_TYPES
    ]
    return accepted, instructions, excluded


def select_spot_check_products(
    products: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    sample_config = config["spot_check"]
    namespace = str(sample_config["namespace"])
    seed = int(sample_config["seed"])
    per_category = int(sample_config["products_per_category"])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        by_category[product["category_l2"]].append(product)
    selected: list[dict[str, Any]] = []
    for category, category_products in sorted(by_category.items()):
        ranked = sorted(
            category_products,
            key=lambda product: _rank(
                namespace, seed, category, product["product_id"]
            ),
        )
        selected.extend(ranked[:per_category])
    return selected


def summarize_dataset(
    products: list[dict[str, Any]],
    instructions: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> dict[str, Any]:
    product_split_counts = Counter(product["split"] for product in products)
    product_category_counts = Counter(product["category_l2"] for product in products)
    task_counts = Counter(row["task_type"] for row in instructions)
    split_task_counts = Counter(
        (row["split"], row["task_type"]) for row in instructions
    )
    exclusion_counts = Counter(row["reason"] for row in excluded)
    exclusion_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in excluded:
        exclusion_category_counts[row["category_l2"]][row["reason"]] += 1
    cleaning_counts: Counter[str] = Counter()
    for product in products:
        for action in product["cleaning_actions"]:
            cleaning_counts[action.split(":", 1)[0]] += 1
    return {
        "usable_product_count": len(products),
        "instruction_count": len(instructions),
        "product_split_counts": dict(sorted(product_split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "split_task_counts": {
            f"{split}:{task}": count
            for (split, task), count in sorted(split_task_counts.items())
        },
        "product_category_counts": dict(sorted(product_category_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "exclusion_category_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(exclusion_category_counts.items())
        },
        "cleaning_action_type_counts": dict(sorted(cleaning_counts.items())),
    }


def validate_dataset(
    products: list[dict[str, Any]],
    instructions: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    product_ids = [product["product_id"] for product in products]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("可用商品存在重复 product_id。")
    excluded_ids = [row["product_id"] for row in excluded]
    if set(product_ids).intersection(excluded_ids):
        raise ValueError("可用商品与过滤商品存在交集。")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instructions:
        grouped[row["product_id"]].append(row)
        source_title = row["traceability"]["source_title"]
        if source_title and source_title in row["instruction"]:
            raise ValueError(f"原标题泄漏到模型输入：{row['product_id']}")
        if row["task_type"] == "selling_points":
            if not isinstance(row["target"], list) or len(row["target"]) != 3:
                raise ValueError(f"卖点数量错误：{row['product_id']}")
    if set(grouped) != set(product_ids):
        raise ValueError("商品与指令记录集合不一致。")
    for product_id, rows in grouped.items():
        if {row["task_type"] for row in rows} != set(TASK_TYPES):
            raise ValueError(f"商品缺少任务：{product_id}")
        if len({row["split"] for row in rows}) != 1:
            raise ValueError(f"同一商品跨 split：{product_id}")
