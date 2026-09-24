"""可解释的跨模态检索规则重排。"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable


def normalize_text(value: Any) -> str:
    """规范化检索规则使用的文本，不改变数字和单位语义。"""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("毫安时", "mah").replace("毫安", "mah")
    return re.sub(r"\s+", "", text)


def candidate_document(candidate: dict[str, Any]) -> str:
    """只拼接推理阶段真实可用的候选商品标题和结构化属性。"""
    attributes = candidate.get("candidate_attributes", candidate.get("attributes", {}))
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    parts = [candidate.get("title", "")]
    for field, values in attributes.items():
        parts.append(field)
        if isinstance(values, list):
            parts.extend(values)
        else:
            parts.append(values)
    return normalize_text(" ".join(str(part) for part in parts))


def parse_query_signals(query_text: str, policy: dict[str, Any]) -> dict[str, Any]:
    """仅从真实查询文本解析品类、语义属性和带单位数字。"""
    normalized = normalize_text(query_text)
    category_l2 = None
    category_l1 = None
    matched_category_alias = None
    for candidate_category, aliases in policy["category_aliases"].items():
        matches = [normalize_text(alias) for alias in aliases if normalize_text(alias) in normalized]
        if not matches:
            continue
        longest = max(matches, key=len)
        if matched_category_alias is None or len(longest) > len(matched_category_alias):
            category_l2 = candidate_category
            category_l1 = policy["category_l1_by_l2"][candidate_category]
            matched_category_alias = longest

    concepts = []
    for concept in policy["attribute_concepts"]:
        categories = concept.get("categories")
        if categories and category_l2 not in categories:
            continue
        query_aliases = concept.get("query_aliases", concept["support_aliases"])
        if any(normalize_text(alias) in normalized for alias in query_aliases):
            concepts.append(concept)

    numeric_constraints = []
    for value, unit in re.findall(
        r"(?<!\d)(\d+(?:\.\d+)?)(mah|毫安时|毫安|dpi|ml|l|w|米|m)(?![a-z])",
        normalized,
        flags=re.IGNORECASE,
    ):
        canonical_unit = "mah" if unit in {"毫安时", "毫安"} else unit.casefold()
        token = f"{value}{canonical_unit}"
        if token not in numeric_constraints:
            numeric_constraints.append(token)
    return {
        "category_l1": category_l1,
        "category_l2": category_l2,
        "category_alias": matched_category_alias,
        "concepts": concepts,
        "numeric_constraints": numeric_constraints,
    }


def category_factor(
    signals: dict[str, Any], candidate: dict[str, Any], policy: dict[str, Any]
) -> tuple[float, str]:
    """按 PRD 起点返回二级相同、一级相同或跨一级品类系数。"""
    weights = policy["category_weights"]
    if signals["category_l2"] is None:
        return 1.0, "query_category_not_parsed"
    if candidate["candidate_category_l2"] == signals["category_l2"]:
        return float(weights["exact_l2"]), "exact_l2"
    candidate_l1 = candidate.get("candidate_category_l1") or policy[
        "category_l1_by_l2"
    ].get(candidate["candidate_category_l2"])
    if candidate_l1 == signals["category_l1"]:
        return float(weights["same_l1"]), "same_l1"
    return float(weights["cross_l1"]), "cross_l1"


def attribute_evidence(
    signals: dict[str, Any], document: str
) -> tuple[list[str], list[str], list[str]]:
    """返回命中、冲突和未判定的查询约束。"""
    matched: list[str] = []
    conflicted: list[str] = []
    unknown: list[str] = []
    for concept in signals["concepts"]:
        support = any(normalize_text(alias) in document for alias in concept["support_aliases"])
        conflict = any(
            normalize_text(alias) in document for alias in concept.get("conflict_aliases", [])
        )
        if conflict:
            conflicted.append(concept["id"])
        elif support:
            matched.append(concept["id"])
        else:
            unknown.append(concept["id"])
    for constraint in signals["numeric_constraints"]:
        if constraint in document:
            matched.append(f"numeric:{constraint}")
        else:
            unknown.append(f"numeric:{constraint}")
    return matched, conflicted, unknown


def rerank_text_candidates(
    query_text: str,
    candidates: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    *,
    candidate_pool_depth: int,
    attribute_weight: float,
    conflict_weight: float,
) -> list[dict[str, Any]]:
    """在大于 Top-10 的候选池中重排；池外候选保持原始顺序。"""
    if candidate_pool_depth <= 10:
        raise ValueError("candidate_pool_depth 必须大于10。")
    ordered = sorted(candidates, key=lambda row: int(row["candidate_rank"]))
    if len(ordered) < candidate_pool_depth:
        raise ValueError("候选数量小于 candidate_pool_depth。")
    signals = parse_query_signals(query_text, policy)
    rerank_pool = []
    for row in ordered[:candidate_pool_depth]:
        document = candidate_document(row)
        matched, conflicted, unknown = attribute_evidence(signals, document)
        constraint_count = len(matched) + len(conflicted) + len(unknown)
        match_ratio = len(matched) / constraint_count if constraint_count else 0.0
        conflict_ratio = len(conflicted) / constraint_count if constraint_count else 0.0
        factor, category_reason = category_factor(signals, row, policy)
        similarity = float(row["similarity_score"])
        final_score = (
            similarity * factor
            + float(attribute_weight) * match_ratio
            - float(conflict_weight) * conflict_ratio
        )
        rerank_pool.append(
            {
                **row,
                "original_rank": int(row["candidate_rank"]),
                "category_factor": factor,
                "category_reason": category_reason,
                "matched_signals": matched,
                "conflicted_signals": conflicted,
                "unknown_signals": unknown,
                "attribute_match_ratio": match_ratio,
                "attribute_conflict_ratio": conflict_ratio,
                "rerank_score": final_score,
            }
        )
    rerank_pool.sort(
        key=lambda row: (-row["rerank_score"], row["original_rank"], str(row["product_id"]))
    )
    tail = [
        {
            **row,
            "original_rank": int(row["candidate_rank"]),
            "category_factor": None,
            "category_reason": "outside_rerank_pool",
            "matched_signals": [],
            "conflicted_signals": [],
            "unknown_signals": [],
            "attribute_match_ratio": None,
            "attribute_conflict_ratio": None,
            "rerank_score": None,
        }
        for row in ordered[candidate_pool_depth:]
    ]
    return [
        {**row, "candidate_rank": rank}
        for rank, row in enumerate([*rerank_pool, *tail], start=1)
    ]


def rerank_candidates(
    query_mode: str,
    candidates: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    *,
    query_text: str | None,
    candidate_pool_depth: int,
    attribute_weight: float,
    conflict_weight: float,
) -> list[dict[str, Any]]:
    """图片查询无独立图片侧信号时严格保持 Baseline 排序。"""
    ordered = sorted(candidates, key=lambda row: int(row["candidate_rank"]))
    if query_mode == "image":
        return [
            {
                **row,
                "original_rank": int(row["candidate_rank"]),
                "rerank_score": None,
                "rerank_policy": "baseline_passthrough_no_image_query_signals",
            }
            for row in ordered
        ]
    if query_mode != "text" or query_text is None:
        raise ValueError("text 模式必须提供真实 query_text。")
    return rerank_text_candidates(
        query_text,
        ordered,
        policy,
        candidate_pool_depth=candidate_pool_depth,
        attribute_weight=attribute_weight,
        conflict_weight=conflict_weight,
    )
