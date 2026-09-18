"""Balanced test-query selection and metrics with complete relevance sets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from statistics import mean
from typing import Any, Sequence


def select_balanced_queries(
    records: Sequence[dict[str, Any]],
    quota_by_category: dict[str, int],
    seed: int,
    excluded_product_ids: set[str],
    replacements: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced subset, then apply recorded same-category replacements."""
    grouped = defaultdict(list)
    for record in records:
        if record["product_id"] not in excluded_product_ids:
            grouped[record["category_l2"]].append(record)

    selected = []
    for category, quota in quota_by_category.items():
        if quota < 1:
            raise ValueError(f"{category} 的查询数量必须大于0。")
        candidates = grouped.get(category, [])
        if len(candidates) < quota:
            raise ValueError(f"{category} 只有 {len(candidates)} 条可选查询，不足 {quota} 条。")
        ranked = sorted(
            candidates,
            key=lambda record: hashlib.sha256(
                f"formal-query:{seed}:{record['product_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:quota])

    replacement_ids = replacements or {}
    if replacement_ids:
        records_by_id = {record["product_id"]: record for record in records}
        selected_ids = {record["product_id"] for record in selected}
        if not set(replacement_ids) <= selected_ids:
            raise ValueError("待替换商品必须属于原始抽样查询集。")
        if len(set(replacement_ids.values())) != len(replacement_ids):
            raise ValueError("替换商品编号不能重复。")
        for old_id, new_id in replacement_ids.items():
            if new_id not in records_by_id or new_id in excluded_product_ids:
                raise ValueError(f"替换商品 {new_id} 不在可用测试库中。")
            if new_id in selected_ids:
                raise ValueError(f"替换商品 {new_id} 已在原始抽样查询集中。")
            if records_by_id[old_id]["category_l2"] != records_by_id[new_id]["category_l2"]:
                raise ValueError(f"替换商品 {new_id} 与原商品 {old_id} 二级品类不一致。")
        updated_selection = []
        for record in selected:
            replacement_id = replacement_ids.get(record["product_id"])
            updated_selection.append(records_by_id[replacement_id] if replacement_id else record)
        selected = updated_selection
    return selected


def build_relevance_rows(
    queries: Sequence[dict[str, Any]],
    gallery: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build exhaustive same-category judgments, excluding each query product itself."""
    candidates_by_category = defaultdict(list)
    for product in gallery:
        candidates_by_category[product["category_l2"]].append(product)

    rows = []
    for query in queries:
        candidates = sorted(
            candidates_by_category[query["category_l2"]],
            key=lambda record: int(record["product_id"]),
        )
        order = 0
        for candidate in candidates:
            if candidate["product_id"] == query["product_id"]:
                continue
            order += 1
            rows.append(
                {
                    "query_id": f"test_{query['product_id']}",
                    "query_category_l1": query["category_l1"],
                    "query_category_l2": query["category_l2"],
                    "query_text": query["query_text"],
                    "query_source_title": query["title"],
                    "query_attributes": query["attributes"],
                    "query_image_path": query["image_path"],
                    "candidate_order": order,
                    "product_id": candidate["product_id"],
                    "title": candidate["title"],
                    "candidate_attributes": candidate["attributes"],
                    "image_path": candidate["image_path"],
                    "relevance_grade": "",
                    "review_notes": "",
                }
            )
    return rows


def validate_query_review(
    review_rows: Sequence[dict[str, str]],
    queries: Sequence[dict[str, Any]],
) -> None:
    """Require human approval and matching text for every frozen test query."""
    reviewed_by_id = {row["query_id"]: row for row in review_rows}
    if len(review_rows) != len(reviewed_by_id) or set(reviewed_by_id) != {
        query["query_id"] for query in queries
    }:
        raise ValueError("查询审核表与正式查询清单不一致。")
    pending = [row["query_id"] for row in review_rows if row["review_query_ok"].strip() != "1"]
    if pending:
        raise ValueError(f"仍有 {len(pending)} 条查询未完成人工确认。")
    for query in queries:
        reviewed = reviewed_by_id[query["query_id"]]
        if reviewed["query_text"].strip() != query["query_text"].strip():
            raise ValueError(f"查询 {query['query_id']} 的审核文本与正式查询不一致。")


def parse_complete_relevance(
    rows: Sequence[dict[str, str]],
    positive_grade_threshold: int,
) -> dict[str, dict[str, int]]:
    """Validate complete 0/1/2 judgments and return grades by query and product."""
    required = {"query_id", "product_id", "candidate_order", "relevance_grade"}
    missing = sorted(required - rows[0].keys()) if rows else sorted(required)
    if missing:
        raise ValueError(f"相关性标注缺少列：{missing}")

    grouped = defaultdict(dict)
    orders = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        grade_text = row["relevance_grade"].strip()
        if grade_text not in {"0", "1", "2"}:
            raise ValueError(f"第 {row_number} 行相关性只能填写 0、1 或 2。")
        query_id = row["query_id"].strip()
        product_id = row["product_id"].strip()
        if product_id in grouped[query_id]:
            raise ValueError(f"查询 {query_id} 重复标注商品 {product_id}。")
        grouped[query_id][product_id] = int(grade_text)
        orders[query_id].append(int(row["candidate_order"]))

    for query_id, query_orders in orders.items():
        if sorted(query_orders) != list(range(1, len(query_orders) + 1)):
            raise ValueError(f"查询 {query_id} 的 candidate_order 必须从1开始连续。")
        if not any(grade >= positive_grade_threshold for grade in grouped[query_id].values()):
            raise ValueError(f"查询 {query_id} 没有相关商品，需要替换该查询。")
    return dict(grouped)


def evaluate_complete_rankings(
    rankings: dict[str, list[str]],
    relevance: dict[str, dict[str, int]],
    cutoff: int,
    positive_grade_threshold: int,
) -> dict[str, Any]:
    """Calculate exact P/R against exhaustively judged same-category test candidates."""
    if cutoff < 1:
        raise ValueError("cutoff 必须大于0。")
    if set(rankings) != set(relevance):
        raise ValueError("检索查询与相关性标注的查询集合不一致。")

    per_query = []
    for query_id in sorted(rankings):
        ranked_ids = rankings[query_id]
        if len(ranked_ids) < cutoff or len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"查询 {query_id} 的检索结果不足或包含重复商品。")
        grades = relevance[query_id]
        total_relevant = sum(grade >= positive_grade_threshold for grade in grades.values())
        top_ids = ranked_ids[:cutoff]
        top_grades = [grades.get(product_id, 0) for product_id in top_ids]
        relevant_at_cutoff = sum(grade >= positive_grade_threshold for grade in top_grades)
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, grade in enumerate(top_grades, start=1)
                if grade >= positive_grade_threshold
            ),
            0.0,
        )
        ideal_grades = sorted(grades.values(), reverse=True)[:cutoff]
        ideal_dcg = discounted_cumulative_gain(ideal_grades)
        per_query.append(
            {
                "query_id": query_id,
                f"precision_at_{cutoff}": relevant_at_cutoff / cutoff,
                f"recall_at_{cutoff}": relevant_at_cutoff / total_relevant,
                f"mrr_at_{cutoff}": reciprocal_rank,
                f"ndcg_at_{cutoff}": discounted_cumulative_gain(top_grades) / ideal_dcg,
                "total_relevant": total_relevant,
                f"relevant_at_{cutoff}": relevant_at_cutoff,
                "retrieved_category_grade_counts": dict(sorted(Counter(top_grades).items())),
            }
        )

    metric_names = (
        f"precision_at_{cutoff}",
        f"recall_at_{cutoff}",
        f"mrr_at_{cutoff}",
        f"ndcg_at_{cutoff}",
    )
    macro = {
        metric: round(mean(float(row[metric]) for row in per_query), 6)
        for metric in metric_names
    }
    return {"query_count": len(per_query), "macro_average": macro, "per_query": per_query}


def discounted_cumulative_gain(grades: Sequence[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )
