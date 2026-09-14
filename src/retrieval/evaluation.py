"""检索人工标注读取和候选池指标计算。"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from src.data.review import find_product_id_mismatches


ASSISTANT_DRAFT_PREFIX = "AI初标"


@dataclass(frozen=True)
class RelevanceJudgment:
    """一条查询对一个候选商品的人工相关性判断。"""

    query_id: str
    rank: int
    product_id: str
    grade: int


def load_annotation_rows(path: Path) -> list[dict[str, str]]:
    """读取 Excel 友好的 UTF-8 BOM CSV。"""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"标注文件为空：{path}")
    mismatches = find_product_id_mismatches(rows)
    if mismatches:
        row_number, current_id, expected_id = mismatches[0]
        raise ValueError(
            f"第 {row_number} 行商品编号为 {current_id!r}，但图片对应 {expected_id!r}。"
            "编号可能被 Excel 转成了科学计数法，请先运行 repair_csv.py。"
        )
    return rows


def annotation_progress(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """统计已填写标签的数量，不计算未完成的效果指标。"""
    completed = sum(bool(row.get("relevance_grade", "").strip()) for row in rows)
    return {
        "total_rows": len(rows),
        "completed_rows": completed,
        "remaining_rows": len(rows) - completed,
        "completion_rate": round(completed / len(rows), 6),
    }


def count_assistant_draft_rows(rows: Sequence[dict[str, str]]) -> int:
    """统计仍需人工确认的 AI 初标行，防止把辅助标签冒充人工结果。"""
    return sum(
        row.get("review_notes", "").strip().startswith(ASSISTANT_DRAFT_PREFIX)
        for row in rows
    )


def parse_completed_judgments(rows: Sequence[dict[str, str]]) -> list[RelevanceJudgment]:
    """将已完成 CSV 转为强类型标注；空值或非法标签会直接报错。"""
    required_fields = {"query_id", "candidate_rank", "product_id", "relevance_grade"}
    missing_fields = sorted(required_fields - rows[0].keys())
    if missing_fields:
        raise ValueError(f"标注文件缺少列：{missing_fields}")

    judgments: list[RelevanceJudgment] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        grade_text = row["relevance_grade"].strip()
        if not grade_text:
            raise ValueError(f"标注尚未完成：第 {row_number} 行 relevance_grade 为空。")
        if grade_text not in {"0", "1", "2"}:
            raise ValueError(f"第 {row_number} 行相关性只能填写 0、1 或 2，实际为 {grade_text!r}。")
        query_id = row["query_id"].strip()
        product_id = row["product_id"].strip()
        pair = (query_id, product_id)
        if pair in seen_pairs:
            raise ValueError(f"重复标注：查询 {query_id}、商品 {product_id}。")
        seen_pairs.add(pair)
        judgments.append(
            RelevanceJudgment(
                query_id=query_id,
                rank=int(row["candidate_rank"]),
                product_id=product_id,
                grade=int(grade_text),
            )
        )
    return judgments


def evaluate_judgments(
    judgments: Sequence[RelevanceJudgment],
    cutoff: int = 10,
    positive_grade_threshold: int = 1,
) -> dict[str, Any]:
    """计算宏平均 P@K、候选池 R@K、MRR@K 和分级 NDCG@K。"""
    if cutoff < 1:
        raise ValueError("cutoff 必须大于或等于 1。")
    grouped: dict[str, list[RelevanceJudgment]] = defaultdict(list)
    for judgment in judgments:
        grouped[judgment.query_id].append(judgment)
    if not grouped:
        raise ValueError("没有可评测的标注。")

    per_query = []
    for query_id in sorted(grouped):
        ranked = sorted(grouped[query_id], key=lambda item: item.rank)
        ranks = [item.rank for item in ranked]
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"查询 {query_id} 存在重复排名。")
        if ranks != list(range(1, len(ranked) + 1)):
            raise ValueError(f"查询 {query_id} 的候选排名必须从 1 开始且连续。")
        if len(ranked) < cutoff:
            raise ValueError(f"查询 {query_id} 只有 {len(ranked)} 个候选，不足以计算 @{cutoff}。")

        top_items = ranked[:cutoff]
        total_relevant = sum(item.grade >= positive_grade_threshold for item in ranked)
        if total_relevant == 0:
            raise ValueError(f"查询 {query_id} 的候选池没有相关商品，需要替换查询或扩充候选池。")
        relevant_at_cutoff = sum(item.grade >= positive_grade_threshold for item in top_items)
        reciprocal_rank = next(
            (1.0 / item.rank for item in top_items if item.grade >= positive_grade_threshold),
            0.0,
        )
        dcg = _discounted_cumulative_gain([item.grade for item in top_items])
        ideal_grades = sorted((item.grade for item in ranked), reverse=True)[:cutoff]
        ideal_dcg = _discounted_cumulative_gain(ideal_grades)
        per_query.append(
            {
                "query_id": query_id,
                f"precision_at_{cutoff}": relevant_at_cutoff / cutoff,
                f"pooled_recall_at_{cutoff}": relevant_at_cutoff / total_relevant,
                f"mrr_at_{cutoff}": reciprocal_rank,
                f"ndcg_at_{cutoff}": dcg / ideal_dcg,
                "relevant_in_pool": total_relevant,
                f"relevant_at_{cutoff}": relevant_at_cutoff,
                "grade_counts": dict(sorted(Counter(item.grade for item in ranked).items())),
            }
        )

    metric_names = [
        f"precision_at_{cutoff}",
        f"pooled_recall_at_{cutoff}",
        f"mrr_at_{cutoff}",
        f"ndcg_at_{cutoff}",
    ]
    macro_average = {
        metric_name: round(mean(float(item[metric_name]) for item in per_query), 6)
        for metric_name in metric_names
    }
    return {"query_count": len(per_query), "macro_average": macro_average, "per_query": per_query}


def _discounted_cumulative_gain(grades: Sequence[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))
