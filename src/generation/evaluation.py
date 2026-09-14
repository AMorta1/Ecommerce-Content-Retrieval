"""文案生成人工标注表读取与指标计算。"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any


MANUAL_FIELDS = (
    "matched_attribute_count",
    "fluency_pass",
    "factual_error_count",
    "category_style_pass",
)
ASSISTANT_DRAFT_PREFIX = "AI初标"


@dataclass(frozen=True)
class GenerationJudgment:
    product_id: str
    core_attribute_count: int
    matched_attribute_count: int
    fluency_pass: int
    factual_error_count: int
    category_style_pass: int


def load_annotation_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"标注表为空：{path}")
    missing = [field for field in MANUAL_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"标注表缺少人工字段：{missing}")
    return rows


def annotation_progress(rows: list[dict[str, str]]) -> dict[str, int]:
    completed = sum(all(row[field].strip() for field in MANUAL_FIELDS) for row in rows)
    return {
        "total_rows": len(rows),
        "completed_rows": completed,
        "remaining_rows": len(rows) - completed,
    }


def count_assistant_draft_rows(rows: list[dict[str, str]]) -> int:
    """统计仍需人工确认的 AI 初标行，防止生成正式人工评测结果。"""
    return sum(
        row.get("review_notes", "").strip().startswith(ASSISTANT_DRAFT_PREFIX)
        for row in rows
    )


def parse_binary(value: str, field: str, product_id: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"商品 {product_id} 的 {field} 必须填写0或1，当前为：{value!r}")
    return int(value)


def parse_nonnegative_integer(value: str, field: str, product_id: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"商品 {product_id} 的 {field} 必须填写整数。") from error
    if parsed < 0:
        raise ValueError(f"商品 {product_id} 的 {field} 不能小于0。")
    return parsed


def parse_judgments(rows: list[dict[str, str]]) -> list[GenerationJudgment]:
    judgments = []
    product_ids: set[str] = set()
    for row in rows:
        product_id = row["product_id"].strip()
        if not product_id or product_id in product_ids:
            raise ValueError(f"商品编号为空或重复：{product_id!r}")
        product_ids.add(product_id)
        core_count = parse_nonnegative_integer(
            row["core_attribute_count"], "core_attribute_count", product_id
        )
        matched_count = parse_nonnegative_integer(
            row["matched_attribute_count"], "matched_attribute_count", product_id
        )
        if core_count == 0 or matched_count > core_count:
            raise ValueError(
                f"商品 {product_id} 的属性命中数必须介于0和核心属性总数 {core_count} 之间。"
            )
        judgments.append(
            GenerationJudgment(
                product_id=product_id,
                core_attribute_count=core_count,
                matched_attribute_count=matched_count,
                fluency_pass=parse_binary(row["fluency_pass"], "fluency_pass", product_id),
                factual_error_count=parse_nonnegative_integer(
                    row["factual_error_count"], "factual_error_count", product_id
                ),
                category_style_pass=parse_binary(
                    row["category_style_pass"], "category_style_pass", product_id
                ),
            )
        )
    return judgments


def evaluate_judgments(judgments: list[GenerationJudgment]) -> dict[str, Any]:
    sample_count = len(judgments)
    matched_total = sum(item.matched_attribute_count for item in judgments)
    attribute_total = sum(item.core_attribute_count for item in judgments)
    factual_error_samples = sum(item.factual_error_count > 0 for item in judgments)
    return {
        "sample_count": sample_count,
        "core_attribute_hit_rate": matched_total / attribute_total,
        "matched_attribute_total": matched_total,
        "core_attribute_total": attribute_total,
        "fluency_pass_rate": sum(item.fluency_pass for item in judgments) / sample_count,
        "factual_error_sample_rate": factual_error_samples / sample_count,
        "average_factual_errors_per_sample": sum(
            item.factual_error_count for item in judgments
        )
        / sample_count,
        "category_style_pass_rate": sum(
            item.category_style_pass for item in judgments
        )
        / sample_count,
    }
