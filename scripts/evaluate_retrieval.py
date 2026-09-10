"""读取完成的人工相关性标注，并计算检索指标。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import resolve_project_path  # noqa: E402
from src.retrieval.evaluation import (  # noqa: E402
    annotation_progress,
    count_assistant_draft_rows,
    evaluate_judgments,
    load_annotation_rows,
    parse_completed_judgments,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_evaluation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="评测配置 JSON")
    parser.add_argument(
        "--allow-assistant-draft",
        action="store_true",
        help="允许计算 AI 初标的临时指标；结果不能作为正式人工评测",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    annotation_path = resolve_project_path(PROJECT_ROOT, config["annotation_path"])
    rows = load_annotation_rows(annotation_path)
    progress = annotation_progress(rows)
    if progress["remaining_rows"]:
        print(json.dumps({"status": "annotation_incomplete", **progress}, ensure_ascii=False, indent=2))
        raise ValueError("人工标注尚未完成；请填完 relevance_grade 列后再计算指标。")

    assistant_draft_rows = count_assistant_draft_rows(rows)
    if assistant_draft_rows and not args.allow_assistant_draft:
        print(
            json.dumps(
                {
                    "status": "assistant_draft_awaiting_human_review",
                    "assistant_draft_rows": assistant_draft_rows,
                    "human_reviewed_rows": len(rows) - assistant_draft_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise ValueError("检测到 AI 初标；人工复核并移除 AI初标 标记后才能生成正式指标。")

    judgments = parse_completed_judgments(rows)
    cutoff = int(config["metric_cutoff"])
    metrics = evaluate_judgments(
        judgments,
        cutoff=cutoff,
        positive_grade_threshold=int(config["positive_grade_threshold"]),
    )
    macro = metrics["macro_average"]
    targets = config["prd_targets"]
    precision_key = f"precision_at_{cutoff}"
    recall_key = f"pooled_recall_at_{cutoff}"
    report = {
        "status": (
            "assistant_draft_pooled_evaluation_completed"
            if assistant_draft_rows
            else "pooled_evaluation_completed"
        ),
        "version": config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotation_file_sha256": file_sha256(annotation_path),
        "annotation_progress": progress,
        "grade_counts": dict(sorted(Counter(judgment.grade for judgment in judgments).items())),
        "annotation_provenance": {
            "human_reviewed_rows": len(rows) - assistant_draft_rows,
            "assistant_draft_rows": assistant_draft_rows,
        },
        "metric_scope": (
            "AI 辅助初标候选池，仅供初步分析；Recall 分母不是全库所有潜在相关商品"
            if assistant_draft_rows
            else "人工标注候选池；Recall 分母不是全库所有潜在相关商品"
        ),
        "metrics": metrics,
        "prd_target_reference": {
            "precision_at_10": {
                "target": targets["precision_at_10"],
                "observed": macro[precision_key],
                "passed_within_pool": macro[precision_key] >= targets["precision_at_10"],
            },
            "recall_at_10": {
                "target": targets["recall_at_10"],
                "observed_pooled_recall": macro[recall_key],
                "passed_within_pool": macro[recall_key] >= targets["recall_at_10"],
            },
        },
        "warning": (
            "该结果含 AI 初标，不能作为正式人工评测或 PRD 达标结论。"
            if assistant_draft_rows
            else "候选池指标适合比较同一标注版本下的方案，不应直接宣称覆盖全库的正式 Recall。"
        ),
    }
    output_path = resolve_project_path(PROJECT_ROOT, config["metrics_path"])
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
