"""仅使用 validation 人工候选池选择规则重排参数。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.evaluation import RelevanceJudgment, evaluate_judgments  # noqa: E402
from src.retrieval.rerank import rerank_candidates  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_rerank_validation_v1.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    for query_id, candidates in grouped.items():
        ranks = sorted(int(row["candidate_rank"]) for row in candidates)
        if ranks != list(range(1, len(candidates) + 1)):
            raise ValueError(f"查询 {query_id} 的候选排名不连续。")
    return grouped


def metrics_for_rows(
    rows_by_query: dict[str, list[dict[str, Any]]], cutoff: int, threshold: int
) -> dict[str, Any]:
    judgments = [
        RelevanceJudgment(
            query_id=query_id,
            rank=int(row["candidate_rank"]),
            product_id=str(row["product_id"]),
            grade=int(row["relevance_grade"]),
        )
        for query_id, rows in rows_by_query.items()
        for row in rows
    ]
    return evaluate_judgments(judgments, cutoff=cutoff, positive_grade_threshold=threshold)


def run_trial(
    rows_by_query: dict[str, list[dict[str, str]]],
    policy: dict[str, Any],
    *,
    depth: int,
    attribute_weight: float,
    conflict_weight: float,
    cutoff: int,
    threshold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reranked = {
        query_id: rerank_candidates(
            "text",
            candidates,
            policy,
            query_text=candidates[0]["query_text"],
            candidate_pool_depth=depth,
            attribute_weight=attribute_weight,
            conflict_weight=conflict_weight,
        )
        for query_id, candidates in rows_by_query.items()
    }
    overall = metrics_for_rows(reranked, cutoff, threshold)
    three_c_ids = {
        query_id
        for query_id, candidates in rows_by_query.items()
        if candidates[0]["expected_category_l2"] in {"耳机", "键盘", "鼠标", "移动电源"}
    }
    three_c = metrics_for_rows(
        {query_id: reranked[query_id] for query_id in three_c_ids}, cutoff, threshold
    )
    return overall, three_c


def selection_key(trial: dict[str, Any]) -> tuple[float, ...]:
    macro = trial["overall"]["macro_average"]
    return (
        float(macro["precision_at_10"]),
        float(macro["pooled_recall_at_10"]),
        float(macro["ndcg_at_10"]),
        float(macro["mrr_at_10"]),
        -float(trial["attribute_weight"] + trial["conflict_weight"]),
        -float(trial["candidate_pool_depth"]),
    )


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_count": metrics["query_count"],
        "macro_average": metrics["macro_average"],
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    policy = load_json(config_path)
    annotation_path = resolve(policy["validation_annotation_pool"])
    queries_path = resolve(policy["validation_queries"])
    if file_sha256(annotation_path) != policy["validation_annotation_sha256"]:
        raise ValueError("validation 人工标注文件 SHA256 不一致。")
    if file_sha256(queries_path) != policy["validation_queries_sha256"]:
        raise ValueError("validation 查询文件 SHA256 不一致。")

    rows_by_query = group_rows(load_rows(annotation_path))
    cutoff = int(policy["metric_cutoff"])
    threshold = int(policy["positive_grade_threshold"])
    baseline = metrics_for_rows(rows_by_query, cutoff, threshold)
    three_c_query_ids = {
        query_id
        for query_id, candidates in rows_by_query.items()
        if candidates[0]["expected_category_l2"] in {"耳机", "键盘", "鼠标", "移动电源"}
    }
    baseline_three_c = metrics_for_rows(
        {query_id: rows_by_query[query_id] for query_id in three_c_query_ids},
        cutoff,
        threshold,
    )
    trials = []
    for depth, attribute_weight, conflict_weight in itertools.product(
        policy["candidate_pool_depths"],
        policy["attribute_weights"],
        policy["conflict_weights"],
    ):
        overall, three_c = run_trial(
            rows_by_query,
            policy,
            depth=int(depth),
            attribute_weight=float(attribute_weight),
            conflict_weight=float(conflict_weight),
            cutoff=cutoff,
            threshold=threshold,
        )
        trials.append(
            {
                "candidate_pool_depth": int(depth),
                "attribute_weight": float(attribute_weight),
                "conflict_weight": float(conflict_weight),
                "overall": compact_metrics(overall),
                "three_c": compact_metrics(three_c),
            }
        )
    selected = max(trials, key=selection_key)
    selected_overall, selected_three_c = run_trial(
        rows_by_query,
        policy,
        depth=selected["candidate_pool_depth"],
        attribute_weight=selected["attribute_weight"],
        conflict_weight=selected["conflict_weight"],
        cutoff=cutoff,
        threshold=threshold,
    )
    selected = {
        **selected,
        "overall": selected_overall,
        "three_c": selected_three_c,
    }
    report = {
        "status": "validation_parameter_selection_completed",
        "version": policy["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test50_executed": False,
        "query_mode_policy": policy["query_mode_policy"],
        "formula": (
            "final_score = similarity_score * category_factor "
            "+ attribute_weight * matched_constraint_ratio "
            "- conflict_weight * conflicted_constraint_ratio"
        ),
        "category_weights": policy["category_weights"],
        "validation_inputs": {
            "annotation_path": policy["validation_annotation_pool"],
            "annotation_sha256": file_sha256(annotation_path),
            "queries_path": policy["validation_queries"],
            "queries_sha256": file_sha256(queries_path),
            "query_count": len(rows_by_query),
            "annotated_pool_depth": min(len(rows) for rows in rows_by_query.values()),
        },
        "selection_order": policy["selection_order"],
        "baseline": compact_metrics(baseline),
        "baseline_three_c": compact_metrics(baseline_three_c),
        "trials": trials,
        "selected": selected,
        "limitations": [
            "validation 只有文本查询的 Top-20 人工候选池，因此候选深度只在15和20之间比较。",
            "Image→Text 没有真实图片侧品类或属性信号，正式首版保持 Baseline 排序。",
            "validation pooled recall 的分母来自已标注 Top-20，不代表全库完整 Recall。"
        ],
        "config_sha256": file_sha256(config_path),
    }
    output_path = resolve(policy["output_path"])
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
