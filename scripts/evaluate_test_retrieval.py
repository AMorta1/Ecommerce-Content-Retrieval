"""Evaluate text and image retrieval on 50 frozen-test queries after human review."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import load_catalog, load_metadata, resolve_project_path  # noqa: E402
from src.retrieval.ernie_vil import ErnieVilEmbedder  # noqa: E402
from src.retrieval.evaluation import (  # noqa: E402
    annotation_progress,
    count_assistant_draft_rows,
    load_annotation_rows,
)
from src.retrieval.faiss_index import load_index, search  # noqa: E402
from src.retrieval.feature_library import catalog_index_path  # noqa: E402
from src.retrieval.formal_evaluation import (  # noqa: E402
    evaluate_complete_rankings,
    parse_complete_relevance,
    validate_query_review,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_test_evaluation.json"
RANKING_FIELDS = [
    "query_mode",
    "query_id",
    "query_product_id",
    "query_category_l1",
    "query_category_l2",
    "rank",
    "source_index_rank",
    "similarity_score",
    "product_id",
    "title",
    "candidate_category_l1",
    "candidate_category_l2",
    "relevance_grade",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="正式评测配置")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_images(
    embedder: ErnieVilEmbedder,
    queries: list[dict[str, Any]],
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        paths = [resolve_project_path(PROJECT_ROOT, query["query_image_path"]) for query in batch]
        batches.append(embedder.encode_images(paths).astype(np.float32, copy=False))
    return np.concatenate(batches, axis=0)


def filter_gallery_results(
    scores: np.ndarray,
    indexes: np.ndarray,
    metadata: list[dict[str, Any]],
    query: dict[str, Any],
    gallery_split: str,
    cutoff: int,
    relevance: dict[str, int],
    query_mode: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Keep test products, remove the query itself, and return the first K."""
    ranked_ids = []
    rows = []
    for source_rank, (score, metadata_index) in enumerate(zip(scores, indexes), start=1):
        product = metadata[int(metadata_index)]
        if product["split"] != gallery_split or product["product_id"] == query["product_id"]:
            continue
        ranked_ids.append(product["product_id"])
        rows.append(
            {
                "query_mode": query_mode,
                "query_id": query["query_id"],
                "query_product_id": query["product_id"],
                "query_category_l1": query["category_l1"],
                "query_category_l2": query["category_l2"],
                "rank": len(ranked_ids),
                "source_index_rank": source_rank,
                "similarity_score": round(float(score), 6),
                "product_id": product["product_id"],
                "title": product["title"],
                "candidate_category_l1": product["category_l1"],
                "candidate_category_l2": product["category_l2"],
                "relevance_grade": relevance.get(product["product_id"], 0),
            }
        )
        if len(ranked_ids) == cutoff:
            break
    if len(ranked_ids) < cutoff:
        raise ValueError(f"查询 {query['query_id']} 在 {gallery_split} 中不足 {cutoff} 个结果。")
    return ranked_ids, rows


def evaluate_breakdowns(
    rankings: dict[str, list[str]],
    relevance: dict[str, dict[str, int]],
    queries: list[dict[str, Any]],
    cutoff: int,
    threshold: int,
) -> dict[str, Any]:
    query_by_id = {query["query_id"]: query for query in queries}
    result = {
        "overall": evaluate_complete_rankings(rankings, relevance, cutoff, threshold),
        "by_category_l1": {},
        "by_category_l2": {},
    }
    for field in ("category_l1", "category_l2"):
        values = sorted({query[field] for query in queries})
        destination = result[f"by_{field}"]
        for value in values:
            query_ids = {
                query_id for query_id, query in query_by_id.items() if query[field] == value
            }
            destination[value] = evaluate_complete_rankings(
                {query_id: rankings[query_id] for query_id in query_ids},
                {query_id: relevance[query_id] for query_id in query_ids},
                cutoff,
                threshold,
            )["macro_average"]
    return result


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


def write_rankings(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=RANKING_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    retrieval_config_path = resolve_project_path(PROJECT_ROOT, config["retrieval_config"])
    retrieval_config = load_json(retrieval_config_path)
    queries_path = resolve_project_path(PROJECT_ROOT, config["queries_path"])
    annotation_path = resolve_project_path(PROJECT_ROOT, config["annotation_path"])
    manifest_path = resolve_project_path(PROJECT_ROOT, config["manifest_path"])
    rankings_path = resolve_project_path(PROJECT_ROOT, config["rankings_path"])
    metrics_path = resolve_project_path(PROJECT_ROOT, config["metrics_path"])
    if rankings_path.exists() or metrics_path.exists():
        raise FileExistsError("正式排名或指标已经存在，拒绝覆盖。")

    manifest = load_json(manifest_path)
    if manifest["version"] != config["version"]:
        raise ValueError("正式评测配置版本与冻结清单不一致。")
    if file_sha256(queries_path) != manifest["queries_sha256"]:
        raise ValueError("正式查询文件发生变化。")
    queries = load_jsonl(queries_path)
    query_review_path = resolve_project_path(PROJECT_ROOT, config["query_review_path"])
    with query_review_path.open("r", encoding="utf-8-sig", newline="") as review_file:
        validate_query_review(list(csv.DictReader(review_file)), queries)
    if file_sha256(query_review_path) != manifest["query_review_sha256"]:
        raise ValueError("查询审核表已确认但冻结清单尚未更新；请运行 prepare_test_retrieval_evaluation.py --confirm-review。")
    rows = load_annotation_rows(annotation_path)
    progress = annotation_progress(rows)
    if progress["remaining_rows"]:
        raise ValueError(f"人工标注还剩 {progress['remaining_rows']} 行未完成。")
    assistant_draft_rows = count_assistant_draft_rows(rows)
    if assistant_draft_rows:
        raise ValueError(f"仍有 {assistant_draft_rows} 行 AI 初标，不能生成正式指标。")

    threshold = int(config["positive_grade_threshold"])
    relevance = parse_complete_relevance(rows, threshold)
    if {query["query_id"] for query in queries} != set(relevance):
        raise ValueError("查询清单与相关性标注不一致。")

    dataset_path = resolve_project_path(PROJECT_ROOT, retrieval_config["dataset_path"])
    catalog = load_catalog(dataset_path, PROJECT_ROOT)
    metadata = load_metadata(resolve_project_path(PROJECT_ROOT, retrieval_config["metadata_path"]))
    features_by_mode = config["catalog_feature_by_query_mode"]
    if config["retrieval_strategy"] == "pure_cross_modal" and features_by_mode != {
        "text": "image",
        "image": "text",
    }:
        raise ValueError("纯跨模态策略必须配置为文本查图片、图片查文本。")
    if features_by_mode != manifest["catalog_feature_by_query_mode"]:
        raise ValueError("查询模式与商品特征库的对应关系发生变化。")
    indexes_by_mode = {}
    for mode, catalog_feature in features_by_mode.items():
        index_path = resolve_project_path(
            PROJECT_ROOT,
            catalog_index_path(retrieval_config, catalog_feature),
        )
        expected_hash = manifest["index_sha256_by_query_mode"][mode]
        if file_sha256(index_path) != expected_hash:
            raise ValueError(f"{mode} 查询使用的商品索引发生变化。")
        index = load_index(index_path)
        if len(metadata) != int(index.ntotal) or len(catalog) != len(metadata):
            raise ValueError(f"{mode} 查询的商品、元数据和索引数量不一致。")
        indexes_by_mode[mode] = index

    started = time.perf_counter()
    embedder = ErnieVilEmbedder(
        model_name=retrieval_config["model_name"],
        device=retrieval_config["device"],
        batch_size=int(retrieval_config["batch_size"]),
    )
    text_features = embedder.encode_texts([query["query_text"] for query in queries])
    image_features = encode_images(embedder, queries, int(retrieval_config["batch_size"]))

    cutoff = int(config["metric_cutoff"])
    results_by_mode = {}
    ranking_rows = []
    for mode, features in (("text", text_features), ("image", image_features)):
        index = indexes_by_mode[mode]
        scores, neighbor_indexes = search(index, features, top_k=int(index.ntotal))
        rankings = {}
        for query_index, query in enumerate(queries):
            query_id = query["query_id"]
            ranked_ids, query_rows = filter_gallery_results(
                scores[query_index],
                neighbor_indexes[query_index],
                metadata,
                query,
                config["gallery_split"],
                cutoff,
                relevance[query_id],
                mode,
            )
            rankings[query_id] = ranked_ids
            ranking_rows.extend(query_rows)
        results_by_mode[mode] = evaluate_breakdowns(
            rankings,
            relevance,
            queries,
            cutoff,
            threshold,
        )

    write_rankings(rankings_path, ranking_rows)
    targets = config["prd_targets"]
    report = {
        "status": "human_evaluation_completed",
        "version": config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": retrieval_config["model_name"],
        "retrieval_strategy": config["retrieval_strategy"],
        "catalog_feature_by_query_mode": features_by_mode,
        "gallery_split": config["gallery_split"],
        "gallery_size": sum(row["split"] == config["gallery_split"] for row in metadata),
        "query_count": len(queries),
        "annotation_progress": progress,
        "annotation_provenance": {
            "human_reviewed_rows": len(rows),
            "assistant_draft_rows": 0,
        },
        "grade_counts": dict(sorted(Counter(int(row["relevance_grade"]) for row in rows).items())),
        "metric_scope": "test 同二级品类候选已完整标注，跨品类按标准品类视为不相关；查询商品自身已排除。",
        "metrics_by_query_mode": results_by_mode,
        "prd_target_reference": {
            mode: {
                "precision_at_10": {
                    "target": targets["precision_at_10"],
                    "observed": result["overall"]["macro_average"]["precision_at_10"],
                },
                "recall_at_10": {
                    "target": targets["recall_at_10"],
                    "observed": result["overall"]["macro_average"]["recall_at_10"],
                },
            }
            for mode, result in results_by_mode.items()
        },
        "timings_seconds": {"total_model_and_search": round(time.perf_counter() - started, 3)},
        "inputs": {
            "queries_sha256": file_sha256(queries_path),
            "annotation_sha256": file_sha256(annotation_path),
            "index_sha256_by_query_mode": manifest["index_sha256_by_query_mode"],
        },
    }
    atomic_write_json(metrics_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
