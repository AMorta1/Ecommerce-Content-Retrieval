"""运行检索基线，并生成供人工判断相关性的 CSV 候选池。"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import load_catalog, load_metadata, resolve_project_path  # noqa: E402
from src.retrieval.ernie_vil import ErnieVilEmbedder  # noqa: E402
from src.retrieval.faiss_index import load_index, search  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_evaluation.json"
ANNOTATION_FIELDS = [
    "query_id",
    "query_text",
    "query_intent",
    "expected_category_l2",
    "difficulty",
    "candidate_rank",
    "source_index_rank",
    "similarity_score",
    "product_id",
    "title",
    "candidate_category_l2",
    "candidate_raw_label",
    "candidate_attributes",
    "image_path",
    "relevance_grade",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="评测配置 JSON")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_queries(path: Path) -> list[dict[str, Any]]:
    required_fields = {
        "query_id",
        "query_text",
        "category_l2",
        "intent",
        "difficulty",
        "judge_attribute_keys",
    }
    queries = []
    query_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            query = json.loads(line)
            missing_fields = sorted(required_fields - query.keys())
            if missing_fields:
                raise ValueError(f"查询文件第 {line_number} 行缺少字段：{missing_fields}")
            if query["query_id"] in query_ids:
                raise ValueError(f"查询编号重复：{query['query_id']}")
            query_ids.add(query["query_id"])
            queries.append(query)
    if not queries:
        raise ValueError(f"查询文件为空：{path}")
    return queries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_candidate_attributes(record: dict[str, Any], keys: list[str]) -> str:
    selected = {key: record["attributes"][key] for key in keys if key in record["attributes"]}
    return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


def build_annotation_rows(
    queries: list[dict[str, Any]],
    scores: Any,
    indexes: Any,
    metadata: list[dict[str, Any]],
    catalog_by_id: dict[str, dict[str, Any]],
    gallery_split: str,
    pool_depth: int,
) -> list[dict[str, Any]]:
    """从全量索引结果中筛出指定数据划分，并保留每条查询前 N 个候选。"""
    rows = []
    for query_index, query in enumerate(queries):
        gallery_rank = 0
        for source_rank, (score, metadata_index) in enumerate(
            zip(scores[query_index], indexes[query_index]), start=1
        ):
            product_metadata = metadata[int(metadata_index)]
            if product_metadata["split"] != gallery_split:
                continue
            gallery_rank += 1
            product = catalog_by_id[product_metadata["product_id"]]
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query_text": query["query_text"],
                    "query_intent": query["intent"],
                    "expected_category_l2": query["category_l2"],
                    "difficulty": query["difficulty"],
                    "candidate_rank": gallery_rank,
                    "source_index_rank": source_rank,
                    "similarity_score": round(float(score), 6),
                    "product_id": product_metadata["product_id"],
                    "title": product_metadata["title"],
                    "candidate_category_l2": product_metadata["category_l2"],
                    "candidate_raw_label": product["raw_label"],
                    "candidate_attributes": select_candidate_attributes(
                        product, query["judge_attribute_keys"]
                    ),
                    "image_path": product_metadata["image_path"],
                    "relevance_grade": "",
                    "review_notes": "",
                }
            )
            if gallery_rank == pool_depth:
                break
        if gallery_rank < pool_depth:
            raise ValueError(
                f"查询 {query['query_id']} 在 {gallery_split} 中只有 {gallery_rank} 个候选，"
                f"不足 {pool_depth} 个。"
            )
    return rows


def write_outputs(
    annotation_path: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """使用临时文件写入，避免中断后留下半份标注表。"""
    if annotation_path.exists() or manifest_path.exists():
        raise FileExistsError("标注表或清单已经存在。为保护人工标签，请修改版本和输出路径后再生成。")
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_temporary = annotation_path.with_name(annotation_path.name + ".tmp")
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        with annotation_temporary.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with manifest_temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(manifest, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(annotation_temporary, annotation_path)
        os.replace(manifest_temporary, manifest_path)
    finally:
        annotation_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    evaluation_config_path = args.config.resolve()
    evaluation_config = load_json(evaluation_config_path)
    retrieval_config_path = resolve_project_path(PROJECT_ROOT, evaluation_config["retrieval_config"])
    retrieval_config = load_json(retrieval_config_path)
    queries_path = resolve_project_path(PROJECT_ROOT, evaluation_config["queries_path"])
    annotation_path = resolve_project_path(PROJECT_ROOT, evaluation_config["annotation_path"])
    manifest_path = resolve_project_path(PROJECT_ROOT, evaluation_config["manifest_path"])
    if annotation_path.exists() or manifest_path.exists():
        raise FileExistsError("当前版本的标注表已经存在，不会覆盖可能填写过的人工标签。")

    pool_depth = int(evaluation_config["pool_depth"])
    if pool_depth < int(evaluation_config["metric_cutoff"]):
        raise ValueError("pool_depth 不能小于 metric_cutoff。")
    queries = load_queries(queries_path)

    dataset_path = resolve_project_path(PROJECT_ROOT, retrieval_config["dataset_path"])
    catalog = load_catalog(dataset_path, PROJECT_ROOT)
    catalog_by_id = {record["product_id"]: record for record in catalog}
    metadata = load_metadata(resolve_project_path(PROJECT_ROOT, retrieval_config["metadata_path"]))
    index = load_index(resolve_project_path(PROJECT_ROOT, retrieval_config["index_path"]))
    if len(metadata) != int(index.ntotal):
        raise RuntimeError("索引向量数量与商品元数据数量不一致。")

    gallery_split = evaluation_config["gallery_split"]
    gallery_size = sum(record["split"] == gallery_split for record in metadata)
    if gallery_size < pool_depth:
        raise ValueError(f"{gallery_split} 商品只有 {gallery_size} 条，无法生成 {pool_depth} 深度的候选池。")

    load_started = time.perf_counter()
    embedder = ErnieVilEmbedder(
        model_name=retrieval_config["model_name"],
        device=retrieval_config["device"],
        batch_size=min(int(retrieval_config["batch_size"]), len(queries)),
    )
    model_load_seconds = time.perf_counter() - load_started
    encoding_started = time.perf_counter()
    query_features = embedder.encode_texts([query["query_text"] for query in queries])
    encoding_seconds = time.perf_counter() - encoding_started
    search_started = time.perf_counter()
    scores, indexes = search(index, query_features, top_k=int(index.ntotal))
    search_seconds = time.perf_counter() - search_started

    rows = build_annotation_rows(
        queries,
        scores,
        indexes,
        metadata,
        catalog_by_id,
        gallery_split,
        pool_depth,
    )
    manifest = {
        "status": "awaiting_human_annotation",
        "version": evaluation_config["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": embedder.model_name,
        "gallery_split": gallery_split,
        "gallery_size": gallery_size,
        "query_count": len(queries),
        "queries_by_category": dict(sorted(Counter(query["category_l2"] for query in queries).items())),
        "pool_depth": pool_depth,
        "annotation_rows": len(rows),
        "query_file_sha256": file_sha256(queries_path),
        "index_sha256": file_sha256(resolve_project_path(PROJECT_ROOT, retrieval_config["index_path"])),
        "timings_seconds": {
            "model_load": round(model_load_seconds, 3),
            "query_encoding": round(encoding_seconds, 3),
            "faiss_full_search": round(search_seconds, 6),
        },
        "label_values": {
            "2": "高度相关：满足品类和主要意图",
            "1": "部分相关：核心品类正确，但只满足部分意图",
            "0": "不相关：品类错误或与关键要求冲突",
        },
        "warning": "当前是未标注候选池，不能计算或宣称正式 P@10、R@10。",
    }
    write_outputs(annotation_path, manifest_path, rows, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
