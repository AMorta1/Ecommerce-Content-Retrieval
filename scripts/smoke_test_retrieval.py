"""用五条人工检查查询，在完整商品索引中验证中文检索链路。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from platform import python_version
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import load_metadata, resolve_project_path  # noqa: E402
from src.retrieval.ernie_vil import ErnieVilEmbedder  # noqa: E402
from src.retrieval.faiss_index import load_index, search  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "week1_v3" / "inference_samples.jsonl"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "retrieval" / "baseline" / "smoke_test.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="五条人工检查样本 JSONL")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="检索配置 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="试跑结果 JSON")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not sample.get("smoke_query") or not sample.get("product_id"):
                raise ValueError(f"第 {line_number} 行缺少 smoke_query 或 product_id。")
            samples.append(sample)
    if not samples:
        raise ValueError(f"输入文件没有可用样本：{path}")
    return samples


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    samples = load_samples(args.input.resolve())
    index = load_index(resolve_project_path(PROJECT_ROOT, config["index_path"]))
    metadata = load_metadata(resolve_project_path(PROJECT_ROOT, config["metadata_path"]))
    if len(metadata) != int(index.ntotal):
        raise RuntimeError("索引向量数量与商品元数据数量不一致。")

    metadata_ids = {record["product_id"] for record in metadata}
    missing_ids = [sample["product_id"] for sample in samples if sample["product_id"] not in metadata_ids]
    if missing_ids:
        raise ValueError(f"检查样本不在当前商品索引中：{missing_ids}")

    load_started = time.perf_counter()
    embedder = ErnieVilEmbedder(
        model_name=config["model_name"],
        device=config["device"],
        batch_size=min(config["batch_size"], len(samples)),
    )
    load_seconds = time.perf_counter() - load_started

    encoding_started = time.perf_counter()
    text_features = embedder.encode_texts([sample["smoke_query"] for sample in samples])
    encoding_seconds = time.perf_counter() - encoding_started

    search_started = time.perf_counter()
    scores, indexes = search(index, text_features, top_k=int(index.ntotal))
    search_seconds = time.perf_counter() - search_started

    rankings = []
    same_product_top1 = 0
    same_product_top10 = 0
    same_category_top10 = 0
    for query_index, sample in enumerate(samples):
        expected_rank = None
        top_candidates = []
        for rank, (score, metadata_index) in enumerate(zip(scores[query_index], indexes[query_index]), start=1):
            product = metadata[int(metadata_index)]
            if rank <= 10:
                top_candidates.append(
                    {
                        "rank": rank,
                        "score": round(float(score), 6),
                        "product_id": product["product_id"],
                        "title": product["title"],
                        "category_l2": product["category_l2"],
                        "image_path": product["image_path"],
                    }
                )
            if product["product_id"] == sample["product_id"]:
                expected_rank = rank
        if expected_rank is None:
            raise RuntimeError(f"未找到检查商品：{sample['product_id']}")
        same_product_top1 += int(expected_rank == 1)
        same_product_top10 += int(expected_rank <= 10)
        query_same_category_top10 = sum(
            candidate["category_l2"] == sample["category_l2"] for candidate in top_candidates
        )
        same_category_top10 += query_same_category_top10
        rankings.append(
            {
                "query_product_id": sample["product_id"],
                "query": sample["smoke_query"],
                "same_product_image_rank": expected_rank,
                "same_category_in_top_10": query_same_category_top10,
                "top_10_candidates": top_candidates,
            }
        )

    report = {
        "status": "full_catalog_smoke_test_completed",
        "model": embedder.model_name,
        "device": embedder.device,
        "python_version": python_version(),
        "query_count": len(samples),
        "candidate_count": int(index.ntotal),
        "embedding_dimension": int(text_features.shape[1]),
        "timings_seconds": {
            "model_load": round(load_seconds, 3),
            "five_query_encoding": round(encoding_seconds, 3),
            "faiss_search_all_candidates": round(search_seconds, 6),
        },
        "gpu_memory_allocated_mib": embedder.gpu_memory_allocated_mib(),
        "same_product_top1_count": same_product_top1,
        "same_product_top10_count": same_product_top10,
        "same_category_top10_count": same_category_top10,
        "top10_candidate_total": len(samples) * min(10, int(index.ntotal)),
        "metric_warning": "查询由目标商品信息人工编写且未做独立相关性标注；同商品排名只是链路检查，不是正式 Recall@10 或 Precision@10。",
        "rankings": rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
