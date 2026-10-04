"""使用中文文本或本地图片检索 Top-K 商品。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import load_metadata, resolve_project_path  # noqa: E402
from src.retrieval.ernie_vil import ErnieVilEmbedder  # noqa: E402
from src.retrieval.faiss_index import load_index, search  # noqa: E402
from src.retrieval.feature_library import catalog_index_path  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="检索配置 JSON")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--text", help="中文商品描述，例如：带麦克风的有线入耳式耳机")
    query_group.add_argument("--image", type=Path, help="本地查询图片")
    parser.add_argument("--top-k", type=int, help="返回结果数；默认读取配置")
    parser.add_argument(
        "--catalog-feature",
        choices=("image", "text", "fused"),
        help="覆盖默认商品库特征；不填写时文本查图片、图片查文本",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式打印结果")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    top_k = args.top_k if args.top_k is not None else config["default_top_k"]
    if top_k < 1:
        raise ValueError("top_k 必须大于或等于 1。")

    query_type = "text" if args.text is not None else "image"
    default_features = config["default_catalog_feature_by_query_mode"]
    if default_features != {"text": "image", "image": "text"}:
        raise ValueError("默认纯跨模态策略必须配置为文本查图片、图片查文本。")
    catalog_feature = args.catalog_feature or default_features[query_type]
    index_path = catalog_index_path(config, catalog_feature)
    index = load_index(resolve_project_path(PROJECT_ROOT, index_path))
    metadata = load_metadata(resolve_project_path(PROJECT_ROOT, config["metadata_path"]))
    if len(metadata) != int(index.ntotal):
        raise RuntimeError(f"索引有 {index.ntotal} 条向量，但元数据有 {len(metadata)} 条。请重新建立索引。")

    load_started = time.perf_counter()
    embedder = ErnieVilEmbedder(
        model_name=config["model_name"],
        device=config["device"],
        batch_size=1,
    )
    model_load_seconds = time.perf_counter() - load_started

    encoding_started = time.perf_counter()
    if args.text is not None:
        query_value = args.text
        query_features = embedder.encode_texts([args.text])
    else:
        query_path = args.image.resolve()
        if not query_path.is_file():
            raise FileNotFoundError(f"查询图片不存在：{query_path}")
        query_value = str(query_path)
        query_features = embedder.encode_images([query_path])
    encoding_seconds = time.perf_counter() - encoding_started

    search_started = time.perf_counter()
    scores, indexes = search(index, query_features, top_k)
    search_seconds = time.perf_counter() - search_started
    results = []
    for rank, (score, metadata_index) in enumerate(zip(scores[0], indexes[0]), start=1):
        product = metadata[int(metadata_index)]
        results.append({"rank": rank, "score": round(float(score), 6), **product})

    response = {
        "query_type": query_type,
        "catalog_feature": catalog_feature,
        "query": query_value,
        "top_k": len(results),
        "timings_seconds": {
            "model_load": round(model_load_seconds, 3),
            "query_encoding": round(encoding_seconds, 3),
            "faiss_search": round(search_seconds, 6),
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return

    print(f"查询：{query_value}")
    print(
        f"耗时：模型加载 {model_load_seconds:.3f}s，查询编码 {encoding_seconds:.3f}s，"
        f"Faiss 检索 {search_seconds * 1000:.3f}ms"
    )
    for result in results:
        print(
            f"{result['rank']}. [{result['category_l2']}] {result['title']}\n"
            f"   相似度={result['score']:.6f}，商品ID={result['product_id']}，图片={result['image_path']}"
        )


if __name__ == "__main__":
    main()
