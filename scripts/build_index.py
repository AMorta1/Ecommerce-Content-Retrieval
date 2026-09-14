"""批量提取商品图片特征，并建立 Faiss 检索索引。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import load_catalog, resolve_project_path, write_metadata  # noqa: E402
from src.retrieval.ernie_vil import ErnieVilEmbedder  # noqa: E402
from src.retrieval.faiss_index import create_exact_index, save_index  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="检索配置 JSON")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有特征和索引")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        config = json.load(input_file)
    required = {
        "model_name",
        "device",
        "batch_size",
        "dataset_path",
        "features_path",
        "index_path",
        "metadata_path",
        "report_path",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"检索配置缺少字段：{missing}")
    if not isinstance(config["batch_size"], int) or config["batch_size"] < 1:
        raise ValueError("batch_size 必须是大于或等于 1 的整数。")
    return config


def encode_catalog_images(
    embedder: ErnieVilEmbedder,
    records: list[dict[str, Any]],
    batch_size: int,
) -> np.ndarray:
    """分批编码，避免一次打开全部图片导致内存或显存不足。"""
    feature_batches: list[np.ndarray] = []
    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        image_paths = [resolve_project_path(PROJECT_ROOT, record["image_path"]) for record in batch]
        feature_batches.append(embedder.encode_images(image_paths).astype(np.float32, copy=False))
        completed = min(start + batch_size, total)
        print(f"图片特征进度：{completed}/{total}", flush=True)
    return np.concatenate(feature_batches, axis=0)


def ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        display = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(f"以下输出已经存在。如需重建，请增加 --overwrite：\n{display}")


def temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    dataset_path = resolve_project_path(PROJECT_ROOT, config["dataset_path"])
    features_path = resolve_project_path(PROJECT_ROOT, config["features_path"])
    index_path = resolve_project_path(PROJECT_ROOT, config["index_path"])
    metadata_path = resolve_project_path(PROJECT_ROOT, config["metadata_path"])
    report_path = resolve_project_path(PROJECT_ROOT, config["report_path"])
    output_paths = [features_path, index_path, metadata_path, report_path]
    ensure_outputs_available(output_paths, args.overwrite)

    records = load_catalog(dataset_path, PROJECT_ROOT)
    print(f"商品库检查完成：{len(records)} 条有效商品。", flush=True)

    started = time.perf_counter()
    load_started = time.perf_counter()
    embedder = ErnieVilEmbedder(
        model_name=config["model_name"],
        device=config["device"],
        batch_size=config["batch_size"],
    )
    model_load_seconds = time.perf_counter() - load_started

    encoding_started = time.perf_counter()
    features = encode_catalog_images(embedder, records, config["batch_size"])
    encoding_seconds = time.perf_counter() - encoding_started
    if features.shape[0] != len(records):
        raise RuntimeError("特征数量与商品数量不一致。")

    index_started = time.perf_counter()
    index = create_exact_index(features)
    index_build_seconds = time.perf_counter() - index_started

    temporary_outputs = [temporary_path(path) for path in (features_path, index_path, metadata_path)]
    try:
        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_outputs[0].open("wb") as output_file:
            np.save(output_file, features, allow_pickle=False)
        save_index(index, temporary_outputs[1])
        write_metadata(temporary_outputs[2], records)
        os.replace(temporary_outputs[0], features_path)
        os.replace(temporary_outputs[1], index_path)
        os.replace(temporary_outputs[2], metadata_path)
    finally:
        for path in temporary_outputs:
            path.unlink(missing_ok=True)

    report = {
        "status": "index_build_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": embedder.model_name,
        "device": embedder.device,
        "batch_size": embedder.batch_size,
        "dataset_path": config["dataset_path"],
        "dataset_sha256": file_sha256(dataset_path),
        "product_count": len(records),
        "embedding_dimension": int(features.shape[1]),
        "feature_dtype": str(features.dtype),
        "index_type": "IndexFlatIP",
        "similarity": "cosine_via_normalized_inner_product",
        "corpus_splits": dict(sorted(Counter(record["split"] for record in records).items())),
        "category_counts": dict(sorted(Counter(record["category_l2"] for record in records).items())),
        "timings_seconds": {
            "model_load": round(model_load_seconds, 3),
            "image_encoding": round(encoding_seconds, 3),
            "index_build": round(index_build_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "average_image_encoding_ms": round(encoding_seconds * 1000 / len(records), 3),
        "gpu_memory_allocated_mib": embedder.gpu_memory_allocated_mib(),
        "outputs": {
            "features_path": config["features_path"],
            "features_sha256": file_sha256(features_path),
            "index_path": config["index_path"],
            "index_sha256": file_sha256(index_path),
            "metadata_path": config["metadata_path"],
            "metadata_sha256": file_sha256(metadata_path),
        },
        "versions": {
            "paddlepaddle-gpu": version("paddlepaddle-gpu"),
            "paddlenlp": version("paddlenlp"),
            "numpy": version("numpy"),
            "faiss-cpu": version("faiss-cpu"),
        },
        "metric_warning": "该索引覆盖全部实验商品，人工相关性标注尚未完成，不能据此报告 Recall@10 或 Precision@10。",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
