"""Extract product text features and build text plus fused Faiss indexes."""

from __future__ import annotations

import argparse
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

from src.retrieval.catalog import load_catalog, load_metadata, resolve_project_path  # noqa: E402
from src.retrieval.ernie_vil import ErnieVilEmbedder  # noqa: E402
from src.retrieval.faiss_index import create_exact_index, save_index  # noqa: E402
from src.retrieval.feature_library import build_product_text, fuse_product_features  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="检索配置 JSON")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文本和融合特征")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_name",
        "device",
        "batch_size",
        "dataset_path",
        "features_path",
        "metadata_path",
        "text_features_path",
        "text_index_path",
        "fused_features_path",
        "fused_index_path",
        "feature_library_report_path",
        "product_text_fields",
        "fusion_weights",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"检索配置缺少字段：{missing}")
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_texts(
    embedder: ErnieVilEmbedder,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batches.append(embedder.encode_texts(batch).astype(np.float32, copy=False))
        print(f"文本特征进度：{min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(batches, axis=0)


def ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        display = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(f"以下输出已经存在。如需重建，请增加 --overwrite：\n{display}")


def atomic_save_array(path: Path, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as output_file:
            np.save(output_file, features, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_index(path: Path, index: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        save_index(index, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    dataset_path = resolve_project_path(PROJECT_ROOT, config["dataset_path"])
    image_features_path = resolve_project_path(PROJECT_ROOT, config["features_path"])
    metadata_path = resolve_project_path(PROJECT_ROOT, config["metadata_path"])
    text_features_path = resolve_project_path(PROJECT_ROOT, config["text_features_path"])
    text_index_path = resolve_project_path(PROJECT_ROOT, config["text_index_path"])
    fused_features_path = resolve_project_path(PROJECT_ROOT, config["fused_features_path"])
    fused_index_path = resolve_project_path(PROJECT_ROOT, config["fused_index_path"])
    report_path = resolve_project_path(PROJECT_ROOT, config["feature_library_report_path"])
    output_paths = [
        text_features_path,
        text_index_path,
        fused_features_path,
        fused_index_path,
        report_path,
    ]
    ensure_outputs_available(output_paths, args.overwrite)

    records = load_catalog(dataset_path, PROJECT_ROOT)
    metadata = load_metadata(metadata_path)
    if [record["product_id"] for record in records] != [row["product_id"] for row in metadata]:
        raise ValueError("商品数据与已有图片索引元数据顺序不一致，请先重建图片索引。")
    image_features = np.load(image_features_path, allow_pickle=False)
    if image_features.shape[0] != len(records):
        raise ValueError("图片特征数量与商品数量不一致。")

    texts = [build_product_text(record, config["product_text_fields"]) for record in records]
    descriptions_used = sum(bool(record.get("description")) for record in records)
    started = time.perf_counter()
    load_started = time.perf_counter()
    embedder = ErnieVilEmbedder(
        model_name=config["model_name"],
        device=config["device"],
        batch_size=int(config["batch_size"]),
    )
    model_load_seconds = time.perf_counter() - load_started
    encoding_started = time.perf_counter()
    text_features = encode_texts(embedder, texts, int(config["batch_size"]))
    text_encoding_seconds = time.perf_counter() - encoding_started

    weights = config["fusion_weights"]
    fused_features = fuse_product_features(
        text_features,
        image_features,
        float(weights["text"]),
        float(weights["image"]),
    )
    text_index = create_exact_index(text_features)
    fused_index = create_exact_index(fused_features)
    atomic_save_array(text_features_path, text_features)
    atomic_save_index(text_index_path, text_index)
    atomic_save_array(fused_features_path, fused_features)
    atomic_save_index(fused_index_path, fused_index)

    report = {
        "status": "feature_library_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config["model_name"],
        "dataset_path": config["dataset_path"],
        "dataset_sha256": file_sha256(dataset_path),
        "product_count": len(records),
        "embedding_dimension": int(text_features.shape[1]),
        "product_text_fields": config["product_text_fields"],
        "products_with_description": descriptions_used,
        "products_using_title_only": len(records) - descriptions_used,
        "fusion_weights": weights,
        "fused_feature_role": "optimization_candidate_not_week1_baseline",
        "normalization": "text/image individually L2; weighted sum then L2",
        "index_type": "Faiss IndexFlatIP",
        "timings_seconds": {
            "model_load": round(model_load_seconds, 3),
            "text_encoding": round(text_encoding_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "outputs": {
            "image_features_sha256": file_sha256(image_features_path),
            "text_features_path": config["text_features_path"],
            "text_features_sha256": file_sha256(text_features_path),
            "text_index_path": config["text_index_path"],
            "text_index_sha256": file_sha256(text_index_path),
            "fused_features_path": config["fused_features_path"],
            "fused_features_sha256": file_sha256(fused_features_path),
            "fused_index_path": config["fused_index_path"],
            "fused_index_sha256": file_sha256(fused_index_path),
            "metadata_path": config["metadata_path"],
            "metadata_sha256": file_sha256(metadata_path),
        },
        "versions": {
            "paddlepaddle-gpu": version("paddlepaddle-gpu"),
            "paddlenlp": version("paddlenlp"),
            "numpy": version("numpy"),
            "faiss-cpu": version("faiss-cpu"),
        },
        "limitation": "当前数据 description 全为空，因此文本特征实际只使用商品标题。",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
