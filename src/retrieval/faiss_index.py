"""Faiss 精确余弦检索索引。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def create_exact_index(features: np.ndarray) -> Any:
    """为已归一化向量建立内积索引，等价于精确余弦检索。"""
    features = _as_float32_matrix(features)
    if not np.all(np.isfinite(features)):
        raise ValueError("特征中包含 NaN 或无穷大。")
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("建立余弦索引前，所有特征必须完成 L2 归一化。")

    import faiss

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    return index


def save_index(index: Any, path: Path) -> None:
    """把索引保存到磁盘。"""
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> Any:
    """从磁盘读取索引。"""
    import faiss

    if not path.is_file():
        raise FileNotFoundError(f"索引文件不存在：{path}")
    return faiss.read_index(str(path))


def search(index: Any, query_features: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """返回每条查询的相似度和元数据行号。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于或等于 1。")
    query_features = _as_float32_matrix(query_features)
    actual_top_k = min(top_k, int(index.ntotal))
    scores, indexes = index.search(query_features, actual_top_k)
    return scores, indexes


def _as_float32_matrix(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"特征必须是非空二维数组，实际形状为 {features.shape}。")
    return np.ascontiguousarray(features, dtype=np.float32)
