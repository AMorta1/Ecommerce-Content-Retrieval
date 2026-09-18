"""Build and select product text, image, and fused feature libraries."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .ernie_vil import normalize_rows


INDEX_PATH_FIELDS = {
    "image": "index_path",
    "text": "text_index_path",
    "fused": "fused_index_path",
}


def build_product_text(record: dict[str, Any], fields: Sequence[str]) -> str:
    """Join available configured text fields; missing descriptions are skipped."""
    parts = []
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if not parts:
        raise ValueError(f"商品 {record.get('product_id', '')} 没有可编码文本。")
    return "\n".join(parts)


def fuse_product_features(
    text_features: np.ndarray,
    image_features: np.ndarray,
    text_weight: float,
    image_weight: float,
) -> np.ndarray:
    """Weighted-average aligned ERNIE-ViL vectors and L2-normalize the result."""
    if text_features.shape != image_features.shape:
        raise ValueError(
            f"文本和图片特征形状必须一致：{text_features.shape} != {image_features.shape}。"
        )
    if text_weight < 0 or image_weight < 0 or text_weight + image_weight <= 0:
        raise ValueError("融合权重必须非负且总和大于0。")
    fused = text_weight * text_features + image_weight * image_features
    return normalize_rows(fused).astype(np.float32, copy=False)


def catalog_index_path(config: dict[str, Any], feature_type: str) -> str:
    """Return the configured index path for image, text, or fused products."""
    if feature_type not in INDEX_PATH_FIELDS:
        raise ValueError(f"不支持的商品特征类型：{feature_type}")
    field = INDEX_PATH_FIELDS[feature_type]
    if field not in config:
        raise ValueError(f"检索配置缺少字段：{field}")
    return str(config[field])
