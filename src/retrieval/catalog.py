"""读取检索商品库，并保存与向量行号一一对应的精简元数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = ("product_id", "title", "category_l1", "category_l2", "image_path", "split")


def load_catalog(dataset_path: Path, project_root: Path) -> list[dict[str, Any]]:
    """读取商品 JSONL，并检查检索阶段依赖的字段和图片。"""
    records: list[dict[str, Any]] = []
    product_ids: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing_fields = [field for field in REQUIRED_FIELDS if not record.get(field)]
            if missing_fields:
                raise ValueError(f"{dataset_path} 第 {line_number} 行缺少字段：{missing_fields}")

            product_id = str(record["product_id"])
            if product_id in product_ids:
                raise ValueError(f"商品编号重复：{product_id}")
            product_ids.add(product_id)

            image_path = resolve_project_path(project_root, record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"商品 {product_id} 的图片不存在：{image_path}")
            records.append(record)

    if not records:
        raise ValueError(f"商品数据为空：{dataset_path}")
    return records


def to_search_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """仅保留展示检索结果所需字段，避免重复保存完整原始数据。"""
    return {
        "product_id": str(record["product_id"]),
        "title": record["title"],
        "category_l1": record["category_l1"],
        "category_l2": record["category_l2"],
        "image_path": record["image_path"],
        "split": record["split"],
    }


def write_metadata(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """按向量顺序写入精简商品信息。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(to_search_metadata(record), ensure_ascii=False) + "\n")


def load_metadata(path: Path) -> list[dict[str, Any]]:
    """读取索引配套的商品信息。"""
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def resolve_project_path(project_root: Path, configured_path: str | Path) -> Path:
    """将配置中的相对路径统一解释为相对项目根目录。"""
    path = Path(configured_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
