"""人工审核表的商品编号校验与恢复。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def product_id_from_image_path(image_path: str) -> str:
    """从项目图片文件名取得未被 Excel 改写的商品编号。"""
    product_id = Path(image_path).stem
    if not product_id.isdigit():
        raise ValueError(f"图片文件名不是数字商品编号：{image_path}")
    return product_id


def find_product_id_mismatches(rows: Sequence[dict[str, str]]) -> list[tuple[int, str, str]]:
    """返回行号、当前编号和图片路径中的正确编号。"""
    mismatches = []
    for row_number, row in enumerate(rows, start=2):
        expected_product_id = product_id_from_image_path(row.get("image_path", ""))
        current_product_id = row.get("product_id", "").strip()
        if current_product_id != expected_product_id:
            mismatches.append((row_number, current_product_id, expected_product_id))
    return mismatches


def restore_product_ids(rows: Sequence[dict[str, str]]) -> int:
    """原地恢复商品编号，返回发生修改的行数。"""
    repaired = 0
    for row in rows:
        expected_product_id = product_id_from_image_path(row.get("image_path", ""))
        if row.get("product_id", "").strip() != expected_product_id:
            row["product_id"] = expected_product_id
            repaired += 1
    return repaired
