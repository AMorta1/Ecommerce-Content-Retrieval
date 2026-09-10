"""检查或修复 Excel 保存 CSV 时改坏的长商品编号。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.review import find_product_id_mismatches, restore_product_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="需要检查的 CSV 文件")
    parser.add_argument("--repair", action="store_true", help="根据 image_path 修复商品编号")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV 缺少表头：{path}")
        rows = list(reader)
        return reader.fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    found_mismatches = False
    for configured_path in args.paths:
        path = configured_path.resolve()
        fieldnames, rows = read_csv(path)
        mismatches = find_product_id_mismatches(rows)
        if not mismatches:
            print(f"编号检查通过：{path}（{len(rows)} 行）")
            continue

        found_mismatches = True
        if not args.repair:
            first_row, current_id, expected_id = mismatches[0]
            print(
                f"发现 {len(mismatches)} 个错误编号：{path}\n"
                f"首个错误位于第 {first_row} 行：{current_id!r}，应为 {expected_id!r}"
            )
            continue

        repaired = restore_product_ids(rows)
        write_csv(path, fieldnames, rows)
        print(f"已修复：{path}（{repaired}/{len(rows)} 行）")

    if found_mismatches and not args.repair:
        raise ValueError("检测到 Excel 商品编号转换。确认无误后增加 --repair 执行恢复。")


if __name__ == "__main__":
    main()
