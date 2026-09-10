"""从测试集分层抽取100条文案生成评测样本。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generation_evaluation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="生成评测配置 JSON")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "test":
                raise ValueError(f"第 {line_number} 行不是 test 样本。")
            records.append(record)
    if not records:
        raise ValueError(f"测试集为空：{path}")
    return records


def select_core_attributes(record: dict[str, Any], config: dict[str, Any]) -> dict[str, list[str]]:
    """选择可用于生成的核心属性，并排除不提供实际信息的占位值。"""
    excluded_global = {str(value).casefold() for value in config["excluded_attribute_values"]}
    excluded_by_attribute = {
        name: {str(value).casefold() for value in values}
        for name, values in config["excluded_values_by_attribute"].items()
    }
    selected = {}
    for name in config["core_attributes"][record["category_l2"]]:
        values = record["attributes"].get(name, [])
        valid_values = [
            value
            for value in values
            if str(value).casefold() not in excluded_global
            and str(value).casefold() not in excluded_by_attribute.get(name, set())
        ]
        if valid_values:
            selected[name] = valid_values
    return selected


def select_samples(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = record["category_l2"]
        if len(select_core_attributes(record, config)) >= int(config["minimum_core_attribute_count"]):
            grouped[category].append(record)

    random_generator = random.Random(int(config["random_seed"]))
    selected = []
    for category, quota in config["category_quotas"].items():
        candidates = sorted(grouped[category], key=lambda record: record["product_id"])
        if len(candidates) < int(quota):
            raise ValueError(f"{category} 只有 {len(candidates)} 条，无法抽取 {quota} 条。")
        for record in random_generator.sample(candidates, int(quota)):
            selected_attributes = select_core_attributes(record, config)
            sample = dict(record)
            sample["generation_input"] = {
                "category_l1": record["category_l1"],
                "category_l2": record["category_l2"],
                "attributes": selected_attributes,
            }
            selected.append(sample)
    return sorted(selected, key=lambda record: (record["category_l1"], record["category_l2"], record["product_id"]))


def rebuild_manifest_samples(
    records: list[dict[str, Any]], product_ids: list[str], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """按已提交清单中的商品编号重建同一份本地评测输入。"""
    records_by_id = {record["product_id"]: record for record in records}
    missing_ids = [product_id for product_id in product_ids if product_id not in records_by_id]
    if missing_ids:
        raise ValueError(f"源测试集缺少清单商品：{missing_ids}")
    selected = []
    for product_id in product_ids:
        record = records_by_id[product_id]
        selected_attributes = select_core_attributes(record, config)
        if len(selected_attributes) < int(config["minimum_core_attribute_count"]):
            raise ValueError(f"商品 {product_id} 不再满足核心属性数量要求。")
        sample = dict(record)
        sample["generation_input"] = {
            "category_l1": record["category_l1"],
            "category_l2": record["category_l2"],
            "attributes": selected_attributes,
        }
        selected.append(sample)
    return selected


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    dataset_path = resolve_project_path(config["dataset_path"])
    samples_path = resolve_project_path(config["samples_path"])
    manifest_path = resolve_project_path(config["manifest_path"])
    if samples_path.exists():
        raise FileExistsError("本地评测样本已经存在；为保证后续对比一致，本脚本不会覆盖。")

    records = load_test_records(dataset_path)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if file_sha256(dataset_path) != manifest["source_dataset_sha256"]:
            raise ValueError("源测试集哈希与已提交清单不一致，不能声称重建了同一评测集。")
        selected = rebuild_manifest_samples(records, manifest["product_ids"], config)
        write_jsonl_atomic(samples_path, selected)
        if file_sha256(samples_path) != manifest["samples_sha256"]:
            samples_path.unlink(missing_ok=True)
            raise ValueError("重建样本哈希与已提交清单不一致，请检查评测配置版本。")
        print(f"已按清单重建 {len(selected)} 条本地评测样本：{samples_path}")
        return

    selected = select_samples(records, config)
    category_l1_counts = Counter(record["category_l1"] for record in selected)
    category_l2_counts = Counter(record["category_l2"] for record in selected)
    expected_count = sum(int(value) for value in config["category_quotas"].values())
    if len(selected) != expected_count:
        raise RuntimeError(f"预期抽取 {expected_count} 条，实际得到 {len(selected)} 条。")

    write_jsonl_atomic(samples_path, selected)
    manifest = {
        "version": "generation_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": config["dataset_path"],
        "source_dataset_sha256": file_sha256(dataset_path),
        "samples_path": config["samples_path"],
        "samples_sha256": file_sha256(samples_path),
        "random_seed": int(config["random_seed"]),
        "minimum_core_attribute_count": int(config["minimum_core_attribute_count"]),
        "sample_count": len(selected),
        "category_l1_counts": dict(sorted(category_l1_counts.items())),
        "category_l2_counts": dict(sorted(category_l2_counts.items())),
        "product_ids": [record["product_id"] for record in selected],
        "note": "样本只来自 test；generation_input 不含原始标题。",
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
