"""Audit prepared data independently, including image hashes and split leakage."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.images import validate_image
from src.data.preprocess import write_json


def load(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_category_review(records, path):
    """首次生成审核表；重复校验时保留同一批商品的人工填写内容。"""
    groups = defaultdict(list)
    for record in records:
        if record["split"] == "train":
            groups[record["category_l2"]].append(record)
    fields = [
        "product_id",
        "category_l1",
        "category_l2",
        "raw_label",
        "title",
        "attributes",
        "image_path",
        "review_label_ok",
        "review_image_ok",
        "notes",
    ]
    review_rows = []
    for category in sorted(groups):
        sample = sorted(
            groups[category],
            key=lambda record: hashlib.sha256(("review:" + record["product_id"]).encode()).hexdigest(),
        )[:5]
        for record in sample:
            row = {key: record[key] for key in fields if key in record}
            row["attributes"] = json.dumps(row["attributes"], ensure_ascii=False)
            review_rows.append(row)

    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        existing_ids = [row["product_id"] for row in existing_rows]
        expected_ids = [row["product_id"] for row in review_rows]
        if existing_ids != expected_ids:
            raise FileExistsError(
                f"已有审核表的商品与当前数据不一致，拒绝覆盖：{path}。"
                "请先归档旧表，或为本次数据使用新的输出目录。"
            )
        print(f"保留已有人工审核表：{path}", flush=True)
        return

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/week1_v3/multimodal")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/data")
    args = parser.parse_args()
    records = load(args.dataset / "products.jsonl")
    if not records:
        raise ValueError("Dataset is empty")
    for key in ("product_id", "content_fingerprint", "image_url_fingerprint", "image_sha256"):
        if len({r[key] for r in records}) != len(records):
            raise ValueError(f"Duplicates in {key}")
    partitions = {}
    for split in ("train", "validation", "test"):
        rows = load(args.dataset / f"{split}.jsonl")
        expected = [r for r in records if r["split"] == split]
        if rows != expected:
            raise ValueError(f"Inconsistent split file: {split}")
        partitions[split] = {r["product_id"] for r in rows}
    split_pairs = (
        ("train", "test"),
        ("train", "validation"),
        ("validation", "test"),
    )
    if any(partitions[left] & partitions[right] for left, right in split_pairs):
        raise ValueError("Split ID leakage")
    for record in records:
        path = (ROOT / record["image_path"]).resolve()
        path.relative_to(ROOT)
        info = validate_image(path.read_bytes())
        if info["sha256"] != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {record['product_id']}")
        if not record["attributes"] or not all(
            isinstance(values, list) and values
            for values in record["attributes"].values()
        ):
            raise ValueError("Invalid multivalue attributes")
    examples = load(args.dataset.parent / "inference_samples.jsonl")
    if len(examples) != 5 or any(r["product_id"] not in partitions["train"] for r in examples):
        raise ValueError("Expected 5 training-only inference examples")
    if len({r["category_l1"] for r in examples}) != 2:
        raise ValueError("Examples must cover both broad categories")
    training_records = {
        record["product_id"]: record
        for record in records
        if record["split"] == "train"
    }
    for example in examples:
        original = training_records[example["product_id"]]
        if example["image_path"] != original["image_path"]:
            raise ValueError("Inference sample references an outdated image path")
        generation_input = example["generation_input"]
        if "title" in generation_input:
            raise ValueError("Generation input must not include the target title")
        for name, values in generation_input["attributes"].items():
            if original["attributes"].get(name) != values:
                raise ValueError(f"Inference attribute changed: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "validated_records": len(records),
        "split_sizes": {key: len(value) for key, value in partitions.items()},
        "by_category": dict(Counter(record["category_l2"] for record in records)),
        "exact_duplicate_id_content_url_image_checks": "passed",
        "all_images_full_decode_and_sha256": "passed",
        "five_training_inference_examples": "passed",
        "limitations": [
            "Not a record-by-record semantic category review",
            "Does not automatically remove perceptually similar images or paraphrased duplicates",
            "Attributes are source claims, not verified facts",
        ],
        "products_jsonl_sha256": hashlib.sha256(
            (args.dataset / "products.jsonl").read_bytes()
        ).hexdigest(),
    }
    write_json(args.output / "validation.json", summary)
    summary_sources = (
        (args.dataset.parent / "summary.json", "preprocessing.json"),
        (args.dataset / "summary.json", "images.json"),
    )
    for src, name in summary_sources:
        write_json(args.output / name, json.loads(src.read_text(encoding="utf-8")))
    # 首次导出时审核列为空；重复验证不能清除人工填写的结果。
    write_category_review(records, args.output / "category_review.csv")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
