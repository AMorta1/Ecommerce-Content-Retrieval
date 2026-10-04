"""Exclude training records that are near duplicates of frozen evaluation data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.deduplication import (
    category_counts,
    collect_training_exclusions,
    repeated_brand_model_groups,
    similar_image_pairs,
)
from src.data.preprocess import write_json, write_jsonl


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_evidence_csv(path: Path, records: list[dict], evidence: dict[str, list[dict]]) -> None:
    fields = [
        "product_id",
        "category_l1",
        "category_l2",
        "title",
        "reasons",
        "matched_frozen_product_ids",
        "matched_frozen_splits",
        "details",
        "decision",
    ]
    by_id = {record["product_id"]: record for record in records}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for product_id in sorted(evidence):
            product = by_id[product_id]
            matches = evidence[product_id]
            writer.writerow(
                {
                    "product_id": product_id,
                    "category_l1": product["category_l1"],
                    "category_l2": product["category_l2"],
                    "title": product["title"],
                    "reasons": ";".join(sorted({item["reason"] for item in matches})),
                    "matched_frozen_product_ids": ";".join(
                        sorted({item["reference_id"] for item in matches})
                    ),
                    "matched_frozen_splits": ";".join(
                        sorted({item["reference_split"] for item in matches})
                    ),
                    "details": ";".join(sorted({item["detail"] for item in matches})),
                    "decision": "exclude_from_training",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/training_deduplication.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = (ROOT / config["source_dataset"]).resolve()
    output = (ROOT / config["output_dataset"]).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")

    records = load_jsonl(source / "products.jsonl")
    training = [record for record in records if record["split"] == "train"]
    frozen_splits = set(config["frozen_splits"])
    brand_model_groups = repeated_brand_model_groups(records)
    image_pairs = similar_image_pairs(
        records,
        ROOT,
        config["image_dhash_max_distance"],
    )
    exclusions = collect_training_exclusions(
        brand_model_groups,
        image_pairs,
        frozen_splits,
    )
    cleaned_training = [
        record for record in training if record["product_id"] not in exclusions
    ]

    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", cleaned_training)
    frozen_files = {split: source / f"{split}.jsonl" for split in sorted(frozen_splits)}
    cross_brand_model = [
        group
        for group in brand_model_groups
        if len({product["split"] for product in group["products"]}) > 1
    ]
    frozen_only_brand_model = [
        group
        for group in cross_brand_model
        if all(product["split"] in frozen_splits for product in group["products"])
    ]
    cross_image_pairs = [
        pair for pair in image_pairs if pair["left"]["split"] != pair["right"]["split"]
    ]
    frozen_only_image_pairs = [
        pair
        for pair in cross_image_pairs
        if pair["left"]["split"] in frozen_splits
        and pair["right"]["split"] in frozen_splits
    ]
    summary = {
        "dataset_version": config["dataset_version"],
        "source_dataset_version": config["source_dataset_version"],
        "policy": config["policy"],
        "frozen_evaluation": {
            split: {
                "records": len(load_jsonl(path)),
                "sha256": file_sha256(path),
            }
            for split, path in frozen_files.items()
        },
        "brand_model_audit": {
            "repeated_groups": len(brand_model_groups),
            "cross_split_groups": len(cross_brand_model),
            "frozen_only_cross_split_groups": len(frozen_only_brand_model),
        },
        "image_dhash_audit": {
            "max_hamming_distance": config["image_dhash_max_distance"],
            "candidate_pairs": len(image_pairs),
            "cross_split_pairs": len(cross_image_pairs),
            "frozen_only_cross_split_pairs": len(frozen_only_image_pairs),
        },
        "training": {
            "before": len(training),
            "excluded": len(exclusions),
            "after": len(cleaned_training),
            "before_by_category": category_counts(training),
            "excluded_by_category": category_counts(
                [record for record in training if record["product_id"] in exclusions]
            ),
            "after_by_category": category_counts(cleaned_training),
            "source_train_jsonl_sha256": file_sha256(source / "train.jsonl"),
            "train_jsonl_sha256": file_sha256(output / "train.jsonl"),
        },
        "limitations": [
            "同品牌型号按保守策略排除，不表示这些商品已被人工确认完全相同。",
            "dHash 只反映低分辨率视觉结构相似，不等同于商品语义相同。",
            "验证集和测试集已冻结；两者之间的疑似近重复只记录、不移动。",
        ],
    }
    write_json(output / "summary.json", summary)

    report_dir = ROOT / "reports/data"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "training_deduplication_week2_v1.json", summary)
    write_evidence_csv(
        report_dir / "training_deduplication_week2_v1.csv",
        training,
        exclusions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
