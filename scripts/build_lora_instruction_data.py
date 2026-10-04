"""Build LoRA v1 instruction data without loading or training a model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.lora_data import (  # noqa: E402
    build_dataset,
    file_sha256,
    load_consistency_review,
    load_jsonl,
    select_spot_check_products,
    summarize_dataset,
    validate_dataset,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/lora_instruction_data_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/lora_instruction_data_v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports/generation/lora/lora_instruction_data_v1_report.json"
DEFAULT_SAMPLES = PROJECT_ROOT / "reports/generation/lora/lora_instruction_data_v1_samples.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    return parser.parse_args()


def _resolve_project_path(value: str) -> Path:
    return (PROJECT_ROOT / value).resolve()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary.replace(path)


def _write_samples(path: Path, products: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category_l2",
        "product_id",
        "split",
        "source_title",
        "used_attributes",
        "title_target",
        "selling_points_target",
        "short_description_target",
        "cleaning_actions",
        "manual_review_status",
        "manual_review_note",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for product in products:
            writer.writerow(
                {
                    "category_l2": product["category_l2"],
                    "product_id": product["product_id"],
                    "split": product["split"],
                    "source_title": product["source_title"],
                    "used_attributes": json.dumps(
                        product["used_attributes"], ensure_ascii=False
                    ),
                    "title_target": product["targets"]["title"],
                    "selling_points_target": json.dumps(
                        product["targets"]["selling_points"], ensure_ascii=False
                    ),
                    "short_description_target": product["targets"][
                        "short_description"
                    ],
                    "cleaning_actions": json.dumps(
                        product["cleaning_actions"], ensure_ascii=False
                    ),
                    "manual_review_status": "",
                    "manual_review_note": "",
                }
            )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = _resolve_project_path(config["source"]["path"])
    review_path = _resolve_project_path(config["consistency_review"]["path"])
    core_path = _resolve_project_path(config["core_attribute_definition"]["path"])

    checks = (
        (source_path, config["source"]["sha256"], "source"),
        (review_path, config["consistency_review"]["sha256"], "review"),
        (core_path, config["core_attribute_definition"]["sha256"], "core definition"),
    )
    for path, expected_hash, label in checks:
        if file_sha256(path) != expected_hash:
            raise ValueError(f"{label} SHA-256 与配置不一致：{path}")

    records = load_jsonl(source_path)
    if len(records) != int(config["source"]["record_count"]):
        raise ValueError("源训练商品数量与配置不一致。")
    review = load_consistency_review(review_path)
    core_definition = json.loads(core_path.read_text(encoding="utf-8"))
    core_attributes = core_definition[config["core_attribute_definition"]["key"]]

    products, instructions, excluded = build_dataset(
        records, review, core_attributes, config
    )
    validate_dataset(products, instructions, excluded)
    output_dir = args.output_dir.resolve()
    product_path = output_dir / "products.jsonl"
    excluded_path = output_dir / "excluded_products.jsonl"
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    summary_path = output_dir / "summary.json"
    train_rows = [row for row in instructions if row["split"] == "train"]
    validation_rows = [row for row in instructions if row["split"] == "validation"]

    _write_jsonl(product_path, products)
    _write_jsonl(excluded_path, excluded)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    summary = summarize_dataset(products, instructions, excluded)
    summary.update(
        {
            "version": config["version"],
            "status": "built_spot_check_pending",
            "model_loaded": False,
            "training_executed": False,
            "test_data_read_or_executed": False,
        }
    )
    _write_json(summary_path, summary)
    samples = select_spot_check_products(products, config)
    _write_samples(args.samples.resolve(), samples)

    artifacts = {}
    for name, path in (
        ("products", product_path),
        ("excluded_products", excluded_path),
        ("train", train_path),
        ("validation", validation_path),
        ("summary", summary_path),
        ("samples", args.samples.resolve()),
    ):
        artifacts[name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    report = {
        "version": config["version"],
        "status": "built_spot_check_pending",
        "config": {
            "path": config_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "sources": {
            "train": config["source"],
            "consistency_review": config["consistency_review"],
            "core_attribute_definition": config["core_attribute_definition"],
        },
        "summary": summary,
        "spot_check": {
            "products_per_category": config["spot_check"][
                "products_per_category"
            ],
            "product_count": len(samples),
            "task_example_count": len(samples) * 3,
            "manual_review_complete": False,
        },
        "artifacts": artifacts,
        "guards": {
            "source_title_in_model_input": False,
            "model_generated_target_count": 0,
            "frozen_validation_used": False,
            "frozen_test_used": False,
            "test_output_created": False,
            "model_loaded": False,
            "training_executed": False,
        },
    }
    _write_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
