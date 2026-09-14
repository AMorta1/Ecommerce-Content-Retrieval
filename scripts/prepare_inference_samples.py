"""Export five inspected training examples with explicit generation attributes."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocess import write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/week1_v3")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference_samples.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with (args.dataset / "multimodal/train.jsonl").open(encoding="utf-8") as handle:
        products = {r["product_id"]: r for r in map(json.loads, handle)}
    examples = []
    for specification in config["samples"]:
        record = dict(products[specification["product_id"]])
        record["generation_input"] = {
            "category_l1": record["category_l1"], "category_l2": record["category_l2"],
            "attributes": {key: record["attributes"][key] for key in specification["attribute_keys"]},
        }
        record["smoke_query"] = specification["query"]
        record["sample_review"] = {
            "method": "assistant_image_and_text_inspection",
            "note": specification["review_note"],
            "facts_verified": False,
        }
        if not (ROOT / record["image_path"]).is_file():
            raise FileNotFoundError(record["image_path"])
        examples.append(record)
    if len(examples) != 5 or len({r["product_id"] for r in examples}) != 5:
        raise ValueError("Expected exactly five distinct examples")
    target = args.dataset / "inference_samples.jsonl"
    write_jsonl(target, examples)
    print(f"Prepared {len(examples)} training examples: {target}")


if __name__ == "__main__":
    main()
