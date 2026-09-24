"""对已有 validation RAG 日志执行只读 grounding 检查。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.grounding import (  # noqa: E402
    VALIDATOR_VERSION,
    validate_generation_grounding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("scope") != "validation_only":
        raise ValueError("只允许复核 validation_only 报告。")
    if source.get("formal_test100_executed") is not False:
        raise ValueError("输入报告没有明确记录 formal_test100_executed=false。")

    results = []
    statuses = Counter()
    classifications = Counter()
    for sample in source["results"]:
        rag_result = sample["rag"][str(args.top_k)]
        validation = validate_generation_grounding(
            rag_result.get("parsed_output"), rag_result["retrieval"]
        )
        statuses[validation["status"]] += 1
        classifications.update(validation["classification_counts"])
        results.append(
            {
                "product_id": sample["product_id"],
                "category_l1": sample["category_l1"],
                "category_l2": sample["category_l2"],
                "validation": validation,
            }
        )

    report = {
        "version": VALIDATOR_VERSION,
        "scope": "validation_only",
        "formal_test100_executed": False,
        "mode": "detect_and_record_only",
        "source_report": {"path": str(input_path), "sha256": sha256(input_path)},
        "top_k": args.top_k,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "sample_count": len(results),
            "status_counts": dict(sorted(statuses.items())),
            "classification_counts": dict(sorted(classifications.items())),
        },
        "results": results,
    }
    write_json(args.output.resolve(), report)
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
