"""Full streaming label census. No third-party dependencies required."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.streaming import iter_products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels, fields, empty = Counter(), Counter(), Counter()
    count = 0
    start = time.monotonic()
    for product_id, record in iter_products(args.input):
        count += 1
        labels[record.get("label", "")] += 1
        fields.update(record.keys())
        empty.update(k for k, v in record.items() if v is None or v == "")
        if count % 500000 == 0:
            print(f"Scanned {count:,} records in {time.monotonic()-start:.1f}s", flush=True)
    with (args.output / "label_counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raw_label", "count"])
        writer.writerows(labels.most_common())
    summary = {"source": str(args.input.resolve()), "source_bytes": args.input.stat().st_size,
               "records": count, "unique_labels": len(labels), "field_presence": dict(fields),
               "empty_fields": dict(empty), "elapsed_seconds": round(time.monotonic()-start, 2)}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
