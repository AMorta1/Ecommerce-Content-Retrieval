"""Cache and validate images; export image-ready JSONL and inference examples."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.images import download_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/week1_v3")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--limit", type=int, help="Only fetch diverse training examples for a smoke test")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    download_dataset(ROOT, args.dataset, args.workers, args.limit)


if __name__ == "__main__":
    main()
