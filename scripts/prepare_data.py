"""Prepare an isolated, balanced Week 1 experiment dataset."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocess import build_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--output", type=Path, help="Defaults to data/processed/<dataset_version>")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / "data/processed" / config["dataset_version"]
    build_dataset(config, ROOT, output, args.config)


if __name__ == "__main__":
    main()
