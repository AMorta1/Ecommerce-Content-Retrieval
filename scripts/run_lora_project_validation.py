"""Prepare/freeze or run the approved Base/LoRA three-task project-validation protocol."""
import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.lora_validation import generate, prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "generate"], required=True)
    parser.add_argument("--variant", choices=["Base", "LoRA"])
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.variant:
            parser.error("prepare freezes both variants")
        prepare()
    else:
        if not args.variant:
            parser.error("generate requires one explicit variant; no automatic retries")
        generate(args.variant)
