"""Isolated development24 integration; no training, test or holdout entrypoint."""
import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.lora_rag_integration import prepare, generate, build_review, summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "smoke", "generate", "review", "summarize"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode in ("smoke", "generate"):
        generate(smoke=args.mode == "smoke")
    elif args.mode == "review":
        build_review()
    else:
        summarize()
