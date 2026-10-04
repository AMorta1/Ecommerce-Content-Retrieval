"""Single-attempt frozen LoRA v1 test100; no tuning, training, retries or resume."""
import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.lora_formal_test import generate, prepare
from src.generation.lora_formal_review import archive, summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "generate", "archive", "summarize"], required=True)
    parser.add_argument("--variant", choices=["Base", "LoRA"])
    parser.add_argument("--confirm-formal-test100", action="store_true")
    args = parser.parse_args()
    if args.mode == "generate":
        if not args.variant or not args.confirm_formal_test100:
            parser.error("generate needs --variant and --confirm-formal-test100; no resume")
        generate(args.variant)
    else:
        if args.variant or args.confirm_formal_test100:
            parser.error("generation flags may only accompany --mode generate")
        {"prepare": prepare, "archive": archive, "summarize": summarize}[args.mode]()
