"""Create the blinded workbook, or summarize it only after complete human review."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.lora_validation_review import build_workbook, summarize_workbook

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["workbook", "summarize"], required=True)
    args = parser.parse_args()
    build_workbook() if args.mode == "workbook" else summarize_workbook()
