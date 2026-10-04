"""Independent offline QLoRA trainer; full training requires explicit confirmation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.lora_training import run


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lora_qlora_train_v1.json")
    parser.add_argument("--mode", choices=["prepare", "acceptance", "train"], default="prepare")
    parser.add_argument("--confirm-formal-training", action="store_true")
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--report-directory", type=Path)
    parser.add_argument("--evaluate-dev", action="store_true",
                        help="Complete lora_train_dev evaluation at acceptance stop; no update.")
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
