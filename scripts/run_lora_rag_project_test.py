"""User-approved additional fixed test100; no tuning, training or holdout."""
import argparse
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.lora_rag_project_test import prepare, generate, build_review, summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'generate', 'review', 'summarize'), required=True)
    parser.add_argument('--confirm-fixed-test100', action='store_true')
    args = parser.parse_args()
    if args.mode == 'generate' and not args.confirm_fixed_test100:
        parser.error('Generation requires --confirm-fixed-test100; never retry/resume')
    {'prepare': prepare, 'generate': generate, 'review': build_review, 'summarize': summarize}[args.mode]()
