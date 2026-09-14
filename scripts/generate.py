"""用一条标准化商品样本验证 Qwen 文案生成链路。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from platform import python_version
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.qwen import QwenGenerator, parse_generation_output  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "week1_v3" / "inference_samples.jsonl"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "generation" / "baseline" / "smoke_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="人工检查样本 JSONL")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="生成配置 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="试跑报告 JSON")
    parser.add_argument("--sample-index", type=int, default=0, help="使用第几条样本，从0开始")
    parser.add_argument("--sample-count", type=int, default=1, help="从起始位置连续测试几条样本")
    parser.add_argument("--offline", action="store_true", help="只读取本地缓存，不访问网络")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_samples(path: Path, sample_index: int, sample_count: int) -> list[dict[str, Any]]:
    if sample_index < 0:
        raise ValueError("sample-index 不能小于0。")
    if sample_count < 1:
        raise ValueError("sample-count 必须大于0。")
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for current_index, line in enumerate(input_file):
            if sample_index <= current_index < sample_index + sample_count:
                sample = json.loads(line)
                if "generation_input" not in sample or "product_id" not in sample:
                    raise ValueError("样本缺少 generation_input 或 product_id。")
                samples.append(sample)
    if len(samples) != sample_count:
        raise IndexError(
            f"从第 {sample_index} 条开始需要 {sample_count} 条样本，实际只读取到 {len(samples)} 条。"
        )
    return samples


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    samples = load_samples(args.input.resolve(), args.sample_index, args.sample_count)
    cache_dir = resolve_project_path(config["cache_dir"])

    load_started = time.perf_counter()
    generator = QwenGenerator(config, cache_dir, local_files_only=args.offline)
    load_seconds = time.perf_counter() - load_started

    results = []
    generation_total_started = time.perf_counter()
    for sample in samples:
        generation_started = time.perf_counter()
        raw_output = generator.generate(sample["generation_input"])
        generation_seconds = time.perf_counter() - generation_started

        parsed_output = None
        parse_error = None
        try:
            parsed_output = parse_generation_output(raw_output)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            parse_error = str(error)
        results.append(
            {
                "product_id": sample["product_id"],
                "generation_input": sample["generation_input"],
                "reference_title_not_given_to_model": sample["title"],
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "parse_error": parse_error,
                "generation_seconds": round(generation_seconds, 3),
            }
        )
        print(
            f"[{len(results)}/{len(samples)}] product_id={sample['product_id']} "
            f"time={generation_seconds:.3f}s parse={'ok' if parsed_output is not None else 'failed'}",
            flush=True,
        )
    generation_total_seconds = time.perf_counter() - generation_total_started
    parsed_count = sum(result["parsed_output"] is not None for result in results)

    report = {
        "status": "passed" if parsed_count == len(results) else "output_parse_failed",
        "model": config["model_name"],
        "prompt_version": config["prompt_version"],
        "generation_parameters": {
            "max_input_tokens": config["max_input_tokens"],
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": config["do_sample"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "seed": config["seed"],
        },
        "quantization": "bitsandbytes_nf4_4bit" if config["load_in_4bit"] else "none",
        "cache_dir": str(cache_dir),
        "python_version": python_version(),
        "sample_count": len(results),
        "parsed_output_count": parsed_count,
        "structured_output_success_rate": round(parsed_count / len(results), 4),
        "timings_seconds": {
            "model_load": round(load_seconds, 3),
            "generation_total": round(generation_total_seconds, 3),
            "generation_average": round(generation_total_seconds / len(results), 3),
        },
        "gpu_memory_mib": generator.gpu_memory_mib(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
