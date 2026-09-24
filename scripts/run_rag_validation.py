"""在 validation 子集上成对运行 Baseline 与 Lightweight RAG。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from platform import python_version
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.qwen import (  # noqa: E402
    QwenGenerator,
    build_messages,
    build_rag_messages,
    parse_generation_output,
)
from src.generation.grounding import validate_generation_grounding  # noqa: E402
from src.generation.rag import (  # noqa: E402
    load_fact_policy,
    load_quality_audit,
    select_global_top_k_facts,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/generation_rag_validation_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[3])
    parser.add_argument(
        "--exclude-results",
        type=Path,
        action="append",
        default=[],
        help="排除既有 validation 结果文件中的 product_id；可重复指定。",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded_product_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        report = load_json(path)
        if report.get("scope") != "validation_only":
            raise ValueError(f"排除样本文件不是 validation_only：{path}")
        results = report.get("results")
        if not isinstance(results, list):
            raise ValueError(f"排除样本文件缺少 results 数组：{path}")
        excluded.update(str(result["product_id"]) for result in results)
    return excluded


def select_samples(
    records: list[dict[str, Any]],
    facts_by_product: dict[str, list[dict[str, Any]]],
    audit: dict[str, dict[str, str]],
    policy: dict[str, Any],
    *,
    per_category: int,
    seed: int,
    excluded_product_ids: set[str] | None = None,
    separate_negative_constraints: bool = False,
) -> list[dict[str, Any]]:
    if per_category < 1:
        raise ValueError("per-category 必须大于0。")
    by_category: dict[str, list[dict[str, Any]]] = {}
    excluded_product_ids = excluded_product_ids or set()
    for record in records:
        product_id = str(record["product_id"])
        if product_id in excluded_product_ids:
            continue
        if audit[product_id]["status"] != "PASS":
            continue
        selection = select_global_top_k_facts(
            facts_by_product[product_id],
            policy,
            top_k=3,
            separate_negative_constraints=separate_negative_constraints,
        )
        if not selection["identity_facts"] or not selection["selected_facts"]:
            continue
        by_category.setdefault(record["category_l2"], []).append(record)

    selected = []
    for category in sorted(policy["category_field_policy"]):
        candidates = by_category.get(category, [])
        candidates.sort(
            key=lambda record: hashlib.sha256(
                f"{seed}:{record['product_id']}".encode("utf-8")
            ).hexdigest()
        )
        if len(candidates) < per_category:
            raise ValueError(
                f"品类 {category} 只有 {len(candidates)} 条可用 PASS 样本，少于 {per_category}。"
            )
        selected.extend(candidates[:per_category])
    return selected


def parse_result(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return parse_generation_output(raw_output), None
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return None, str(error)


def run_generation(
    generator: QwenGenerator,
    generation_input: dict[str, Any],
    *,
    prompt_version: str,
    seed: int,
    rag_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_output = generator.generate(
        generation_input,
        prompt_version=prompt_version,
        rag_context=rag_context,
        seed=seed,
    )
    elapsed = time.perf_counter() - started
    parsed_output, parse_error = parse_result(raw_output)
    messages = (
        build_rag_messages(generation_input, rag_context, prompt_version=prompt_version)
        if rag_context is not None
        else build_messages(generation_input, prompt_version=prompt_version)
    )
    return {
        "prompt_version": prompt_version,
        "seed": seed,
        "messages": messages,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "parse_error": parse_error,
        "generation_seconds": round(elapsed, 3),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary.replace(path)


def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    top_k_values = sorted(set(args.top_k))
    if any(top_k < 1 for top_k in top_k_values):
        raise ValueError("所有 Top-K 都必须大于0。")
    config_path = args.config.resolve()
    config = load_json(config_path)
    validation_path = project_path(config["validation_source"])
    facts_path = project_path(config["fact_units_path"])
    policy_path = project_path(config["fact_policy_path"])
    audit_path = project_path(config["audit_path"])
    exclusion_paths = [path.resolve() for path in args.exclude_results]
    excluded_product_ids = load_excluded_product_ids(exclusion_paths)
    records = load_jsonl(validation_path)
    policy = load_fact_policy(policy_path)
    audit = load_quality_audit(audit_path, "validation")
    fact_units = load_jsonl(facts_path)
    facts_by_product: dict[str, list[dict[str, Any]]] = {}
    for fact in fact_units:
        facts_by_product.setdefault(str(fact["product_id"]), []).append(fact)
    record_ids = {str(record["product_id"]) for record in records}
    if record_ids != set(audit) or record_ids != set(facts_by_product):
        raise ValueError("validation、质量审计和事实知识库的商品集合不一致。")

    samples = select_samples(
        records,
        facts_by_product,
        audit,
        policy,
        per_category=args.per_category,
        seed=int(config["seed"]),
        excluded_product_ids=excluded_product_ids,
        separate_negative_constraints=bool(config.get("separate_negative_constraints", False)),
    )
    cache_dir = project_path(config["cache_dir"]).resolve()
    load_started = time.perf_counter()
    generator = QwenGenerator(config, cache_dir, local_files_only=True)
    load_seconds = time.perf_counter() - load_started
    model_identity = generator.model_identity()
    if model_identity["revision"] != config["revision"]:
        raise RuntimeError(
            f"实际模型 revision {model_identity['revision']} 与配置 {config['revision']} 不一致。"
        )
    if not model_identity["snapshot_path"]:
        raise RuntimeError("无法确认实际加载的本地模型 snapshot。")

    results = []
    for sample_index, record in enumerate(samples):
        product_id = str(record["product_id"])
        paired_seed = int(config["seed"]) + sample_index
        generation_input = {
            "category_l1": record["category_l1"],
            "category_l2": record["category_l2"],
            "attributes": record["attributes"],
        }
        baseline = run_generation(
            generator,
            generation_input,
            prompt_version=config["baseline_prompt_version"],
            seed=paired_seed,
        )
        rag_results = {}
        for top_k in top_k_values:
            context = select_global_top_k_facts(
                facts_by_product[product_id],
                policy,
                top_k=top_k,
                separate_negative_constraints=bool(
                    config.get("separate_negative_constraints", False)
                ),
            )
            rag_result = run_generation(
                generator,
                generation_input,
                prompt_version=config["prompt_version"],
                seed=paired_seed,
                rag_context=context,
            )
            if config.get("grounding_validator", {}).get("enabled", False):
                rag_result["grounding_validation"] = validate_generation_grounding(
                    rag_result["parsed_output"], context
                )
            rag_results[str(top_k)] = {"retrieval": context, **rag_result}
        results.append(
            {
                "sample_index": sample_index,
                "product_id": product_id,
                "category_l1": record["category_l1"],
                "category_l2": record["category_l2"],
                "audit_status": audit[product_id]["status"],
                "generation_input": generation_input,
                "reference_title_not_given_to_model": record["title"],
                "baseline": baseline,
                "rag": rag_results,
            }
        )
        parsed = int(baseline["parsed_output"] is not None) + sum(
            result["parsed_output"] is not None for result in rag_results.values()
        )
        print(
            f"[{len(results)}/{len(samples)}] product_id={product_id} "
            f"category={record['category_l2']} parsed={parsed}/{1 + len(top_k_values)}",
            flush=True,
        )

    report = {
        "version": config["version"],
        "status": "completed",
        "scope": "validation_only",
        "formal_test100_executed": False,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_identity,
        "quantization": "bitsandbytes_nf4_4bit" if config["load_in_4bit"] else "none",
        "generation_parameters": {
            key: config[key]
            for key in (
                "max_input_tokens",
                "max_new_tokens",
                "do_sample",
                "temperature",
                "top_p",
                "seed",
            )
        },
        "pairing": config["pairing"],
        "sample_selection": {
            "method": (
                "PASS only; deterministic hash order within each category; "
                "exclude product_ids from recorded validation result files"
                if exclusion_paths
                else "PASS only; deterministic hash order within each category"
            ),
            "per_category": args.per_category,
            "sample_count": len(samples),
            "top_k_values": top_k_values,
            "excluded_product_count": len(excluded_product_ids),
        },
        "artifacts": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "validation": {"path": str(validation_path), "sha256": sha256(validation_path)},
            "facts": {"path": str(facts_path), "sha256": sha256(facts_path)},
            "policy": {"path": str(policy_path), "sha256": sha256(policy_path)},
            "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
            "excluded_validation_results": [
                {"path": str(path), "sha256": sha256(path)} for path in exclusion_paths
            ],
        },
        "runtime": {
            "python_version": python_version(),
            "model_load_seconds": round(load_seconds, 3),
            "gpu_memory_mib": generator.gpu_memory_mib(),
        },
        "results": results,
    }
    write_json(args.output.resolve(), report)
    print(f"saved={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
