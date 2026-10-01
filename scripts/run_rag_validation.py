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
    select_rag_v2_context,
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


def paired_sample_seed(base_seed: int, stable_sample_index: int) -> int:
    """所有比较变体共享的逐样本 seed；变体名不得参与计算。"""
    return base_seed + stable_sample_index


def verify_p2_holdout_guard(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    validation_path: Path,
    audit_path: Path,
) -> set[str]:
    """验证冻结 holdout 的清单、输入哈希和 P2 禁跑标记。"""
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("RAG v2 generation-output holdout manifest SHA256 不匹配。")
    manifest = load_json(manifest_path)
    if manifest.get("version") != "rag_v2_generation_output_holdout_v1":
        raise ValueError("RAG v2 holdout manifest 版本不匹配。")
    if manifest.get("execution_guard", {}).get("p2_v2_generation_allowed") is not False:
        raise ValueError("RAG v2 holdout 未明确禁止 P2 V2 generation。")
    if manifest.get("source", {}).get("sha256") != sha256(validation_path):
        raise ValueError("RAG v2 holdout 的 validation 源数据 SHA256 不匹配。")
    if manifest.get("quality_audit", {}).get("sha256") != sha256(audit_path):
        raise ValueError("RAG v2 holdout 的质量审计 SHA256 不匹配。")
    project_root = manifest_path.parents[3]
    excluded_ids: set[str] = set()
    for prior in manifest.get("excluded_prior_generation_results", []):
        prior_path = project_root / prior["path"]
        if sha256(prior_path) != prior["sha256"]:
            raise ValueError(f"RAG v2 holdout 的历史结果 SHA256 不匹配：{prior_path}")
        report = load_json(prior_path)
        prior_results = report.get("results", [])
        if len(prior_results) != int(prior["sample_count"]):
            raise ValueError(f"RAG v2 holdout 的历史结果样本数不匹配：{prior_path}")
        excluded_ids.update(str(result["product_id"]) for result in prior_results)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 32:
        raise ValueError("RAG v2 holdout 必须恰好包含 32 条样本。")
    product_ids = [str(sample["product_id"]) for sample in samples]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("RAG v2 holdout product_id 存在重复。")
    payload = ("\n".join(product_ids) + "\n").encode("utf-8")
    expected_ids_hash = manifest.get("selection", {}).get("product_id_list_sha256")
    if hashlib.sha256(payload).hexdigest() != expected_ids_hash:
        raise ValueError("RAG v2 holdout product_id 清单 SHA256 不匹配。")
    selection = manifest["selection"]
    if len(excluded_ids) != int(selection["prior_generation_product_union_count"]):
        raise ValueError("RAG v2 holdout 的历史 generation 商品并集数量不匹配。")
    records = load_jsonl(validation_path)
    audit = load_quality_audit(audit_path, "validation")
    candidates_by_category: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        product_id = str(record["product_id"])
        if product_id in excluded_ids or audit[product_id]["status"] != "PASS":
            continue
        candidates_by_category.setdefault(record["category_l2"], []).append(record)
    namespace = str(selection["namespace"])
    seed = int(selection["seed"])
    per_category = int(selection["per_category"])
    expected_samples = []
    candidate_counts = {}
    for category in sorted(candidates_by_category):
        candidates = candidates_by_category[category]
        candidate_counts[category] = len(candidates)
        candidates.sort(
            key=lambda record: hashlib.sha256(
                f"{namespace}:{seed}:{category}:{record['product_id']}".encode("utf-8")
            ).hexdigest()
        )
        expected_samples.extend(
            (category, str(record["product_id"]))
            for record in candidates[:per_category]
        )
    actual_samples = [
        (str(sample["category_l2"]), str(sample["product_id"])) for sample in samples
    ]
    if actual_samples != expected_samples:
        raise ValueError("RAG v2 holdout ID 未按冻结抽样规则复现。")
    if selection.get("candidate_counts") != candidate_counts:
        raise ValueError("RAG v2 holdout candidate_counts 与冻结抽样规则不一致。")
    return set(product_ids)


def load_development_product_ids(path: Path) -> list[str]:
    report = load_json(path)
    if report.get("scope") != "validation_only":
        raise ValueError("P2 开发样本来源必须是 validation_only 结果。")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("P2 开发样本来源缺少 results 数组。")
    product_ids = [str(result["product_id"]) for result in results]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("P2 开发样本来源 product_id 重复。")
    return product_ids


def select_development_samples(
    records: list[dict[str, Any]],
    development_product_ids: list[str],
    audit: dict[str, dict[str, str]],
    *,
    per_category: int,
    protected_product_ids: set[str],
) -> list[dict[str, Any]]:
    """从已有 RAG 开发样本中按原始顺序选择，绝不触碰冻结 holdout。"""
    by_id = {str(record["product_id"]): record for record in records}
    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    for product_id in development_product_ids:
        if product_id in protected_product_ids:
            raise ValueError(f"P2 开发样本与冻结 holdout 重叠：{product_id}")
        record = by_id.get(product_id)
        if record is None:
            raise ValueError(f"P2 开发样本不在 validation 源数据中：{product_id}")
        if audit[product_id]["status"] != "PASS":
            raise ValueError(f"P2 开发样本不是 PASS：{product_id}")
        selected_by_category.setdefault(record["category_l2"], []).append(record)

    selected = []
    for category in sorted(selected_by_category):
        candidates = selected_by_category[category]
        if len(candidates) < per_category:
            raise ValueError(
                f"P2 开发样本中品类 {category} 只有 {len(candidates)} 条，少于 {per_category}。"
            )
        selected.extend(candidates[:per_category])
    return selected


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
        "prompt_preflight": dict(generator.last_prompt_preflight or {}),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary.replace(path)


def mandatory_output_coverage(
    parsed_output: dict[str, Any] | None,
    rag_context: dict[str, Any],
) -> dict[str, Any]:
    """提供开发期的字面覆盖诊断，不替代人工 attribute-hit 评测。"""
    facts = rag_context.get("mandatory_core_facts", [])
    if parsed_output is None:
        return {
            "method": "literal_value_diagnostic_not_official_attribute_hit",
            "expected_count": len(facts),
            "covered_count": 0,
            "missing_fact_ids": [fact["fact_id"] for fact in facts],
        }
    output_text = "\n".join(
        [
            str(parsed_output["generated_title"]),
            *[str(point) for point in parsed_output["selling_points"]],
            str(parsed_output["short_description"]),
        ]
    )
    missing = [
        fact["fact_id"]
        for fact in facts
        if not all(str(value) in output_text for value in fact["normalized_values"])
    ]
    return {
        "method": "literal_value_diagnostic_not_official_attribute_hit",
        "expected_count": len(facts),
        "covered_count": len(facts) - len(missing),
        "missing_fact_ids": missing,
    }


def run_v2_paired_comparison(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    started_at: str,
) -> None:
    """运行 P2/P2.6 配置声明的同商品、同 seed 配对比较。"""
    if args.top_k != [3]:
        raise ValueError("RAG v2 P2 固定 Supplemental Top-3，不接受其他 --top-k。")
    validation_path = project_path(config["validation_source"]).resolve()
    audit_path = project_path(config["audit_path"]).resolve()
    holdout_path = project_path(config["protected_holdout"]["path"]).resolve()
    development_path = project_path(config["development_sample_source"]).resolve()
    protected_ids = verify_p2_holdout_guard(
        holdout_path,
        expected_manifest_sha256=config["protected_holdout"]["sha256"],
        validation_path=validation_path,
        audit_path=audit_path,
    )
    records = load_jsonl(validation_path)
    audit = load_quality_audit(audit_path, "validation")
    record_ids = {str(record["product_id"]) for record in records}
    if record_ids != set(audit):
        raise ValueError("validation 与质量审计的商品集合不一致。")
    development_ids = load_development_product_ids(development_path)
    samples = select_development_samples(
        records,
        development_ids,
        audit,
        per_category=args.per_category,
        protected_product_ids=protected_ids,
    )
    selected_ids = {str(record["product_id"]) for record in samples}
    overlap = selected_ids & protected_ids
    if overlap:
        raise ValueError(f"P2 拒绝运行冻结 generation-output holdout：{sorted(overlap)}")

    variant_resources: dict[str, dict[str, Any]] = {}
    for variant in config["comparison_variants"]:
        name = str(variant["name"])
        if name in variant_resources:
            raise ValueError(f"比较变体名称重复：{name}")
        facts_path = project_path(variant["fact_units_path"]).resolve()
        policy_path = project_path(variant["fact_policy_path"]).resolve()
        facts_by_product: dict[str, list[dict[str, Any]]] = {}
        for fact in load_jsonl(facts_path):
            facts_by_product.setdefault(str(fact["product_id"]), []).append(fact)
        if record_ids != set(facts_by_product):
            raise ValueError(f"{name} 的事实知识库与 validation 商品集合不一致。")
        variant_resources[name] = {
            "config": variant,
            "facts_path": facts_path,
            "policy_path": policy_path,
            "facts_by_product": facts_by_product,
            "policy": load_fact_policy(policy_path),
        }
    expected_names = [str(name) for name in config["pairing"]["variants"]]
    if list(variant_resources) != expected_names:
        raise ValueError(f"比较变体必须与 pairing.variants 按相同顺序配置：{expected_names}")

    # 所有保护检查均在首次构造模型、调用 Qwen 之前完成。
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
        sample_seed = paired_sample_seed(int(config["seed"]), sample_index)
        generation_input = {
            "category_l1": record["category_l1"],
            "category_l2": record["category_l2"],
            "attributes": record["attributes"],
        }
        variant_results = {}
        for name, resource in variant_resources.items():
            variant = resource["config"]
            facts = resource["facts_by_product"][product_id]
            if variant["context_selector"] == "rag_v1_global_top_k":
                context = select_global_top_k_facts(
                    facts,
                    resource["policy"],
                    top_k=3,
                    separate_negative_constraints=True,
                )
            elif variant["context_selector"] == "rag_v2_four_zone":
                context = select_rag_v2_context(facts, resource["policy"], top_k=3)
            else:
                raise ValueError(f"不支持的 context_selector：{variant['context_selector']}")
            generation = run_generation(
                generator,
                generation_input,
                prompt_version=variant["prompt_version"],
                seed=sample_seed,
                rag_context=context,
            )
            if config.get("grounding_validator", {}).get("enabled", False):
                generation["grounding_validation"] = (
                    validate_generation_grounding(generation["parsed_output"], context)
                    if generation["parsed_output"] is not None
                    else None
                )
            generation["mandatory_output_coverage"] = mandatory_output_coverage(
                generation["parsed_output"], context
            )
            variant_results[name] = {"retrieval": context, **generation}
        if len({result["seed"] for result in variant_results.values()}) != 1:
            raise RuntimeError(f"商品 {product_id} 的比较变体未使用同一 seed。")
        results.append(
            {
                "sample_index": sample_index,
                "product_id": product_id,
                "category_l1": record["category_l1"],
                "category_l2": record["category_l2"],
                "audit_status": audit[product_id]["status"],
                "paired_seed": sample_seed,
                "generation_input": generation_input,
                "reference_title_not_given_to_model": record["title"],
                "variants": variant_results,
            }
        )
        parsed = sum(
            result["parsed_output"] is not None for result in variant_results.values()
        )
        print(
            f"[{len(results)}/{len(samples)}] product_id={product_id} "
            f"category={record['category_l2']} paired_seed={sample_seed} "
            f"parsed={parsed}/{len(expected_names)}",
            flush=True,
        )

    summary = {}
    for name in expected_names:
        variant_results = [row["variants"][name] for row in results]
        expected_mandatory = sum(
            result["prompt_preflight"].get("mandatory_expected_count", 0)
            for result in variant_results
        )
        verified_mandatory = sum(
            result["prompt_preflight"].get("mandatory_verified_count", 0)
            for result in variant_results
        )
        summary[name] = {
            "sample_count": len(variant_results),
            "parsed_output_count": sum(
                result["parsed_output"] is not None for result in variant_results
            ),
            "grounding_pass_count": sum(
                result.get("grounding_validation", {}).get("status") == "PASS"
                for result in variant_results
                if result.get("grounding_validation") is not None
            ),
            "average_generation_seconds": round(
                sum(result["generation_seconds"] for result in variant_results)
                / len(variant_results),
                3,
            ),
            "mandatory_injection_expected_count": expected_mandatory,
            "mandatory_injection_verified_count": verified_mandatory,
            "mandatory_injection_complete": expected_mandatory == verified_mandatory,
            "official_attribute_hit_metrics": "not_computed_requires_human_review",
        }

    report = {
        "version": config["version"],
        "status": "completed",
        "scope": "validation_development_only",
        "formal_test100_executed": False,
        "generation_output_holdout_executed": False,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_identity,
        "quantization": "bitsandbytes_nf4_4bit" if config["load_in_4bit"] else "none",
        "pairing": config["pairing"],
        "sample_selection": {
            "source": str(development_path),
            "per_category": args.per_category,
            "sample_count": len(samples),
            "protected_holdout_overlap_count": 0,
        },
        "denominator_contract": config["denominator_contract"],
        "summary": summary,
        "artifacts": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "validation": {"path": str(validation_path), "sha256": sha256(validation_path)},
            "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
            "development_sample_source": {
                "path": str(development_path),
                "sha256": sha256(development_path),
            },
            "protected_holdout": {
                "path": str(holdout_path),
                "sha256": sha256(holdout_path),
            },
            "variants": {
                name: {
                    "facts": {
                        "path": str(resource["facts_path"]),
                        "sha256": sha256(resource["facts_path"]),
                    },
                    "policy": {
                        "path": str(resource["policy_path"]),
                        "sha256": sha256(resource["policy_path"]),
                    },
                }
                for name, resource in variant_resources.items()
            },
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


def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    top_k_values = sorted(set(args.top_k))
    if any(top_k < 1 for top_k in top_k_values):
        raise ValueError("所有 Top-K 都必须大于0。")
    config_path = args.config.resolve()
    config = load_json(config_path)
    if config.get("mode") == "rag_v2_paired_comparison":
        run_v2_paired_comparison(
            args=args,
            config=config,
            config_path=config_path,
            started_at=started_at,
        )
        return
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
        paired_seed = paired_sample_seed(int(config["seed"]), sample_index)
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
