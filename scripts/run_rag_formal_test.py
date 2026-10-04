"""按冻结的 rag_v1 formal 配置单次运行 generation test100。"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from platform import python_version
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.grounding import validate_generation_grounding  # noqa: E402
from src.generation.qwen import (  # noqa: E402
    QwenGenerator,
    build_rag_messages,
    parse_generation_output,
)
from src.generation.rag import (  # noqa: E402
    build_fact_units,
    load_fact_policy,
    load_quality_audit,
    select_global_top_k_facts,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/generation_rag_formal_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports/generation/rag/rag_v1_formal_manifest.json"
FORMAL_VERSION = "rag_v1_formal"
FORMAL_PROMPT_VERSION = "rag_v6"
FORMAL_POLICY_VERSION = "rag_fact_policy_v3"
FORMAL_TOP_K = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="只核对冻结配置、哈希、模型快照与输出路径；不解析 test100，不加载模型。",
    )
    mode.add_argument(
        "--confirm-formal-test100",
        action="store_true",
        help="显式确认按冻结配置执行一次正式 test100。",
    )
    return parser.parse_args()


def project_path(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象：{path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"JSONL 第 {line_number} 行不是对象：{path}")
            records.append(record)
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def function_source_sha256(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"找不到待校验函数 {function_name}：{path}")
    segment = ast.get_source_segment(source, function)
    if segment is None:
        raise ValueError(f"无法提取待校验函数源码 {function_name}：{path}")
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA256 不匹配：expected={expected}, observed={observed}, path={path}"
        )
    return observed


def formal_output_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    paths = {
        "test100_fact_units": project_path(config["rag"]["test100_fact_units_path"], project_root)
    }
    for name, value in config["formal_outputs"].items():
        if name == "overwrite_existing":
            continue
        paths[name] = project_path(value, project_root)
    if len(set(paths.values())) != len(paths):
        raise ValueError("formal 输出路径存在重复。")
    return paths


def ensure_outputs_absent(paths: dict[str, Path]) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        display = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"formal 输出已经存在，拒绝覆盖：\n{display}")


def validate_frozen_values(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    if config.get("version") != FORMAL_VERSION:
        raise ValueError(f"配置版本必须为 {FORMAL_VERSION}。")
    if config.get("status") != "frozen_pending_single_test100_execution":
        raise ValueError("formal 配置状态不是待单次执行的冻结状态。")
    if config.get("formal_test100_executed") is not False:
        raise ValueError("formal 配置必须记录 formal_test100_executed=false。")
    if manifest.get("status") != "frozen_pending_single_test100_execution":
        raise ValueError("formal manifest 状态不是待单次执行的冻结状态。")
    if manifest.get("formal_test100_executed") is not False:
        raise ValueError("formal manifest 必须记录 formal_test100_executed=false。")
    if config["prompt"].get("version") != FORMAL_PROMPT_VERSION:
        raise ValueError(f"Prompt 版本必须为 {FORMAL_PROMPT_VERSION}。")
    if config["rag"].get("policy_version") != FORMAL_POLICY_VERSION:
        raise ValueError(f"RAG policy 版本必须为 {FORMAL_POLICY_VERSION}。")
    if int(config["rag"].get("top_k", -1)) != FORMAL_TOP_K:
        raise ValueError(f"formal Top-K 必须为 {FORMAL_TOP_K}。")
    if config["rag"]["joint_ranking"].get("method") != "reciprocal_rank_fusion":
        raise ValueError("formal 联合排序必须使用 reciprocal_rank_fusion。")
    if config["rag"]["joint_ranking"].get("rrf_k") != 60:
        raise ValueError("formal RRF k 必须为 60。")
    if not config["rag"]["negative_constraints"].get("enabled"):
        raise ValueError("formal 只读否定约束必须启用。")
    if config["formal_outputs"].get("overwrite_existing") is not False:
        raise ValueError("formal 输出必须禁止覆盖。")
    execution = config.get("execution", {})
    if execution.get("entrypoint") != "scripts/run_rag_formal_test.py":
        raise ValueError("formal 执行入口没有锁定到专用脚本。")
    if execution.get("seed_rule") != "base_seed_plus_stable_sample_index":
        raise ValueError("formal 随机种子规则不正确。")
    if execution.get("local_files_only") is not True:
        raise ValueError("formal 模型加载必须限制为本地缓存。")


def preflight(
    config_path: Path,
    manifest_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """只做冻结资产和输出保护检查，不解析 test100，也不加载模型。"""
    config_path = config_path.resolve()
    manifest_path = manifest_path.resolve()
    config = load_json(config_path)
    manifest = load_json(manifest_path)
    validate_frozen_values(config, manifest)

    require_hash(
        config_path,
        manifest["formal_config"]["sha256"],
        "formal 配置",
    )
    policy_path = project_path(config["rag"]["policy_path"], project_root)
    require_hash(policy_path, config["rag"]["policy_sha256"], "RAG policy")
    if manifest["rag_policy"]["sha256"] != config["rag"]["policy_sha256"]:
        raise ValueError("formal 配置与 manifest 记录的 RAG policy 哈希不一致。")

    prompt_path_value, prompt_symbol = config["prompt"]["implementation"].split("::", 1)
    prompt_path = project_path(prompt_path_value, project_root)
    prompt_sha = function_source_sha256(prompt_path, prompt_symbol)
    if prompt_sha != config["prompt"]["implementation_sha256"]:
        raise ValueError("Prompt 实现 SHA256 与 formal 配置不一致。")
    if prompt_sha != manifest["prompt"]["sha256"]:
        raise ValueError("Prompt 实现 SHA256 与 formal manifest 不一致。")

    test100_path = project_path(config["frozen_test100"]["path"], project_root)
    require_hash(test100_path, config["frozen_test100"]["sha256"], "冻结 test100")
    if manifest["frozen_test100"]["sha256"] != config["frozen_test100"]["sha256"]:
        raise ValueError("formal 配置与 manifest 记录的 test100 哈希不一致。")

    audit_path = project_path(config["rag"]["quality_audit_path"], project_root)
    require_hash(audit_path, config["rag"]["quality_audit_sha256"], "事实质量审计")
    if manifest["quality_audit"]["sha256"] != config["rag"]["quality_audit_sha256"]:
        raise ValueError("formal 配置与 manifest 记录的事实质量审计哈希不一致。")
    baseline_manifest_path = project_path(
        config["baseline_comparison"]["manifest_path"], project_root
    )
    require_hash(
        baseline_manifest_path,
        config["baseline_comparison"]["manifest_sha256"],
        "Baseline manifest",
    )
    if manifest["model"]["name"] != config["model"]["name"]:
        raise ValueError("formal 配置与 manifest 记录的模型名称不一致。")
    if manifest["model"]["revision"] != config["model"]["revision"]:
        raise ValueError("formal 配置与 manifest 记录的模型 revision 不一致。")
    if manifest["generation_parameters"] != config["generation_parameters"]:
        raise ValueError("formal 配置与 manifest 记录的生成参数不一致。")
    if manifest["validator"]["version"] != config["validator"]["version"]:
        raise ValueError("formal 配置与 manifest 记录的 validator 版本不一致。")

    runtime_artifacts = manifest["code_version"].get("formal_runtime_artifacts", [])
    if not runtime_artifacts:
        raise ValueError("formal manifest 未记录正式执行所需的代码文件哈希。")
    for artifact in runtime_artifacts:
        artifact_path = project_path(artifact["path"], project_root)
        require_hash(artifact_path, artifact["sha256"], f"formal 运行代码 {artifact['path']}")

    snapshot_path = Path(config["model"]["observed_snapshot_path"])
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"冻结模型 snapshot 不存在：{snapshot_path}")

    outputs = formal_output_paths(config, project_root)
    ensure_outputs_absent(outputs)
    return {
        "config": config,
        "manifest": manifest,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "policy_path": policy_path,
        "test100_path": test100_path,
        "audit_path": audit_path,
        "output_paths": outputs,
        "verified": {
            "config_sha256": file_sha256(config_path),
            "manifest_sha256": file_sha256(manifest_path),
            "prompt_sha256": prompt_sha,
            "policy_sha256": file_sha256(policy_path),
            "test100_sha256": file_sha256(test100_path),
            "audit_sha256": file_sha256(audit_path),
        },
    }


def build_formal_inputs(
    records: list[dict[str, Any]],
    audit: dict[str, dict[str, str]],
    policy: dict[str, Any],
    *,
    dataset_version: str,
    source_path: str,
    source_sha256: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """保留每条冻结样本，构建事实单元和对应的 formal RAG 上下文。"""
    product_ids = [str(record.get("product_id", "")) for record in records]
    if any(not product_id for product_id in product_ids):
        raise ValueError("test100 存在缺少 product_id 的记录。")
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("test100 存在重复 product_id。")
    if set(product_ids) != set(audit):
        raise ValueError(
            "test100 与质量审计商品集合不一致："
            f"缺少审计={len(set(product_ids) - set(audit))}，"
            f"多余审计={len(set(audit) - set(product_ids))}"
        )

    all_facts: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for sample_index, record in enumerate(records):
        product_id = str(record["product_id"])
        generation_input = record.get("generation_input")
        if not isinstance(generation_input, dict):
            raise ValueError(f"商品 {product_id} 缺少冻结 generation_input。")
        fact_record = dict(record)
        fact_record["split"] = "test100"
        facts = build_fact_units(
            fact_record,
            audit[product_id],
            policy,
            dataset_version=dataset_version,
            source_path=source_path,
            source_sha256=source_sha256,
        )
        context = select_global_top_k_facts(
            facts,
            policy,
            top_k=top_k,
            separate_negative_constraints=True,
        )
        if context["product_id"] != product_id:
            raise ValueError(f"商品 {product_id} 的事实上下文 product_id 不一致。")
        build_rag_messages(generation_input, context, prompt_version=FORMAL_PROMPT_VERSION)
        all_facts.extend(facts)
        prepared.append(
            {
                "sample_index": sample_index,
                "product_id": product_id,
                "category_l1": record["category_l1"],
                "category_l2": record["category_l2"],
                "audit_status": audit[product_id]["status"],
                "generation_input": generation_input,
                "reference_title_not_given_to_model": record.get("title", ""),
                "rag_context": context,
            }
        )
    return all_facts, prepared


def qwen_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": config["model"]["name"],
        "revision": config["model"]["revision"],
        "cache_dir": config["model"]["cache_dir"],
        "device": config["model"]["device"],
        "load_in_4bit": config["model"]["load_in_4bit"],
        "bnb_4bit_quant_type": config["model"]["bnb_4bit_quant_type"],
        "bnb_4bit_compute_dtype": config["model"]["bnb_4bit_compute_dtype"],
        **config["generation_parameters"],
        "prompt_version": config["prompt"]["version"],
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_model_output(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return parse_generation_output(raw_output), None
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return None, str(error)


def run_formal_test(preflight_result: dict[str, Any]) -> None:
    config = preflight_result["config"]
    test100_path = preflight_result["test100_path"]
    policy = load_fact_policy(preflight_result["policy_path"])
    if policy.get("version") != FORMAL_POLICY_VERSION:
        raise ValueError("加载后的 RAG policy 版本不正确。")
    records = load_jsonl(test100_path)
    expected_count = int(config["frozen_test100"]["sample_count"])
    if len(records) != expected_count:
        raise ValueError(f"冻结 test100 应有 {expected_count} 条，实际为 {len(records)} 条。")
    audit = load_quality_audit(preflight_result["audit_path"], "test100")
    source_path = config["frozen_test100"]["path"]
    all_facts, prepared = build_formal_inputs(
        records,
        audit,
        policy,
        dataset_version="week1_v3",
        source_path=source_path,
        source_sha256=config["frozen_test100"]["sha256"],
        top_k=int(config["rag"]["top_k"]),
    )

    runtime_config = qwen_runtime_config(config)
    cache_dir = project_path(runtime_config["cache_dir"])
    load_started = time.perf_counter()
    generator = QwenGenerator(runtime_config, cache_dir, local_files_only=True)
    load_seconds = time.perf_counter() - load_started
    model_identity = generator.model_identity()
    if model_identity["revision"] != config["model"]["revision"]:
        raise RuntimeError("实际加载的模型 revision 与 formal 配置不一致。")
    if not model_identity["snapshot_path"]:
        raise RuntimeError("无法确认实际加载的本地模型 snapshot。")

    started_at = datetime.now(timezone.utc).isoformat()
    results = []
    validator_rows = []
    base_seed = int(config["generation_parameters"]["seed"])
    for prepared_record in prepared:
        sample_index = int(prepared_record["sample_index"])
        seed = base_seed + sample_index
        generation_started = time.perf_counter()
        raw_output = generator.generate(
            prepared_record["generation_input"],
            prompt_version=config["prompt"]["version"],
            rag_context=prepared_record["rag_context"],
            seed=seed,
        )
        generation_seconds = time.perf_counter() - generation_started
        parsed_output, parse_error = parse_model_output(raw_output)
        messages = build_rag_messages(
            prepared_record["generation_input"],
            prepared_record["rag_context"],
            prompt_version=config["prompt"]["version"],
        )
        grounding = validate_generation_grounding(
            parsed_output, prepared_record["rag_context"]
        )
        result = {
            **prepared_record,
            "prompt_version": config["prompt"]["version"],
            "seed": seed,
            "messages": messages,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "parse_error": parse_error,
            "generation_seconds": round(generation_seconds, 3),
            "grounding_validation": grounding,
        }
        results.append(result)
        validator_rows.append(
            {
                "sample_index": sample_index,
                "product_id": prepared_record["product_id"],
                "audit_status": prepared_record["audit_status"],
                "parse_error": parse_error,
                "grounding_validation": grounding,
            }
        )
        print(
            f"[{len(results)}/{len(prepared)}] product_id={prepared_record['product_id']} "
            f"parse={'ok' if parsed_output is not None else 'failed'}",
            flush=True,
        )

    outputs = preflight_result["output_paths"]
    ensure_outputs_absent(outputs)
    completed_at = datetime.now(timezone.utc).isoformat()
    report = {
        "version": FORMAL_VERSION,
        "status": "completed",
        "scope": "formal_test100",
        "formal_test100_executed": True,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "model": model_identity,
        "prompt_version": config["prompt"]["version"],
        "rag_policy_version": config["rag"]["policy_version"],
        "top_k": config["rag"]["top_k"],
        "generation_parameters": config["generation_parameters"],
        "seed_rule": config["execution"]["seed_rule"],
        "sample_count": len(results),
        "audit_status_counts": dict(
            sorted(Counter(record["audit_status"] for record in prepared).items())
        ),
        "parsed_output_count": sum(
            result["parsed_output"] is not None for result in results
        ),
        "model_load_seconds": round(load_seconds, 3),
        "python_version": python_version(),
        "artifacts": preflight_result["verified"],
        "results": results,
    }
    atomic_write_jsonl(outputs["test100_fact_units"], all_facts)
    atomic_write_json(outputs["raw_outputs_path"], report)
    validator_report = {
        "version": config["validator"]["version"],
        "status": "completed",
        "scope": "formal_test100",
        "formal_test100_executed": True,
        "mode": config["validator"]["mode"],
        "replaces_human_evaluation": False,
        "source_outputs": {
            "path": config["formal_outputs"]["raw_outputs_path"],
            "sha256": file_sha256(outputs["raw_outputs_path"]),
        },
        "sample_count": len(validator_rows),
        "results": validator_rows,
    }
    atomic_write_json(outputs["validator_report_path"], validator_report)
    print(
        json.dumps(
            {
                "status": "completed",
                "formal_test100_executed": True,
                "sample_count": len(results),
                "outputs": {
                    name: str(path) for name, path in outputs.items() if path.exists()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    result = preflight(args.config, args.manifest)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "formal_test100_executed": False,
                    "test100_content_parsed": False,
                    "model_loaded": False,
                    "verified": result["verified"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_formal_test(result)


if __name__ == "__main__":
    main()
