"""Isolated final test100 orchestration; inherited P7 inference stays immutable."""
from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from collections import Counter

from scripts.prepare_generation_evaluation import select_core_attributes
from .lora_data import TASK_TYPES, build_instruction_text, select_reliable_core_attributes
from .lora_training import ROOT, atomic_json, digest, json_digest, memory, now, read_json, resolve, verify_checkpoint
from .lora_validation import (CODE_FILES as P7_CODE_FILES, PACKAGE_NAMES, append_jsonl,
                              assemble, blind_mapping, load_jsonl, model_snapshot, parse_task,
                              seed_for, verify_protocol as verify_validation_protocol)
from .qwen import preflight_chat_prompt
from .qlora_memory import prepare_kbit_with_cpu_staged_large_layers
from .rag import _canonical_identity_field, _identity_conflict, _split_blocked_fields

CONFIG = "configs/lora_project_test100_v1.json"
DOCUMENT = "docs/lora_project_test100_v1_protocol.md"
OUTPUT = "reports/generation/lora/project_test100_v1"
ALLOWED = {
    "validation_config": "configs/lora_project_validation_v1.json",
    "validation_protocol": "reports/generation/lora/project_validation_v1/protocol_manifest.json",
    "source": "data/processed/week1_v3/generation_evaluation_samples.jsonl",
    "project_test_parent": "data/processed/week1_v3/multimodal/test.jsonl",
    "selection_manifest": "reports/generation/evaluation/annotation_manifest.json",
    "source_audit": "reports/data/week2_rag_fact_quality_audit.csv",
}
NEW_CODE = ("scripts/run_lora_project_test.py", "src/generation/lora_formal_test.py",
            "src/generation/lora_formal_review.py", "tests/test_lora_formal_test.py")


def checked_config():
    cfg = read_json(resolve(CONFIG))
    for key, allowed in ALLOWED.items():
        if resolve(cfg[key]["path"]) != resolve(allowed):
            raise ValueError(f"Forbidden input before read: {key}")
    if cfg["scope"] != "project_test_frozen100_only" or resolve(cfg["output"]) != resolve(OUTPUT):
        raise ValueError("Only the authorized fixed test100 scope/output is allowed")
    expected = {"single_attempt_per_variant": True, "resume_allowed": False,
                "output_based_adjustment": False, "parameter_updates": False,
                "rag_holdout_files_read": False, "rag_v2_generation": False,
                "old_model_outputs_or_labels_as_inputs": False}
    if cfg["guards"] != expected or cfg["review"]["products"] != 100 or cfg["source"]["products"] != 100:
        raise ValueError("Formal scope/guard changed")
    if cfg["review"]["selection"] != "all_frozen_test100_before_first_generation" or cfg["source_audit"]["scope"] != "test100":
        raise ValueError("No output-based selection or audit changes")
    for key in ALLOWED:
        if digest(resolve(cfg[key]["path"])) != cfg[key]["sha256"]:
            raise ValueError(f"Frozen source/config changed: {key}")
    parent, validation = verify_validation_protocol()
    return cfg, parent, validation


def audit_metadata(rows, scope):
    """Same flagged-field/identity gates as P7 fact metadata, no RAG selection."""
    result = {}
    for row in rows:
        if row["scope"] != scope:
            continue
        pid = str(row["product_id"])
        if pid in result or row["status"] not in {"PASS", "REVIEW", "CONFLICT"}:
            raise ValueError("Duplicate/invalid source audit")
        blocked = _split_blocked_fields(row)
        result[pid] = {"status": row["status"], "blocked_fields": sorted(blocked | {
            _canonical_identity_field(f) or f for f in blocked}),
            "identity_block": _identity_conflict(row, blocked), "note": row.get("review_note", "")}
    return result


def reliable_attributes(record, definition, p5, meta):
    reliable, actions = select_reliable_core_attributes(record, definition["core_attributes"][record["category_l2"]], p5)
    for field in list(reliable):
        # P7 only withdraws flagged fields for REVIEW/CONFLICT, not PASS.
        flagged = meta["status"] in {"REVIEW", "CONFLICT"} and field in meta["blocked_fields"]
        if meta["identity_block"] or flagged:
            actions.append(f"withhold_source_audit:{field}")
            del reliable[field]
    return reliable, actions


def build_inputs(records, audit, p5, definition, inherited):
    samples, seen = [], set()
    for record in records:
        pid = str(record["product_id"])
        if record.get("split") != "test" or pid in seen or pid not in audit:
            raise ValueError("Non-test/duplicate/unaudited frozen sample")
        seen.add(pid)
        core = select_core_attributes(record, definition)
        if core != record["generation_input"]["attributes"] or not core:
            raise ValueError("Frozen test100 original core denominator changed")
        meta = audit[pid]
        used, actions = reliable_attributes(record, definition, p5, meta)
        tasks = {}
        for task in TASK_TYPES:
            messages = [{"role": "system", "content": inherited["prompt"]["system"]},
                        {"role": "user", "content": build_instruction_text(record["category_l1"], record["category_l2"], task, used)}]
            tasks[task] = {"messages": messages, "prompt_sha256": json_digest(messages),
                           "seed": seed_for(pid, task, inherited["pairing"]["base_seed"])}
        samples.append({"product_id": pid, "category_l1": record["category_l1"], "category_l2": record["category_l2"],
                        "source_title": record["title"], "source_attributes": record["attributes"],
                        "evaluation_attributes": core, "core_attribute_count": len(core),
                        "used_attributes": used, "input_quality_actions": actions, "source_quality": meta, "tasks": tasks})
    if seen != set(audit) or len(samples) != 100:
        raise ValueError("Must retain all 100 frozen source/audit products")
    seeds = [t["seed"] for s in samples for t in s["tasks"].values()]
    if len(set(seeds)) != 300:
        raise ValueError("Task seed collision")
    return samples


def prepare():
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    cfg, parent, validation = checked_config()
    out = resolve(OUTPUT)
    if out.exists():
        raise FileExistsError("Never overwrite a frozen formal protocol")
    definition = read_json(resolve(parent["evaluation_definition"]["path"]))
    p5 = read_json(resolve(parent["p5_config"]["path"]))
    with resolve(cfg["source_audit"]["path"]).open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    # Full readonly P7 input comparison proves the adapter-independent audit path
    # did not introduce new field gates while adapting to the final split.
    val_audit = audit_metadata(audit_rows, "validation")
    val_inputs = load_jsonl(resolve(parent["output"]) / "frozen_inputs.jsonl")
    for s in val_inputs:
        record = {"attributes": s["source_attributes"], "category_l2": s["category_l2"]}
        used, actions = reliable_attributes(record, definition, p5, val_audit[s["product_id"]])
        if used != s["used_attributes"] or actions != s["input_quality_actions"]:
            raise ValueError("Source audit gate is not equivalent to frozen validation")
    records = load_jsonl(resolve(cfg["source"]["path"]))
    selection = read_json(resolve(cfg["selection_manifest"]["path"]))
    if ([str(r["product_id"]) for r in records] != selection["product_ids"]
            or selection["samples_sha256"] != cfg["source"]["sha256"]
            or selection["source_dataset_sha256"] != cfg["project_test_parent"]["sha256"]):
        raise ValueError("No resampling/reordering of the historical test100")
    test_audit = audit_metadata(audit_rows, "test100")
    samples = build_inputs(records, test_audit, p5, definition, parent)
    training = read_json(resolve(parent["frozen_training_config"]["path"]))
    for role in ("lora_train", "lora_train_dev"):
        entry = training["data"][role]
        if digest(resolve(entry["path"])) != entry["sha256"]:
            raise ValueError("P5 data changed")
        if {s["product_id"] for s in samples} & {str(r["product_id"]) for r in load_jsonl(resolve(entry["path"]))}:
            raise ValueError("Test100 overlaps LoRA update/dev data")
    snapshot = model_snapshot(parent)
    for name, expected in validation["model_metadata_sha256"].items():
        if digest(snapshot / name) != expected:
            raise ValueError("Frozen base metadata changed")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if json_digest(tokenizer.chat_template) != validation["chat_template_sha256"]:
        raise ValueError("Frozen chat template changed")
    generation = GenerationConfig.from_pretrained(str(snapshot), local_files_only=True)
    generation.update(**parent["decoding"], pad_token_id=tokenizer.eos_token_id)
    generation.validate()
    previous_inference = read_json(resolve(parent["output"]) / "inference_base.json")
    if generation.to_dict() != previous_inference["decoding_resolved"]:
        raise ValueError("Effective decoding differs from validation")
    for s in samples:
        for task in TASK_TYPES:
            _, meta = preflight_chat_prompt(tokenizer, s["tasks"][task]["messages"],
                max_input_tokens=parent["prompt"]["max_input_tokens"],
                mandatory_lines=[json.dumps(s["used_attributes"], ensure_ascii=False, sort_keys=False)])
            s["tasks"][task]["preflight"] = meta
    checkpoint = resolve(parent["adapter"]["path"]).parent
    state = read_json(checkpoint / "trainer_state.json")
    verify_checkpoint(checkpoint, state["identity"])
    if state["identity"]["run_kind"] != "formal_training" or state["identity"]["config_sha256"] != parent["frozen_training_config"]["sha256"]:
        raise ValueError("Not the frozen formal best adapter")
    environment = {"python": sys.version, "packages": {n: importlib.metadata.version(n) for n in PACKAGE_NAMES}, "cuda": torch.version.cuda}
    if environment != validation["environment"]:
        raise RuntimeError("Use the validated inference environment without upgrades")
    out.mkdir(parents=True)
    for s in samples:
        append_jsonl(out / "frozen_inputs.jsonl", s)
    mapping = blind_mapping(samples, cfg["review"]["blind_seed"])
    for p in mapping:
        p["pair_id"] = p["pair_id"].replace("PV", "PT")
    atomic_json(out / "blind_mapping.json", {"status": "sealed_until_review_complete", "pairs": mapping})
    atomic_json(out / "human_scope_manifest.json", {"status": "frozen_before_first_generation", "selection": "all_frozen_test100",
                "products": 100, "seed": cfg["review"]["blind_seed"], "selected_ids_in_source_order": [s["product_id"] for s in samples],
                "by_category": dict(Counter(s["category_l2"] for s in samples)), "generated_outputs_used": False,
                "anonymous_mapping_sha256": digest(out / "blind_mapping.json")})
    common = {"protocol_version": cfg["version"], "model": parent["model"], "base_snapshot": str(snapshot),
              "prompt": parent["prompt"], "decoding_resolved": generation.to_dict(), "pairing": parent["pairing"]}
    for variant in ("Base", "LoRA"):
        atomic_json(out / f"inference_{variant.lower()}.json", {**common, "variant": variant,
                    "adapter": parent["adapter"] if variant == "LoRA" else None})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (out / "environment_freeze.txt").write_text(freeze, encoding="utf-8")
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    paths = [CONFIG, DOCUMENT, *NEW_CODE, *P7_CODE_FILES, *[cfg[k]["path"] for k in ALLOWED]]
    paths += [parent[k]["path"] for k in ("p5_config", "evaluation_definition", "frozen_training_config")]
    paths += [parent["adapter"]["path"] + "/" + n for n in ("adapter_model.safetensors", "adapter_config.json")]
    paths += [p.relative_to(ROOT).as_posix() for p in out.iterdir() if p.is_file()]
    manifest = {"status": "frozen_before_first_generation", "version": cfg["version"], "created_at_utc": now(),
        "config": cfg, "inherited_inference_config": parent, "git_head": git("rev-parse", "HEAD"),
        "git_status": git("-c", "core.quotepath=false", "status", "--short"),
        "code_snapshots_encoding": "hex_original_bytes", "code_snapshots": {n: resolve(n).read_bytes().hex() for n in (*P7_CODE_FILES, *NEW_CODE)},
        "files": [{"path": p, "sha256": digest(resolve(p))} for p in sorted(set(paths))],
        "chat_template_sha256": json_digest(tokenizer.chat_template),
        "task_prompt_bundle_sha256": json_digest([{"product_id": s["product_id"], "tasks": s["tasks"]} for s in samples]),
        "model_metadata_sha256": validation["model_metadata_sha256"], "base_model_revision": parent["model"]["revision"],
        "base_snapshot": str(snapshot), "adapter": parent["adapter"], "environment": environment,
        "source_statistics": {"products": 100, "core_attribute_total": sum(s["core_attribute_count"] for s in samples),
            "by_category": dict(Counter(s["category_l2"] for s in samples)),
            "source_quality": dict(Counter(s["source_quality"]["status"] for s in samples)),
            "input_core_attribute_total": sum(len(s["used_attributes"]) for s in samples),
            "input_core_below_three": sum(len(s["used_attributes"]) < 3 for s in samples),
            "max_prompt_tokens": max(t["preflight"]["input_token_count"] for s in samples for t in s["tasks"].values())},
        "input_policy_equivalence_checked_against_validation_products": len(val_inputs),
        "test_history_disclosure": cfg["test_history_disclosure"], "parameter_updates": 0,
        "rag_holdout_files_read": False, "rag_v2_generation": False}
    atomic_json(out / "protocol_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], **manifest["source_statistics"]}, ensure_ascii=False), flush=True)


def verify_protocol():
    cfg, parent, validation = checked_config()
    manifest = read_json(resolve(OUTPUT) / "protocol_manifest.json")
    if cfg != manifest["config"] or parent != manifest["inherited_inference_config"] or manifest["status"] != "frozen_before_first_generation":
        raise ValueError("Formal protocol changed")
    for item in manifest["files"]:
        if digest(resolve(item["path"])) != item["sha256"]:
            raise ValueError(f"Formal frozen file changed: {item['path']}")
    return cfg, parent, manifest


def claim_run(out, variant):
    if variant not in ("Base", "LoRA"):
        raise ValueError("Unknown variant")
    folder = out / variant.lower()
    folder.mkdir()  # Atomic fail if the attempt already exists, even if incomplete.
    atomic_json(folder / "attempt_started.json", {"variant": variant, "started_at_utc": now(),
                "single_attempt_consumed": True, "resume_allowed": False})
    return folder


def generate(variant):
    cfg, parent, frozen = verify_protocol()
    out = resolve(OUTPUT)
    run_dir = claim_run(out, variant)
    report = {"status": "running", "variant": variant, "started_at_utc": now(), "parameter_updates": 0,
              "project_test_read": True, "rag_holdout_files_read": False, "rag_v2_generation": False}
    begun = time.perf_counter()
    try:
        # Keep the P7 model construction, preflight, sampling, parsing and timing path.
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
        if ({n: importlib.metadata.version(n) for n in PACKAGE_NAMES} != frozen["environment"]["packages"]
                or sys.version != frozen["environment"]["python"] or torch.version.cuda != frozen["environment"]["cuda"]):
            raise RuntimeError("Frozen inference environment changed")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA/BF16 unavailable; no fallback")
        random.seed(parent["pairing"]["base_seed"])
        torch.manual_seed(parent["pairing"]["base_seed"])
        torch.cuda.manual_seed_all(parent["pairing"]["base_seed"])
        snapshot = model_snapshot(parent)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        if json_digest(tokenizer.chat_template) != frozen["chat_template_sha256"]:
            raise ValueError("Chat template changed")
        inference = read_json(out / f"inference_{variant.lower()}.json")
        generation_cfg = GenerationConfig.from_dict(inference["decoding_resolved"])
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, device_map={"": 0},
            quantization_config=quant, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model, _ = prepare_kbit_with_cpu_staged_large_layers(model, prepare_model_for_kbit_training,
                                                            use_gradient_checkpointing=False)
        model.config.use_cache = True
        if variant == "LoRA":
            model = PeftModel.from_pretrained(model, str(resolve(parent["adapter"]["path"])),
                                             is_trainable=False, local_files_only=True)
        model.eval()
        if any(p.requires_grad for p in model.parameters()) or not model.is_loaded_in_4bit:
            raise RuntimeError("Expected frozen, 4-bit inference-only model")
        before = {n: p._version for n, p in model.named_parameters()}
        report.update(pid=__import__("os").getpid(), adapter_loaded=variant == "LoRA", base_snapshot=str(snapshot),
                      gpu=torch.cuda.get_device_name(0), generation_config=generation_cfg.to_dict())
        samples = load_jsonl(out / "frozen_inputs.jsonl")
        for index, sample in enumerate(samples, 1):
            task_results = {}
            for task in TASK_TYPES:
                task_input = sample["tasks"][task]
                inputs, preflight = preflight_chat_prompt(tokenizer, task_input["messages"],
                    max_input_tokens=parent["prompt"]["max_input_tokens"],
                    mandatory_lines=[json.dumps(sample["used_attributes"], ensure_ascii=False, sort_keys=False)])
                if preflight != task_input["preflight"]:
                    raise ValueError("Frozen tokenizer preflight changed")
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                seed = task_input["seed"]
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    generated = model.generate(**inputs, generation_config=generation_cfg)
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                item = {"product_id": sample["product_id"], "variant": variant, "task_type": task,
                        "seed": seed, "prompt_sha256": task_input["prompt_sha256"], "preflight": preflight,
                        "raw_text": text, "parsed": parse_task(task, text), "generated_tokens": len(ids),
                        "hit_max_new_tokens": len(ids) >= parent["decoding"]["max_new_tokens"],
                        "latency_seconds": latency, "memory": memory()}
                task_results[task] = item
                append_jsonl(run_dir / "task_outputs.jsonl", item)
                del generated, inputs
            append_jsonl(run_dir / "product_outputs.jsonl", assemble(sample, task_results, variant))
            print(json.dumps({"variant": variant, "products_complete": index, "total": len(samples),
                              "elapsed_seconds": round(time.perf_counter() - begun, 2)}, ensure_ascii=False), flush=True)
        if before != {n: p._version for n, p in model.named_parameters()} or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Inference modified parameters or accumulated gradients")
        verify_protocol()
        report.update(status="passed", completed_at_utc=now(), products=len(samples), task_calls=len(samples) * 3,
                      wall_seconds=time.perf_counter() - begun, model_parameters_unchanged=True)
        atomic_json(run_dir / "run_report.json", report)
    except Exception as error:
        report.update(status="failed", failed_at_utc=now(), error=str(error))
        atomic_json(run_dir / "run_report.json", report)
        raise
