"""Frozen, task-conditioned Base/LoRA project-validation inputs and inference."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.prepare_generation_evaluation import select_core_attributes
from .lora_data import TASK_TYPES, build_instruction_text, select_reliable_core_attributes
from .lora_training import ROOT, atomic_json, digest, json_digest, memory, now, read_json, resolve, verify_checkpoint
from .qlora_memory import prepare_kbit_with_cpu_staged_large_layers
from .qwen import preflight_chat_prompt

CONFIG_PATH = "configs/lora_project_validation_v1.json"
ALLOWED_INPUTS = {
    "source": "data/processed/week1_v3/multimodal/validation.jsonl",
    "validation_quality_facts": "data/processed/week2_rag_mandatory_core_v2/validation_fact_units.jsonl",
    "p5_config": "configs/lora_instruction_data_v1.json",
    "evaluation_definition": "configs/generation_evaluation.json",
    "frozen_training_config": "configs/lora_qlora_train_v1.json",
}
CODE_FILES = ("scripts/run_lora_project_validation.py", "scripts/review_lora_project_validation.py",
              "src/generation/lora_validation.py", "src/generation/lora_validation_review.py",
              "src/generation/lora_data.py", "src/generation/qlora_memory.py",
              "src/generation/lora_training.py", "src/generation/qwen.py", "src/generation/rag.py",
              "src/generation/evaluation.py", "scripts/prepare_generation_evaluation.py")
PACKAGE_NAMES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors")
PROTOCOL_DOCUMENT = "docs/lora_project_validation_v1_protocol.md"


def checked_config():
    cfg = read_json(resolve(CONFIG_PATH))
    # Reject every forbidden path before hashing/opening any input.
    for key, allowed in ALLOWED_INPUTS.items():
        entry = cfg[key]
        if resolve(entry["path"]) != resolve(allowed):
            raise ValueError(f"Forbidden input path for {key}")
    for key, allowed in ALLOWED_INPUTS.items():
        entry = cfg[key]
        if digest(resolve(allowed)) != entry["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {key}")
    if any(cfg["guards"].values()) or cfg["scope"] != "project_validation_only":
        raise ValueError("P7 safety guards changed")
    if cfg["model"]["revision"] != "a09a35458c702b33eeacc393d103063234e8bc28":
        raise ValueError("Base revision changed")
    adapter = resolve(cfg["adapter"]["path"])
    expected = resolve("artifacts/lora/lora_v1/checkpoints/checkpoint-step-001359/adapter")
    if adapter != expected or digest(adapter / "adapter_model.safetensors") != cfg["adapter"]["weights_sha256"]:
        raise ValueError("Frozen best adapter changed")
    if resolve(cfg["output"]) != resolve("reports/generation/lora/project_validation_v1"):
        raise ValueError("P7 output directory changed")
    return cfg


def seed_for(product_id, task_type, base_seed=42):
    if task_type not in TASK_TYPES:
        raise ValueError("Unknown task")
    value = f"p7_lora_v1:{base_seed}:{product_id}:{task_type}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 2147483647


def quality_metadata(facts):
    grouped = defaultdict(list)
    for fact in facts:
        if fact.get("split") != "validation":
            raise ValueError("Quality facts must be project-validation-only")
        grouped[str(fact["product_id"])].append(fact)
    output = {}
    for pid, rows in grouped.items():
        statuses = {r["evidence"]["audit_status"] for r in rows}
        if len(statuses) != 1 or not statuses <= {"PASS", "REVIEW", "CONFLICT"}:
            raise ValueError("Inconsistent source audit")
        blocked, identity_block = set(), False
        for fact in rows:
            reasons = set(fact.get("quality_reasons", []))
            identity_block |= "identity_or_category_conflict" in reasons
            if reasons & {"field_requires_review", "field_conflict"}:
                blocked.update((fact["field_name"], fact["canonical_field"]))
        output[pid] = {"status": statuses.pop(), "blocked_fields": sorted(blocked),
                       "identity_block": identity_block, "note": rows[0]["evidence"]["audit_note"]}
    return output


def build_inputs(records, facts, p5, definition, cfg):
    audit = quality_metadata(facts)
    samples = []
    seen = set()
    for record in records:
        pid = str(record["product_id"])
        if record.get("split") != "validation" or pid in seen:
            raise ValueError("Non-validation or duplicate source product")
        seen.add(pid)
        meta = audit[pid]
        original_core = select_core_attributes(record, definition)
        reliable, actions = select_reliable_core_attributes(record, definition["core_attributes"][record["category_l2"]], p5)
        for field in list(reliable):
            if meta["identity_block"] or field in meta["blocked_fields"]:
                actions.append(f"withhold_source_audit:{field}")
                del reliable[field]
        if not original_core:
            raise ValueError("Cannot silently alter a zero official denominator")
        tasks = {}
        for task in TASK_TYPES:
            instruction = build_instruction_text(record["category_l1"], record["category_l2"], task, reliable)
            messages = [{"role": "system", "content": cfg["prompt"]["system"]},
                        {"role": "user", "content": instruction}]
            tasks[task] = {"messages": messages, "prompt_sha256": json_digest(messages),
                           "seed": seed_for(pid, task, cfg["pairing"]["base_seed"])}
        samples.append({"product_id": pid, "category_l1": record["category_l1"], "category_l2": record["category_l2"],
                        "source_title": record["title"], "source_attributes": record["attributes"],
                        "evaluation_attributes": original_core, "core_attribute_count": len(original_core),
                        "used_attributes": reliable, "input_quality_actions": actions,
                        "source_quality": meta, "tasks": tasks})
    if set(audit) != seen or len(samples) != cfg["source"]["products"]:
        raise ValueError("Project-validation source/audit coverage mismatch")
    seeds = [v["seed"] for s in samples for v in s["tasks"].values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Per-product/task seed collision")
    return samples


def blind_mapping(samples, seed):
    rank = lambda ns, pid: hashlib.sha256(f"{ns}:{seed}:{pid}".encode()).hexdigest()
    assignment = sorted(samples, key=lambda s: rank("p7_ab_assignment", s["product_id"]))
    base_as_a = {s["product_id"] for s in assignment[:len(samples) // 2]}
    order = sorted(samples, key=lambda s: rank("p7_review_order", s["product_id"]))
    return [{"pair_id": f"PV{i:03d}", "product_id": s["product_id"],
             "A": "Base" if s["product_id"] in base_as_a else "LoRA",
             "B": "LoRA" if s["product_id"] in base_as_a else "Base"} for i, s in enumerate(order, 1)]


def parse_task(task, text):
    """Parse format only. Raw text is always retained and used in human review."""
    value = text.strip()
    if task != "selling_points":
        return {"success": bool(value) and (task != "title" or "\n" not in value), "value": value,
                "format": "plain_text", "error": "empty_output" if not value else
                ("multiline_title" if task == "title" and "\n" in value else None)}
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list) and len(parsed) == 3 and all(isinstance(v, str) and v.strip() for v in parsed):
            return {"success": True, "value": parsed, "format": "json_array", "error": None}
    except (ValueError, TypeError):
        pass
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    pattern = r"^(?:[-*•]|\d+[.、)）]|[一二三][、.])\s*(.+)$"
    matched = [re.match(pattern, line) for line in lines]
    if len(lines) == 3 and all(matched):
        return {"success": True, "value": [m.group(1) for m in matched], "format": "three_bullets", "error": None}
    return {"success": False, "value": None, "format": "unparsed_raw_retained", "error": "not_exactly_three_points"}


def assemble(sample, tasks, variant):
    if set(tasks) != set(TASK_TYPES):
        raise ValueError("Incomplete three-task product")
    return {"product_id": sample["product_id"], "variant": variant,
            "tasks": tasks, "assembled": {"generated_title": tasks["title"]["parsed"]["value"],
            "selling_points": tasks["selling_points"]["parsed"]["value"],
            "short_description": tasks["short_description"]["parsed"]["value"],
            "complete_raw_text": "\n".join(tasks[t]["raw_text"] for t in TASK_TYPES),
            "structure_success": all(tasks[t]["parsed"]["success"] for t in TASK_TYPES)},
            "latency_seconds": {**{t: tasks[t]["latency_seconds"] for t in TASK_TYPES},
                                "three_task_total": sum(tasks[t]["latency_seconds"] for t in TASK_TYPES)}}


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_snapshot(cfg):
    return resolve(cfg["model"]["cache_dir"]) / ("models--" + cfg["model"]["name"].replace("/", "--")) / "snapshots" / cfg["model"]["revision"]


def prepare():
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    cfg = checked_config()
    out = resolve(cfg["output"])
    if out.exists():
        raise FileExistsError("Refusing to overwrite frozen P7 protocol")
    facts = load_jsonl(resolve(cfg["validation_quality_facts"]["path"]))
    for fact in facts:
        if fact.get("source_path") != cfg["source"]["path"] or fact.get("source_sha256") != cfg["source"]["sha256"]:
            raise ValueError("Quality facts do not originate from frozen project_validation")
    samples = build_inputs(load_jsonl(resolve(cfg["source"]["path"])), facts,
                           read_json(resolve(cfg["p5_config"]["path"])),
                           read_json(resolve(cfg["evaluation_definition"]["path"])), cfg)
    training = read_json(resolve(cfg["frozen_training_config"]["path"]))
    for role in ("lora_train", "lora_train_dev"):
        entry = training["data"][role]
        if digest(resolve(entry["path"])) != entry["sha256"]:
            raise ValueError("P5 hash changed")
        if {s["product_id"] for s in samples} & {str(r["product_id"]) for r in load_jsonl(resolve(entry["path"]))}:
            raise ValueError("Project-validation overlaps LoRA update/dev products")
    snapshot = model_snapshot(cfg)
    for name, expected in (("tokenizer_config.json", training["model"]["tokenizer_config_sha256"]),
                           ("config.json", training["model"]["model_config_sha256"])):
        if digest(snapshot / name) != expected:
            raise ValueError("Frozen base metadata changed")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    generation = GenerationConfig.from_pretrained(str(snapshot), local_files_only=True)
    generation.update(**cfg["decoding"], pad_token_id=tokenizer.eos_token_id)
    generation.validate()
    preflights = []
    for sample in samples:
        payload = json.dumps(sample["used_attributes"], ensure_ascii=False, sort_keys=False)
        for task in TASK_TYPES:
            _, metadata = preflight_chat_prompt(tokenizer, sample["tasks"][task]["messages"],
                max_input_tokens=cfg["prompt"]["max_input_tokens"], mandatory_lines=[payload])
            sample["tasks"][task]["preflight"] = metadata
            preflights.append(metadata["input_token_count"])
    checkpoint = resolve(cfg["adapter"]["path"]).parent
    identity = read_json(checkpoint / "trainer_state.json")["identity"]
    verify_checkpoint(checkpoint, identity)
    if identity["run_kind"] != "formal_training" or identity["config_sha256"] != cfg["frozen_training_config"]["sha256"]:
        raise ValueError("Wrong adapter provenance")
    out.mkdir(parents=True)
    for sample in samples:
        append_jsonl(out / "frozen_inputs.jsonl", sample)
    mapping = blind_mapping(samples, cfg["pairing"]["blind_seed"])
    atomic_json(out / "blind_mapping.json", {"status": "sealed_until_review_complete", "pairs": mapping})
    common = {"protocol_version": cfg["version"], "model": cfg["model"], "base_snapshot": str(snapshot),
              "prompt": cfg["prompt"], "decoding_resolved": generation.to_dict(), "pairing": cfg["pairing"]}
    for variant in ("Base", "LoRA"):
        atomic_json(out / f"inference_{variant.lower()}.json", {**common, "variant": variant,
                    "adapter": cfg["adapter"] if variant == "LoRA" else None})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (out / "environment_freeze.txt").write_text(freeze, encoding="utf-8")
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    manifests = [CONFIG_PATH, PROTOCOL_DOCUMENT, "tests/test_lora_validation.py", *CODE_FILES, *[cfg[k]["path"] for k in ALLOWED_INPUTS],
                 cfg["adapter"]["path"] + "/adapter_model.safetensors", cfg["adapter"]["path"] + "/adapter_config.json"]
    manifests += [training["data"][role]["path"] for role in ("lora_train", "lora_train_dev")]
    manifests += [str(p.relative_to(ROOT)).replace("\\", "/") for p in out.iterdir() if p.is_file()]
    code_bytes = {name: resolve(name).read_bytes().hex() for name in CODE_FILES}
    frozen = {"version": cfg["version"], "status": "frozen_before_first_generation", "created_at_utc": now(),
              "config": cfg, "git_head": git("rev-parse", "HEAD"), "git_status": git("-c", "core.quotepath=false", "status", "--short"),
              "code_snapshots_encoding": "hex_original_bytes", "code_snapshots": code_bytes,
              "files": [{"path": name, "sha256": digest(resolve(name))} for name in sorted(set(manifests))],
              "chat_template_sha256": json_digest(tokenizer.chat_template),
              "task_prompt_bundle_sha256": json_digest([{ "product_id": s["product_id"], "tasks": s["tasks"]} for s in samples]),
              "model_metadata_sha256": {name: digest(snapshot / name) for name in ("config.json", "tokenizer_config.json", "generation_config.json")},
              "environment": {"python": sys.version, "packages": {n: importlib.metadata.version(n) for n in PACKAGE_NAMES}, "cuda": torch.version.cuda},
              "source_statistics": {"products": len(samples), "core_attribute_total": sum(s["core_attribute_count"] for s in samples),
                  "by_category": dict(Counter(s["category_l2"] for s in samples)), "source_quality": dict(Counter(s["source_quality"]["status"] for s in samples)),
                  "input_core_attribute_total": sum(len(s["used_attributes"]) for s in samples),
                  "input_core_below_three": sum(len(s["used_attributes"]) < 3 for s in samples), "max_prompt_tokens": max(preflights)},
              "holdout_boundary": cfg["holdout_boundary_user_confirmed"], "parameter_updates": 0,
              "project_test_read": False, "rag_holdout_files_read": False, "rag_v2_generation": False}
    atomic_json(out / "protocol_manifest.json", frozen)
    print(json.dumps({"status": frozen["status"], **frozen["source_statistics"]}, ensure_ascii=False), flush=True)


def verify_protocol():
    cfg = checked_config()
    out = resolve(cfg["output"])
    frozen = read_json(out / "protocol_manifest.json")
    if frozen["config"] != cfg or frozen["status"] != "frozen_before_first_generation":
        raise ValueError("P7 frozen protocol changed")
    for item in frozen["files"]:
        if digest(resolve(item["path"])) != item["sha256"]:
            raise ValueError(f"P7 frozen file changed: {item['path']}")
    return cfg, frozen


def generate(variant):
    if variant not in ("Base", "LoRA"):
        raise ValueError("Unknown variant")
    cfg, frozen = verify_protocol()
    out = resolve(cfg["output"])
    run_dir = out / variant.lower()
    if run_dir.exists():
        raise FileExistsError("No automatic rerun/resume of an existing P7 generation")
    run_dir.mkdir()
    report = {"status": "running", "variant": variant, "started_at_utc": now(), "parameter_updates": 0,
              "project_test_read": False, "rag_holdout_files_read": False, "rag_v2_generation": False}
    begun = time.perf_counter()
    try:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
        if ({n: importlib.metadata.version(n) for n in PACKAGE_NAMES} != frozen["environment"]["packages"]
                or sys.version != frozen["environment"]["python"] or torch.version.cuda != frozen["environment"]["cuda"]):
            raise RuntimeError("Frozen inference environment changed")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA/BF16 unavailable; no fallback")
        random.seed(cfg["pairing"]["base_seed"])
        torch.manual_seed(cfg["pairing"]["base_seed"])
        torch.cuda.manual_seed_all(cfg["pairing"]["base_seed"])
        snapshot = model_snapshot(cfg)
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
            model = PeftModel.from_pretrained(model, str(resolve(cfg["adapter"]["path"])),
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
                    max_input_tokens=cfg["prompt"]["max_input_tokens"],
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
                        "hit_max_new_tokens": len(ids) >= cfg["decoding"]["max_new_tokens"],
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
        report.update(status="passed", completed_at_utc=now(), products=200, task_calls=600,
                      wall_seconds=time.perf_counter() - begun, model_parameters_unchanged=True)
        atomic_json(run_dir / "run_report.json", report)
    except Exception as error:
        report.update(status="failed", failed_at_utc=now(), error=str(error))
        atomic_json(run_dir / "run_report.json", report)
        raise
