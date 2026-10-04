"""Offline train-only updates, dev-only selection, and resumable adapter checkpoints.

No dependency on a smoke trainer. Reuses ONLY the frozen low-memory loss/preparer.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .qlora_memory import prepare_kbit_with_cpu_staged_large_layers, qwen_chunked_forward_loss

ROOT = Path(__file__).resolve().parents[2]
PINNED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATHS = {"lora_train": "data/processed/lora_instruction_data_v1/train.jsonl",
              "lora_train_dev": "data/processed/lora_instruction_data_v1/validation.jsonl"}
DATA_HASHES = {"lora_train": "c31b00f3d19d503b3f7c7529cc2ce7e115143214b0d9ab40f07412becd6a5434",
               "lora_train_dev": "20a32d5f67a8ac6dd0743bb90445522fe28cd4099ed318e71a4f0e8e33fa0a94"}


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    return (ROOT / path).resolve()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def step_counts(records, accumulation, epochs):
    if min(records, accumulation, epochs) <= 0:
        raise ValueError("Counts must be positive.")
    per_epoch = math.ceil(records / accumulation)
    return {"micro_steps_per_epoch": records, "optimizer_steps_per_epoch": per_epoch,
            "total_optimizer_steps": per_epoch * epochs,
            "last_group_records": records % accumulation or accumulation}


def epoch_order(records, seed, epoch):
    order = list(range(records))
    random.Random(seed + epoch).shuffle(order)
    return order


def accumulation_groups(order, accumulation, cursor=0):
    if cursor < 0 or cursor > len(order) or (cursor % accumulation and cursor != len(order)):
        raise ValueError("Resume cursor must be at an optimizer boundary.")
    for start in range(cursor, len(order), accumulation):
        yield order[start:start + accumulation]


def validate_config(config, mode, confirm=False, stop=None):
    if mode not in {"prepare", "acceptance", "train"}:
        raise ValueError("Unsupported execution mode.")
    if mode == "train" and not confirm:
        raise PermissionError("Full training requires --confirm-formal-training after user approval.")
    if mode == "acceptance" and (stop is None or not 1 <= stop <= config["execution"]["acceptance_max_global_steps"]):
        raise ValueError("Acceptance is bounded to global steps 1..4.")
    if mode != "acceptance" and stop is not None:
        raise ValueError("--stop-at-step is only allowed for bounded acceptance.")
    if config["execution"]["acceptance_max_global_steps"] != 4 or any(config["guards"].values()):
        raise ValueError("Safety guards changed.")
    model, eng = config["model"], config["engineering"]
    if model["revision"] != PINNED_REVISION or model["name"] != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("Frozen model identity/revision changed.")
    expected_model = {"offline": True, "load_in_4bit": True, "quant_type": "nf4",
                      "double_quant": True, "compute_dtype": "bfloat16"}
    expected_eng = {"frozen_embedding_head_dtype": "bfloat16", "norm_and_trainable_lora_dtype": "float32",
                    "autocast_dtype": "bfloat16", "loss_chunk_tokens": 16, "micro_batch_size": 1,
                    "gradient_accumulation_steps": 8, "max_sequence_length": 320, "truncation": False,
                    "fixed_length_padding": False, "gradient_checkpointing": True,
                    "gradient_checkpointing_use_reentrant": False, "use_cache": False,
                    "loss": "native_shape_bf16_logits_assistant_only_chunked_fp32_mean_causal_cross_entropy",
                    "initialization": "cpu_stage_large_frozen_layers_during_peft_preparation"}
    if any(model[key] != value for key, value in expected_model.items()) or any(eng[key] != value for key, value in expected_eng.items()):
        raise ValueError("Verified engineering parameters changed.")
    if config["lora"] != {"task_type": "CAUSAL_LM", "target_modules": ["q_proj", "v_proj"], "r": 8,
                           "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
                           "expected_trainable_parameters": 2523136}:
        raise ValueError("Frozen LoRA configuration changed.")
    if config["evaluation"]["dataset"] != "lora_train_dev" or not config["evaluation"]["no_grad"]:
        raise ValueError("Evaluation must use no-grad internal dev only.")
    if config["training"]["scheduler"] != "constant_lambda_1" or config["training"]["early_stopping"]:
        raise ValueError("First-version scheduler/early-stopping policy changed.")
    if config["training"]["max_grad_norm"] is not None or config["training"]["warmup_steps"] != 0:
        raise ValueError("Unimplemented clipping/warmup must not be silently enabled.")
    if config["evaluation"]["frequency"] != "each_epoch_end" or config["checkpoint"]["keep"] != "latest_and_best":
        raise ValueError("Unexpected evaluation/retention policy.")
    if resolve(config["data"]["manifest"]) != resolve("reports/generation/lora/lora_instruction_data_v1_final_manifest.json"):
        raise ValueError("Forbidden manifest path.")
    if resolve(eng["reference_smoke_config"]) != resolve("configs/lora_qlora_low_memory_smoke_v1.json"):
        raise ValueError("Forbidden smoke reference path.")
    # Validate allowed paths BEFORE opening any data, even when hashes look valid.
    for role, expected_path in DATA_PATHS.items():
        entry = config["data"][role]
        if resolve(entry["path"]) != resolve(expected_path) or entry["sha256"] != DATA_HASHES[role]:
            raise ValueError(f"Forbidden data path/hash for {role}.")
        if digest(resolve(expected_path)) != DATA_HASHES[role]:
            raise ValueError(f"Frozen data hash mismatch for {role}.")
    if digest(resolve(config["data"]["manifest"])) != config["data"]["manifest_sha256"]:
        raise ValueError("P5 manifest hash mismatch.")
    if digest(resolve(eng["reference_smoke_config"])) != eng["reference_smoke_config_sha256"]:
        raise ValueError("Frozen smoke reference changed.")
    if digest(Path(__file__).with_name("qlora_memory.py")) != eng["memory_module_sha256"]:
        raise ValueError("Frozen memory implementation changed.")


def load_records(config, role):
    if role not in DATA_PATHS or resolve(config["data"][role]["path"]) != resolve(DATA_PATHS[role]):
        raise ValueError("Only lora_train / lora_train_dev may be opened.")
    entry = config["data"][role]
    if digest(resolve(entry["path"])) != DATA_HASHES[role]:
        raise ValueError("Data changed after preflight.")
    with resolve(entry["path"]).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for row in records:
        if row["split"] != entry["source_split"] or row["template_version"] != config["data"]["template_version"]:
            raise ValueError("Unexpected split/template version.")
        if row["task_type"] not in config["data"]["task_types"] or row["data_version"] != config["data"]["version"]:
            raise ValueError("Unexpected task/data version.")
    keys = {(str(row["product_id"]), row["task_type"]) for row in records}
    if len(records) != entry["records"] or len(keys) != len(records):
        raise ValueError("Record count or uniqueness mismatch.")
    if len({str(row["product_id"]) for row in records}) != entry["products"]:
        raise ValueError("Product count mismatch.")
    return records


def token_ids(value):
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise ValueError("Invalid chat-template token output.")
    return value


def tokenize_record(tokenizer, record, maximum):
    import torch
    messages = record["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Record must end in an assistant target.")
    prompt = token_ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
    full = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    if full[:len(prompt)] != prompt or len(full) > maximum:
        raise ValueError("Prompt prefix mismatch or sequence too long; truncation forbidden.")
    labels = [-100] * len(prompt) + full[len(prompt):]
    if not any(value != -100 for value in labels[1:]):
        raise ValueError("No shifted supervised target.")
    return {"input_ids": torch.tensor([full], dtype=torch.long),
            "attention_mask": torch.ones((1, len(full)), dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long), "sequence_tokens": len(full),
            "product_id": str(record["product_id"]), "task_type": record["task_type"]}


def memory():
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {key: round(function() / 1024**2, 1) for key, function in
            (("allocated_mib", torch.cuda.memory_allocated), ("reserved_mib", torch.cuda.memory_reserved),
             ("peak_allocated_mib", torch.cuda.max_memory_allocated), ("peak_reserved_mib", torch.cuda.max_memory_reserved))}


def capture_rng():
    import torch
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def tensor_tree_digest(value):
    """Hash nested optimizer/scheduler/RNG state without pickle representation noise."""
    import torch
    hasher = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            hasher.update(str((str(item.dtype), tuple(item.shape))).encode())
            hasher.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                hasher.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        else:
            hasher.update(repr(item).encode())
    visit(value)
    return hasher.hexdigest()


def parameter_signature(model):
    quant = model.config.quantization_config
    return {"trainable": {name: str(param.dtype) for name, param in model.named_parameters() if param.requires_grad},
            "embedding": str(model.get_input_embeddings().weight.dtype),
            "head": str(model.get_output_embeddings().weight.dtype),
            "norm_dtypes": sorted({str(param.dtype) for name, param in model.named_parameters() if "norm" in name}),
            "use_cache": model.config.use_cache,
            "quantization": quant.to_dict() if hasattr(quant, "to_dict") else quant}


def evaluate_dev(model, items, loss_fn, device="cuda", progress=None):
    """No parameter updates, no configuration/dtype changes, preserve RNG/mode."""
    import torch
    rng, was_training = capture_rng(), model.training
    begun = time.perf_counter()
    before = {name: parameter._version for name, parameter in model.named_parameters()}
    signature = parameter_signature(model) if hasattr(model, "get_input_embeddings") else None
    model.eval()
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows, tasks = [], defaultdict(list)
    try:
        with torch.no_grad():
            for index, item in enumerate(items, 1):
                batch = {key: item[key].to(device) for key in ("input_ids", "attention_mask", "labels")}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = loss_fn(model, batch)
                value = float(loss.detach())
                if not math.isfinite(value) or loss.requires_grad:
                    raise RuntimeError("Dev evaluation produced non-finite loss or gradient graph.")
                rows.append({"product_id": item["product_id"], "task_type": item["task_type"],
                             "sequence_tokens": item["sequence_tokens"], "loss": value})
                tasks[item["task_type"]].append(value)
                del loss, batch
                if progress and index % 50 == 0:
                    progress(f"lora_train_dev evaluation {index}/{len(items)}")
        if before != {name: parameter._version for name, parameter in model.named_parameters()}:
            raise RuntimeError("Dev evaluation modified model parameters.")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Dev evaluation produced parameter gradients.")
        if signature is not None and signature != parameter_signature(model):
            raise RuntimeError("Dev evaluation changed model engineering configuration.")
        return {"dataset": "lora_train_dev", "records": len(rows),
                "dev_loss": sum(row["loss"] for row in rows) / len(rows),
                "task_losses": {task: sum(values) / len(values) for task, values in tasks.items()},
                "no_grad": True, "parameter_updates": 0, "configuration_unchanged": True,
                "seconds": time.perf_counter() - begun, "memory": memory(), "records_detail": rows}
    finally:
        model.train(was_training)
        restore_rng(rng)


def file_manifest(directory):
    return [{"path": str(path.relative_to(directory)).replace("\\", "/"),
             "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "checkpoint_manifest.json"]


def save_checkpoint(directory, model, optimizer, scheduler, state, identity):
    """Commit a complete optimizer-boundary checkpoint, then publish its pointer."""
    import torch
    directory = Path(directory)
    temporary = directory.with_name(directory.name + ".incomplete")
    if directory.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {directory}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    packed = {"format_version": 1, "identity": identity, "state": state,
              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": capture_rng()}
    torch.save(packed, temporary / "training_state.pt")
    atomic_json(temporary / "trainer_state.json", {
        "identity": identity, "state": state, "scheduler": scheduler.state_dict(),
        "optimizer_digest": tensor_tree_digest(packed["optimizer"]),
        "scheduler_digest": tensor_tree_digest(packed["scheduler"]),
        "rng_digest": tensor_tree_digest(packed["rng"]),
    })
    atomic_json(temporary / "config_snapshot.json", identity["config"])
    atomic_json(temporary / "checkpoint_manifest.json", {
        "complete": True, "global_step": state["global_step"], "run_kind": identity["run_kind"],
        "files": file_manifest(temporary), "saved_at_utc": now(),
    })
    os.replace(temporary, directory)
    return directory


def verify_checkpoint(directory, identity):
    directory = Path(directory).resolve()
    if directory.name.endswith(".incomplete"):
        raise ValueError("Incomplete checkpoint cannot be resumed.")
    manifest = read_json(directory / "checkpoint_manifest.json")
    if not manifest["complete"] or manifest["run_kind"] != identity["run_kind"]:
        raise ValueError("Checkpoint run kind mismatch; acceptance/smoke cannot initialize formal training.")
    for item in manifest["files"]:
        path = (directory / item["path"]).resolve()
        if directory not in path.parents or digest(path) != item["sha256"]:
            raise ValueError("Checkpoint checksum/path verification failed.")
    saved_identity = read_json(directory / "trainer_state.json")["identity"]
    if saved_identity != identity:
        raise ValueError("Checkpoint config/data/model/code/environment identity mismatch.")


def restore_checkpoint_state(directory, optimizer, scheduler, identity):
    import torch
    verify_checkpoint(directory, identity)
    # Only this runner's local, hash-verified checkpoint may be unpickled.
    packed = torch.load(Path(directory) / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(packed["optimizer"])
    scheduler.load_state_dict(packed["scheduler"])
    if tensor_tree_digest(optimizer.state_dict()) != tensor_tree_digest(packed["optimizer"]):
        raise RuntimeError("Optimizer state did not restore exactly.")
    if scheduler.state_dict() != packed["scheduler"]:
        raise RuntimeError("Scheduler state did not restore exactly.")
    restore_rng(packed["rng"])
    return packed["state"], {
        "restored_global_step": packed["state"]["global_step"],
        "restored_epoch": packed["state"]["epoch"], "restored_cursor": packed["state"]["cursor"],
        "optimizer_digest": tensor_tree_digest(packed["optimizer"]),
        "scheduler_digest": tensor_tree_digest(packed["scheduler"]),
        "rng_digest": tensor_tree_digest(packed["rng"]), "optimizer_scheduler_rng_restored": True,
    }


def prune_checkpoints(root, keep):
    """Delete only complete obsolete checkpoints inside this exact run directory."""
    root = Path(root).resolve()
    keep = {Path(path).resolve() for path in keep}
    for path in root.iterdir():
        target = path.resolve()
        if target.parent != root or target in keep or not target.is_dir():
            continue
        suffix = target.name.removeprefix("checkpoint-step-")
        if not target.name.startswith("checkpoint-step-") or not suffix.isdigit():
            continue
        if (target / "checkpoint_manifest.json").is_file() and read_json(target / "checkpoint_manifest.json")["complete"]:
            shutil.rmtree(target)


def validate_output(path, root):
    path = resolve(path)
    if resolve(root) not in path.parents:
        raise ValueError(f"Output must remain strictly beneath {root}.")
    return path


def run(args):
    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config, args.mode, args.confirm_formal_training, args.stop_at_step)
    counts = step_counts(config["data"]["lora_train"]["records"],
                         config["engineering"]["gradient_accumulation_steps"], config["training"]["epochs"])
    if args.mode == "prepare":
        print(json.dumps({"status": "prepared_only_no_model_or_training", "counts": counts,
                          "config_sha256": digest(config_path)}, ensure_ascii=False, indent=2))
        return
    if args.evaluate_dev and args.mode != "acceptance":
        raise ValueError("--evaluate-dev is only an acceptance override; train uses epoch-end evaluation.")
    if args.mode == "train" and (args.output_directory or args.report_directory):
        raise ValueError("Formal output paths must come from the frozen config.")
    if args.mode == "acceptance" and (not args.output_directory or not args.report_directory):
        raise ValueError("Acceptance needs independent output/report directories.")
    out = validate_output(args.output_directory or config["output"]["directory"], "artifacts/lora")
    reports = validate_output(args.report_directory or config["output"]["report_directory"], "reports/generation/lora")
    if args.mode == "acceptance" and (out == resolve(config["output"]["directory"]) or reports == resolve(config["output"]["report_directory"])):
        raise ValueError("Acceptance cannot touch formal output/report paths.")
    resume = args.resume.resolve() if args.resume else None
    if resume:
        if resume.parent != out / "checkpoints" or not out.is_dir():
            raise ValueError("Resume checkpoint must belong to this run's checkpoint directory.")
        latest = read_json(out / "latest.json")
        if resolve(latest["path"]) != resume:
            raise ValueError("Only the complete latest checkpoint can resume this run.")
        if digest(resume / "checkpoint_manifest.json") != latest["manifest_sha256"]:
            raise ValueError("Latest checkpoint manifest changed.")
    elif out.exists() or reports.exists():
        raise FileExistsError("Fresh run refuses existing output/report directory.")
    out.mkdir(parents=True, exist_ok=bool(resume))
    reports.mkdir(parents=True, exist_ok=bool(resume))
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    report_path = reports / f"invocation_{invocation}.json"
    log_path = reports / f"events_{invocation}.jsonl"
    report = {"status": "running", "mode": args.mode, "started_at_utc": now(), "pid": os.getpid(),
              "counts": counts, "config_sha256": digest(config_path),
              "project_validation_read": False, "project_test_read": False, "rag_holdout_read": False}
    def event(kind, **values):
        item = {"at_utc": now(), "kind": kind, **values}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if kind != "micro_step":
            print(json.dumps(item, ensure_ascii=False), flush=True)
    begun = time.perf_counter()
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA + BF16 required; no automatic precision/device fallback.")
        seed = config["training"]["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors")}
        environment = {"python": sys.version, "packages": packages, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (reports / f"environment_{invocation}.txt").write_text(freeze.stdout, encoding="utf-8")
        model_cfg = config["model"]
        snapshot = (resolve(model_cfg["cache_dir"]) / ("models--" + model_cfg["name"].replace("/", "--"))
                    / "snapshots" / model_cfg["revision"])
        for name, expected in (("tokenizer_config.json", model_cfg["tokenizer_config_sha256"]),
                               ("config.json", model_cfg["model_config_sha256"])):
            if digest(snapshot / name) != expected:
                raise ValueError(f"Frozen snapshot metadata changed: {name}")
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        identity = {"run_kind": "formal_training" if args.mode == "train" else "acceptance_only",
                    "config": config, "config_sha256": digest(config_path), "base_snapshot": str(snapshot),
                    "chat_template_sha256": json_digest(tokenizer.chat_template), "environment": environment,
                    "code_sha256": {name: digest(resolve(name)) for name in
                                    ("scripts/train_lora.py", "src/generation/lora_training.py", "src/generation/qlora_memory.py")}}
        report["identity"] = identity
        if resume:
            verify_checkpoint(resume, identity)
        train_records = load_records(config, "lora_train")
        train_items = [tokenize_record(tokenizer, row, config["engineering"]["max_sequence_length"]) for row in train_records]
        train_products = {str(row["product_id"]) for row in train_records}
        dev_cache = None
        def dev_items():
            nonlocal dev_cache
            if dev_cache is None:
                rows = load_records(config, "lora_train_dev")
                if train_products & {str(row["product_id"]) for row in rows}:
                    raise ValueError("Train/dev product overlap.")
                dev_cache = [tokenize_record(tokenizer, row, config["engineering"]["max_sequence_length"]) for row in rows]
            return dev_cache
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, device_map={"": 0},
                                                  quantization_config=quant, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if not base.is_loaded_in_4bit:
            raise RuntimeError("4-bit load failed.")
        base.config.use_cache = False
        base, changes = prepare_kbit_with_cpu_staged_large_layers(
            base, prepare_model_for_kbit_training, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if resume:
            model = PeftModel.from_pretrained(base, str(resume / "adapter"), is_trainable=True, local_files_only=True)
        else:
            lc = {key: value for key, value in config["lora"].items() if key != "expected_trainable_parameters"}
            model = get_peft_model(base, LoraConfig(**lc))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        report["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
        signature = parameter_signature(model)
        if report["trainable_parameters"] != config["lora"]["expected_trainable_parameters"]:
            raise RuntimeError("Trainable parameter count changed.")
        if signature["embedding"] != "torch.bfloat16" or signature["head"] != "torch.bfloat16" or signature["norm_dtypes"] != ["torch.float32"]:
            raise RuntimeError("Frozen layer dtypes changed.")
        if any(dtype != "torch.float32" or "lora_" not in name for name, dtype in signature["trainable"].items()):
            raise RuntimeError("Only FP32 LoRA parameters may be trained.")
        report["model_signature"] = signature
        report["initialization_memory"] = memory()
        report["precision_storage_changes"] = changes
        training = config["training"]
        optimizer = torch.optim.AdamW(trainable, lr=training["learning_rate"], betas=tuple(training["betas"]),
                                     eps=training["eps"], weight_decay=training["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        state = {"epoch": 0, "cursor": 0, "global_step": 0, "global_micro_step": 0,
                 "train_seconds": 0.0, "epoch_train_loss_sum": 0.0, "epoch_train_records": 0,
                 "train_peak_allocated_mib": 0.0, "best": None, "completed_epochs": 0}
        if resume:
            state, report["resume"] = restore_checkpoint_state(resume, optimizer, scheduler, identity)
            if args.mode == "acceptance" and state["global_step"] >= args.stop_at_step:
                raise ValueError("Acceptance stop must be after resumed step.")
        report["start_state"] = dict(state)
        limit = args.stop_at_step if args.mode == "acceptance" else counts["total_optimizer_steps"]
        all_step_rows = []
        evaluations = []
        loss_fn = lambda m, b: qwen_chunked_forward_loss(m, b, config["engineering"]["loss_chunk_tokens"])
        checkpoints = out / "checkpoints"
        def checkpoint():
            directory = checkpoints / f"checkpoint-step-{state['global_step']:06d}"
            save_checkpoint(directory, model, optimizer, scheduler, dict(state), identity)
            relative = str(directory.relative_to(ROOT)).replace("\\", "/")
            atomic_json(out / "latest.json", {"path": relative, "global_step": state["global_step"],
                                             "manifest_sha256": digest(directory / "checkpoint_manifest.json")})
            if state["best"] and state["best"]["global_step"] == state["global_step"]:
                atomic_json(out / "best.json", {"path": relative, **state["best"]})
            if args.mode == "train":
                keep = [directory]
                if (out / "best.json").exists():
                    keep.append(resolve(read_json(out / "best.json")["path"]))
                prune_checkpoints(checkpoints, keep)
            event("checkpoint", global_step=state["global_step"], path=relative)
            return directory
        event("start", mode=args.mode, resume=str(resume) if resume else None,
              global_step=state["global_step"], total_planned_optimizer_steps=counts["total_optimizer_steps"])
        while state["global_step"] < limit and state["epoch"] < training["epochs"]:
            epoch = state["epoch"]
            order = epoch_order(len(train_items), seed, epoch)
            order_hash = json_digest(order)
            for indices in accumulation_groups(order, config["engineering"]["gradient_accumulation_steps"], state["cursor"]):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                group_start = time.perf_counter()
                group_losses, ids = [], []
                lr = optimizer.param_groups[0]["lr"]
                for index in indices:
                    item = train_items[index]  # Never dev_cache: update source is train-only.
                    batch = {key: item[key].to("cuda") for key in ("input_ids", "attention_mask", "labels")}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = loss_fn(model, batch)
                    value = float(loss.detach())
                    if not math.isfinite(value):
                        raise RuntimeError("Non-finite train loss.")
                    (loss / len(indices)).backward()
                    group_losses.append(value)
                    ids.append({"product_id": item["product_id"], "task_type": item["task_type"]})
                    state["global_micro_step"] += 1
                    event("micro_step", global_micro_step=state["global_micro_step"], loss=value,
                          product_id=item["product_id"], task_type=item["task_type"], source="lora_train")
                    del loss, batch
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
                    raise RuntimeError("Missing/non-finite LoRA gradients.")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - group_start
                state["global_step"] += 1
                state["cursor"] += len(indices)
                state["train_seconds"] += elapsed
                state["epoch_train_loss_sum"] += sum(group_losses)
                state["epoch_train_records"] += len(indices)
                observed_memory = memory()
                state["train_peak_allocated_mib"] = max(state["train_peak_allocated_mib"], observed_memory["peak_allocated_mib"])
                row = {"global_step": state["global_step"], "epoch": epoch + state["cursor"] / len(train_items),
                       "cursor": state["cursor"], "epoch_order_sha256": order_hash,
                       "train_loss": sum(group_losses) / len(group_losses), "micro_losses": group_losses,
                       "records": ids, "actual_accumulation_records": len(indices), "learning_rate": lr,
                       "scheduler_last_epoch": scheduler.last_epoch, "seconds": elapsed, "memory": observed_memory}
                all_step_rows.append(row)
                event("optimizer_step", **row)
                epoch_end = state["cursor"] == len(train_items)
                if epoch_end:
                    epoch_train_loss = state["epoch_train_loss_sum"] / state["epoch_train_records"]
                    result = evaluate_dev(model, dev_items(), loss_fn, progress=lambda msg: event("dev_progress", message=msg))
                    result.update(global_step=state["global_step"], completed_epoch=epoch + 1, epoch_train_loss=epoch_train_loss)
                    evaluations.append(result)
                    atomic_json(reports / f"dev_epoch_{epoch + 1}.json", result)
                    if state["best"] is None or result["dev_loss"] < state["best"]["dev_loss"]:
                        state["best"] = {"global_step": state["global_step"], "dev_loss": result["dev_loss"], "epoch": epoch + 1}
                    state.update(epoch=epoch + 1, cursor=0, completed_epochs=epoch + 1,
                                 epoch_train_loss_sum=0.0, epoch_train_records=0)
                    event("epoch_end", epoch=epoch + 1, dev_loss=result["dev_loss"], epoch_train_loss=epoch_train_loss)
                if state["global_step"] % config["checkpoint"]["save_steps"] == 0 or epoch_end:
                    checkpoint()
                if state["global_step"] >= limit:
                    break
        if args.evaluate_dev:
            result = evaluate_dev(model, dev_items(), loss_fn, progress=lambda msg: event("dev_progress", message=msg))
            result.update(global_step=state["global_step"], engineering_only=True)
            evaluations.append(result)
            atomic_json(reports / f"dev_acceptance_step_{state['global_step']:06d}.json", result)
        last = checkpoints / f"checkpoint-step-{state['global_step']:06d}"
        if not last.exists():
            last = checkpoint()
        for role in DATA_PATHS:
            if digest(resolve(DATA_PATHS[role])) != DATA_HASHES[role]:
                raise RuntimeError("Frozen P5 data changed during invocation.")
        report.update(status="passed", completed_at_utc=now(), wall_seconds=time.perf_counter() - begun,
                      final_state=state, optimizer_steps=all_step_rows, evaluations=evaluations,
                      final_optimizer_digest=tensor_tree_digest(optimizer.state_dict()),
                      final_scheduler_digest=tensor_tree_digest(scheduler.state_dict()),
                      final_rng_digest=tensor_tree_digest(capture_rng()),
                      latest_checkpoint=str(last.relative_to(ROOT)).replace("\\", "/"),
                      best_checkpoint=read_json(out / "best.json") if (out / "best.json").exists() else None)
        atomic_json(report_path, report)
        event("complete", status="passed", mode=args.mode, global_step=state["global_step"], report=str(report_path))
    except Exception as error:
        report.update(status="failed", failed_at_utc=now(), error=str(error), traceback=traceback.format_exc())
        atomic_json(report_path, report)
        raise
