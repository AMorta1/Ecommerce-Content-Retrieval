"""Independent bounded local QLoRA precision/loss smoke and longest-record stress."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# Offline is process-local, and the exact snapshot directory is loaded below.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qlora_smoke import (
    adapter_file_manifest, load_json, load_jsonl, memory_snapshot, project_path,
    select_smoke_records, sha256_file, tokenize_supervised_record, utc_now,
)
from src.generation.qlora_memory import (
    chunked_causal_loss, qwen_chunked_forward_loss, restore_frozen_large_layers_bf16,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Telemetry:
    """Sample driver and Windows process dedicated/shared memory, not estimates."""

    def __init__(self):
        self.samples = []
        self.errors = []
        self.phase = "preflight"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        item = {"at_utc": utc_now(), "phase": self.phase}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=10, check=True, creationflags=0x08000000,
            )
            total, used, free = map(float, result.stdout.strip().splitlines()[0].split(","))
            item.update(device_total_mib=total, device_used_mib=used, device_free_mib=free)
            command = (
                "Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
                f"| Where-Object {{$_.Name -like 'pid_{os.getpid()}_*'}} "
                "| Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted | ConvertTo-Json -Compress"
            )
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, timeout=15, check=True, creationflags=0x08000000,
            )
            entries = json.loads(result.stdout) if result.stdout.strip() else []
            if isinstance(entries, dict):
                entries = [entries]
            if entries:
                for name, field in (("dedicated", "DedicatedUsage"), ("shared", "SharedUsage")):
                    item[f"process_{name}_mib"] = sum(row[field] for row in entries) / 1024**2
                item["process_counter_entries"] = entries
        except Exception as error:
            self.errors.append({"at_utc": utc_now(), "phase": self.phase, "error": str(error)})
        self.samples.append(item)

    def loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(2)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=30)


def batch_on_cuda(item):
    return {key: item[key].to("cuda") for key in ("input_ids", "attention_mask", "labels")}


def protected_hashes():
    # Hashing dev and manifests does not evaluate them or read test examples.
    paths = [
        "data/processed/lora_instruction_data_v1/train.jsonl",
        "data/processed/lora_instruction_data_v1/validation.jsonl",
        "reports/generation/lora/lora_instruction_data_v1_final_manifest.json",
        "reports/generation/rag/rag_v2_generation_output_holdout_v1.json",
        "configs/lora_qlora_smoke_v1.json",
        "scripts/run_qlora_smoke.py",
    ]
    return {path: sha256_file(project_path(path)) for path in paths}


def validate(config):
    if config["scope"] != "engineering_smoke_only" or any(config["guards"].values()):
        raise ValueError("Only guarded engineering smoke is allowed.")
    ref_path = project_path(config["reference_config"])
    if sha256_file(ref_path) != config["reference_config_sha256"]:
        raise ValueError("Frozen reference config hash mismatch.")
    ref = load_json(ref_path)
    if ref["input"]["path"] != "data/processed/lora_instruction_data_v1/train.jsonl":
        raise ValueError("Only frozen P5 lora_train is permitted.")
    if sha256_file(project_path(ref["input"]["path"])) != ref["input"]["expected_sha256"]:
        raise ValueError("Frozen P5 lora_train hash mismatch.")
    # Bounds, not configurable formal training knobs.
    if ref["lora"]["r"] != 8 or ref["optimization"]["micro_steps"] != 24:
        raise ValueError("Reference rank/sample count changed.")
    if config["stress"]["records"] != 8 or config["stress"]["repeats"] != 8:
        raise ValueError("Stress must remain bounded to 8 records x 8 repeats.")
    for key, root in (("output_directory", "reports/generation/lora"),
                      ("adapter_directory", "artifacts/lora")):
        target = project_path(config[key])
        if project_path(root) not in target.parents or target.exists():
            raise ValueError(f"Refusing unsafe or existing output: {target}")
    return ref


def run(config_path):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = load_json(config_path)
    ref = validate(config)
    out = project_path(config["output_directory"])
    adapter_dir = project_path(config["adapter_directory"])
    out.mkdir(parents=True, exist_ok=False)
    monitor = Telemetry()
    report = {"version": config["version"], "scope": config["scope"],
              "started_at_utc": utc_now(), "status": "running", "guards": config["guards"],
              "protected_hashes_before": protected_hashes()}
    stage = "preflight"

    def progress(message):
        print(f"[{utc_now()}] {message}", flush=True)
        with (out / "progress.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] {message}\n")

    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA and BF16 hardware support required.")
        seed = ref["optimization"]["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        report["environment"] = {
            "python": sys.version, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "bf16_supported": True,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
        }
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=True)
        (out / "environment_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        check = subprocess.run([sys.executable, "-m", "pip", "check"],
                               capture_output=True, text=True)
        report["pip_check"] = {"returncode": check.returncode, "stdout": check.stdout,
                               "stderr": check.stderr}
        if check.returncode:
            raise RuntimeError("pip check failed; no automatic environment repair.")

        mc = ref["model"]
        snapshot = (project_path(mc["cache_dir"]) / ("models--" + mc["name"].replace("/", "--"))
                    / "snapshots" / mc["revision"])
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        records = load_jsonl(project_path(ref["input"]["path"]))
        selected, products = select_smoke_records(records, ref["input"])
        max_length = ref["optimization"]["max_sequence_length"]
        all_tokens = [tokenize_supervised_record(tokenizer, row, max_length) for row in records]
        lookup = {(str(row["product_id"]), row["task_type"]): tokens
                  for row, tokens in zip(records, all_tokens)}
        smoke_tokens = [lookup[(str(row["product_id"]), row["task_type"])] for row in selected]
        ranked = sorted(zip(records, all_tokens), key=lambda pair:
                        (-pair[1]["sequence_tokens"], str(pair[0]["product_id"]), pair[0]["task_type"]))
        longest = ranked[:config["stress"]["records"]]
        report["data"] = {
            "split": "lora_train", "all_train_records_tokenized": len(records),
            "truncation": False, "max_sequence_length": max_length,
            "all_train_max_tokens": max(item["sequence_tokens"] for item in all_tokens),
            "smoke_products": products,
            "smoke_records": [{"product_id": row["product_id"], "task_type": row["task_type"],
                               "sequence_tokens": tokens["sequence_tokens"]}
                              for row, tokens in zip(selected, smoke_tokens)],
            "stress_records": [{"product_id": row["product_id"], "category_l2": row["category_l2"],
                                "task_type": row["task_type"], "sequence_tokens": tokens["sequence_tokens"],
                                "supervised_tokens": tokens["supervised_tokens"]}
                               for row, tokens in longest],
        }
        report["model"] = {**mc, "actual_load_path": str(snapshot), "network_access": False}
        progress(f"Preflight: {len(records)} train tasks, longest {report['data']['all_train_max_tokens']} tokens; P5 unchanged.")
        monitor.start()
        stage = monitor.phase = "loading_precision_validation"
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=mc["bnb_4bit_quant_type"],
                                 bnb_4bit_use_double_quant=mc["bnb_4bit_use_double_quant"],
                                 bnb_4bit_compute_dtype=torch.bfloat16)

        def load_base():
            base = AutoModelForCausalLM.from_pretrained(
                str(snapshot), local_files_only=True, device_map={"": 0},
                quantization_config=quant, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
            if not base.is_loaded_in_4bit:
                raise RuntimeError("4-bit loading not active.")
            base.config.use_cache = False
            return base

        base = prepare_model_for_kbit_training(
            load_base(), use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lc = ref["lora"]
        model = get_peft_model(base, LoraConfig(
            task_type=lc["task_type"], target_modules=lc["target_modules"], r=lc["r"],
            lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"], bias=lc["bias"],
        ))
        trainables = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainables or any("lora_" not in name or parameter.dtype != torch.float32
                                 for name, parameter in trainables):
            raise RuntimeError("Unexpected trainable parameter identity/dtype.")
        report["lora"] = {**lc, "trainable_parameters": sum(p.numel() for _, p in trainables)}
        if report["lora"]["trainable_parameters"] != 2523136:
            raise RuntimeError("Trainable parameter count differs from frozen rank=8 smoke.")
        model.eval()
        first = batch_on_cuda(smoke_tokens[0])
        chunk = config["loss"]["chunk_tokens"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before = float(qwen_chunked_forward_loss(model, first, chunk))
        changes = restore_frozen_large_layers_bf16(model)
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after = float(qwen_chunked_forward_loss(model, first, chunk))
            dense = float(model(**first).loss)
        delta = abs(before - after)
        if delta > config["loss"]["precision_change_max_absolute_loss_delta"]:
            raise RuntimeError(f"Precision loss delta too large: {delta}")
        if not math.isclose(dense, after, rel_tol=config["loss"]["loss_relative_tolerance"],
                            abs_tol=config["loss"]["loss_absolute_tolerance"]):
            raise RuntimeError(f"Dense vs chunk loss mismatch: {dense} vs {after}")

        # Compare gradients through the ACTUAL Qwen LoRA path, dropout disabled.
        def gradients(use_chunk):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = qwen_chunked_forward_loss(model, first, chunk) if use_chunk else model(**first).loss
            loss.backward()
            result = {name: parameter.grad.detach().cpu().clone() for name, parameter in trainables}
            model.zero_grad(set_to_none=True)
            return result

        dense_grad = gradients(False)
        chunk_grad = gradients(True)
        maximum_delta = max(float((dense_grad[name] - chunk_grad[name]).abs().max()) for name in dense_grad)
        squared_difference = sum(float((dense_grad[name] - chunk_grad[name]).square().sum()) for name in dense_grad)
        squared_reference = sum(float(dense_grad[name].square().sum()) for name in dense_grad)
        relative_l2 = math.sqrt(squared_difference / max(squared_reference, 1e-30))
        gradient_close = all(torch.allclose(dense_grad[name], chunk_grad[name],
                            atol=config["loss"]["gradient_absolute_tolerance"],
                            rtol=config["loss"]["gradient_relative_tolerance"]) for name in dense_grad)
        report["precision_loss_validation"] = {
            "storage_changes": changes, "before_cast_loss": before, "after_cast_loss": after,
            "precision_absolute_loss_delta": delta, "dense_loss": dense,
            "dense_chunk_absolute_loss_delta": abs(dense - after),
            "actual_lora_gradient_max_absolute_delta": maximum_delta,
            "actual_lora_gradient_relative_l2": relative_l2,
            "actual_lora_gradient_close": gradient_close,
            "validation_memory": memory_snapshot(torch),
        }
        if not gradient_close:
            raise RuntimeError(f"Actual LoRA gradients differ beyond tolerance: max={maximum_delta}, relative_l2={relative_l2}")
        del dense_grad, chunk_grad, first
        gc.collect()
        torch.cuda.empty_cache()
        progress(f"Precision/loss validated: precision delta={delta:.8g}; chunk loss delta={abs(dense-after):.8g}; gradient max delta={maximum_delta:.8g}.")

        optimizer = torch.optim.AdamW((p for _, p in trainables), lr=ref["optimization"]["learning_rate"])
        accumulation = ref["optimization"]["gradient_accumulation_steps"]
        initial_parameters = {name: p.detach().cpu().clone() for name, p in trainables}

        def run_steps(items, phase):
            nonlocal stage
            stage = monitor.phase = phase
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begun = time.perf_counter()
            rows = []
            snapshots = []
            for index, item in enumerate(items, 1):
                started = time.perf_counter()
                batch = batch_on_cuda(item)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = qwen_chunked_forward_loss(model, batch, chunk)
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite loss in {phase}/{index}.")
                (loss / accumulation).backward()
                if index % accumulation == 0:
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in trainables):
                        raise RuntimeError(f"Missing/non-finite adapter gradients in {phase}/{index}.")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                del loss, batch
                torch.cuda.synchronize()
                rows.append({"micro_step": index, "sequence_tokens": item["sequence_tokens"],
                             "loss": value, "seconds": time.perf_counter() - started,
                             **memory_snapshot(torch)})
                if index % accumulation == 0:
                    snapshots.append(rows[-1])
                    progress(f"{phase}: micro {index}/{len(items)}, optimizer {index // accumulation}, loss {value:.5f}, peak allocated {rows[-1]['peak_allocated_mib']} MiB.")
            result = {"micro_steps": len(rows), "optimizer_steps": len(rows) // accumulation,
                      "finite_loss_and_gradients": True, "seconds": time.perf_counter() - begun,
                      "steps": rows, "memory": memory_snapshot(torch),
                      "optimizer_boundary_snapshots": snapshots}
            result["median_micro_step_seconds"] = statistics.median(row["seconds"] for row in rows)
            return result

        report["smoke24"] = run_steps(smoke_tokens, "smoke24")
        if not any(not torch.equal(initial_parameters[name], p.detach().cpu()) for name, p in trainables):
            raise RuntimeError("Optimizer steps did not change any adapter weights.")
        report["smoke24"]["optimizer_changed_parameters"] = True
        del initial_parameters
        write_json(out / "smoke24_checkpoint_report.json", report)
        progress("24-task smoke passed; proceeding ONLY to bounded longest-training-record stress.")
        report["longest_stress"] = run_steps([tokens for _, tokens in longest] * config["stress"]["repeats"], "longest_stress")
        snapshots = report["longest_stress"]["optimizer_boundary_snapshots"]
        growth = max(row["allocated_mib"] for row in snapshots[1:]) - snapshots[0]["allocated_mib"]
        report["longest_stress"]["allocated_growth_after_optimizer_warmup_mib"] = growth
        report["longest_stress"]["memory_growth_pass"] = growth <= config["stress"]["max_allocated_growth_mib"]

        stage = monitor.phase = "adapter_save_reload"
        model.eval()
        first = batch_on_cuda(smoke_tokens[0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            saved_loss = float(qwen_chunked_forward_loss(model, first, chunk))
        saved_weights = {name: p.detach().cpu().clone() for name, p in trainables}
        model.save_pretrained(adapter_dir, safe_serialization=True)
        report["adapter"] = {"path": config["adapter_directory"],
                             "files": adapter_file_manifest(adapter_dir),
                             "formal_training_initialization_allowed": False}
        del optimizer, model, base, trainables, first
        gc.collect()
        torch.cuda.empty_cache()
        reloaded_base = prepare_model_for_kbit_training(load_base(), use_gradient_checkpointing=False)
        restore_frozen_large_layers_bf16(reloaded_base)
        reloaded = PeftModel.from_pretrained(reloaded_base, str(adapter_dir), is_trainable=False,
                                           local_files_only=True)
        reloaded.eval()
        parameters = dict(reloaded.named_parameters())
        if not all(torch.equal(weight, parameters[name].detach().cpu()) for name, weight in saved_weights.items()):
            raise RuntimeError("Saved/reloaded adapter weights mismatch.")
        first = batch_on_cuda(smoke_tokens[0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reload_loss = float(qwen_chunked_forward_loss(reloaded, first, chunk))
        if not math.isfinite(reload_loss) or not math.isclose(saved_loss, reload_loss, abs_tol=2e-5, rel_tol=2e-5):
            raise RuntimeError(f"Reload forward mismatch: {saved_loss} vs {reload_loss}.")
        report["adapter"].update(reload_succeeded=True, weights_identical=True,
                                 saved_loss=saved_loss, reloaded_loss=reload_loss)
        del parameters, saved_weights, first, reloaded, reloaded_base
        gc.collect()
        torch.cuda.empty_cache()
        monitor.stop()
        run_samples = [row for row in monitor.samples if row["phase"] in {"smoke24", "longest_stress"}]
        driver = [row["device_free_mib"] for row in run_samples if "device_free_mib" in row]
        shared = [row["process_shared_mib"] for row in run_samples if "process_shared_mib" in row]
        telemetry_complete = bool(driver and shared)
        headroom_pass = (telemetry_complete and min(driver) >= config["stress"]["minimum_sampled_device_free_mib"]
                         and max(shared) <= config["stress"]["maximum_sampled_process_shared_mib"])
        report["headroom"] = {
            "telemetry_available": telemetry_complete,
            "minimum_sampled_device_free_mib": min(driver) if driver else None,
            "maximum_sampled_process_shared_mib": max(shared) if shared else None,
            "passed": headroom_pass, "thresholds": config["stress"],
            "sampling_not_instantaneous_peak_proof": True,
        }
        report["protected_hashes_after"] = protected_hashes()
        if report["protected_hashes_before"] != report["protected_hashes_after"]:
            raise RuntimeError("A frozen input/reference changed during smoke.")
        report.update(status="passed", completed_at_utc=utc_now(),
                      engineering_pass=True,
                      formal_configuration_ready_for_review=(headroom_pass and report["longest_stress"]["memory_growth_pass"]),
                      config_sha256=sha256_file(config_path),
                      code_sha256={path: sha256_file(project_path(path)) for path in
                                   ("scripts/run_qlora_low_memory_smoke.py", "src/generation/qlora_memory.py")})
        write_json(out / "smoke_report.json", report)
        progress(f"Completed engineering smoke; headroom gate={headroom_pass}. No formal training/effect evaluation performed.")
        return report
    except Exception as error:
        monitor.stop()
        report.update(status="failed", failed_at_utc=utc_now(), failure_stage=stage,
                      error=str(error), traceback=traceback.format_exc())
        write_json(out / "failure_report.json", report)
        raise
    finally:
        write_json(out / "gpu_telemetry.json", {"samples": monitor.samples, "errors": monitor.errors})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lora_qlora_low_memory_smoke_v1.json")
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps({"status": result["status"], "headroom": result["headroom"],
                      "report": str(project_path(load_json(args.config)["output_directory"]) / "smoke_report.json")}, ensure_ascii=False))
