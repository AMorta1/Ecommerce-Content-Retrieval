"""Run the bounded P6 QLoRA engineering smoke on 24 frozen P5 train records."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(raw_path: str) -> Path:
    return (PROJECT_ROOT / raw_path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def stable_rank(namespace: str, seed: int, category: str, product_id: str) -> str:
    payload = f"{namespace}:{seed}:{category}:{product_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_smoke_records(
    records: list[dict[str, Any]], input_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    required_split = input_config["required_split"]
    task_types = list(input_config["task_types"])
    required_tasks = set(task_types)
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    for record in records:
        if record.get("split") != required_split:
            raise ValueError(
                f"Only split={required_split!r} is allowed, got {record.get('split')!r}."
            )
        category = str(record["category_l2"])
        product_id = str(record["product_id"])
        task_type = str(record["task_type"])
        if task_type not in required_tasks:
            raise ValueError(f"Unexpected task_type={task_type!r}.")
        key = (category, product_id)
        if task_type in grouped[key]:
            raise ValueError(f"Duplicate task {key}/{task_type}.")
        grouped[key][task_type] = record

    by_category: dict[str, list[str]] = defaultdict(list)
    for (category, product_id), task_map in grouped.items():
        if set(task_map) == required_tasks:
            by_category[category].append(product_id)

    expected_categories = int(input_config["expected_category_count"])
    if len(by_category) != expected_categories:
        raise ValueError(
            f"Expected {expected_categories} categories, found {len(by_category)}."
        )

    namespace = str(input_config["selection_namespace"])
    seed = int(input_config["selection_seed"])
    products_per_category = int(input_config["products_per_category"])
    selection: list[dict[str, str]] = []
    selected_records: list[dict[str, Any]] = []
    for category in sorted(by_category):
        ranked = sorted(
            by_category[category],
            key=lambda product_id: stable_rank(namespace, seed, category, product_id),
        )
        selected_ids = ranked[:products_per_category]
        if len(selected_ids) != products_per_category:
            raise ValueError(f"Category {category} has insufficient complete products.")
        for product_id in selected_ids:
            selection.append({"category_l2": category, "product_id": product_id})
            task_map = grouped[(category, product_id)]
            selected_records.extend(task_map[task] for task in task_types)

    if len(selection) != int(input_config["expected_product_count"]):
        raise ValueError("Smoke product count does not match config.")
    if len(selected_records) != int(input_config["expected_record_count"]):
        raise ValueError("Smoke record count does not match config.")
    return selected_records, selection


def token_id_list(value: Any) -> list[int]:
    """Normalize Transformers chat-template outputs to one flat token-id list."""
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token_id, int) for token_id in value):
        raise TypeError("Tokenizer output cannot be normalized to a flat token-id list.")
    return value


def tokenize_supervised_record(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, Any]:
    import torch

    messages = record["messages"]
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Each record must end with one assistant target message.")
    prompt_messages = messages[:-1]
    prompt_ids = token_id_list(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = token_id_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens are not an exact prefix of the supervised sequence.")
    if len(full_ids) > max_length:
        raise ValueError(
            f"Sequence has {len(full_ids)} tokens, exceeding max_length={max_length}; truncation is forbidden."
        )
    labels = list(full_ids)
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
    supervised_tokens = sum(label != -100 for label in labels)
    if supervised_tokens <= 0:
        raise ValueError("Record has no supervised assistant tokens.")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "sequence_tokens": len(full_ids),
        "prompt_tokens": len(prompt_ids),
        "supervised_tokens": supervised_tokens,
    }


def memory_snapshot(torch: Any) -> dict[str, float]:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }


def adapter_file_manifest(adapter_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    ]


def validate_config(config: dict[str, Any], config_path: Path) -> tuple[Path, Path, Path]:
    if config["scope"] != "engineering_smoke_only":
        raise ValueError("Config scope must be engineering_smoke_only.")
    guards = config["guards"]
    if any(
        guards[key]
        for key in (
            "formal_training",
            "full_epoch_over_p5_train",
            "project_validation_used",
            "project_test_used",
            "rag_v2_holdout_used",
            "p5_data_mutation_allowed",
            "effect_evaluation_allowed",
        )
    ):
        raise ValueError("A smoke safety guard is not false.")

    input_path = project_path(config["input"]["path"])
    expected_input = project_path("data/processed/lora_instruction_data_v1/train.jsonl")
    if input_path != expected_input:
        raise ValueError("Smoke input must be the frozen P5 train JSONL.")
    if sha256_file(input_path) != config["input"]["expected_sha256"]:
        raise ValueError("Frozen P5 train hash mismatch.")

    output_dir = project_path(config["output"]["directory"])
    p5_data_dir = project_path("data/processed/lora_instruction_data_v1")
    if output_dir == p5_data_dir or p5_data_dir in output_dir.parents:
        raise ValueError("Smoke output may not be placed under the frozen P5 data directory.")
    adapter_dir = output_dir / config["output"]["adapter_subdirectory"]
    report_path = output_dir / config["output"]["report_filename"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {output_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return input_path, output_dir, adapter_dir


def run(config_path: Path) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = load_json(config_path)
    input_path, output_dir, adapter_dir = validate_config(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    failure_path = output_dir / config["output"]["failure_filename"]
    report_path = output_dir / config["output"]["report_filename"]
    started_at = utc_now()

    try:
        seed = int(config["optimization"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the bounded QLoRA smoke.")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        all_records = load_jsonl(input_path)
        records, selected_products = select_smoke_records(all_records, config["input"])

        model_config = config["model"]
        cache_dir = project_path(model_config["cache_dir"])
        snapshot_dir = (
            cache_dir
            / ("models--" + model_config["name"].replace("/", "--"))
            / "snapshots"
            / model_config["revision"]
        )
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"Pinned local model snapshot is missing: {snapshot_dir}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["name"],
            revision=model_config["revision"],
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        max_length = int(config["optimization"]["max_sequence_length"])
        tokenized = [tokenize_supervised_record(tokenizer, record, max_length) for record in records]
        sequence_lengths = [item["sequence_tokens"] for item in tokenized]
        prompt_lengths = [item["prompt_tokens"] for item in tokenized]
        supervised_lengths = [item["supervised_tokens"] for item in tokenized]

        compute_dtype = getattr(torch, model_config["bnb_4bit_compute_dtype"])
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=bool(model_config["bnb_4bit_use_double_quant"]),
            bnb_4bit_compute_dtype=compute_dtype,
        )

        def load_base_model() -> Any:
            return AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                revision=model_config["revision"],
                cache_dir=str(cache_dir),
                local_files_only=True,
                device_map={"": 0},
                quantization_config=quantization_config,
                dtype=compute_dtype,
                low_cpu_mem_usage=True,
            )

        base_model = load_base_model()
        if not getattr(base_model, "is_loaded_in_4bit", False):
            raise RuntimeError("Model did not report is_loaded_in_4bit=True.")
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=bool(config["optimization"]["gradient_checkpointing"]),
        )
        lora_config = LoraConfig(
            task_type=config["lora"]["task_type"],
            target_modules=list(config["lora"]["target_modules"]),
            r=int(config["lora"]["r"]),
            lora_alpha=int(config["lora"]["lora_alpha"]),
            lora_dropout=float(config["lora"]["lora_dropout"]),
            bias=config["lora"]["bias"],
            inference_mode=False,
        )
        model = get_peft_model(base_model, lora_config)
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        if trainable_parameters <= 0:
            raise RuntimeError("LoRA attached but no trainable parameters were found.")

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["optimization"]["learning_rate"]),
        )
        gradient_accumulation_steps = int(config["optimization"]["gradient_accumulation_steps"])
        configured_micro_steps = int(config["optimization"]["micro_steps"])
        configured_optimizer_steps = int(config["optimization"]["optimizer_steps"])
        if configured_micro_steps != len(tokenized):
            raise ValueError("Configured micro_steps must equal the 24 selected smoke records.")
        if configured_micro_steps // gradient_accumulation_steps != configured_optimizer_steps:
            raise ValueError("Configured optimizer step count is inconsistent.")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        optimizer_steps = 0
        for index, item in enumerate(tokenized, start=1):
            batch = {
                key: value.to(model.device)
                for key, value in item.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at micro step {index}: {loss}")
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            (loss / gradient_accumulation_steps).backward()
            if index % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            del outputs, loss, batch

        if optimizer_steps != configured_optimizer_steps:
            raise RuntimeError("Observed optimizer step count does not match config.")
        if not all(math.isfinite(loss) for loss in losses):
            raise RuntimeError("At least one recorded loss is non-finite.")

        memory_after_training = memory_snapshot(torch)
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_files = adapter_file_manifest(adapter_dir)
        if not any(item["path"].endswith("adapter_model.safetensors") for item in adapter_files):
            raise RuntimeError("Adapter safetensors file was not saved.")

        del optimizer, model, base_model
        gc.collect()
        torch.cuda.empty_cache()

        reload_base = load_base_model()
        reload_base.config.use_cache = False
        reloaded_model = PeftModel.from_pretrained(
            reload_base,
            adapter_dir,
            is_trainable=False,
            local_files_only=True,
        )
        adapter_names = sorted(reloaded_model.peft_config)
        if not adapter_names:
            raise RuntimeError("Reloaded model has no PEFT adapter config.")
        reload_trainable_parameters = sum(
            parameter.numel() for parameter in reloaded_model.parameters() if parameter.requires_grad
        )
        memory_after_reload = memory_snapshot(torch)

        report = {
            "version": config["version"],
            "status": "passed",
            "scope": config["scope"],
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "config": {
                "path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(config_path),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            },
            "input": {
                "path": str(input_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(input_path),
                "all_train_records_seen_for_selection": len(all_records),
                "selected_products": selected_products,
                "selected_product_count": len(selected_products),
                "selected_record_count": len(records),
                "task_counts": {
                    task: sum(record["task_type"] == task for record in records)
                    for task in config["input"]["task_types"]
                },
            },
            "model": {
                "name": model_config["name"],
                "revision": model_config["revision"],
                "snapshot_path": str(snapshot_dir),
                "local_files_only": True,
                "loaded_in_4bit": True,
                "quant_type": model_config["bnb_4bit_quant_type"],
                "double_quant": bool(model_config["bnb_4bit_use_double_quant"]),
                "compute_dtype": model_config["bnb_4bit_compute_dtype"],
            },
            "lora": {
                **config["lora"],
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "trainable_percent": round(trainable_parameters / total_parameters * 100, 6),
            },
            "tokenization": {
                "truncation": False,
                "max_sequence_length": max_length,
                "sequence_tokens": {
                    "min": min(sequence_lengths),
                    "max": max(sequence_lengths),
                    "mean": round(sum(sequence_lengths) / len(sequence_lengths), 2),
                },
                "prompt_tokens": {
                    "min": min(prompt_lengths),
                    "max": max(prompt_lengths),
                },
                "supervised_tokens": {
                    "min": min(supervised_lengths),
                    "max": max(supervised_lengths),
                },
            },
            "execution": {
                "micro_steps": len(losses),
                "optimizer_steps": optimizer_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "finite_loss": True,
                "losses": losses,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "effect_interpretation": "forbidden; engineering smoke only",
            },
            "memory": {
                "after_training": memory_after_training,
                "after_adapter_reload": memory_after_reload,
            },
            "adapter": {
                "directory": str(adapter_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "files": adapter_files,
                "reload_succeeded": True,
                "adapter_names": adapter_names,
                "reload_trainable_parameters": reload_trainable_parameters,
            },
            "guards": config["guards"],
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    except Exception as error:
        failure = {
            "version": config.get("version"),
            "status": "failed",
            "scope": config.get("scope"),
            "started_at_utc": started_at,
            "failed_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "guards": config.get("guards"),
        }
        failure_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/lora_qlora_smoke_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    report = run(config_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
