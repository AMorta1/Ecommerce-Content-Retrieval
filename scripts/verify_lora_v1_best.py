"""Independent offline best-adapter reload; train-only mechanical inference, not evaluation."""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.lora_training import (
    atomic_json, digest, json_digest, load_records, memory, now, parameter_signature,
    read_json, resolve, tokenize_record, verify_checkpoint,
)
from src.generation.qlora_memory import prepare_kbit_with_cpu_staged_large_layers, qwen_chunked_forward_loss


def run():
    frozen = read_json(ROOT / "reports/generation/lora/lora_v1_pretraining_manifest.json")
    config = frozen["config"]
    reports = resolve(config["output"]["report_directory"])
    result_path = reports / "best_adapter_reload.json"
    if result_path.exists():
        raise FileExistsError("Refusing to overwrite independent reload verification.")
    begun = time.perf_counter()
    result = {"status": "running", "started_at_utc": now(), "pid": os.getpid(),
              "engineering_only": True, "project_validation_read": False,
              "project_test_read": False, "rag_holdout_read": False,
              "optimizer_scheduler_state_loaded": False,
              "verification_code_sha256": digest(Path(__file__))}
    try:
        for item in frozen["file_snapshots"]:
            if digest(resolve(item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen file changed: {item['path']}")
        invocations = list(reports.glob("invocation_*.json"))
        if len(invocations) != 1:
            raise ValueError("Expected exactly one fresh formal-training invocation.")
        training = read_json(invocations[0])
        if training["status"] != "passed" or training["final_state"]["completed_epochs"] != 3:
            raise ValueError("Formal three-epoch training has not completed.")
        if training["start_state"]["global_step"] != 0 or "resume" in training:
            raise ValueError("Expected approved fresh-init training, not resume.")
        evaluations = training["evaluations"]
        if len(evaluations) != 3 or [e["global_step"] for e in evaluations] != [453, 906, 1359]:
            raise ValueError("Best selection must use three epoch-end evaluations only.")
        if any(e["dataset"] != "lora_train_dev" or e["records"] != 402 for e in evaluations):
            raise ValueError("Incomplete or forbidden dev evaluation.")
        selected = min(evaluations, key=lambda e: e["dev_loss"])
        pointer = read_json(resolve(config["output"]["directory"]) / "best.json")
        if pointer["global_step"] != selected["global_step"] or pointer["dev_loss"] != selected["dev_loss"]:
            raise ValueError("Best pointer differs from complete-dev minimum.")
        checkpoint = resolve(pointer["path"])
        if checkpoint.parent != resolve(config["output"]["directory"]) / "checkpoints":
            raise ValueError("Best checkpoint outside this formal run.")
        verify_checkpoint(checkpoint, training["identity"])

        import torch
        from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch.manual_seed(config["training"]["seed"])
        torch.cuda.manual_seed_all(config["training"]["seed"])
        snapshot = frozen["base_model"]["local_snapshot"]
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if json_digest(tokenizer.chat_template) != frozen["prompt"]["chat_template_sha256"]:
            raise ValueError("Chat template changed.")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, device_map={"": 0},
                                                  quantization_config=quant, dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True)
        base.config.use_cache = False
        base, _ = prepare_kbit_with_cpu_staged_large_layers(
            base, prepare_model_for_kbit_training, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model = PeftModel.from_pretrained(base, str(checkpoint / "adapter"), is_trainable=False,
                                         local_files_only=True)
        model.eval()
        if not model.is_loaded_in_4bit or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Expected frozen 4-bit base + inference-only adapter.")
        weights_path = checkpoint / "adapter/adapter_model.safetensors"
        expected = load_file(str(weights_path), device="cpu")
        actual = get_peft_model_state_dict(model)
        if set(actual) != set(expected) or any(not torch.equal(actual[k].detach().cpu(), expected[k]) for k in actual):
            raise RuntimeError("Reloaded adapter weights differ from saved weights.")
        del actual, expected
        rows = load_records(config, "lora_train")
        first_id = str(rows[0]["product_id"])
        samples = [r for r in rows if str(r["product_id"]) == first_id]
        if {r["task_type"] for r in samples} != set(config["data"]["task_types"]) or len(samples) != 3:
            raise ValueError("Expected all three frozen train tasks for the first product.")
        torch.cuda.reset_peak_memory_stats()
        outputs = []
        with torch.no_grad():
            for row in samples:
                item = tokenize_record(tokenizer, row, config["engineering"]["max_sequence_length"])
                batch = {k: item[k].to("cuda") for k in ("input_ids", "attention_mask", "labels")}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    value = float(qwen_chunked_forward_loss(model, batch, config["engineering"]["loss_chunk_tokens"]))
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite independently reloaded forward loss.")
                del batch
                prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
                                                       add_generation_prompt=True, return_tensors="pt",
                                                       return_dict=True)
                prompt = {k: v.to("cuda") for k, v in prompt.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generated = model.generate(**prompt, max_new_tokens=128, do_sample=False,
                                               use_cache=False, pad_token_id=tokenizer.eos_token_id)
                new_ids = generated[0, prompt["input_ids"].shape[1]:].tolist()
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                if not text.strip():
                    raise RuntimeError("Empty mechanical inference output.")
                outputs.append({"source": "lora_train", "product_id": first_id,
                                "task_type": row["task_type"], "forward_loss": value,
                                "new_tokens": len(new_ids), "output": text,
                                "quality_judgment": "not_performed"})
                del prompt, generated
        torch.cuda.synchronize()
        result.update(status="passed", completed_at_utc=now(), seconds=time.perf_counter() - begun,
                      checkpoint=str(checkpoint), best=pointer, base_snapshot=snapshot,
                      base_revision=config["model"]["revision"], adapter_sha256=digest(weights_path),
                      independent_process=True, saved_weights_exactly_reloaded=True,
                      signature=parameter_signature(model), inference={"do_sample": False,
                      "max_new_tokens": 128, "use_cache": False, "purpose": "mechanical_only"},
                      samples=outputs, memory=memory())
        atomic_json(result_path, result)
        print(f"Independent best adapter reload passed: {result_path}", flush=True)
    except Exception as error:
        result.update(status="failed", failed_at_utc=now(), error=str(error))
        atomic_json(result_path, result)
        raise


if __name__ == "__main__":
    run()
