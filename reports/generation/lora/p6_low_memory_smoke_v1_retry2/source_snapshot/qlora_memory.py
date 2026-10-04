"""Memory controls for frozen Qwen weights and assistant-only causal loss."""

from __future__ import annotations

from typing import Any


def restore_frozen_large_layers_bf16(model: Any) -> list[dict[str, Any]]:
    """Cast only frozen embedding/head; leave norms and LoRA weights unchanged."""
    import torch

    changes = []
    seen = set()
    for name, module in (
        ("input_embedding", model.get_input_embeddings()),
        ("output_head", model.get_output_embeddings()),
    ):
        weight = module.weight
        if weight.requires_grad:
            raise ValueError(f"{name} must be frozen before changing its storage dtype.")
        if weight.__class__.__name__ == "Params4bit":
            raise ValueError(f"{name} is quantized; this cast is for dense frozen weights only.")
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        before_dtype = str(weight.dtype)
        before_bytes = weight.numel() * weight.element_size()
        weight.data = weight.data.to(torch.bfloat16)
        changes.append({
            "layer": name,
            "before_dtype": before_dtype,
            "after_dtype": str(weight.dtype),
            "parameters": weight.numel(),
            "saved_bytes": before_bytes - weight.numel() * weight.element_size(),
            "requires_grad": weight.requires_grad,
        })
    return changes


def prepare_kbit_with_cpu_staged_large_layers(model: Any, prepare: Any, **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Let PEFT prepare normally, but stage dense frozen large layers on CPU.

    PEFT's temporary FP32 embedding/head copies then consume host RAM, not VRAM.
    The final model has exactly the same dtypes as prepare + restore above.
    This changes initialization storage only, not rank, targets or loss.
    """
    import torch

    staged = []
    seen = set()
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        weight = module.weight
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        if weight.__class__.__name__ == "Params4bit":
            raise ValueError("CPU staging is restricted to dense embedding/head weights.")
        weight.requires_grad_(False)
        staged.append((weight, weight.device))
        weight.data = weight.data.to(device="cpu", dtype=torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prepared = prepare(model, **kwargs)
    if prepared is not model:
        raise ValueError("Expected PEFT preparation to return the same model object.")
    changes = restore_frozen_large_layers_bf16(prepared)
    for weight, device in staged:
        weight.data = weight.data.to(device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prepared, changes


def chunked_causal_loss(hidden_states: Any, head: Any, labels: Any, chunk_tokens: int = 16) -> Any:
    """Same shifted mean CE, chunking FP32 loss but preserving head GEMM shape.

    Retain the native BF16 head projection shape to avoid BF16 kernel-shape
    changes in its backward. Only assistant logits enter FP32 CE, in checkpointed
    chunks; no full-sequence FP32 logits/CE allocation is retained. The causal
    shift, valid-token denominator and -100 semantics are unchanged.
    """
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")
    if head.weight.requires_grad:
        raise ValueError("This memory path requires a frozen output head.")
    if labels.shape != hidden_states.shape[:2]:
        raise ValueError("labels and hidden_states must have matching batch/sequence dimensions.")
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("No valid causal targets after shifting.")
    logits = head(hidden_states)
    selected_logits = logits[:, :-1, :][valid]
    selected_targets = targets[valid]

    def loss_sum(scores: Any, target: Any) -> Any:
        return functional.cross_entropy(scores.float(), target, reduction="sum")

    total = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    for start in range(0, count, chunk_tokens):
        scores = selected_logits[start : start + chunk_tokens]
        target = selected_targets[start : start + chunk_tokens]
        if torch.is_grad_enabled() and scores.requires_grad:
            piece = checkpoint(loss_sum, scores, target, use_reentrant=False)
        else:
            piece = loss_sum(scores, target)
        total = total + piece
    return total / count


def qwen_chunked_forward_loss(model: Any, batch: dict[str, Any], chunk_tokens: int) -> Any:
    """Use the Qwen decoder with injected LoRA; bypass full FP32 logits/CE."""
    base = model.get_base_model()
    if base.config.model_type != "qwen2":
        raise ValueError("This smoke is restricted to the frozen Qwen2 architecture.")
    outputs = base.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    return chunked_causal_loss(outputs.last_hidden_state, base.lm_head, batch["labels"], chunk_tokens)
