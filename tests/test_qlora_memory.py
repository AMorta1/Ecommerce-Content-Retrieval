from __future__ import annotations

import pytest
import torch

from src.generation.qlora_memory import (
    chunked_causal_loss, prepare_kbit_with_cpu_staged_large_layers,
    restore_frozen_large_layers_bf16,
)


@pytest.mark.parametrize("chunk_tokens", [1, 3, 32])
def test_chunked_loss_and_hidden_gradients_match_dense_causal_ce(chunk_tokens):
    torch.manual_seed(42)
    head = torch.nn.Linear(7, 19, bias=False)
    head.requires_grad_(False)
    first = torch.randn(2, 8, 7, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 19, (2, 8))
    labels[:, :3] = -100
    labels[1, 6] = -100
    dense = torch.nn.functional.cross_entropy(
        head(first[:, :-1, :]).float().reshape(-1, 19), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    chunked = chunked_causal_loss(second, head, labels, chunk_tokens)
    dense.backward()
    chunked.backward()
    torch.testing.assert_close(chunked, dense, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(second.grad, first.grad, rtol=1e-6, atol=1e-6)
    assert head.weight.grad is None


def test_chunked_loss_rejects_no_targets_or_trainable_head():
    head = torch.nn.Linear(4, 9, bias=False)
    states = torch.randn(1, 3, 4)
    labels = torch.full((1, 3), -100)
    with pytest.raises(ValueError, match="frozen output head"):
        chunked_causal_loss(states, head, labels)
    head.requires_grad_(False)
    with pytest.raises(ValueError, match="No valid causal targets"):
        chunked_causal_loss(states, head, labels)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9, bias=False)
        self.norm = torch.nn.LayerNorm(4)
        self.adapter = torch.nn.Linear(4, 2, bias=False)
        self.embedding.requires_grad_(False)
        self.head.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


def test_precision_cast_preserves_norm_adapter_and_freezing():
    model = TinyModel()
    changes = restore_frozen_large_layers_bf16(model)
    assert len(changes) == 2
    assert model.embedding.weight.dtype == torch.bfloat16
    assert model.head.weight.dtype == torch.bfloat16
    assert not model.embedding.weight.requires_grad
    assert not model.head.weight.requires_grad
    assert model.norm.weight.dtype == torch.float32
    assert model.adapter.weight.dtype == torch.float32
    assert model.adapter.weight.requires_grad
    assert sum(change["saved_bytes"] for change in changes) == 144


def test_precision_cast_rejects_trainable_embedding():
    model = TinyModel()
    model.embedding.requires_grad_(True)
    with pytest.raises(ValueError, match="must be frozen"):
        restore_frozen_large_layers_bf16(model)


def test_cpu_staging_keeps_preparation_contract_and_final_weights():
    model = TinyModel()
    originals = [model.embedding.weight.detach().clone().bfloat16(), model.head.weight.detach().clone().bfloat16()]
    observed = []

    def prepare(candidate, use_gradient_checkpointing):
        assert candidate is model
        assert use_gradient_checkpointing is False
        for module in (candidate.embedding, candidate.head):
            assert module.weight.device.type == "cpu"
            assert not module.weight.requires_grad
            observed.append(module.weight.dtype)
            module.weight.data = module.weight.data.float()
        return candidate

    result, changes = prepare_kbit_with_cpu_staged_large_layers(model, prepare, use_gradient_checkpointing=False)
    assert result is model
    assert observed == [torch.bfloat16, torch.bfloat16]
    assert sum(row["saved_bytes"] for row in changes) == 144
    assert torch.equal(model.embedding.weight, originals[0])
    assert torch.equal(model.head.weight, originals[1])
    assert model.norm.weight.dtype == torch.float32
