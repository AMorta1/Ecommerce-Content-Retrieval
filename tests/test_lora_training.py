from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.generation import lora_training as training


def test_project_counts_and_partial_group():
    assert training.step_counts(3618, 8, 3) == {
        "micro_steps_per_epoch": 3618, "optimizer_steps_per_epoch": 453,
        "total_optimizer_steps": 1359, "last_group_records": 2,
    }
    groups = list(training.accumulation_groups(list(range(3618)), 8))
    assert len(groups) == 453 and len(groups[-1]) == 2
    assert groups[-1] == [3616, 3617]


def test_partial_accumulation_mean_matches_actual_group_not_eight():
    parameter = torch.tensor(1.0, requires_grad=True)
    losses = [parameter * 2, parameter * 4]
    for loss in losses:
        (loss / len(losses)).backward()
    assert parameter.grad.item() == 3.0


def test_epoch_order_deterministic_independent_of_global_rng():
    first = training.epoch_order(3618, 42, 0)
    random.seed(200)
    assert first == training.epoch_order(3618, 42, 0)
    assert first != training.epoch_order(3618, 42, 1)
    assert sorted(first) == list(range(3618))


def test_resume_only_at_optimizer_boundary():
    with pytest.raises(ValueError, match="boundary"):
        list(training.accumulation_groups(list(range(12)), 8, cursor=3))
    assert list(training.accumulation_groups(list(range(12)), 8, cursor=8)) == [[8, 9, 10, 11]]


def config():
    return training.read_json(training.ROOT / "configs/lora_qlora_train_v1.json")


def test_full_training_requires_explicit_permission_before_data_io(monkeypatch):
    monkeypatch.setattr(training, "digest", lambda _: pytest.fail("Must not open data before guard"))
    with pytest.raises(PermissionError, match="confirm-formal-training"):
        training.validate_config(config(), "train")


@pytest.mark.parametrize("stop", [None, 0, 5, 453, 1359])
def test_acceptance_cannot_run_epoch_or_full_training(stop, monkeypatch):
    monkeypatch.setattr(training, "digest", lambda _: pytest.fail("Bounds precede IO"))
    with pytest.raises(ValueError, match="bounded"):
        training.validate_config(config(), "acceptance", stop=stop)


@pytest.mark.parametrize("key,value", [("r", 4), ("lora_dropout", 0.0), ("target_modules", ["q_proj"])])
def test_frozen_lora_parameters_rejected(key, value):
    candidate = config()
    candidate["lora"][key] = value
    with pytest.raises(ValueError, match="LoRA"):
        training.validate_config(candidate, "acceptance", stop=2)


def test_test_data_path_is_rejected_before_open(monkeypatch):
    candidate = config()
    candidate["data"]["lora_train"]["path"] = "forbidden/project_test.jsonl"
    monkeypatch.setattr(training, "digest", lambda _: pytest.fail("Forbidden path must not be read"))
    with pytest.raises(ValueError, match="Forbidden data"):
        training.validate_config(candidate, "acceptance", stop=2)
    with pytest.raises(ValueError, match="Only lora_train"):
        training.load_records(candidate, "project_test")


def test_tokenizer_parity_with_frozen_smoke():
    from scripts.run_qlora_smoke import tokenize_supervised_record
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return {"input_ids": [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]}
    row = {"product_id": "1", "task_type": "title",
           "messages": [{"role": "user", "content": "事实"}, {"role": "assistant", "content": "标题"}]}
    first = training.tokenize_record(Tokenizer(), row, 320)
    second = tokenize_supervised_record(Tokenizer(), row, 320)
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(first[key], second[key])
    with pytest.raises(ValueError, match="truncation forbidden"):
        training.tokenize_record(Tokenizer(), row, 4)


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.dropout = torch.nn.Dropout(0.05)

    def forward(self, tensor):
        return self.linear(self.dropout(tensor)).square().mean()

    def save_pretrained(self, directory, safe_serialization):
        directory.mkdir()
        torch.save(self.state_dict(), directory / "adapter_test.pt")


def test_complete_checkpoint_cpu_resume_replays_rng_optimizer_scheduler_and_loss(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    model = TinyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    inputs = torch.ones(2, 3)
    def step(m, opt, sched):
        opt.zero_grad(set_to_none=True)
        loss = m(inputs)
        loss.backward()
        opt.step()
        sched.step()
        return loss.item()
    step(model, optimizer, scheduler)
    state = {"global_step": 1, "epoch": 0, "cursor": 8}
    identity = {"run_kind": "acceptance_only", "config": {"seed": 42}}
    directory = tmp_path / "checkpoint-step-000001"
    training.save_checkpoint(directory, model, optimizer, scheduler, state, identity)
    expected = step(model, optimizer, scheduler)
    expected_weights = copy.deepcopy(model.state_dict())
    expected_optimizer = training.tensor_tree_digest(optimizer.state_dict())
    expected_scheduler = scheduler.state_dict()
    reloaded = TinyAdapter()
    reloaded.load_state_dict(torch.load(directory / "adapter/adapter_test.pt", weights_only=True))
    resumed_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=2e-4)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    restored, verification = training.restore_checkpoint_state(directory, resumed_optimizer, resumed_scheduler, identity)
    assert restored == state and verification["optimizer_scheduler_rng_restored"]
    assert step(reloaded, resumed_optimizer, resumed_scheduler) == expected
    assert training.tensor_tree_digest(resumed_optimizer.state_dict()) == expected_optimizer
    assert resumed_scheduler.state_dict() == expected_scheduler
    for name, weight in expected_weights.items():
        assert torch.equal(weight, reloaded.state_dict()[name])


def test_checkpoint_corruption_and_formal_acceptance_identity_rejected(tmp_path):
    model = TinyAdapter()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    identity = {"run_kind": "acceptance_only", "config": {}}
    directory = tmp_path / "checkpoint-step-000001"
    training.save_checkpoint(directory, model, optimizer, scheduler,
                             {"global_step": 1, "epoch": 0, "cursor": 8}, identity)
    with pytest.raises(ValueError, match="run kind"):
        training.verify_checkpoint(directory, {"run_kind": "formal_training", "config": {}})
    (directory / "training_state.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        training.verify_checkpoint(directory, identity)


def test_dev_evaluation_no_grad_no_updates_preserves_rng_and_training_mode():
    model = TinyAdapter()
    model.train()
    before = copy.deepcopy(model.state_dict())
    rng = training.tensor_tree_digest(training.capture_rng())
    items = [{"input_ids": torch.ones(1, 3), "attention_mask": torch.ones(1, 3),
              "labels": torch.ones(1, 3), "product_id": str(i), "task_type": "title", "sequence_tokens": 3}
             for i in range(4)]
    observed = []
    def loss_fn(m, batch):
        observed.append(torch.is_grad_enabled())
        return m(batch["input_ids"])
    result = training.evaluate_dev(model, items, loss_fn, device="cpu")
    assert observed == [False] * 4
    assert result["records"] == 4 and result["parameter_updates"] == 0
    assert model.training and all(p.grad is None for p in model.parameters())
    assert training.tensor_tree_digest(training.capture_rng()) == rng
    for name, weight in before.items():
        assert torch.equal(weight, model.state_dict()[name])


def test_retention_keeps_latest_best_and_unrelated_directories(tmp_path):
    paths = []
    for step in (151, 302, 453):
        directory = tmp_path / f"checkpoint-step-{step:06d}"
        directory.mkdir()
        (directory / "checkpoint_manifest.json").write_text(json.dumps({"complete": True}))
        paths.append(directory)
    unrelated = tmp_path / "user-files"
    unrelated.mkdir()
    training.prune_checkpoints(tmp_path, [paths[0], paths[2]])
    assert paths[0].exists() and paths[2].exists() and unrelated.exists()
    assert not paths[1].exists()


def test_outputs_cannot_target_p5_or_root():
    with pytest.raises(ValueError, match="strictly beneath"):
        training.validate_output("data/processed/lora_instruction_data_v1", "artifacts/lora")
    with pytest.raises(ValueError, match="strictly beneath"):
        training.validate_output("artifacts/lora", "artifacts/lora")
