from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "run_qlora_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_qlora_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TensorLike:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def test_token_id_list_accepts_transformers_batch_encoding_shape():
    assert MODULE.token_id_list({"input_ids": [11, 12, 13], "attention_mask": [1, 1, 1]}) == [
        11,
        12,
        13,
    ]


def test_token_id_list_flattens_single_tensor_batch():
    assert MODULE.token_id_list({"input_ids": TensorLike([[21, 22]])}) == [21, 22]


def test_token_id_list_rejects_non_integer_tokens():
    with pytest.raises(TypeError, match="flat token-id list"):
        MODULE.token_id_list({"input_ids": ["21", "22"]})
