# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from core.training.trainer import (
    UnslothTrainer,
    _is_supported_local_quark_qwen35_checkpoint,
)


def _supported_config():
    tensor = {"dtype": "fp4", "group_size": 32, "scale_format": "e8m0"}
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
        },
        "quantization_config": {
            "quant_method": "quark",
            "quant_config": {
                "quant_mode": "eager_mode",
                "global_quant_config": {
                    "weight": {**tensor, "is_dynamic": False},
                    "input_tensors": {**tensor, "is_dynamic": True},
                },
            },
            "json_export_config": {"weight_format": "real_quantized"},
        },
    }


def test_local_quark_detection_is_exact(tmp_path):
    config = _supported_config()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert _is_supported_local_quark_qwen35_checkpoint(str(tmp_path))

    config["text_config"]["num_hidden_layers"] = 32
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert not _is_supported_local_quark_qwen35_checkpoint(str(tmp_path))

    config["text_config"]["num_hidden_layers"] = 64
    config["quantization_config"]["quant_config"]["global_quant_config"]["weight"][
        "group_size"
    ] = 64
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert not _is_supported_local_quark_qwen35_checkpoint(str(tmp_path))


def test_quark_adapter_save_requires_every_lora_pair_and_dense_endpoints(tmp_path):
    trainer = object.__new__(UnslothTrainer)
    UnslothTrainer.__init__(trainer)
    trainer._experimental_quark_qlora = True
    trainer.model = SimpleNamespace(_unsloth_quark_lora_wrapper_count=2)
    complete = {
        "base.layer0.lora_A.default.weight": torch.zeros(1),
        "base.layer0.lora_B.default.weight": torch.zeros(1),
        "base.layer1.lora_A.default.weight": torch.zeros(1),
        "base.layer1.lora_B.default.weight": torch.zeros(1),
        "base.embed_tokens.weight": torch.zeros(1),
        "base.lm_head.weight": torch.zeros(1),
    }
    save_file(complete, tmp_path / "adapter_model.safetensors")
    trainer._verify_quark_adapter_save(tmp_path)

    del complete["base.layer1.lora_B.default.weight"]
    save_file(complete, tmp_path / "adapter_model.safetensors")
    with pytest.raises(RuntimeError, match="incomplete"):
        trainer._verify_quark_adapter_save(tmp_path)
