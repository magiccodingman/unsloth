# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from core.training.trainer import (
    EXPERIMENTAL_QUARK_ACTIVATION_CHUNK_SIZE,
    EXPERIMENTAL_QUARK_FIRST_CUDA_LAYER_COUNT,
    EXPERIMENTAL_QUARK_FUSED_CE_TARGET_GB,
    EXPERIMENTAL_QUARK_OPTIMIZER,
    UnslothTrainer,
    _build_experimental_quark_device_map,
    _configure_experimental_quark_fused_ce_workspace,
    _is_supported_local_quark_qwen35_checkpoint,
    _resolve_experimental_quark_optimizer,
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


def test_quark_fused_ce_workspace_is_capped_after_zoo_import(monkeypatch):
    fused_ce = SimpleNamespace(TARGET_GB="4")
    monkeypatch.setenv("UNSLOTH_CE_LOSS_TARGET_GB", "3")
    monkeypatch.setattr("importlib.import_module", lambda name: fused_ce)

    effective = _configure_experimental_quark_fused_ce_workspace()

    assert effective == EXPERIMENTAL_QUARK_FUSED_CE_TARGET_GB
    assert os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] == str(effective)
    assert fused_ce.TARGET_GB == str(effective)


def test_quark_fused_ce_workspace_preserves_stricter_user_cap(monkeypatch):
    fused_ce = SimpleNamespace(TARGET_GB="4")
    monkeypatch.setenv("UNSLOTH_CE_LOSS_TARGET_GB", "0.25")
    monkeypatch.setattr("importlib.import_module", lambda name: fused_ce)

    assert _configure_experimental_quark_fused_ce_workspace() == 0.25
    assert fused_ce.TARGET_GB == "0.25"


def test_quark_device_map_keeps_dense_endpoints_apart_and_favors_clean_gpu():
    device_map = _build_experimental_quark_device_map()

    assert EXPERIMENTAL_QUARK_FUSED_CE_TARGET_GB == 0.25
    assert EXPERIMENTAL_QUARK_ACTIVATION_CHUNK_SIZE == 2048
    assert EXPERIMENTAL_QUARK_FIRST_CUDA_LAYER_COUNT == 28
    assert device_map["model.language_model.embed_tokens"] == 0
    assert device_map["lm_head"] == 1
    assert device_map["model.language_model.layers.27"] == 0
    assert device_map["model.language_model.layers.28"] == 1
    assert sum(
        device == 0
        for name, device in device_map.items()
        if "language_model.layers." in name
    ) == 28
    assert sum(
        device == 1
        for name, device in device_map.items()
        if "language_model.layers." in name
    ) == 36


def test_quark_dense_vocab_cpt_pages_adamw8bit_state():
    assert EXPERIMENTAL_QUARK_OPTIMIZER == "paged_adamw_8bit"
    assert (
        _resolve_experimental_quark_optimizer("adamw_8bit", enabled=True)
        == EXPERIMENTAL_QUARK_OPTIMIZER
    )
    assert (
        _resolve_experimental_quark_optimizer("paged_adamw_8bit", enabled=True)
        == EXPERIMENTAL_QUARK_OPTIMIZER
    )
    assert (
        _resolve_experimental_quark_optimizer("adamw_8bit", enabled=False)
        == "adamw_8bit"
    )
    assert (
        _resolve_experimental_quark_optimizer("adamw_torch", enabled=True)
        == "adamw_torch"
    )


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
