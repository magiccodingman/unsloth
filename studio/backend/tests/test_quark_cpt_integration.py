# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import core.training.trainer as trainer_module
from core.training.trainer import (
    EXPERIMENTAL_QUARK_ACTIVATION_CHUNK_SIZE,
    EXPERIMENTAL_QUARK_FIRST_CUDA_LAYER_COUNT,
    EXPERIMENTAL_QUARK_FUSED_CE_TARGET_GB,
    EXPERIMENTAL_QUARK_OPTIMIZER,
    UnslothTrainer,
    _build_experimental_quark_device_map,
    _configure_experimental_quark_fused_ce_workspace,
    _disable_experimental_quark_double_buffering,
    _evict_paged_optimizer_state_to_cpu,
    _install_experimental_quark_loss_only_evaluation,
    _install_experimental_quark_paged_optimizer_eviction,
    _install_experimental_quark_paged_optimizer_resume,
    _is_supported_local_quark_qwen35_checkpoint,
    _restore_paged_optimizer_state_buffers,
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


def test_quark_disables_zoo_double_buffering_and_clears_cached_choice(monkeypatch):
    calls = []
    fake_disabled = SimpleNamespace(cache_clear=lambda: calls.append("clear"))
    fake_module = SimpleNamespace(_double_buffer_disabled=fake_disabled)
    monkeypatch.setitem(
        trainer_module.sys.modules,
        "unsloth_zoo.gradient_checkpointing",
        fake_module,
    )
    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "0")

    _disable_experimental_quark_double_buffering()

    assert os.environ["UNSLOTH_DISABLE_DOUBLE_BUFFER"] == "1"
    assert calls == ["clear"]


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


def test_quark_periodic_evaluation_never_requests_full_logits(monkeypatch):
    calls = []

    class FakeTrainer:
        eval_dataset = [object()]
        args = SimpleNamespace(prediction_loss_only=False, device=torch.device("cpu"))

        @staticmethod
        def _prepare_inputs(inputs):
            return inputs

        @staticmethod
        def compute_loss_context_manager():
            return nullcontext()

        @staticmethod
        def _get_num_items_in_batch(inputs, device):
            assert device.type == "cpu"
            return inputs[0]["labels"].numel()

        @staticmethod
        def compute_loss(
            model,
            inputs,
            *,
            return_outputs,
            num_items_in_batch,
        ):
            calls.append(
                {
                    "return_logits": os.environ.get("UNSLOTH_RETURN_LOGITS"),
                    "return_outputs": return_outputs,
                    "num_items": num_items_in_batch,
                }
            )
            return torch.tensor(1.25)

    trainer = FakeTrainer()
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")

    assert _install_experimental_quark_loss_only_evaluation(trainer)
    assert trainer.args.prediction_loss_only is True
    loss, logits, labels = trainer.prediction_step(
        object(),
        {"labels": torch.ones(7, dtype=torch.long)},
        prediction_loss_only=False,
    )

    assert loss.item() == 1.25
    assert logits is None
    assert labels is None
    assert calls == [
        {"return_logits": "0", "return_outputs": False, "num_items": 7}
    ]
    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "1"


def test_quark_loss_only_evaluation_requires_an_eval_dataset():
    trainer = SimpleNamespace(eval_dataset=None)
    assert not _install_experimental_quark_loss_only_evaluation(trainer)
    assert not hasattr(trainer, "prediction_step")


def test_quark_resume_recreates_serialized_optimizer_state_as_pages():
    class FakeParameter:
        device = SimpleNamespace(type="cuda")

    class FakePagedBuffer:
        is_paged = True

        def __init__(self):
            self.copied = None

        def copy_(self, value, non_blocking):
            assert non_blocking is False
            self.copied = value.clone()
            return self

    parameter = FakeParameter()
    state1 = torch.arange(100_000, dtype=torch.uint8)
    state2 = torch.arange(100_000, dtype=torch.uint8)
    # torch.save/load preserves these stale attributes even though the loaded
    # tensors use ordinary CPU storage rather than CUDA managed memory.
    state1.is_paged = True
    state2.is_paged = True

    class FakeOptimizer:
        is_paged = True

        def __init__(self):
            self.state = {}
            self.load_move_to_device = None
            self.buffers = []

        def load_state_dict(self, state_dict, move_to_device=True):
            self.load_move_to_device = move_to_device
            self.state = {parameter: dict(state_dict["state"])}

        def get_state_buffer(self, restored_parameter, dtype):
            assert restored_parameter is parameter
            assert dtype == torch.uint8
            buffer = FakePagedBuffer()
            self.buffers.append(buffer)
            return buffer

    optimizer = FakeOptimizer()

    class FakeTrainer:
        def __init__(self):
            self.optimizer = optimizer

        def _load_optimizer_and_scheduler(self, checkpoint):
            assert checkpoint == "checkpoint-390"
            self.optimizer.load_state_dict(
                {"state": {"state1": state1, "state2": state2}}
            )

    class FakeAcceleratedOptimizer:
        def __init__(self, base_optimizer):
            self.optimizer = base_optimizer

        def load_state_dict(self, state_dict):
            return self.optimizer.load_state_dict(state_dict)

    base_optimizer = optimizer
    trainer = FakeTrainer()
    trainer.optimizer = FakeAcceleratedOptimizer(base_optimizer)
    assert _install_experimental_quark_paged_optimizer_resume(trainer)
    trainer._load_optimizer_and_scheduler("checkpoint-390")

    assert optimizer.load_move_to_device is False
    assert len(optimizer.buffers) == 2
    assert optimizer.state[parameter]["state1"] is optimizer.buffers[0]
    assert optimizer.state[parameter]["state2"] is optimizer.buffers[1]
    assert torch.equal(optimizer.buffers[0].copied, state1)
    assert torch.equal(optimizer.buffers[1].copied, state2)


def test_quark_paged_state_restore_skips_cpu_and_small_buffers():
    class FakeParameter:
        def __init__(self, device):
            self.device = device

    cpu_parameter = FakeParameter(torch.device("cpu"))
    cuda_parameter = FakeParameter(SimpleNamespace(type="cuda"))

    class FakeOptimizer:
        state = {
            cpu_parameter: {"state1": torch.zeros(100_000, dtype=torch.uint8)},
            cuda_parameter: {"state1": torch.zeros(99_999, dtype=torch.uint8)},
        }

        @staticmethod
        def get_state_buffer(parameter, dtype):
            raise AssertionError("no buffer should be paged")

    assert _restore_paged_optimizer_state_buffers(FakeOptimizer()) == (0, 0)


def test_quark_paged_optimizer_state_is_evicted_after_each_step(monkeypatch):
    state1 = torch.zeros(100_000, dtype=torch.uint8)
    state2 = torch.zeros(200_000, dtype=torch.uint8)
    state1.is_paged = True
    state1.page_deviceid = 1
    state2.is_paged = True
    state2.page_deviceid = 0
    calls = []

    class FakeOptimizer:
        is_paged = True

        def __init__(self):
            self.state = {object(): {"state1": state1, "state2": state2}}

        @staticmethod
        def step():
            calls.append("step")
            return "updated"

    class FakeTrainer:
        optimizer = None

        def create_optimizer(self):
            self.optimizer = FakeOptimizer()
            return self.optimizer

    def prefetch(value):
        calls.append(("prefetch", value.page_deviceid))

    def synchronize(device_id):
        calls.append(("synchronize", device_id))

    def empty_cache(device_id):
        calls.append(("empty_cache", device_id))

    optimizer = FakeOptimizer()
    assert _evict_paged_optimizer_state_to_cpu(
        optimizer,
        prefetch_fn=prefetch,
        synchronize_fn=synchronize,
        empty_cache_fn=empty_cache,
    ) == (2, 300_000)
    assert calls == [
        ("prefetch", 1),
        ("prefetch", 0),
        ("synchronize", 0),
        ("empty_cache", 0),
        ("synchronize", 1),
        ("empty_cache", 1),
    ]

    calls.clear()
    trainer = FakeTrainer()
    assert _install_experimental_quark_paged_optimizer_eviction(trainer)
    created = trainer.create_optimizer()
    monkeypatch.setattr(
        trainer_module,
        "_evict_paged_optimizer_state_to_cpu",
        lambda candidate: (2, 300_000),
    )
    assert created.step() == "updated"
    assert calls == ["step"]

    monkeypatch.setattr(
        trainer_module,
        "_evict_paged_optimizer_state_to_cpu",
        lambda candidate: (0, 0),
    )
    with pytest.raises(RuntimeError, match="could not evict any paged optimizer state"):
        created.step()


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
