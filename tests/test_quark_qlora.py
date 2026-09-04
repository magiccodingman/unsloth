from __future__ import annotations

import types

import torch

import unsloth.models.quark as quark_module
from unsloth.models.quark import (
    _quark_chunked_base_forward,
    _quark_forward_with_cache_release,
    _quark_lora_forward,
    is_quark_qwen35_mxfp4_config,
)


def test_quark_forward_releases_cache_before_and_after_each_microbatch(monkeypatch):
    calls = []

    class Model:
        @staticmethod
        def _unsloth_quark_original_forward(value):
            calls.append(("forward", value))
            return value + 1

    monkeypatch.setattr(
        quark_module,
        "_release_quark_cuda_cache",
        lambda: calls.append("release"),
    )
    assert _quark_forward_with_cache_release(Model(), 41) == 42
    assert calls == ["release", ("forward", 41), "release"]


def _supported_config():
    tensor = {
        "dtype": "fp4",
        "group_size": 32,
        "scale_format": "e8m0",
    }
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"model_type": "qwen3_5_text"},
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


def test_quark_opt_in_config_match_is_exact():
    config = _supported_config()
    assert is_quark_qwen35_mxfp4_config(config)
    config["quantization_config"]["quant_config"]["global_quant_config"]["weight"]["group_size"] = 64
    assert not is_quark_qwen35_mxfp4_config(config)


class _PackedBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((4, 2), dtype=torch.uint8), requires_grad=False)
        self.seen_dtype = None

    def forward(self, x):
        self.seen_dtype = x.dtype
        return x


class _Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = _PackedBase()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)})
        self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
        self.scaling = {"default": 0.5}
        self.active_adapters = ["default"]
        self.lora_variant = {}
        self.disable_adapters = False
        self.merged = False

    def _check_forward_args(self, *args, **kwargs):
        return None

    def _cast_input_dtype(self, x, dtype):
        return x.to(dtype)


def test_quark_lora_forward_never_casts_activation_to_packed_uint8():
    wrapper = _Wrapper()
    wrapper.forward = types.MethodType(_quark_lora_forward, wrapper)
    inputs = torch.randn((1, 3, 4), dtype=torch.bfloat16, requires_grad=True)
    output = wrapper(inputs)
    output.float().square().mean().backward()
    assert wrapper.base_layer.seen_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    assert wrapper.lora_B["default"].weight.grad is not None
    assert torch.isfinite(wrapper.lora_B["default"].weight.grad).all()


class _ChunkableBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(20, dtype=torch.bfloat16).reshape(5, 4) / 20,
            requires_grad=False,
        )
        self.bias = None
        self.weight_calls = 0

    def _get_qinput(self, x):
        # Row-local stand-in for dynamic group quantization.
        return torch.round(x * 2) / 2

    def _get_qweight(self, weight):
        self.weight_calls += 1
        return weight

    def _get_qbias(self, bias):
        return bias

    def _get_qoutput(self, output):
        return output

    def forward(self, x):
        return torch.nn.functional.linear(
            self._get_qinput(x),
            self._get_qweight(self.weight),
        )


def test_quark_activation_row_chunking_is_exact_and_differentiable():
    base = _ChunkableBase()
    inputs = torch.randn((1, 7, 4), dtype=torch.bfloat16, requires_grad=True)
    expected = base(inputs)
    base.weight_calls = 0
    actual = _quark_chunked_base_forward(base, inputs, 2)
    assert torch.equal(actual, expected)
    assert base.weight_calls == 1
    actual.float().square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
