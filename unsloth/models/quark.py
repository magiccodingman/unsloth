# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Narrow, opt-in support for imported AMD Quark W4A4 MXFP4 QLoRA bases.

This module deliberately does not mark Quark quantization generally trainable.
It recognizes and validates the representation exercised by the Qwen3.5 CPT
experiments, and refuses generic dense merging of adapters into packed FP4.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import types

import torch
import torch.nn.functional as F


def _quark_chunked_base_forward(base_layer, x, chunk_size, *args, **kwargs):
    """Run exact row-wise W4A4 QDQ in bounded chunks with one weight dequantization.

    Quark's dynamic input quantization groups values only along the last
    dimension, so splitting flattened leading rows does not change quantization
    groups or numerical results. It does bound the otherwise sequence-linear
    activation-QDQ temporary used by the CUDA hardware-emulation kernel.
    """
    if chunk_size is None or x.ndim < 2 or x.numel() == 0 or args or kwargs:
        return base_layer(x, *args, **kwargs)
    rows = x.numel() // x.shape[-1]
    if rows <= chunk_size:
        return base_layer(x, *args, **kwargs)

    dtype = x.dtype
    flat = x.reshape(rows, x.shape[-1])
    qweight = base_layer._get_qweight(base_layer.weight).to(dtype)
    qbias = base_layer._get_qbias(base_layer.bias)
    if qweight.device != flat.device:
        qweight = qweight.to(flat.device)
    if qbias is not None:
        qbias = qbias.to(device=flat.device, dtype=dtype)

    outputs = []
    for input_chunk in flat.split(chunk_size, dim=0):
        qinput = base_layer._get_qinput(input_chunk).to(dtype)
        qoutput = F.linear(qinput, qweight, bias=qbias)
        outputs.append(base_layer._get_qoutput(qoutput).to(dtype))
    output = torch.cat(outputs, dim=0)
    return output.reshape(*x.shape[:-1], output.shape[-1])


def _release_quark_cuda_cache() -> None:
    """Finish queued multi-GPU work and release inactive cache on every device."""
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()


def _quark_forward_with_cache_release(self, *args, **kwargs):
    """Bound inactive cache and async work around each accumulated forward."""
    # The previous microbatch's backward is asynchronous.  Finish it and return
    # its inactive allocator segments before the next forward starts, rather
    # than allowing eight accumulation cycles to pressure the display GPU.
    _release_quark_cuda_cache()
    outputs = self._unsloth_quark_original_forward(*args, **kwargs)
    _release_quark_cuda_cache()
    return outputs


def _quark_lora_forward(self, x, *args, **kwargs):
    """PEFT Linear.forward without treating a packed uint8 weight as compute dtype.

    Unsloth Zoo's generated PEFT forward normally casts activations to a dense
    base layer's weight dtype when autocast is off. QParamsLinear is an
    nn.Linear subclass, but its uint8 weight is packed storage, not its compute
    dtype. Casting x to uint8 makes Quark cast its dequantized weight back to
    uint8 and eventually sends byte tensors to F.linear.
    """
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    try:
        from peft.tuners.lora.layer import VARIANT_KWARG_KEYS
    except ImportError:
        VARIANT_KWARG_KEYS = ("alora_offsets",)
    variant_kwargs = {key: kwargs.pop(key, None) for key in VARIANT_KWARG_KEYS}

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if adapter_names is not None:
        return self._mixed_batch_forward(
            x,
            *args,
            adapter_names = adapter_names,
            **variant_kwargs,
            **kwargs,
        )
    if self.merged:
        return self.base_layer(x, *args, **kwargs)

    result = _quark_chunked_base_forward(
        self.base_layer,
        x,
        getattr(self, "_unsloth_quark_activation_chunk_size", None),
        *args,
        **kwargs,
    )
    result_dtype = result.dtype
    lora_A_keys = self.lora_A.keys()
    for active_adapter in self.active_adapters:
        if active_adapter not in lora_A_keys:
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        lora_input = self._cast_input_dtype(x, lora_A.weight.dtype)
        if active_adapter not in self.lora_variant:
            result = result + lora_B(lora_A(dropout(lora_input))) * scaling
        else:
            result = self.lora_variant[active_adapter].forward(
                self,
                active_adapter = active_adapter,
                x = lora_input,
                result = result,
                **variant_kwargs,
                **kwargs,
            )
    return result.to(result_dtype)


def _config_value(config, key, default = None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _enum_text(value) -> str:
    enum_value = getattr(value, "value", value)
    if isinstance(enum_value, str):
        return enum_value.lower()
    return str(getattr(value, "name", value)).lower()


def is_quark_qwen35_mxfp4_config(config) -> bool:
    quantization_config = _config_value(config, "quantization_config")
    quant_method = _config_value(quantization_config, "quant_method", "")
    quant_method = _enum_text(quant_method)
    if quant_method != "quark" or _config_value(config, "model_type") != "qwen3_5":
        return False

    text_config = _config_value(config, "text_config")
    native_quant_config = _config_value(quantization_config, "quant_config", quantization_config)
    global_config = _config_value(native_quant_config, "global_quant_config", {})
    weight = _config_value(global_config, "weight", {})
    inputs = _config_value(global_config, "input_tensors", {})
    export = _config_value(
        quantization_config,
        "json_export_config",
        _config_value(quantization_config, "export", {}),
    )
    architectures = _config_value(config, "architectures", []) or []
    return all((
        _config_value(text_config, "model_type") == "qwen3_5_text",
        "Qwen3_5ForConditionalGeneration" in architectures,
        _enum_text(_config_value(native_quant_config, "quant_mode")) == "eager_mode",
        _enum_text(_config_value(weight, "dtype")) == "fp4",
        _config_value(weight, "group_size") == 32,
        _config_value(weight, "scale_format") == "e8m0",
        _config_value(weight, "is_dynamic") is False,
        _enum_text(_config_value(inputs, "dtype")) == "fp4",
        _config_value(inputs, "group_size") == 32,
        _config_value(inputs, "scale_format") == "e8m0",
        _config_value(inputs, "is_dynamic") is True,
        _config_value(export, "weight_format") == "real_quantized",
    ))


def ensure_quark_transformers_compatibility() -> None:
    """Install the Quark 0.12 QConfig alias expected by Transformers 5.5."""
    try:
        import quark.torch.quantization.config.config as quark_config_module
    except ImportError as error:
        raise ImportError(
            "Unsloth: experimental Quark QLoRA requires AMD Quark matching the "
            "installed Torch/CUDA build."
        ) from error
    if not hasattr(quark_config_module, "Config") and hasattr(quark_config_module, "QConfig"):
        quark_config_module.Config = quark_config_module.QConfig


@contextmanager
def quark_qwen35_load_context(config, enabled = False):
    """Work around Transformers' nested qwen3_5_text sidecar-key rewrite.

    The context is gated by an explicit caller opt-in and an exact configuration
    check. It changes the process-global conversion table only around one
    synchronous ``from_pretrained`` call and restores it even on failure.
    """
    if not enabled:
        yield False
        return
    if not is_quark_qwen35_mxfp4_config(config):
        raise ValueError(
            "Unsloth: experimental_quark_qlora=True currently supports only an "
            "imported Qwen3.5 Quark eager W4A4 FP4/group-32/E8M0 checkpoint with "
            "real-quantized packed weights."
        )

    ensure_quark_transformers_compatibility()

    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    original = get_checkpoint_conversion_mapping("qwen3_5_text")
    register_checkpoint_conversion_mapping("qwen3_5_text", [], overwrite = True)
    try:
        yield True
    finally:
        register_checkpoint_conversion_mapping("qwen3_5_text", original, overwrite = True)


def validate_loaded_quark_qwen35_mxfp4(model) -> int:
    """Validate packed-resident decoder modules and dense endpoint dtypes."""
    if not is_quark_qwen35_mxfp4_config(model.config):
        raise RuntimeError("Unsloth: loaded model no longer matches the opted-in Quark format.")
    try:
        from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
    except ImportError as error:
        raise RuntimeError("Unsloth: AMD Quark disappeared after model loading.") from error

    packed = []
    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, QParamsLinear):
            continue
        packed.append((name, module))
        scale = getattr(getattr(module, "weight_quantizer", None), "scale", None)
        expected_scale_width = (module.in_features + 31) // 32
        if module.weight.dtype != torch.uint8:
            failures.append(f"{name}.weight dtype={module.weight.dtype}")
        if tuple(module.weight.shape) != (module.out_features, (module.in_features + 1) // 2):
            failures.append(f"{name}.weight shape={tuple(module.weight.shape)}")
        if not isinstance(scale, torch.Tensor) or scale.dtype != torch.uint8:
            failures.append(f"{name}.weight_quantizer.scale dtype={getattr(scale, 'dtype', None)}")
        elif tuple(scale.shape) != (module.out_features, expected_scale_width):
            failures.append(f"{name}.weight_quantizer.scale shape={tuple(scale.shape)}")

    if not packed:
        failures.append("no QParamsLinear modules")
    for endpoint_name, endpoint in (
        ("embed_tokens", model.get_input_embeddings()),
        ("lm_head", model.get_output_embeddings()),
    ):
        if endpoint is None or endpoint.weight.dtype not in (torch.float16, torch.bfloat16):
            failures.append(
                f"{endpoint_name} dtype={getattr(getattr(endpoint, 'weight', None), 'dtype', None)}"
            )
    if failures:
        raise RuntimeError(
            "Unsloth: refusing experimental Quark QLoRA because packed-state "
            "validation failed: " + "; ".join(failures[:20])
        )

    model._unsloth_quark_qlora_validated = True
    model._unsloth_quark_packed_module_count = len(packed)
    return len(packed)


def offload_quark_frozen_vision_tower(model) -> int:
    """Keep the unused vision tower on CPU for the text-only CPT path."""
    if not getattr(model, "_unsloth_quark_qlora_validated", False):
        return 0
    outer_model = getattr(model, "model", None)
    vision = getattr(outer_model, "visual", None)
    if vision is None:
        return 0
    named_tensors = tuple(vision.named_parameters()) + tuple(vision.named_buffers())
    tensors = tuple(tensor for _, tensor in named_tensors)
    cuda_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if tensor.device.type == "cuda"
    )
    vision.requires_grad_(False)
    vision.to("cpu")
    remaining_cuda = [
        name
        for name, tensor in (*vision.named_parameters(), *vision.named_buffers())
        if tensor.device.type == "cuda"
    ]
    if remaining_cuda:
        raise RuntimeError(
            "Unsloth: failed to move the excluded Quark vision tower to CPU; "
            f"CUDA tensors remain: {remaining_cuda[:20]}."
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model._unsloth_quark_vision_cpu_bytes = cuda_bytes
    return cuda_bytes


@contextmanager
def protect_quark_config_from_dtype_rewrite(model):
    """Hide native Quark config objects from Zoo's recursive dtype normalizer.

    That normalizer treats every attribute named ``dtype`` as a torch compute
    dtype. Quark's QTensorConfig.dtype is instead its FP4 format enum; replacing
    it corrupts QConfig and makes its next ``to_dict`` call fail.
    """
    if not getattr(model, "_unsloth_quark_qlora_validated", False):
        yield
        return
    saved = []
    seen = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        quantization_config = getattr(config, "quantization_config", None)
        if _enum_text(_config_value(quantization_config, "quant_method", "")) != "quark":
            continue
        saved.append((config, quantization_config))
        config.quantization_config = {"quant_method": "quark"}
    try:
        yield
    finally:
        for config, quantization_config in saved:
            config.quantization_config = quantization_config


def finalize_quark_qlora_peft_model(
    model,
    activation_chunk_size = None,
    release_cache_after_forward = False,
    offload_redundant_dense_originals = False,
):
    """Keep adapters BF16 and narrowly bypass HF's policy-only Quark guard."""
    if not getattr(model, "_unsloth_quark_qlora_validated", False):
        return model

    # PEFT and Unsloth have now performed all wrapping/preparation. Re-run the
    # packed-state validation here so a persistent dequantization or scale cast
    # cannot hide between load validation and the first training step.
    packed_count = validate_loaded_quark_qwen35_mxfp4(model)

    endpoint_dtype = model.get_input_embeddings().weight.dtype
    if endpoint_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"Unsloth: Quark endpoint dtype changed unexpectedly to {endpoint_dtype}.")

    try:
        from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
    except ImportError as error:
        raise RuntimeError("Unsloth: AMD Quark disappeared after PEFT injection.") from error

    if activation_chunk_size is not None:
        if type(activation_chunk_size) is not int or activation_chunk_size <= 0:
            raise ValueError(
                "Unsloth: experimental_quark_activation_chunk_size must be a positive integer or None."
            )

    lora_tensors = 0
    quark_wrappers = 0
    for module in model.modules():
        for collection_name in ("lora_A", "lora_B"):
            collection = getattr(module, collection_name, None)
            if collection is None:
                continue
            for adapter in collection.values():
                adapter.to(dtype = endpoint_dtype)
                lora_tensors += sum(1 for _ in adapter.parameters())
        if isinstance(getattr(module, "base_layer", None), QParamsLinear):
            module.forward = types.MethodType(_quark_lora_forward, module)
            module._unsloth_quark_safe_lora_forward = True
            module._unsloth_quark_activation_chunk_size = activation_chunk_size
            quark_wrappers += 1
    if lora_tensors == 0:
        raise RuntimeError("Unsloth: Quark QLoRA validation found no LoRA tensors after PEFT injection.")
    if quark_wrappers == 0:
        raise RuntimeError("Unsloth: Quark QLoRA validation found no wrapped QParamsLinear modules.")

    dense_saved_tensors = 0
    for name, parameter in model.named_parameters():
        if ".modules_to_save." not in name or not parameter.requires_grad:
            continue
        dense_saved_tensors += 1
        if parameter.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "Unsloth: refusing experimental Quark QLoRA because a dense "
                f"modules_to_save parameter was upcast: {name} dtype={parameter.dtype}."
            )

    offloaded_original_bytes = 0
    offloaded_original_modules = 0
    if offload_redundant_dense_originals:
        try:
            from peft.utils.other import ModulesToSaveWrapper
        except ImportError as error:
            raise RuntimeError("Unsloth: PEFT disappeared after adapter injection.") from error
        for name, module in model.named_modules():
            if not isinstance(module, ModulesToSaveWrapper):
                continue
            if not (name.endswith("embed_tokens") or name.endswith("lm_head")):
                continue
            original = module.original_module
            original_parameters = tuple(original.parameters())
            if any(parameter.requires_grad for parameter in original_parameters):
                raise RuntimeError(
                    f"Unsloth: refusing to offload trainable PEFT original module {name}."
                )
            offloaded_original_bytes += sum(
                parameter.numel() * parameter.element_size()
                for parameter in original_parameters
                if parameter.device.type == "cuda"
            )
            original.to("cpu")
            if any(parameter.device.type != "cpu" for parameter in original_parameters):
                raise RuntimeError(f"Unsloth: failed to offload redundant PEFT original {name}.")
            offloaded_original_modules += 1
        if offloaded_original_modules != 2:
            raise RuntimeError(
                "Unsloth: expected exactly the embed_tokens and lm_head PEFT originals "
                f"to offload, found {offloaded_original_modules}."
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Transformers' guard uses this per-model marker before consulting the Quark
    # class-wide is_trainable=False property. Do not globally bless other Quark models.
    model._hf_peft_config_loaded = True
    model._unsloth_quark_lora_dtype = endpoint_dtype
    model._unsloth_quark_lora_wrapper_count = quark_wrappers
    model._unsloth_quark_post_peft_packed_module_count = packed_count
    model._unsloth_quark_dense_saved_tensor_count = dense_saved_tensors
    model._unsloth_quark_cpu_original_module_count = offloaded_original_modules
    model._unsloth_quark_cpu_original_bytes = offloaded_original_bytes
    model._unsloth_quark_activation_chunk_size = activation_chunk_size
    model._unsloth_quark_release_cache_after_forward = bool(release_cache_after_forward)

    if release_cache_after_forward:
        model._unsloth_quark_original_forward = model.forward
        model.forward = types.MethodType(_quark_forward_with_cache_release, model)

    def _reject_packed_merge(self, *args, **kwargs):
        raise RuntimeError(
            "Unsloth: generic PEFT merge_and_unload() cannot merge a dense LoRA "
            "delta into packed Quark FP4 [out, in/2] weights. Save the adapter "
            "separately; a Quark-aware dequantize/add/requantize/export path is required."
        )

    model.merge_and_unload = types.MethodType(_reject_packed_merge, model)
    return model
