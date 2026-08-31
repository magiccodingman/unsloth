# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility fixes for Transformers FlashAttention integration."""

from functools import wraps
import sys


_PATCH_MARKER = "_unsloth_non_2d_position_ids_not_packed"


def _position_ids_ndim(position_ids):
    """Return a tensor-like object's rank without importing torch."""
    ndim = getattr(position_ids, "ndim", None)
    if ndim is not None:
        return ndim

    dim = getattr(position_ids, "dim", None)
    if callable(dim):
        try:
            return dim()
        except Exception:
            pass
    return None


def _guard_non_2d_position_ids(original):
    """Wrap Transformers' packed-sequence probe with the rank check it needs."""
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def _is_packed_sequence(position_ids, *args, **kwargs):
        # Transformers uses 2-D position ids as packed-sequence metadata.
        # Qwen3.5 uses 3-D position ids for its multiple rotary-position axes;
        # treating those as packed metadata constructs invalid cu_seqlens and
        # can send flash_attn_varlen_func out of bounds.
        ndim = _position_ids_ndim(position_ids)
        if ndim is not None and ndim != 2:
            return False
        return original(position_ids, *args, **kwargs)

    setattr(_is_packed_sequence, _PATCH_MARKER, True)
    return _is_packed_sequence


def fix_transformers_flash_attention_packed_sequence_detection():
    """Prevent non-2D position ids from selecting FlashAttention's varlen path.

    Qwen3.5 represents position ids with rank 3 (for example
    ``[3, batch, sequence]``). Transformers 5.x releases can mistake those ids
    for packed-sequence metadata in ``_is_packed_sequence``. The resulting
    ``cu_seqlens`` describe more tokens than exist in Q/K/V, which can make
    FlashAttention's varlen CUDA kernel perform an illegal memory access.

    Keep FlashAttention enabled for normal attention. Only the incorrect
    packed-sequence decision is suppressed for non-2D position ids.

    Returns ``True`` when the relevant Transformers helper exists and is
    patched (or was already patched), otherwise ``False``.
    """
    try:
        import transformers.modeling_flash_attention_utils as fa_utils
    except Exception:
        return False

    original = getattr(fa_utils, "_is_packed_sequence", None)
    if not callable(original):
        return False
    if getattr(original, _PATCH_MARKER, False):
        return True

    patched = _guard_non_2d_position_ids(original)
    fa_utils._is_packed_sequence = patched

    # Some Transformers modules may already have imported the helper by value.
    # Replace only references that still point at the exact original callable;
    # modules imported later will naturally receive the patched module attribute.
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("transformers."):
            continue
        try:
            if getattr(module, "_is_packed_sequence", None) is original:
                setattr(module, "_is_packed_sequence", patched)
        except Exception:
            continue

    return True
