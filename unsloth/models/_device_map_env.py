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

"""Environment overrides for the head-aware multi-GPU device-map planner.

Studio deliberately asks for ``device_map="unsloth_balanced"`` on multi-GPU CUDA,
but its training loader does not currently expose ``device_map_planner_kwargs``.
That makes planner tuning inaccessible to a Studio launch even when measured
training activations need more room than the planner's generic heuristic keeps.

These opt-in environment variables bridge that gap without changing default
placement or overriding an explicit Python API request:

``UNSLOTH_DEVICE_MAP_ACTIVATION_RESERVE_GIB``
    Either one GiB value for every device (``"2"``), or a comma-separated
    per-logical-device mapping (``"0:1.5,1:2.5"``).

``UNSLOTH_DEVICE_MAP_FREE_SPACE_POLICY``
    ``"balanced"`` or ``"head_max"``.

``UNSLOTH_DEVICE_MAP_PREFER_HEAD_DEVICE``
    Logical CUDA device index to prefer for the output head.

The variables affect only Unsloth's planned maps (``unsloth`` and
``unsloth_balanced``). Explicit ``device_map_planner_kwargs`` passed by Python
callers always win over the environment.
"""

from __future__ import annotations

import functools
import os
from typing import Any

_GIB = 1024 ** 3
_RESERVE_ENV = "UNSLOTH_DEVICE_MAP_ACTIVATION_RESERVE_GIB"
_POLICY_ENV = "UNSLOTH_DEVICE_MAP_FREE_SPACE_POLICY"
_HEAD_ENV = "UNSLOTH_DEVICE_MAP_PREFER_HEAD_DEVICE"
_PLANNED_MAPS = frozenset(("unsloth", "unsloth_balanced"))


def _gib_to_bytes(value: str) -> int:
    try:
        gib = float(value.strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a GiB number, got {value!r}") from error
    if not (gib >= 0.0):  # also rejects NaN
        raise ValueError(f"GiB value must be >= 0, got {value!r}")
    return int(gib * _GIB)


def _parse_activation_reserve(raw: str) -> int | dict[int, int]:
    raw = raw.strip()
    if not raw:
        raise ValueError(f"{_RESERVE_ENV} must not be empty")
    if ":" not in raw:
        return _gib_to_bytes(raw)

    reserve: dict[int, int] = {}
    for item in raw.split(","):
        if ":" not in item:
            raise ValueError(
                f"{_RESERVE_ENV} must be one GiB value or device:GiB pairs; got {raw!r}"
            )
        device_text, gib_text = item.split(":", 1)
        try:
            device = int(device_text.strip())
        except ValueError as error:
            raise ValueError(
                f"{_RESERVE_ENV} has an invalid device index {device_text!r}"
            ) from error
        if device < 0:
            raise ValueError(f"{_RESERVE_ENV} device indices must be >= 0")
        if device in reserve:
            raise ValueError(f"{_RESERVE_ENV} repeats logical device {device}")
        reserve[device] = _gib_to_bytes(gib_text)
    return reserve


def _environment_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    reserve = os.environ.get(_RESERVE_ENV)
    if reserve is not None:
        overrides["activation_reserve_bytes"] = _parse_activation_reserve(reserve)

    policy = os.environ.get(_POLICY_ENV)
    if policy is not None:
        policy = policy.strip().lower()
        if policy not in ("balanced", "head_max"):
            raise ValueError(f"{_POLICY_ENV} must be 'balanced' or 'head_max', got {policy!r}")
        overrides["free_space_policy"] = policy

    head = os.environ.get(_HEAD_ENV)
    if head is not None:
        try:
            head_device = int(head.strip())
        except ValueError as error:
            raise ValueError(f"{_HEAD_ENV} must be a logical CUDA device index") from error
        if head_device < 0:
            raise ValueError(f"{_HEAD_ENV} must be >= 0")
        overrides["prefer_head_device"] = head_device

    return overrides


def install_device_map_environment_overrides() -> None:
    """Install the opt-in planner override before model loaders import the resolver."""
    from . import loader_utils

    original = loader_utils.resolve_unsloth_device_map
    if getattr(original, "_unsloth_device_map_env_wrapper", False):
        return

    @functools.wraps(original)
    def resolve_with_environment(
        device_map,
        model_name,
        *,
        planner_kwargs=None,
        **kwargs,
    ):
        if isinstance(device_map, str) and device_map in _PLANNED_MAPS:
            environment = _environment_overrides()
            if environment:
                merged = dict(planner_kwargs or {})
                applied = {}
                for key, value in environment.items():
                    if key not in merged:
                        merged[key] = value
                        applied[key] = value
                planner_kwargs = merged
                if applied:
                    printable = dict(applied)
                    reserve = printable.get("activation_reserve_bytes")
                    if isinstance(reserve, int):
                        printable["activation_reserve_bytes"] = f"{reserve / _GIB:.3f} GiB"
                    elif isinstance(reserve, dict):
                        printable["activation_reserve_bytes"] = {
                            device: f"{amount / _GIB:.3f} GiB"
                            for device, amount in reserve.items()
                        }
                    print(f"Unsloth: device-map environment overrides: {printable}")

        return original(
            device_map,
            model_name,
            planner_kwargs=planner_kwargs,
            **kwargs,
        )

    resolve_with_environment._unsloth_device_map_env_wrapper = True
    loader_utils.resolve_unsloth_device_map = resolve_with_environment
