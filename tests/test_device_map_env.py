import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_device_map_env.py"
SPEC = importlib.util.spec_from_file_location("_device_map_env_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _clear(monkeypatch):
    for name in (
        MODULE._SAFETY_ENV,
        MODULE._RESERVE_ENV,
        MODULE._POLICY_ENV,
        MODULE._HEAD_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_no_environment_means_no_planner_override(monkeypatch):
    _clear(monkeypatch)
    assert MODULE._environment_overrides() == {}


def test_head_safety_gib_becomes_bytes(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(MODULE._SAFETY_ENV, "1.25")
    assert MODULE._environment_overrides() == {
        "safety_bytes": int(1.25 * 1024**3),
    }


def test_per_device_activation_reserve_uses_logical_indices(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(MODULE._RESERVE_ENV, "0:1.5,1:2.25")
    assert MODULE._environment_overrides() == {
        "activation_reserve_bytes": {
            0: int(1.5 * 1024**3),
            1: int(2.25 * 1024**3),
        }
    }


def test_policy_and_head_device_are_parsed(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(MODULE._POLICY_ENV, "HEAD_MAX")
    monkeypatch.setenv(MODULE._HEAD_ENV, "1")
    assert MODULE._environment_overrides() == {
        "free_space_policy": "head_max",
        "prefer_head_device": 1,
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("safety", "nan"),
        ("reserve", "0:1,garbage"),
        ("policy", "greedy"),
        ("head", "cuda:1"),
    ],
)
def test_invalid_environment_fails_loudly(monkeypatch, name, value):
    _clear(monkeypatch)
    env = {
        "safety": MODULE._SAFETY_ENV,
        "reserve": MODULE._RESERVE_ENV,
        "policy": MODULE._POLICY_ENV,
        "head": MODULE._HEAD_ENV,
    }[name]
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError):
        MODULE._environment_overrides()
