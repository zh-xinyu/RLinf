# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded waiting for GPU cleanup between serial experiment stages."""

import pytest

from toolkits.robocasa_mopd import owned_ray


def test_gpu_wait_handles_cleanup_without_changing_threshold(monkeypatch):
    elapsed = [0.0]
    calls = []
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "GPU_FREE_MB": "0"}

    def check(count, actual_env):
        calls.append((count, actual_env.copy()))
        if len(calls) < 3:
            raise RuntimeError("Selected GPUs are busy; refusing to start")
        return "available"

    monkeypatch.setattr(owned_ray, "check_gpus", check)
    monkeypatch.setattr(owned_ray.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        owned_ray.time,
        "sleep",
        lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    )
    assert (
        owned_ray.wait_for_gpus(4, env, timeout_seconds=10, poll_seconds=2)
        == "available"
    )
    assert calls == [(4, env)] * 3
    assert elapsed[0] == 4


def test_gpu_wait_has_finite_deadline(monkeypatch):
    elapsed = [0.0]

    def busy(*args):
        raise RuntimeError("busy")

    monkeypatch.setattr(owned_ray, "check_gpus", busy)
    monkeypatch.setattr(owned_ray.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        owned_ray.time,
        "sleep",
        lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    )
    with pytest.raises(RuntimeError, match="remain busy"):
        owned_ray.wait_for_gpus(4, {}, timeout_seconds=3, poll_seconds=2)
    assert elapsed[0] == 3


def test_gpu_wait_does_not_retry_configuration_errors(monkeypatch):
    def invalid(*args):
        raise ValueError("wrong GPU count")

    monkeypatch.setattr(owned_ray, "check_gpus", invalid)
    monkeypatch.setattr(
        owned_ray.time, "sleep", lambda seconds: pytest.fail("must not wait")
    )
    with pytest.raises(ValueError, match="wrong GPU count"):
        owned_ray.wait_for_gpus(4, {})


def test_gpu_wait_returns_immediately_for_idle_gpus(monkeypatch):
    monkeypatch.setattr(owned_ray, "check_gpus", lambda *args: "available")
    monkeypatch.setattr(
        owned_ray.time, "sleep", lambda seconds: pytest.fail("must not wait")
    )
    assert owned_ray.wait_for_gpus(4, {}) == "available"
