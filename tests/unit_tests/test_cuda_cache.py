# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""Actor cache retention and optional reusable preallocation are opt-in."""

import pytest
import torch

from rlinf.utils.cuda_cache import retain_cuda_cache


def test_retention_skips_small_cache_and_releases_large_cache(monkeypatch):
    calls = []
    reserved = [10 * 1024**3]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: reserved[0])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("released"))
    retain_cuda_cache(18)
    torch.cuda.empty_cache()
    assert calls == []
    reserved[0] = 19 * 1024**3
    torch.cuda.empty_cache()
    assert calls == ["released"]
    with pytest.raises(RuntimeError, match="already configured"):
        retain_cuda_cache(18)


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan")])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        retain_cuda_cache(limit)


@pytest.mark.parametrize("floor", [-1, 19, float("inf"), float("nan")])
def test_invalid_preallocation(floor):
    with pytest.raises(ValueError, match="preallocation"):
        retain_cuda_cache(18, preallocate_gib=floor)


def test_preallocation_is_reusable_and_refilled_after_release(monkeypatch):
    gib = 1024**3
    memory = {"reserved": 0, "allocated": 0}
    released = []

    class Block:
        def __init__(self, size):
            self.size = size
            memory["allocated"] += size
            memory["reserved"] = max(memory.values())

        def __del__(self):
            memory["allocated"] -= self.size

    def empty_cache():
        released.append(True)
        memory["reserved"] = memory["allocated"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: memory["reserved"])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (48 * gib, 48 * gib))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch, "empty", lambda size, **kwargs: Block(size))
    rng = torch.get_rng_state().clone()
    retain_cuda_cache(2, preallocate_gib=1)
    assert memory == {"reserved": gib, "allocated": 0}
    live = Block(gib // 2)
    torch.cuda.empty_cache()
    assert memory == {"reserved": gib, "allocated": gib // 2}
    del live
    memory["reserved"] = 3 * gib
    torch.cuda.empty_cache()
    assert released == [True]
    assert memory == {"reserved": gib, "allocated": 0}
    assert torch.equal(rng, torch.get_rng_state())


def test_preallocation_refuses_insufficient_headroom(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (30 * 1024**3, 48 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(RuntimeError, match="headroom"):
        retain_cuda_cache(28, preallocate_gib=26)
    assert not getattr(torch.cuda.empty_cache, "_rlinf_cache_retention", False)
