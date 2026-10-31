# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""Bounded actor cache retention is opt-in and never preallocates memory."""

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
