# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit rollout seeds control flow noise independently of model initialization."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def seed_worker(seed, rank=0):
    worker = SimpleNamespace(
        cfg=OmegaConf.create({"rollout": {"seed": seed}}),
        _rank=rank,
        log_info=lambda message: None,
    )
    MultiStepRolloutWorker._seed_rollout_rng(worker)


def test_noise_reproducible_after_arbitrary_model_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seed_worker(7, 2)
    expected = torch.randn(2, 16, 32)
    numpy_expected = np.random.normal(size=5)
    torch.randn(999)
    seed_worker(7, 2)
    torch.testing.assert_close(torch.randn(2, 16, 32), expected, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.normal(size=5), numpy_expected)
    seed_worker(7, 3)
    assert not torch.equal(torch.randn(2, 16, 32), expected)


def test_omitted_seed_does_not_reset_existing_stream():
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    seed_worker(None)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "7"])
def test_invalid_explicit_seed_is_rejected(bad):
    with pytest.raises(ValueError, match="nonnegative integer"):
        seed_worker(bad)
