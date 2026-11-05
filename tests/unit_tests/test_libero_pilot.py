# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Gradient accumulation and held-out split contracts of the pilot."""

import pytest
import torch

from rlinf.algorithms.losses import compute_analytic_local_kl_loss
from toolkits.libero_gr00t.pilot import validate_recipe


def test_microbatch_and_equal_suite_means_match_global_m3_gradient():
    torch.manual_seed(4)
    x = torch.randn(4, 3, 2, 7)
    target = torch.randn_like(x)
    weights = torch.rand(4, 3) + 0.2
    model = torch.nn.Linear(7, 7, bias=False)
    loss, _ = compute_analytic_local_kl_loss(
        model(x.flatten(0, 1)), target.flatten(0, 1), weights.flatten()
    )
    loss.backward()
    expected = model.weight.grad.clone()
    model.zero_grad()
    gradients = []
    for suite in range(4):
        model.zero_grad()
        for sample in range(3):
            loss, _ = compute_analytic_local_kl_loss(
                model(x[suite, sample][None]),
                target[suite, sample][None],
                weights[suite, sample][None],
            )
            (loss / 3).backward()
        gradients.append(model.weight.grad.clone())
    torch.testing.assert_close(torch.stack(gradients).mean(0), expected)


def test_recipe_rejects_overlapping_initial_states_and_biased_batch():
    cfg = {
        "world_size": 4,
        "global_batch_size": 64,
        "micro_batch_size": 1,
        "train_initial_states": [0, 2],
        "eval_initial_states": [1],
        "rounds": 20,
        "reuse": 2,
        "reuse_mode": "refresh",
        "eval_interval_rounds": 10,
    }
    validate_recipe(cfg)
    with pytest.raises(ValueError, match="disjoint"):
        validate_recipe(dict(cfg, eval_initial_states=[2]))
    with pytest.raises(ValueError, match="divisible"):
        validate_recipe(dict(cfg, global_batch_size=63))
    with pytest.raises(ValueError, match="microbatch"):
        validate_recipe(dict(cfg, micro_batch_size=2))
