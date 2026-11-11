# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Task query allocation changes objective shares without extra loss weights."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.suite_distillation import (
    finalize_suite_statistics,
    suite_query_quotas,
)
from rlinf.config import validate_suite_distillation_cfg
from rlinf.utils.rollout_allocation import RolloutAllocator
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

NAMES = ["a", "b", "c", "d"]
WEIGHTS = dict(zip(NAMES, [0.5, 0.5, 2.5, 0.5]))


def test_query_budget_and_default_remainder():
    assert suite_query_quotas(None, NAMES, 66) == [17, 17, 16, 16]
    assert suite_query_quotas(WEIGHTS, NAMES, 64) == [8, 8, 40, 8]
    quotas = suite_query_quotas(WEIGHTS, NAMES, 65)
    assert sum(quotas) == 65 and min(quotas) > 0
    with pytest.raises(ValueError, match="at least one"):
        suite_query_quotas(dict(zip(NAMES, [0.001, 1, 1, 1])), NAMES, 64)
    with pytest.raises(ValueError, match="suite_query_weights"):
        suite_query_quotas({"a": 1}, NAMES, 64)


@pytest.mark.parametrize("weighted", [False, True])
def test_actual_sampler_obeys_quotas_and_short_pool_replacement(weighted):
    ids = torch.repeat_interleave(torch.arange(4), 20)
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {
                    "suite_balanced_sampling": True,
                    "suite_query_weights": WEIGHTS if weighted else None,
                },
                "actor": {"global_batch_size": 256, "seed": 0},
            }
        ),
        rollout_batch={"forward_inputs": {"teacher_ids": ids.reshape(1, -1)}},
        _teacher_names=lambda: NAMES,
        _world_size=4,
        _rank=0,
        log_info=lambda _: None,
    )
    selected = EmbodiedFSDPActor._suite_balanced_sample(worker, 80, 1, 80)
    expected = [8, 8, 40, 8] if weighted else [16] * 4
    assert torch.bincount(ids[selected], minlength=4).tolist() == expected
    assert len(selected) == 64
    if weighted:
        assert len(selected[ids[selected] == 2].unique()) == 20


def test_macro_loss_stays_task_equal_for_unequal_counts():
    stats = torch.tensor([[8, 16, 8], [16, 32, 8], [400, 800, 40], [32, 64, 8]])
    metrics = finalize_suite_statistics(stats, NAMES, weighted=True)
    assert metrics["actor/macro_velocity_mse"] == 4.25
    assert metrics["actor/macro_analytic_kl"] == 8.5


def test_query_weights_require_unweighted_stratified_updates():
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "suite_query_weights": WEIGHTS,
                "log_suite_velocity_metrics": True,
                "suite_balanced_sampling": True,
                "reward_type": "chunk_level",
                "loss_type": "analytic_local_kl",
            },
            "actor": {"global_batch_size": 256, "training_backend": "fsdp"},
            "rollout": {"expert_models": dict.fromkeys(NAMES, {})},
        }
    )
    validate_suite_distillation_cfg(cfg)
    cfg.algorithm.suite_loss_weights = WEIGHTS
    with pytest.raises(ValueError, match="exclusive"):
        validate_suite_distillation_cfg(cfg)
    del cfg.algorithm.suite_loss_weights
    cfg.algorithm.suite_balanced_sampling = False
    with pytest.raises(ValueError, match="balanced sampling"):
        validate_suite_distillation_cfg(cfg)


def test_static_allocation_can_start_on_first_round():
    tasks = [
        {"task_id": i, "task": name, "suite": name, "horizon": 20}
        for i, name in enumerate(NAMES)
    ]
    config = {
        "mode": "static",
        "warmup_rounds": 0,
        "window_rounds": 5,
        "static_quotas": dict(zip(NAMES, [4, 4, 20, 4])),
    }
    allocator = RolloutAllocator(tasks, 4, 8, 1, config)
    plan = allocator.plan()
    assert allocator.quotas == [4, 4, 20, 4]
    for worker in range(4):
        assert {row["suite"] for row in plan if row["worker"] == worker} == set(NAMES)
    with pytest.raises(ValueError, match="nonnegative"):
        RolloutAllocator(tasks, 4, 8, 1, dict(config, warmup_rounds=-1))
