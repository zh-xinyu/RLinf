# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exact simulator counts must be summed, never averaged across workers."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.utils.physics_step_counts import (
    physics_session_metrics,
    sum_physics_step_counts,
)


def counts(policy, settling, resets):
    return {
        "policy_simulation_steps": policy,
        "settling_simulation_steps": settling,
        "total_simulation_steps": policy + settling,
        "reset_episodes": resets,
    }


def test_stage_and_rank_sum_preserves_exact_total_not_mean():
    rank0 = sum_physics_step_counts([counts(100, 20, 2), counts(80, 10, 1)])
    rank1 = sum_physics_step_counts([counts(300, 40, 4)])
    combined = sum_physics_step_counts([rank0, rank1])
    assert combined == counts(480, 70, 7)
    metrics = physics_session_metrics(combined)
    assert metrics == {
        "interaction/physics_policy_steps_session": 480,
        "interaction/physics_settling_steps_session": 70,
        "interaction/physics_total_steps_session": 550,
        "interaction/physics_reset_episodes_session": 7,
    }


@pytest.mark.parametrize("rows", [[], [None], [{"policy_simulation_steps": 1}]])
def test_missing_instrumentation_is_not_silently_counted_as_zero(rows):
    with pytest.raises(ValueError):
        sum_physics_step_counts(rows)


def test_inconsistent_total_is_rejected_before_rank_aggregation():
    row = counts(2, 3, 1)
    row["total_simulation_steps"] = 4
    with pytest.raises(ValueError, match="policy plus settling"):
        sum_physics_step_counts([row])


@pytest.mark.parametrize("value", [-1, 1.0, True])
def test_counter_values_are_nonnegative_integers(value):
    row = counts(2, 3, 1)
    row["reset_episodes"] = value
    with pytest.raises(ValueError, match="nonnegative integer"):
        sum_physics_step_counts([row])


def test_runner_flag_controls_rpc_and_valid_mask_alias():
    from rlinf.runners.embodied_runner import EmbodiedRunner

    calls = []
    logs = []

    def remote_counts():
        calls.append(True)
        return SimpleNamespace(wait=lambda: [counts(10, 2, 1), counts(30, 4, 2)])

    runner = SimpleNamespace(
        cfg=OmegaConf.create({"runner": {}}),
        env=SimpleNamespace(get_physics_step_counts=remote_counts),
        optimizer_update_steps=True,
        cumulative_env_steps=25,
        metric_logger=SimpleNamespace(
            log=lambda values, step: logs.append((values, step))
        ),
    )
    EmbodiedRunner._log_physics_step_counts(runner, 2)
    assert calls == logs == []
    runner.cfg.runner.log_physics_step_counts = True
    EmbodiedRunner._log_physics_step_counts(runner, 2)
    assert len(calls) == 1
    assert logs[0][1] == 2
    assert logs[0][0]["interaction/physics_policy_steps_session"] == 40
    assert logs[0][0]["interaction/physics_total_steps_session"] == 46
    assert logs[0][0]["interaction/valid_policy_steps"] == 25
    assert "interaction/env_steps" not in logs[0][0]


def test_worker_reads_training_stages_only():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = SimpleNamespace(
        env_list=[
            SimpleNamespace(physics_step_counts=counts(10, 2, 1)),
            SimpleNamespace(physics_step_counts=counts(30, 4, 2)),
        ],
        eval_env_list=[SimpleNamespace(physics_step_counts=counts(1000, 200, 10))],
    )
    assert EnvWorker.get_physics_step_counts(worker) == counts(40, 6, 3)
