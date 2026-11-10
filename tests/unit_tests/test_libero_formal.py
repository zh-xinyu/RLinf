# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Formal arm budgets and inherited mixed-prefix/controller contracts."""

import sys
from types import SimpleNamespace

import pytest

from toolkits.libero_gr00t import launch_formal as formal
from toolkits.libero_gr00t import launch_production as production


@pytest.mark.parametrize("arm", formal.ARMS)
def test_formal_recipe_has_matching_physical_and_evaluation_budgets(
    tmp_path, monkeypatch, arm
):
    monkeypatch.setenv("OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    paths = {
        key: tmp_path / key
        for key in (
            "student",
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        )
    }
    updates, save = formal.ARMS[arm]
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / arm,
        gpus=[0, 1, 2, 3],
        config_name=f"libero_mopd_formal_{arm}",
        updates=updates,
        save_every_updates=save,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        seed=0,
    )
    cfg = production.build_config(args, paths)
    reuse = 1 if arm in {"m0", "m3"} else 2
    adaptive = arm in {"m2", "m3"}
    assert cfg.runner.max_steps == 100 * reuse
    assert cfg.runner.save_interval == 25 * reuse
    assert cfg.runner.logger.project_name == "FORMAL_libero_gr00t"
    assert cfg.runner.logger.experiment_name == "libero_formal"
    assert cfg.env.train.total_num_envs == 32 and cfg.env.train.rollout_epoch == 1
    assert list(cfg.env.train.task_suite_horizons.values()) == [240, 240, 320, 480]
    assert list(cfg.env.train.task_suite_execution_lengths.values()) == [5, 5, 5, 10]
    assert cfg.actor.global_batch_size == 512 and cfg.actor.micro_batch_size == 2
    assert cfg.actor.optim.lr == cfg.actor.optim.value_lr == 3e-5
    assert cfg.actor.optim.lr_scheduler == "constant"
    assert cfg.actor.optim.lr_warmup_steps == 4
    assert cfg.actor.fsdp_config.enable_gradient_accumulation is False
    assert cfg.algorithm.flow_refresh_updates == reuse
    assert cfg.algorithm.sync_suite_loss_weights == adaptive
    allocation = cfg.runner.rollout_allocation
    assert allocation.mode == ("adaptive" if adaptive else "uniform")
    assert (
        allocation.rho,
        allocation.epsilon,
        allocation.warmup_rounds,
        allocation.window_rounds,
        allocation.interval_rounds,
        allocation.min_task_episodes,
        allocation.max_quota_change,
    ) == (0.5, 0.01, 20, 20, 10, 5, 3)
    assert cfg.algorithm.log_observation_pool_metrics
    assert cfg.runner.step_unit == "optimizer_update"
    assert cfg.runner.val_check_interval == -1
    assert all(
        expert.opd_rotary_precision == "bf16"
        for expert in cfg.rollout.expert_models.values()
    )


@pytest.mark.parametrize(
    "option",
    [
        "--learning-rate",
        "--global-batch-size",
        "--trajectories-per-round",
        "--allocation-smoke",
        "--updates",
        "--run-purpose",
    ],
)
def test_formal_entrypoint_rejects_recipe_overrides(option):
    with pytest.raises(ValueError, match="Unsupported formal"):
        formal.formal_arguments(["--arm", "m1", option, "1"])


def test_formal_entrypoint_restores_argv_after_delegation_failure(monkeypatch):
    original = sys.argv
    seen = []

    def fake_main():
        seen.extend(sys.argv)
        raise RuntimeError("owned run failure")

    monkeypatch.setattr(production, "main", fake_main)
    with pytest.raises(RuntimeError):
        formal.main(["--arm", "m2", "--dry-run"])
    assert sys.argv is original
    for key, value in [
        ("--updates", "200"),
        ("--save-every-updates", "50"),
        ("--execution-protocol", "official-mixed"),
        ("--run-purpose", "formal"),
    ]:
        assert seen[seen.index(key) + 1] == value
    with pytest.raises(ValueError, match="Duplicate"):
        formal.formal_arguments(["--arm", "m1", "--arm=m2"])
