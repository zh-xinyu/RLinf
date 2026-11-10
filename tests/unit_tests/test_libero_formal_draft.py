# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Formal draft guards and all12 composed recipes, without Ray or GPU work."""

import sys
from types import SimpleNamespace

import pytest

from toolkits.libero_gr00t import launch_formal_draft as draft
from toolkits.libero_gr00t import launch_production as production


@pytest.mark.parametrize(
    "option",
    [
        ["--config-name", "other"],
        ["--updates", "2"],
        ["--updates=300"],
        ["--save-every-updates", "1"],
        ["--rollout-steps", "32"],
        ["--global-batch-size", "64"],
        ["--learning-rate", "0.01"],
        ["--allocation-smoke"],
        ["--actor-cache-retention-gib", "1"],
        ["--up", "2"],
    ],
)
def test_recipe_overrides_are_rejected(option):
    with pytest.raises(ValueError):
        draft.draft_arguments(["--arm", "m1", *option])


def test_duplicate_options_and_non_native_profile_are_rejected():
    with pytest.raises(ValueError):
        draft.draft_arguments(["--arm", "m0", "--arm=m1"])
    with pytest.raises(ValueError):
        draft.draft_arguments(["--arm", "m1", "--seed", "0", "--seed=1"])
    with pytest.raises(SystemExit):
        draft.draft_arguments(
            ["--arm", "m1", "--student-padding-profile", "canonical-fixed570"]
        )


def test_real_main_delegation_restores_process_local_registration(monkeypatch):
    original_name, original_argv = production.CONFIG_NAME, sys.argv
    seen = []

    def fake_main():
        seen.append((production.CONFIG_NAME, list(sys.argv)))
        raise RuntimeError("owned production failure")

    monkeypatch.setattr(production, "main", fake_main)
    with pytest.raises(RuntimeError):
        draft.main(["--arm", "m3", "--dry-run"])
    name, argv = seen[0]
    assert name == "libero_mopd_formal_draft_m3"
    assert argv[argv.index("--updates") + 1] == "150"
    assert argv[argv.index("--save-every-updates") + 1] == "25"
    assert argv[argv.index("--rollout-steps") + 1] == "520"
    assert production.CONFIG_NAME == original_name and sys.argv is original_argv


@pytest.mark.parametrize("arm", list(draft.ARMS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_twelve_draft_recipe_invariants(tmp_path, monkeypatch, arm, seed):
    monkeypatch.setenv("OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    updates, save = draft.ARMS[arm]
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
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / f"{arm}_{seed}",
        gpus=[0, 1, 2, 3],
        config_name=f"libero_mopd_formal_draft_{arm}",
        updates=updates,
        save_every_updates=save,
        rollout_steps=520,
        seed=seed,
    )
    cfg = production.build_config(args, paths)
    reuse = 1 if arm in {"m0", "m3"} else 2
    adaptive = arm in {"m2", "m3"}
    assert cfg.algorithm.flow_refresh_updates == reuse
    assert cfg.runner.max_steps == 150 * reuse
    assert cfg.runner.save_interval == 25 * reuse
    assert cfg.runner.val_check_interval == -1
    assert cfg.env.train.total_num_envs == 32 and cfg.env.train.rollout_epoch == 1
    assert cfg.actor.global_batch_size == 256 and cfg.actor.micro_batch_size == 1
    assert cfg.actor.optim.lr == cfg.actor.optim.value_lr == 3e-5
    assert (
        cfg.actor.optim.lr_warmup_steps == 4
        and cfg.actor.optim.lr_scheduler == "constant"
    )
    assert cfg.actor.fsdp_config.enable_gradient_accumulation is False
    assert (
        cfg.actor.cuda_cache_retention_gib == cfg.actor.cuda_cache_preallocate_gib == 0
    )
    assert (
        cfg.actor.seed
        == cfg.rollout.seed
        == cfg.env.train.seed
        == cfg.runner.rollout_allocation.seed
        == seed
    )
    assert list(cfg.env.train.task_suite_horizons.values()) == [240, 320, 320, 520]
    assert cfg.actor.model.opd_fixed_text_padding is False
    assert cfg.actor.model.get("opd_canonical_fixed_padding", False) is False
    assert cfg.runner.rollout_allocation.mode == ("adaptive" if adaptive else "uniform")
    assert cfg.algorithm.sync_suite_loss_weights == adaptive
    allocation = cfg.runner.rollout_allocation
    assert (
        allocation.rho,
        allocation.epsilon,
        allocation.warmup_rounds,
        allocation.window_rounds,
        allocation.interval_rounds,
        allocation.min_task_episodes,
        allocation.max_quota_change,
    ) == (0.5, 0.01, 20, 20, 10, 5, 3)
    assert cfg.runner.logger.project_name == "libero_gr00t_pilot"
    assert list(cfg.runner.logger.logger_backends) == ["tensorboard", "wandb"]
