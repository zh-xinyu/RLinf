# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""T64 keeps the accepted numerical recipe and proportional quota adjustment."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from toolkits.libero_gr00t import launch_formal as formal
from toolkits.libero_gr00t import launch_production as production


@pytest.mark.parametrize("arm", formal.ARMS)
def test_t64_only_changes_budget_quota_cap_and_memory_policy(
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
        gpus=[4, 5, 6, 7],
        config_name=f"libero_mopd_formal_{arm}",
        updates=updates,
        save_every_updates=save,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        seed=0,
    )
    base = production.build_config(args, paths)
    args.config_name = f"libero_mopd_formal_t64_{arm}"
    cfg = production.build_config(args, paths)
    assert cfg.env.train.total_num_envs == 64
    assert cfg.runner.rollout_allocation.max_quota_change == 6
    assert cfg.actor.cuda_cache_preallocate_gib == 26
    assert cfg.actor.cuda_cache_retention_gib == 28
    assert cfg.runner.max_steps // cfg.algorithm.flow_refresh_updates == 100
    before, after = [OmegaConf.to_container(c, resolve=True) for c in (base, cfg)]
    before["env"]["train"]["total_num_envs"] = 64
    before["runner"]["rollout_allocation"]["max_quota_change"] = 6
    before["runner"]["logger"]["wandb_group"] = after["runner"]["logger"]["wandb_group"]
    before["actor"]["cuda_cache_preallocate_gib"] = 26
    before["actor"]["cuda_cache_retention_gib"] = 28
    assert before == after


def test_t64_cli_selects_dedicated_config_without_tuning_other_parameters():
    argv = formal.formal_arguments(
        ["--arm", "m2", "--trajectory-budget", "64", "--dry-run"]
    )
    assert argv[argv.index("--config-name") + 1] == "libero_mopd_formal_t64_m2"
    assert argv[argv.index("--updates") + 1] == "200"
    assert argv[argv.index("--save-every-updates") + 1] == "50"
    with pytest.raises(SystemExit):
        formal.formal_arguments(["--arm", "m1", "--trajectory-budget", "48"])


def test_formal_allows_resource_cache_controls_without_changing_recipe():
    argv = formal.formal_arguments(
        [
            "--arm",
            "m1",
            "--trajectory-budget",
            "64",
            "--actor-cache-retention-gib",
            "20",
            "--actor-cache-preallocate-gib",
            "16",
            "--skip-finished-rollout",
        ]
    )
    assert argv[argv.index("--updates") + 1] == "200"
    assert argv[argv.index("--actor-cache-retention-gib") + 1] == "20"
    assert argv[argv.index("--actor-cache-preallocate-gib") + 1] == "16"
    assert "--skip-finished-rollout" in argv
