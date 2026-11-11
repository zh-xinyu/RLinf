# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Uniform controls must isolate reuse without changing the learning recipe."""

from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from toolkits.libero_gr00t import launch_production as production
from toolkits.libero_gr00t import launch_uniform as uniform


@pytest.mark.parametrize("repeats", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_uniform_recipe_preserves_budget_and_seed(tmp_path, repeats, seed):
    argv = uniform.uniform_arguments(
        ["--reuse-updates", str(repeats), "--seed", str(seed)]
    )
    assert argv[argv.index("--config-name") + 1] == f"libero_mopd_uniform_r{repeats}"
    assert argv[argv.index("--updates") + 1] == str(100 * repeats)
    assert argv[argv.index("--save-every-updates") + 1] == str(10 * repeats)
    assert argv[argv.index("--seed") + 1] == str(seed)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(production.REPO / "examples/embodiment/config"),
    ):
        native = compose(config_name=f"libero_mopd_uniform_r{repeats}")
    assert native.runner.max_steps == 100 * repeats
    assert native.runner.save_interval == 10 * repeats
    assert native.actor.micro_batch_size == 4
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / "run",
        gpus=[0, 1, 2, 3],
        config_name=f"libero_mopd_uniform_r{repeats}",
        updates=100 * repeats,
        save_every_updates=10 * repeats,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        seed=seed,
    )
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
    cfg = production.build_config(args, paths)
    assert cfg.runner.rollout_allocation.mode == "uniform"
    assert cfg.runner.rollout_allocation.teacher_reference_file is None
    assert cfg.algorithm.flow_refresh_updates == repeats
    assert cfg.algorithm.sync_suite_query_weights is False
    assert cfg.runner.logger.experiment_name == "libero_uniform"
    assert cfg.actor.micro_batch_size == 4
    assert cfg.actor.global_batch_size == 512
    args.config_name = "libero_mopd_uniform_r1"
    args.updates = 100
    args.save_every_updates = 10
    control = production.build_config(args, paths)
    # The whole recipe must match R1 except reuse, its derived budget and labels.
    before, after = [OmegaConf.to_container(c, resolve=True) for c in (control, cfg)]
    for c in (before, after):
        c["runner"].pop("max_steps")
        c["runner"].pop("save_interval")
        c["runner"]["logger"].pop("wandb_run_name")
        c["algorithm"].pop("flow_refresh_updates")
    assert before == after


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_uniform_launcher_cli_reaches_runtime_without_starting_work(tmp_path, repeats):
    with pytest.raises(FileNotFoundError, match="missing_runtime_python"):
        uniform.main(
            [
                "--reuse-updates",
                str(repeats),
                "--python",
                str(tmp_path / "missing_runtime_python"),
                "--models",
                str(tmp_path / "models"),
                "--student",
                str(tmp_path / "student"),
                "--spatial-teacher",
                str(tmp_path / "spatial"),
                "--gpus",
                "0",
                "1",
                "2",
                "3",
                "--output",
                str(tmp_path / "plan"),
            ]
        )
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--global-batch-size", "128"],
        ["--arm", "m2"],
        ["--reuse-updates", "2"],
        ["--trajectory-budget", "32"],
    ],
)
def test_uniform_rejects_unmatched_recipe_overrides(extra):
    with pytest.raises(ValueError):
        uniform.uniform_arguments(["--reuse-updates", "4", *extra])


def test_uniform_rejects_unplanned_reuse_budget():
    with pytest.raises(SystemExit):
        uniform.uniform_arguments(["--reuse-updates", "8"])


@pytest.mark.parametrize("microbatch", [2, 4, 8])
@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_microbatch_admission_preserves_formal_recipe(microbatch, repeats, tmp_path):
    argv = uniform.uniform_arguments(
        ["--reuse-updates", str(repeats), "--microbatch-size", str(microbatch)]
    )
    assert argv[argv.index("--native-microbatch-size") + 1] == str(microbatch)
    assert argv[argv.index("--updates") + 1] == str(100 * repeats)
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / "run",
        gpus=[0, 1, 2, 3],
        config_name=argv[argv.index("--config-name") + 1],
        updates=100 * repeats,
        save_every_updates=10 * repeats,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        seed=0,
        native_microbatch_size=microbatch,
    )
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
    cfg = production.build_config(args, paths)
    assert cfg.actor.micro_batch_size == microbatch
    assert cfg.actor.global_batch_size == 512
    assert cfg.algorithm.native_text_microbatches is True
    assert cfg.algorithm.flow_refresh_updates == repeats
    assert cfg.runner.max_steps == 100 * repeats
    assert cfg.runner.save_interval == 10 * repeats
    assert cfg.runner.rollout_allocation.mode == "uniform"


def test_uniform_rejects_unplanned_microbatch():
    with pytest.raises(SystemExit):
        uniform.uniform_arguments(["--reuse-updates", "2", "--microbatch-size", "16"])
