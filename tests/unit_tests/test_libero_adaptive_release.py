# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive uses the shared execution contract without changing Uniform recipes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from toolkits.libero_gr00t import launch_adaptive as adaptive
from toolkits.libero_gr00t import launch_production as production
from toolkits.libero_gr00t import run_uniform as entry

NAMES = ("student", "libero_spatial", "libero_object", "libero_goal", "libero_10")


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_adaptive_config_matches_uniform_except_allocation_and_query_share(
    tmp_path, monkeypatch, repeats
):
    monkeypatch.setenv("OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    paths = {key: tmp_path / key for key in NAMES}
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / "run",
        gpus=[0, 1, 2, 3],
        seed=0,
        config_name=f"libero_mopd_adaptive_release_r{repeats}",
        updates=100 * repeats,
        save_every_updates=10 * repeats,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
    )
    selected = production.build_config(args, paths)
    assert selected.runner.rollout_allocation.mode == "adaptive"
    assert selected.runner.logger.experiment_name == "libero_adaptive"
    assert selected.algorithm.sync_suite_query_weights is True
    assert selected.algorithm.sync_suite_loss_weights is False
    assert selected.actor.micro_batch_size == 4
    args.config_name = f"libero_mopd_uniform_r{repeats}"
    control = production.build_config(args, paths)
    before, after = [
        OmegaConf.to_container(c, resolve=True) for c in (control, selected)
    ]
    for c in (before, after):
        c["runner"].pop("logger")
        # Owner-side allocator tuning is intentional; the numerical training,
        # model, execution and evaluation contract must stay aligned instead.
        c["runner"].pop("rollout_allocation")
        c["algorithm"].pop("sync_suite_query_weights")
    assert before == after
    forwarded = adaptive.adaptive_arguments(
        ["--reuse-updates", str(repeats), "--microbatch-size", "8"]
    )
    assert (
        forwarded[forwarded.index("--config-name") + 1]
        == f"libero_mopd_adaptive_release_r{repeats}"
    )
    assert forwarded[forwarded.index("--updates") + 1] == str(100 * repeats)


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_adaptive_production_cli_accepts_new_configs(tmp_path, repeats):
    with pytest.raises(FileNotFoundError, match="missing_runtime"):
        adaptive.main(
            [
                "--reuse-updates",
                str(repeats),
                "--python",
                str(tmp_path / "missing_runtime"),
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
                str(tmp_path / "output"),
            ]
        )


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_adaptive_manual_entry_binds_reference_and_auto_evaluates(
    tmp_path, monkeypatch, repeats
):
    config = {key: str(tmp_path / key) for key in entry.REQUIRED}
    (tmp_path / "csv_dir").mkdir()
    bundle = tmp_path / "published_bundle.json"
    bundle.write_text("{}")
    config["teacher_reference_bundle"] = str(bundle)
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(config))
    calls, exports = [], []
    run = tmp_path / "results" / f"Adaptive-R{repeats}_seed1"

    def child(command, **kwargs):
        calls.append(command)
        if command[2].endswith("reference_bundle"):
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True)
            output.write_text("{}")
        elif command[2].endswith("launch_adaptive"):
            assert kwargs["env"]["OPD_LIBERO_TEACHER_REFERENCE"].endswith(".json")
            assert (
                kwargs["env"]["OPD_WANDB_RUN_NAME"]
                == f"Adaptive-R{repeats}_T64_B512_seed1"
            )
            run.mkdir(parents=True)
            (run / "status").write_text("OK\n")
            (run / "resolved_config.yaml").write_text("adaptive recipe")
            for r in range(10, 101, 10):
                weights = (
                    run
                    / f"libero_adaptive/checkpoints/global_step_{r * repeats}/actor/model_state_dict/full_weights.pt"
                )
                weights.parent.mkdir(parents=True)
                weights.touch()
        else:
            assert command[2].endswith("launch_parallel_evaluation")

    monkeypatch.setattr(entry.subprocess, "run", child)
    monkeypatch.setattr(
        entry, "record_evaluation", lambda *a, **k: exports.append((a, k))
    )
    entry.main(
        [
            "--method",
            "adaptive",
            "--machine",
            str(machine),
            "--r",
            str(repeats),
            "--seed",
            "1",
            "--run",
        ]
    )
    assert len(calls) == 13  # bind, train, shared initial and ten saved checkpoints.
    assert len(exports) == 11
    assert all(kwargs["method"] == "adaptive" for args, kwargs in exports)
    assert all(
        "libero_adaptive" in c[c.index("--weights") + 1]
        for c in calls
        if "--weights" in c
    )
    calls.clear()
    entry.main(
        [
            "--method",
            "adaptive",
            "--machine",
            str(machine),
            "--r",
            str(repeats),
            "--seed",
            "1",
            "--evaluate-all",
            "--run",
        ]
    )
    assert len(calls) == 11  # Evaluation recovery needs no calibration bundle/bind.


def test_unpublished_adaptive_bundle_stops_before_training(tmp_path, monkeypatch):
    config = {key: str(tmp_path / key) for key in entry.REQUIRED}
    (tmp_path / "csv_dir").mkdir()
    config["teacher_reference_bundle"] = str(tmp_path / "not_published.json")
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(config))
    monkeypatch.setattr(
        entry.subprocess, "run", lambda *a, **k: pytest.fail("unexpected child")
    )
    with pytest.raises(FileNotFoundError, match="not been published"):
        entry.main(
            [
                "--method",
                "adaptive",
                "--machine",
                str(machine),
                "--r",
                "2",
                "--seed",
                "0",
                "--run",
            ]
        )
