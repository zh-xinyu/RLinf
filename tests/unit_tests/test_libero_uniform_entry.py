# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Manual experiment commands must agree with labels, paths and frozen budgets."""

import json
import sys

import pytest

from toolkits.libero_gr00t import launch_production as production
from toolkits.libero_gr00t import launch_uniform as uniform
from toolkits.libero_gr00t import run_uniform as entry


@pytest.fixture(autouse=True)
def mock_result_export(monkeypatch):
    """Dispatch-only tests use stub evaluators; CSV audits have a separate suite."""
    monkeypatch.setattr(entry, "record_evaluation", lambda *a, **k: None)


def machine_file(tmp_path, **overrides):
    config = {
        key: str(tmp_path / key)
        for key in ("python", "models", "spatial_teacher", "student", "results")
    }
    config.update(overrides)
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("reuse", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("run", [False, True])
def test_manual_commands_agree_with_frozen_recipe_and_logging(
    tmp_path, monkeypatch, reuse, seed, run
):
    path = machine_file(tmp_path, microbatch_size=8, gpus=[4, 5, 6, 7])
    monkeypatch.setenv("OPD_WANDB_RUN_NAME", "stale_previous_run")
    calls = []
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    argv = ["--machine", str(path), "--r", str(reuse), "--seed", str(seed)]
    entry.main(argv + (["--run", "--train-only"] if run else []))
    (command,), kwargs = calls[0]
    assert command[:3] == [
        str(tmp_path / "python"),
        "-m",
        "toolkits.libero_gr00t.launch_uniform",
    ]
    expected = f"Uniform-R{reuse}_seed{seed}" + ("" if run else "_plan")
    assert command[command.index("--output") + 1] == str(
        tmp_path / "results" / expected
    )
    assert ("--run" in command) == run
    assert command[command.index("--gpus") + 1 : command.index("--gpus") + 5] == [
        "4",
        "5",
        "6",
        "7",
    ]
    forwarded = uniform.uniform_arguments(command[3:])
    assert forwarded[forwarded.index("--updates") + 1] == str(100 * reuse)
    assert forwarded[forwarded.index("--save-every-updates") + 1] == str(10 * reuse)
    assert forwarded[forwarded.index("--native-microbatch-size") + 1] == "8"
    assert (
        kwargs["env"]["OPD_WANDB_RUN_NAME"] == f"Uniform-R{reuse}_T64_B512_seed{seed}"
    )
    assert kwargs["env"]["OPD_WANDB_GROUP"] == f"T64_uniform_seed{seed}"
    assert kwargs["env"]["WANDB_MODE"] == "offline"
    assert kwargs["check"] is True
    assert kwargs["cwd"] == entry.REPO
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 1e-4},
        {"models": "relative/models"},
        {"gpus": [0, 0, 1, 2]},
        {"microbatch_size": True},
        {"cpus": 0},
        {"actor_cache_preallocate_gib": 30},
        {"wandb_mode": "typo"},
    ],
)
def test_machine_configuration_rejects_recipe_changes_and_invalid_resources(
    tmp_path, overrides
):
    with pytest.raises(ValueError):
        entry.machine_config(machine_file(tmp_path, **overrides))


def test_explicit_output_and_paths_with_spaces_are_preserved(tmp_path, monkeypatch):
    path = machine_file(
        tmp_path, python=str(tmp_path / "runtime with spaces/bin/python")
    )
    output = tmp_path / "retry run"
    calls = []
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    entry.main(
        ["--machine", str(path), "--r", "2", "--seed", "1", "--output", str(output)]
    )
    command = calls[0][0][0]
    assert command[0] == str(tmp_path / "runtime with spaces/bin/python")
    assert command[command.index("--output") + 1] == str(output)


@pytest.mark.parametrize("environment_file", ["libero_env.sh", "env.sh"])
def test_production_accepts_portable_and_historical_runtime_files(
    tmp_path, monkeypatch, environment_file
):
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    python = runtime / "bin/python"
    python.touch()
    (runtime / environment_file).touch()
    argv = [
        "--reuse-updates",
        "1",
        "--python",
        str(python),
        "--models",
        str(tmp_path / "models"),
        "--student",
        str(tmp_path / "missing_student"),
        "--spatial-teacher",
        str(tmp_path / "spatial"),
        "--output",
        str(tmp_path / "plan"),
        "--gpus",
        "0",
        "1",
        "2",
        "3",
    ]
    monkeypatch.setattr(sys, "argv", ["launcher", *uniform.uniform_arguments(argv)])
    with pytest.raises(FileNotFoundError, match="missing_student"):
        production.main()
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize("reuse", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("round_", [0, 10, 100])
def test_evaluation_round_selects_correct_weights_and_fixed_protocol(
    tmp_path, monkeypatch, reuse, seed, round_
):
    path = machine_file(tmp_path)
    calls = []
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    entry.main(
        [
            "--machine",
            str(path),
            "--r",
            str(reuse),
            "--seed",
            str(seed),
            "--eval-round",
            str(round_),
            "--run",
        ]
    )
    command = calls[0][0][0]
    assert command[2] == "toolkits.libero_gr00t.launch_parallel_evaluation"
    assert command[command.index("--seed") + 1] == "1234"
    assert command[command.index("--envs-per-task") + 1] == "5"
    assert command[command.index("--cpus") + 1] == "8"
    label = f"Uniform-R{reuse}_seed{seed}"
    if round_ == 0:
        assert "--baseline" in command and "--weights" not in command
        expected_config = (
            tmp_path / "results" / (label + "_plan") / "resolved_config.yaml"
        )
        expected_output = tmp_path / "results/evaluations/initial_student"
    else:
        assert "--baseline" not in command
        expected_config = tmp_path / "results" / label / "resolved_config.yaml"
        weights = command[command.index("--weights") + 1]
        assert weights == str(
            expected_config.parent
            / "libero_uniform/checkpoints"
            / f"global_step_{round_ * reuse}/actor/model_state_dict/full_weights.pt"
        )
        expected_output = (
            tmp_path / "results/evaluations" / f"{label}_round{round_:03d}"
        )
    assert command[command.index("--config") + 1] == str(expected_config)
    assert command[command.index("--output") + 1] == str(expected_output)
    assert command[-1] == "--run"


def test_eval_dry_run_and_relocated_training_config(tmp_path, monkeypatch):
    path = machine_file(tmp_path)
    relocated = tmp_path / "relocated run/resolved_config.yaml"
    calls = []
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    entry.main(
        [
            "--machine",
            str(path),
            "--r",
            "4",
            "--seed",
            "1",
            "--eval-round",
            "20",
            "--config",
            str(relocated),
        ]
    )
    command = calls[0][0][0]
    assert "--run" not in command
    assert command[command.index("--output") + 1].endswith(
        "Uniform-R4_seed1_round020_plan"
    )
    assert command[command.index("--weights") + 1].startswith(str(relocated.parent))
    assert "global_step_80" in command[command.index("--weights") + 1]
