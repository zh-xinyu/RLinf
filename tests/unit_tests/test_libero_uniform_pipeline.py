# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Training must exit successfully before scheduled checkpoint evaluation."""

import json
import subprocess

import pytest

from toolkits.libero_gr00t import launch_parallel_evaluation as evaluator
from toolkits.libero_gr00t import run_uniform as entry
from toolkits.libero_gr00t.parallel_evaluation import audit_episodes


def setup_machine(tmp_path, repeats=1):
    config = {key: str(tmp_path / key) for key in entry.REQUIRED}
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(config))
    run = tmp_path / "results" / f"Uniform-R{repeats}_seed0"
    return machine, run


def completed_training(run, repeats):
    run.mkdir(parents=True, exist_ok=True)
    (run / "status").write_text("OK\n")
    (run / "resolved_config.yaml").write_text("frozen training recipe")
    for round_ in range(10, 101, 10):
        weights = (
            run
            / f"libero_uniform/checkpoints/global_step_{repeats * round_}/actor/model_state_dict/full_weights.pt"
        )
        weights.parent.mkdir(parents=True)
        weights.touch()


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_default_training_then_all_evaluations_and_evaluate_only(
    tmp_path, monkeypatch, repeats
):
    machine, run = setup_machine(tmp_path, repeats)
    calls = []

    def child(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command)
        if command[2].endswith("launch_uniform"):
            completed_training(run, repeats)
        else:
            assert (run / "status").read_text().strip() == "OK"
            assert "--reuse-complete" in command

    monkeypatch.setattr(entry.subprocess, "run", child)
    args = ["--machine", str(machine), "--r", str(repeats), "--seed", "0", "--run"]
    entry.main(args)
    assert len(calls) == 12
    assert "--baseline" in calls[1]
    for round_, command in zip(range(10, 101, 10), calls[2:], strict=True):
        assert (
            f"global_step_{round_ * repeats}/"
            in command[command.index("--weights") + 1]
        )
        assert command[command.index("--seed") + 1] == "1234"
    state = json.loads((run / "post_training_evaluation.json").read_text())
    assert state["status"] == "COMPLETE"
    assert len(state["rounds"]) == 11
    assert all(row["status"] == "COMPLETE" for row in state["rounds"].values())
    calls.clear()
    entry.main([*args, "--evaluate-all"])
    assert len(calls) == 11
    assert all(command[2].endswith("launch_parallel_evaluation") for command in calls)


def test_no_evaluation_after_training_failure(tmp_path, monkeypatch):
    machine, run = setup_machine(tmp_path)
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(entry.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        entry.main(["--machine", str(machine), "--r", "1", "--seed", "0", "--run"])
    assert len(calls) == 1
    assert not (run / "post_training_evaluation.json").exists()


def test_missing_checkpoint_rejects_evaluation_before_any_child(tmp_path, monkeypatch):
    machine, run = setup_machine(tmp_path)
    completed_training(run, 1)
    next(run.rglob("full_weights.pt")).unlink()
    monkeypatch.setattr(
        entry.subprocess, "run", lambda *a, **k: pytest.fail("unexpected child")
    )
    with pytest.raises(FileNotFoundError):
        entry.main(
            [
                "--machine",
                str(machine),
                "--r",
                "1",
                "--seed",
                "0",
                "--evaluate-all",
                "--run",
            ]
        )


def test_failed_evaluation_preserved_then_new_attempt_on_explicit_retry(
    tmp_path, monkeypatch
):
    machine, run = setup_machine(tmp_path)
    completed_training(run, 1)
    calls = []

    def fail(command, **kwargs):
        output = entry.Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "failed_record").write_text("keep original evidence")
        raise subprocess.CalledProcessError(1, command)

    args = [
        "--machine",
        str(machine),
        "--r",
        "1",
        "--seed",
        "0",
        "--evaluate-all",
        "--run",
    ]
    monkeypatch.setattr(entry.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        entry.main(args)
    receipt = run / "post_training_evaluation.json"
    assert json.loads(receipt.read_text())["status"] == "FAILED"
    monkeypatch.setattr(
        entry.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    entry.main(args)
    assert calls[0][calls[0].index("--output") + 1].endswith("initial_student_attempt2")
    assert (tmp_path / "results/evaluations/initial_student/failed_record").exists()
    assert json.loads(receipt.read_text())["status"] == "COMPLETE"


def evaluation_folder(tmp_path):
    expected = {
        "base_checkpoint_identity": {"content_sha256": "base"},
        "metadata_sha256": "metadata",
        "weights_sha256": None,
        "tasks": [0],
        "trials": [0],
        "execution_lengths": {"libero_spatial": 5},
        "horizons": {"libero_spatial": 240},
        "rng_protocol": "episode-private",
        "seed": 1234,
        "inference_model_config": {"precision": "bf16"},
        "source_sha256": {"evaluator": "frozen-source"},
        "mode": "run",
    }
    output = tmp_path / "evaluation"
    task = output / "task_00"
    task.mkdir(parents=True)
    rows = [
        {
            "task_id": 0,
            "trial_id": 0,
            "suite": "libero_spatial",
            "success_once": True,
            "simulation_steps": 5,
            "settling_steps": 10,
            "horizon": 240,
        }
    ]
    (output / "manifest.json").write_text(json.dumps(expected))
    (task / "status").write_text("OK\n")
    (task / "episodes.json").write_text(json.dumps(rows))
    (output / "summary.json").write_text(json.dumps(audit_episodes(rows, [0], [0])))
    (output / "episodes.csv").touch()
    return output, expected


def test_complete_evaluation_reuse_audits_raw_outcomes(tmp_path):
    output, expected = evaluation_folder(tmp_path)
    evaluator.reuse_completed(output, expected)
    (output / "task_00/episodes.json").write_text("[]")
    with pytest.raises(ValueError, match="coverage"):
        evaluator.reuse_completed(output, expected)


@pytest.mark.parametrize(
    "key",
    [
        "base_checkpoint_identity",
        "seed",
        "weights_sha256",
        "inference_model_config",
        "source_sha256",
    ],
)
def test_stale_evaluation_identity_cannot_be_reused(tmp_path, key):
    output, expected = evaluation_folder(tmp_path)
    expected[key] = "different"
    with pytest.raises(ValueError, match=key):
        evaluator.reuse_completed(output, expected)


def test_failed_task_and_inconsistent_summary_cannot_be_reused(tmp_path):
    output, expected = evaluation_folder(tmp_path)
    (output / "task_00/status").write_text("FAILED\n")
    with pytest.raises(ValueError, match="Incomplete"):
        evaluator.reuse_completed(output, expected)
    (output / "task_00/status").write_text("OK\n")
    (output / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="summary"):
        evaluator.reuse_completed(output, expected)
