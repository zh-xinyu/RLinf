# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Verify machine configuration, experiment routing, and result delivery without GPUs."""

import argparse
import csv
import json
import socket
from pathlib import Path

import pytest

from toolkits.robocasa_mopd import experiment, results, run


def machine_file(tmp_path):
    path = tmp_path / "machine.json"
    value = {
        "hostname": socket.gethostname().split(".")[0],
        "runtime_root": str(tmp_path / "runtime"),
        "resource_root": str(tmp_path / "resources"),
        "student_checkpoint": str(tmp_path / "Joint-SFT-Full"),
        "output_root": str(tmp_path / "output"),
        "wandb_entity": "team",
        "wandb_mode": "offline",
    }
    path.write_text(json.dumps(value))
    return path


def test_machine_does_not_depend_on_shell_overlays(tmp_path, monkeypatch):
    machine = experiment.load_machine(machine_file(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/wrong")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("GPU_FREE_MB", "1")
    env = experiment.environment(machine)
    assert env["PYTHONPATH"] == str(experiment.REPO)
    assert env["WANDB_ENTITY"] == "team"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "HF_HUB_OFFLINE" not in env and "GPU_FREE_MB" not in env
    assert env["OPD_MODEL_ROOT"].endswith(
        "teachers/gr00t_n1-5/foundation_model_learning"
    )
    assert not Path(machine["output_root"]).exists()


def test_machine_rejects_wrong_host_and_unknown_keys(tmp_path):
    path = machine_file(tmp_path)
    value = json.loads(path.read_text())
    value["hostname"] = "different-machine"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hostname"):
        experiment.load_machine(path)
    value["unknown"] = "setting"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exactly"):
        experiment.load_machine(path)


@pytest.mark.parametrize("mode", ["disabled", "offline"])
def test_local_logging_needs_no_account_or_login(tmp_path, monkeypatch, mode):
    path = machine_file(tmp_path)
    value = json.loads(path.read_text())
    value.update(wandb_entity="", wandb_mode=mode)
    path.write_text(json.dumps(value))
    machine = experiment.load_machine(path)
    monkeypatch.setenv("WANDB_ENTITY", "unrelated-team")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    env = experiment.environment(machine)
    assert "WANDB_ENTITY" not in env and "WANDB_DISABLED" not in env
    assert env["WANDB_MODE"] == mode
    assert experiment.commands_for(argparse.Namespace(command="login"), machine) == []


def test_online_logging_uses_collaborator_account(tmp_path):
    path = machine_file(tmp_path)
    value = json.loads(path.read_text())
    value.update(wandb_entity="", wandb_mode="online")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="username or team"):
        experiment.load_machine(path)
    value["wandb_entity"] = "collaborator-account"
    path.write_text(json.dumps(value))
    machine = experiment.load_machine(path)
    assert experiment.environment(machine)["WANDB_ENTITY"] == "collaborator-account"
    assert experiment.commands_for(argparse.Namespace(command="login"), machine) == [
        [str(Path(machine["runtime_root"]) / "venv/bin/wandb"), "login"]
    ]


@pytest.mark.parametrize("arm", list(experiment.ARMS))
def test_disabling_wandb_only_changes_logger_backend(tmp_path, monkeypatch, arm):
    from omegaconf import OmegaConf

    machine = experiment.load_machine(machine_file(tmp_path))
    for key, value in experiment.environment(machine).items():
        if key.startswith(("OPD_", "EMBODIED_")):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("WANDB_MODE", "online")
    online = run.training_config(arm, "logger-test")[0]
    monkeypatch.setenv("WANDB_MODE", "disabled")
    local, _, command = run.training_config(arm, "logger-test")
    assert local.runner.logger.logger_backends == ["tensorboard"]
    assert "runner.logger.logger_backends=[tensorboard]" in command
    assert "wandb" in online.runner.logger.logger_backends
    online.runner.logger.logger_backends = ["tensorboard"]
    assert OmegaConf.to_container(online, resolve=True) == OmegaConf.to_container(
        local, resolve=True
    )


def test_outputs_binding_never_replaces_existing_data(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(experiment, "REPO", repo)
    machine = experiment.load_machine(machine_file(tmp_path))
    experiment.bind_outputs(machine)
    assert (repo / "outputs").resolve() == Path(machine["output_root"])
    experiment.bind_outputs(machine)
    marker = repo / "outputs/keep"
    marker.write_text("important")
    with pytest.raises(ValueError, match="no data moved"):
        experiment.bind_outputs({**machine, "output_root": str(tmp_path / "different")})
    assert marker.read_text() == "important"


@pytest.mark.parametrize("arm", list(experiment.ARMS))
def test_matrix_entry_and_checkpoint_schedule(tmp_path, monkeypatch, arm):
    machine = experiment.load_machine(machine_file(tmp_path))
    for key, value in experiment.environment(machine).items():
        if key.startswith(("OPD_", "EMBODIED_")):
            monkeypatch.setenv(key, value)
    name, repeats = experiment.ARMS[arm]
    cfg, output, _ = run.training_config(arm, name)
    assert cfg.runner.logger.log_path == str(output)
    assert output.name == name
    assert cfg.algorithm.flow_refresh_updates == repeats
    args = argparse.Namespace(command="eval", arm=arm, round=None)
    commands = experiment.commands_for(args, machine)
    assert len(commands) == 6
    for r, command in zip(experiment.ROUNDS, commands):
        expected = (
            Path(cfg.runner.logger.log_path)
            / name
            / "checkpoints"
            / f"global_step_{r * repeats}"
            / "actor/model_state_dict/full_weights.pt"
        )
        assert experiment.weights_path(arm, r) == expected
        assert command[-1] == str(expected)
        assert "--intermediate" not in command


def test_all_twenty_five_evaluations_and_template_budgets(tmp_path):
    machine = experiment.load_machine(machine_file(tmp_path))
    command = experiment.commands_for(
        argparse.Namespace(command="eval", arm="joint-sft-full", round=None), machine
    )[0]
    assert "--weights" not in command and "--intermediate" not in command
    rows = results.template_rows()
    assert len(rows) == 25
    assert sum(row["episodes_per_task"] * row["num_tasks"] for row in rows) == 62500
    assert all(
        "task_macro_sr" not in row and "training_trajectories" not in row
        for row in rows
    )
    assert len({row["eval_directory"] for row in rows}) == 25


def fake_eval(folder, monkeypatch):
    manifest = json.loads(
        (experiment.REPO / "evaluations/robocasa365/target50_manifest.json").read_text()
    )
    folder.mkdir()
    tasks = [
        {"suite": suite, "task": name, "horizon": horizon}
        for suite, entries in manifest.items()
        for name, horizon in entries.items()
    ]
    inputs = json.loads(
        (experiment.REPO / "docs/opd/collaboration/checkpoints.json").read_text()
    )["checkpoints"][experiment.INITIALIZATION]
    plan = {
        "mode": "formal",
        "protocol": "rlinf-robocasa365-target50-native-v1",
        "seed": 7,
        "episodes_per_task": 50,
        "tasks": tasks,
        "weights": None,
        "base_model_sha256": {
            name: record["sha256"]
            for name, record in inputs.items()
            if name.endswith((".safetensors", ".safetensors.index.json"))
        },
        "base_config_sha256": inputs["config.json"]["sha256"],
        "base_metadata_sha256": inputs["experiment_cfg/metadata.json"]["sha256"],
    }
    (folder / "plan.json").write_text(json.dumps(plan))
    with (folder / "per_task_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "suite",
                "task",
                "horizon",
                "episodes",
                "success_once_count",
                "success_once",
            ],
        )
        writer.writeheader()
        for task in tasks:
            count = 50 if task["suite"] == "atomic_seen" else 0
            writer.writerow(
                {
                    **task,
                    "episodes": 50,
                    "success_once_count": count,
                    "success_once": count / 50,
                }
            )
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "tasks": 50,
                "episodes": 2500,
                "success_once": 0.36,
                "suites": {
                    suite: {"success_once": 1.0 if suite == "atomic_seen" else 0.0}
                    for suite in manifest
                },
            }
        )
    )
    return plan


def test_result_recomputation_distinguishes_suite_and_task_macro(tmp_path, monkeypatch):
    folder = tmp_path / "eval"
    fake_eval(folder, monkeypatch)
    scores = results.evaluation_scores(folder, None)
    assert scores["suite_macro_sr"] == pytest.approx(1 / 3)
    assert scores["task_macro_sr"] == pytest.approx(0.36)
    assert not list(folder.glob("*.pt"))


def test_result_rejects_twenty_episode_or_wrong_checkpoint(tmp_path, monkeypatch):
    folder = tmp_path / "eval"
    plan = fake_eval(folder, monkeypatch)
    with pytest.raises(ValueError, match="different experiment"):
        results.evaluation_scores(folder, experiment.weights_path("m1", 25))
    plan["episodes_per_task"] = 20
    (folder / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="50 episodes"):
        results.evaluation_scores(folder, None)


@pytest.mark.parametrize(
    "field", ["base_model_sha256", "base_config_sha256", "base_metadata_sha256"]
)
def test_result_rejects_different_initialization(tmp_path, monkeypatch, field):
    folder = tmp_path / "eval"
    plan = fake_eval(folder, monkeypatch)
    plan[field] = {"old-model": "hash"} if field == "base_model_sha256" else "old-hash"
    (folder / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="initialization"):
        results.evaluation_scores(folder, None)


def test_result_rejects_incomplete_task_results(tmp_path, monkeypatch):
    folder = tmp_path / "eval"
    fake_eval(folder, monkeypatch)
    path = folder / "per_task_results.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="all50"):
        results.evaluation_scores(folder, None)


def test_result_rejects_incorrect_summary(tmp_path, monkeypatch):
    folder = tmp_path / "eval"
    fake_eval(folder, monkeypatch)
    path = folder / "summary.json"
    summary = json.loads(path.read_text())
    summary["success_once"] = 0.4
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Summary disagrees"):
        results.evaluation_scores(folder, None)


def test_no_results_remain_pending_without_fabricated_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "REPO", tmp_path)
    rows = results.collect()
    assert len(rows) == 25 and all(row["status"] == "PENDING" for row in rows)
    assert all("task_macro_sr" not in row for row in rows)


def test_round_time_uses_completed_update_steps(tmp_path, monkeypatch):
    from torch.utils.tensorboard import SummaryWriter

    monkeypatch.setattr(experiment, "REPO", tmp_path)
    folder = tmp_path / "outputs/train" / experiment.ARMS["m1"][0]
    with SummaryWriter(str(folder / "tensorboard")) as writer:
        writer.add_scalar("time/rollout_round", 20.0, 2)
        writer.add_scalar("time/rollout_round", 30.0, 4)
    assert results.round_times("m1") == {2: 20.0, 4: 30.0}


def test_complete_result_without_wandb(tmp_path, monkeypatch):
    from torch.utils.tensorboard import SummaryWriter

    directory = tmp_path / "outputs/eval" / experiment.eval_id("m0", 25)
    directory.parent.mkdir(parents=True)
    plan = fake_eval(directory, monkeypatch)
    plan["weights"] = str(experiment.weights_path("m0", 25))
    plan["weights_sha256"] = "checkpoint-hash"
    (directory / "plan.json").write_text(json.dumps(plan))
    (directory / "status").write_text("OK\n")
    folder = tmp_path / "outputs/train" / experiment.ARMS["m0"][0]
    with SummaryWriter(str(folder / "tensorboard")) as writer:
        for update in range(1, 26):
            writer.add_scalar("time/rollout_round", 10.0, update)
    (folder / "optimizer_updates.jsonl").write_text(
        json.dumps(
            {
                "update_step": 25,
                "interaction/rollout_rounds": 25,
                "interaction/rollout_trajectories": 2400,
                "interaction/simulation_steps": 123456,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(experiment, "REPO", tmp_path)
    # Public manifests identify tasks and initialization without model files or W&B.
    for relative in (
        "evaluations/robocasa365/target50_manifest.json",
        "docs/opd/collaboration/checkpoints.json",
    ):
        manifest = tmp_path / relative
        manifest.parent.mkdir(parents=True, exist_ok=True)
        original = Path(__file__).resolve().parents[2] / relative
        manifest.write_bytes(original.read_bytes())
    row = next(
        r
        for r in results.collect()
        if r["experiment"] == "M0" and r["rollout_round"] == 25
    )
    assert row["status"] == "COMPLETE"
    assert not row.get("wandb_url") and not row["notes"]
    assert row["training_round_seconds"] == 250.0
    assert row["suite_macro_sr"] == pytest.approx(1 / 3)
