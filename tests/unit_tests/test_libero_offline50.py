# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen recipes, exact initializer identity, and automatic tiered measurements."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from toolkits.libero_gr00t import launch_production as production
from toolkits.libero_gr00t import run_offline50 as entry
from toolkits.libero_gr00t.offline50_recipe import config_name


@pytest.mark.parametrize(
    "method,interval", [("uniform", None), ("adaptive", 5), ("adaptive", 10)]
)
@pytest.mark.parametrize("repeats", [1, 2])
def test_six_recipes_preserve_shared_numerics_and_change_only_interval(
    tmp_path, monkeypatch, method, interval, repeats
):
    monkeypatch.setenv("OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / "run",
        gpus=[0, 1, 2, 3],
        config_name=config_name(method, repeats, interval),
        updates=80 * repeats,
        save_every_updates=10 * repeats,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        seed=0,
        native_microbatch_size=4,
    )
    paths = {
        key: tmp_path / key
        for key in [
            "student",
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        ]
    }
    cfg = production.build_config(args, paths)
    assert cfg.runner.max_steps == 80 * repeats
    assert cfg.runner.save_interval == 10 * repeats
    assert cfg.algorithm.flow_refresh_updates == repeats
    assert cfg.actor.global_batch_size == 512 and cfg.actor.micro_batch_size == 4
    assert cfg.actor.optim.lr == pytest.approx(3e-5)
    assert cfg.actor.optim.lr_warmup_steps == 4
    assert cfg.env.train.total_num_envs == 64
    assert list(cfg.env.train.task_suite_execution_lengths.values()) == [5, 5, 5, 10]
    assert cfg.runner.logger.experiment_name == "libero_" + method
    assert cfg.algorithm.sync_suite_loss_weights is False
    assert cfg.algorithm.sync_suite_query_weights is (method == "adaptive")
    a = cfg.runner.rollout_allocation
    assert a.mode == method
    assert a.priority_power == 2 and a.window_rounds == a.warmup_rounds == 10
    if method == "adaptive":
        assert a.interval_rounds == interval and a.max_quota_change == 8
        original = OmegaConf.to_container(cfg, resolve=True)
        args.config_name = config_name(method, repeats, 15 - interval)
        other = OmegaConf.to_container(
            production.build_config(args, paths), resolve=True
        )
        original["runner"]["rollout_allocation"].pop("interval_rounds")
        other["runner"]["rollout_allocation"].pop("interval_rounds")
        assert original == other


def test_initializer_verification_rejects_a_different_student(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    cfg = repo / "toolkits/libero_gr00t/configs"
    cfg.mkdir(parents=True)
    student = tmp_path / "student"
    student.mkdir()
    f = student / "weights"
    f.write_bytes(b"old50")
    manifest = {
        "checkpoint_identity": {
            "content_sha256": "expected",
            "files": [
                {
                    "path": "weights",
                    "size_bytes": 5,
                    "sha256": hashlib.sha256(b"old50").hexdigest(),
                }
            ],
        }
    }
    (cfg / "offline50_student.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(entry, "REPO", repo)
    assert entry.verify_initializer(student) == "expected"
    f.write_bytes(b"sft10")
    with pytest.raises(ValueError, match="Not the agreed"):
        entry.verify_initializer(student)


def machine(tmp_path):
    cfg = {
        k: str(tmp_path / k)
        for k in [
            "python",
            "models",
            "student",
            "spatial_teacher",
            "results",
            "csv_dir",
        ]
    }
    Path(cfg["csv_dir"]).mkdir()
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


def fake_training(command):
    run = Path(command[command.index("--output") + 1])
    run.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text("frozen recipe")
    (run / "status").write_text("OK\n")
    r = int(command[command.index("--updates") + 1]) // 80
    method = (
        "adaptive"
        if "adaptive" in command[command.index("--config-name") + 1]
        else "uniform"
    )
    for n in range(10, 81, 10):
        f = (
            run
            / f"libero_{method}/checkpoints/global_step_{n * r}/actor/model_state_dict/full_weights.pt"
        )
        f.parent.mkdir(parents=True)
        f.touch()
    return run


def test_training_then_final_first_and_five_small_curve_evaluations(
    tmp_path, monkeypatch
):
    path, cfg = machine(tmp_path)
    calls = []
    exports = []
    monkeypatch.setattr(entry, "verify_initializer", lambda p: "identity")
    monkeypatch.setenv("WANDB_RUN_ID", "stale")

    def dispatch(command, **kwargs):
        calls.append(command)
        if command[2] == "toolkits.libero_gr00t.launch_production":
            assert "WANDB_RUN_ID" not in kwargs["env"]
            fake_training(command)

    monkeypatch.setattr(entry.subprocess, "run", dispatch)
    monkeypatch.setattr(
        entry, "record_evaluation", lambda *a, **kw: exports.append((a, kw))
    )
    entry.main(["--machine", str(path), "--r", "2", "--run"])
    assert len(calls) == 8  # one training, seven unique measurements
    evaluation = calls[1:]
    sizes = [c.index("--seed") - c.index("--trials") - 1 for c in evaluation]
    assert sizes == [50, 50, 20, 20, 20, 20, 20]
    assert sum(sizes) * 40 == 8000  # includes the shared initial measurement once
    assert "global_step_160" in evaluation[1][evaluation[1].index("--weights") + 1]
    assert all("--resume-incomplete" in c for c in evaluation)
    assert len(exports) == 9  # initial/final export both50 and20 without new rollouts
    run = Path(calls[0][calls[0].index("--output") + 1])
    assert (
        json.loads((run / "post_training_evaluation_n20.json").read_text())["status"]
        == "COMPLETE"
    )
    calls.clear()
    entry.main(["--machine", str(path), "--r", "2", "--evaluate-all", "--run"])
    assert len(calls) == 7 and all("launch_production" not in c[2] for c in calls)


def test_failed_training_does_not_start_measurements(tmp_path, monkeypatch):
    path, _ = machine(tmp_path)
    calls = []
    monkeypatch.setattr(entry, "verify_initializer", lambda p: "identity")

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(entry.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        entry.main(["--machine", str(path), "--r", "1", "--run"])
    assert len(calls) == 1


def test_variant_labels_and_explicit_selection(tmp_path):
    labels = set()
    for method, interval in [("uniform", None), ("adaptive", 5), ("adaptive", 10)]:
        for r in (1, 2):
            labels.add(
                entry.label(
                    argparse.Namespace(method=method, interval=interval, r=r, seed=0)
                )
            )
    assert len(labels) == 6
    with pytest.raises(SystemExit):
        entry.main(
            ["--machine", str(tmp_path / "unused"), "--method", "adaptive", "--r", "1"]
        )


def test_existing_machine_settings_are_preserved_and_outputs_are_namespaced(tmp_path):
    path, cfg = machine(tmp_path)
    cfg.update(gpus=[4, 5, 6, 7], cpus=48, microbatch_size=8, wandb_mode="disabled")
    path.write_text(json.dumps(cfg))
    before = path.read_bytes()
    args = argparse.Namespace(machine=path, student=tmp_path / "offline50")
    selected = entry.effective_settings(args)
    assert path.read_bytes() == before
    for key in (
        "python",
        "models",
        "spatial_teacher",
        "gpus",
        "cpus",
        "microbatch_size",
        "wandb_mode",
    ):
        assert selected[key] == cfg[key]
    assert selected["student"] == str(tmp_path / "offline50")
    assert selected["results"] == str(tmp_path / "results/offline50_80r")
    assert selected["csv_dir"] == str(tmp_path / "csv_dir/offline50_80r")


def test_three_server_assignments_cover_all_six_and_balance_update_budget():
    from toolkits.libero_gr00t.queue_offline50 import ASSIGNMENTS

    items = [arm for pair in ASSIGNMENTS.values() for arm in pair]
    assert len(set(items)) == 6
    assert all(sum(80 * r for _, r, _ in pair) == 240 for pair in ASSIGNMENTS.values())


def test_server_queue_trains_both_before_automatic_evaluation(tmp_path, monkeypatch):
    from toolkits.libero_gr00t import queue_offline50 as queue

    path, _ = machine(tmp_path)
    calls = []
    monkeypatch.setattr(
        queue.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    queue.main(
        [
            "--machine",
            str(path),
            "--student",
            str(tmp_path / "offline50"),
            "--server",
            "1",
            "--run",
        ]
    )
    assert len(calls) == 4
    assert all("--train-only" in c for c in calls[:2])
    assert all("--evaluate-all" in c for c in calls[2:])
    assert all(
        c[c.index("--student") + 1] == str(tmp_path / "offline50") for c in calls
    )


def test_twenty_episode_upgrade_reuses_training_and_preserves_old_receipt(
    tmp_path, monkeypatch
):
    machine_path, _ = machine(tmp_path)
    args = argparse.Namespace(
        machine=machine_path, student=None, method="uniform", r=1, interval=None, seed=0
    )
    settings = entry.effective_settings(args)
    run = Path(settings["results"]) / entry.label(args)
    fake_training(
        [
            "--output",
            str(run),
            "--updates",
            "80",
            "--config-name",
            "libero_mopd_uniform_offline50_r1",
        ]
    )
    original = entry.contract(args, "identity", run / "resolved_config.yaml")
    original["curve_episodes_per_task"] = 10
    (run / "offline50_experiment.json").write_text(json.dumps(original))
    old_receipt = run / "post_training_evaluation.json"
    old_receipt.write_text("preserve legacy ten-episode receipt")
    monkeypatch.setattr(entry, "verify_initializer", lambda _: "identity")
    calls = []
    monkeypatch.setattr(
        entry.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    monkeypatch.setattr(entry, "record_evaluation", lambda *args, **kwargs: None)
    entry.main(["--machine", str(machine_path), "--r", "1", "--evaluate-all", "--run"])
    assert len(calls) == 7
    assert all(c[2].endswith("launch_parallel_evaluation") for c in calls)
    assert old_receipt.read_text() == "preserve legacy ten-episode receipt"
    new = json.loads((run / "post_training_evaluation_n20.json").read_text())
    assert new["status"] == "COMPLETE"
    assert new["identity"]["curve_episodes_per_task"] == 20
    assert {key for key in new["measurements"] if key.endswith(":20")} == {
        "10:20",
        "20:20",
        "30:20",
        "40:20",
        "60:20",
    }
    changed = {**original, "r": 2}
    assert not entry.compatible_training_contract(changed, new["identity"])
