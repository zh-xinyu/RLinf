# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify the every-five-round train/evaluate protocol and durable CSV results."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from toolkits.robotwin_mopd.author_pool import load_pool
from toolkits.robotwin_mopd.new4 import contracts as c
from toolkits.robotwin_mopd.new4 import evaluation as v
from toolkits.robotwin_mopd.new4 import experiment as e
from toolkits.robotwin_mopd.worker_evaluate import compose_config


@pytest.fixture
def machine(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("ROBOTWIN_PATH", str(tmp_path / "runtime/repos/RoboTwin"))
    return {
        "runtime_root": str(tmp_path / "runtime"),
        "resource_root": str(tmp_path / "resources"),
        "output_root": str(tmp_path / "outputs"),
        "pace_reference_root": str(tmp_path / "PACE"),
        "profile": "blackwell-cu128-candidate",
        "ray_port": 6451,
        "dashboard_port": 8331,
        "wandb_mode": "disabled",
        "actor_micro_batch_size": 16,
    }


def write_result(machine, job, round_, task, *, successes=37):
    pool = load_pool(v.pool_path(machine), task_names=c.TASKS)
    checkpoint = e.checkpoint_path(machine, job, round_)
    directory = v.task_directory(machine, job, round_, task)
    directory.mkdir(parents=True)
    episodes = [
        {
            "episode": index,
            "actual_reset_seed": row["scene_seed"],
            "instruction": row["instruction"],
            "actual_scene_info": row["episode_info"],
            "task": task,
            "success_once": index < successes,
            "simulation_steps": 100,
        }
        for index, row in enumerate(pool["tasks"][task])
    ]
    result = {
        "task": task,
        "checkpoint": str(checkpoint),
        "pool_content_sha256": pool["pool_content_sha256"],
        "pool_protocol": pool["protocol"],
        "requested_episodes": 100,
        "completed_episodes": 100,
        "episodes": episodes,
        "evaluation_chain": "EmbodiedEvalRunner/EnvWorker/MultiStepRolloutWorker",
        "verified_runtime": pool["protocol"]["runtime_profile"],
        "successes": successes,
        "success_rate": successes / 100,
    }
    (directory / "summary.json").write_text(json.dumps(result))
    cfg = OmegaConf.create(
        {
            "rollout": {
                "model": {
                    "num_steps": 10,
                    "num_action_chunks": 25,
                    "opd_task_norm_stats": {
                        name: str(
                            Path(machine["resource_root"])
                            / "student/assets"
                            / name
                            / "norm_stats.json"
                        )
                        for name in c.TASKS
                    },
                }
            }
        }
    )
    OmegaConf.save(cfg, directory / "resolved_config.yaml")
    return directory, result


def test_exact_pool_and_inference_config_use_production_workers(machine, tmp_path):
    command = v.worker_command(machine, c.TASKS[0], Path("/checkpoint"), tmp_path, 5)
    assert command[command.index("-m") + 1] == "toolkits.robotwin_mopd.worker_evaluate"
    assert command[command.index("--episodes") + 1] == "100"
    args = SimpleNamespace(
        task=c.TASKS[0],
        tasks=c.TASKS,
        checkpoint=Path("/checkpoint"),
        output=tmp_path,
        gpu=5,
        episodes=100,
        pool=v.pool_path(machine),
        override=c.overrides(machine, {"method": "uniform_r2", "seed": 2}),
        norm_stats=[
            (
                name,
                str(
                    Path(machine["resource_root"])
                    / "student/assets"
                    / name
                    / "norm_stats.json"
                ),
            )
            for name in c.TASKS
        ],
    )
    pool = load_pool(args.pool, task_names=c.TASKS)
    cfg = compose_config(args, pool)
    assert cfg.runner.only_eval and "train" not in cfg.env
    assert cfg.env.eval.total_num_envs == 1 and cfg.env.eval.rollout_epoch == 100
    assert (
        cfg.rollout.model.num_action_chunks == 25 and cfg.rollout.model.num_steps == 10
    )
    assert (
        cfg.rollout.model.openpi.nft_full_horizon
        and cfg.env.eval.instruction_type == "unseen"
    )
    assert "expert_models" not in cfg.rollout
    assert pool["protocol"]["author_cohort"] == 2
    assert list(pool["tasks"]) == c.TASKS


def test_partial_tasks_never_produce_macro_and_csv_rebuild_preserves_completed_scores(
    machine,
):
    job = c.jobs()[0]
    write_result(machine, job, 5, c.TASKS[0], successes=37)
    result = v.checkpoint_results(machine, job, 5, e.checkpoint_path(machine, job, 5))
    assert result["state"] == "PARTIAL" and result["macro_sr"] is None
    for task, successes in zip(c.TASKS[1:], (20, 40, 60), strict=True):
        write_result(machine, job, 5, task, successes=successes)
    e.summarize(machine)
    path = Path(machine["output_root"]) / "deliverables/checkpoint_results.csv"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 80 and rows[0]["task_macro_sr"] == "0.3925"
    assert rows[0]["evaluation_state"] == "COMPLETE"
    assert rows[1]["task_macro_sr"] == "" and rows[1]["evaluation_state"] == "PENDING"
    e.summarize(machine)
    assert list(csv.DictReader(path.open())) == rows
    experiments = list(csv.DictReader((path.parent / "experiment_results.csv").open()))
    assert len(experiments) == 8 and experiments[0]["reported_round"] == "50"
    assert experiments[0]["task_macro_sr"] == ""  # no picking the best/only checkpoint
    assert experiments[0]["evaluated_checkpoints"] == "1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint", "/other"),
        ("completed_episodes", 99),
        ("pool_content_sha256", "wrong"),
        ("success_rate", 0.99),
        ("evaluation_chain", "standalone"),
    ],
)
def test_wrong_identity_or_counts_are_rejected_not_marked_pending(
    machine, field, value
):
    job, task = c.jobs()[0], c.TASKS[0]
    directory, result = write_result(machine, job, 5, task)
    result[field] = value
    (directory / "summary.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        e.summarize(machine)


def test_scene_instruction_or_horizon_mismatch_is_rejected(machine):
    job, task = c.jobs()[0], c.TASKS[0]
    directory, original = write_result(machine, job, 5, task)
    for field, value in (
        ("actual_reset_seed", 0),
        ("instruction", "different"),
        ("simulation_steps", 701),
        ("success_once", 1),
        ("actual_scene_info", {}),
    ):
        result = copy.deepcopy(original)
        result["episodes"][0][field] = value
        (directory / "summary.json").write_text(json.dumps(result))
        with pytest.raises((ValueError, RuntimeError)):
            v.read_result(machine, directory, task, e.checkpoint_path(machine, job, 5))


def test_evaluate_job_covers_all_ten_checkpoints_and_publishes_each(
    machine, monkeypatch
):
    events = []
    monkeypatch.setattr(e, "audit_run", lambda *args: {})
    monkeypatch.setattr(
        e, "checkpoint_audit", lambda path, *args, **kwargs: events.append(path.name)
    )
    monkeypatch.setattr(e, "summarize", lambda _: events.append("csv"))

    def evaluate(m, p, j, round_, checkpoint, *, on_complete):
        events.append(round_)
        on_complete()

    monkeypatch.setattr(v, "evaluate_checkpoint", evaluate)
    e.evaluate_job(machine, Path("/machine.json"), c.jobs()[0])
    assert events == [
        value
        for round_ in range(5, 51, 5)
        for value in (f"global_step_{round_ * 2}", round_, "csv")
    ]


def test_concurrent_tasks_reuse_complete_results_and_never_overwrite_partial(
    machine, monkeypatch
):
    job = c.jobs()[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,4,5")
    launches = []
    monkeypatch.setattr(c, "environment", lambda _: {})

    class Process:
        def __init__(self, command, **kwargs):
            launches.append(command)
            task = command[command.index("--task") + 1]
            write_result(machine, job, 5, task)

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(v.subprocess, "Popen", Process)
    write_result(machine, job, 5, c.TASKS[0])
    result = v.evaluate_checkpoint(
        machine, Path("/machine.json"), job, 5, e.checkpoint_path(machine, job, 5)
    )
    assert len(launches) == 3 and result["macro_sr"] == 0.37
    v.evaluate_checkpoint(
        machine, Path("/machine.json"), job, 5, e.checkpoint_path(machine, job, 5)
    )
    assert len(launches) == 3
    partial = v.task_directory(machine, job, 10, c.TASKS[0])
    partial.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Partial evaluation"):
        v.evaluate_checkpoint(
            machine, Path("/machine.json"), job, 10, e.checkpoint_path(machine, job, 10)
        )
    assert len(launches) == 3
