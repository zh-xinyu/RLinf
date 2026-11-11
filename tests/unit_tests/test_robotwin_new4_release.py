# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check the shipped eight-job recipe, relocation and fail-closed train queue."""

import copy
import json
import sys
from pathlib import Path

import pytest

from toolkits.robotwin_mopd.new4 import contracts as c
from toolkits.robotwin_mopd.new4 import experiment as e
from toolkits.robotwin_mopd.new4.validate import compose_config


@pytest.fixture
def machine(tmp_path, monkeypatch):
    for key in ("ROBOTWIN_RUNTIME_ROOT", "ROBOTWIN_PATH"):
        monkeypatch.setenv(key, str(tmp_path / "runtime"))
    return {
        "runtime_root": str(tmp_path / "runtime"),
        "resource_root": str(tmp_path / "resources"),
        "output_root": str(tmp_path / "outputs"),
        "pace_reference_root": str(tmp_path / "PACE"),
        "profile": "blackwell-cu128-candidate",
        "wandb_mode": "disabled",
        "actor_micro_batch_size": 16,
        "node_label": "test",
        "ray_port": 6451,
        "dashboard_port": 8331,
    }


def test_eight_jobs_cover_complete_third_seed_and_r4_repeats():
    actual = {(row["method"], row["seed"]) for row in c.jobs()}
    assert actual == {
        ("uniform_r1", 2),
        ("adaptive_r1", 2),
        ("uniform_r2", 2),
        ("adaptive_r2", 2),
        ("uniform_r4", 1),
        ("adaptive_r4", 1),
        ("uniform_r4", 2),
        ("adaptive_r4", 2),
    }
    assert (
        sum(50 * c.RECIPE["methods"][row["method"]]["reuse"] for row in c.jobs())
        == 1100
    )
    assert len({c.job_id(row) for row in c.jobs()}) == 8


@pytest.mark.parametrize("job", c.jobs(), ids=c.job_id)
def test_every_shipped_recipe_preserves_science_and_uses_correct_seed(machine, job):
    from rlinf.config import (
        validate_flow_refresh_cfg,
        validate_rollout_allocation_cfg,
        validate_suite_distillation_cfg,
    )

    cfg = compose_config(machine, job)
    validate_flow_refresh_cfg(cfg)
    validate_rollout_allocation_cfg(cfg)
    validate_suite_distillation_cfg(cfg)
    spec = c.RECIPE["methods"][job["method"]]
    reuse, seed = spec["reuse"], job["seed"]
    assert cfg.actor.seed == seed
    assert cfg.env.train.seed == cfg.runner.rollout_allocation.seed == 42 + 10000 * seed
    assert cfg.rollout.seed == 1000000000 + 10000 * seed
    assert list(cfg.env.train.task_names) == c.TASKS
    assert dict(cfg.env.train.task_horizons) == c.RECIPE["task_horizons"]
    assert set(cfg.rollout.expert_models) == set(c.TASKS)
    assert set(cfg.actor.model.opd_task_norm_stats) == set(c.TASKS)
    assert cfg.actor.global_batch_size == 256 and cfg.actor.micro_batch_size == 16
    assert cfg.env.train.total_num_envs == 32
    assert cfg.runner.max_steps == 50 * reuse
    assert list(cfg.runner.save_steps) == list(range(5 * reuse, 51 * reuse, 5 * reuse))
    assert cfg.actor.optim.lr == 1e-5 and cfg.actor.optim.lr_warmup_steps == 4
    assert cfg.actor.optim.lr_scheduler == "constant"
    assert cfg.algorithm.loss_type == "analytic_local_kl"
    assert cfg.actor.model.num_steps == 10 and cfg.actor.model.num_action_chunks == 25
    assert cfg.algorithm.sync_suite_query_weights == spec["adaptive"]
    assert cfg.algorithm.sync_suite_loss_weights is False
    assert cfg.runner.val_check_interval == -1
    assert cfg.runner.rolling_full_checkpoint is True
    assert (
        cfg.actor.cuda_cache_preallocate_gib == cfg.actor.cuda_cache_retention_gib == 0
    )
    a = cfg.runner.rollout_allocation
    assert (a.warmup_rounds, a.window_rounds, a.interval_rounds) == (10, 10, 5)
    assert (
        a.rho,
        a.epsilon,
        a.priority_power,
        a.min_task_episodes,
        a.max_quota_change,
    ) == (1, 0, 2, 40, 6)
    assert "checkpoint-500" not in c.train_command(machine, job)


def test_target_smoke_really_requests_two_r2_rounds_then_full_resume(machine):
    job = c.select_job("uniform_r2_s2")
    command = c.train_command(machine, job, smoke=True)
    assert "runner.max_steps=4" in command and "runner.save_steps=[2,4]" in command
    assert "--smoke" not in command  # legacy flag would force only two updates
    resumed = c.train_command(
        machine, job, smoke=True, resume_dir=Path("/saved/global_step_4")
    )
    assert "runner.max_steps=6" in resumed
    assert "+runner.resume_mode=adaptive_round_boundary" in resumed
    assert any(
        value.startswith("runner.resume_dir=") and "global_step_4" in value
        for value in resumed
    )
    assert "--num-gpus" in command and command[command.index("--num-gpus") + 1] == "4"


def test_frozen_contract_rejects_microbatch_or_recipe_change(machine):
    c.freeze(machine)
    c.freeze(machine)
    changed = dict(machine, actor_micro_batch_size=8)
    with pytest.raises(ValueError, match="contract changed"):
        c.freeze(changed)


def test_existing_incomplete_run_is_not_silently_overwritten(machine, monkeypatch):
    job = c.select_job("uniform_r2_s2")
    e.job_directory(machine, job).mkdir(parents=True)
    calls = []
    monkeypatch.setattr(e.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(
        e,
        "audit_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("incomplete")),
    )
    with pytest.raises(ValueError, match="incomplete"):
        e.execute_job(machine, job, {})
    assert calls == []


def test_queue_trains_then_evaluates_before_next_training(machine, monkeypatch):
    calls = []
    monkeypatch.setattr(e, "require_deployment", lambda _: None)
    monkeypatch.setattr(e, "summarize", lambda _: None)
    monkeypatch.setattr(
        e, "execute_job", lambda m, j, env: calls.append(("train", c.job_id(j)))
    )
    monkeypatch.setattr(
        e, "evaluate_job", lambda m, p, j: calls.append(("evaluate", c.job_id(j)))
    )
    chosen = c.jobs()[:2]
    e.run_queue(machine, Path("/machine.json"), chosen, {})
    assert calls == [
        (stage, c.job_id(job)) for job in chosen for stage in ("train", "evaluate")
    ]
    calls.clear()
    e.run_queue(machine, Path("/machine.json"), chosen, {}, evaluation_only=True)
    assert calls == [("evaluate", c.job_id(job)) for job in chosen]


def test_reference_relocation_requires_content_identity_and_preserves_counts(
    machine, monkeypatch, tmp_path
):
    package = tmp_path / "package"
    package.mkdir()
    root = Path(machine["resource_root"])
    payloads = {"openpi-data/big_vision/paligemma_tokenizer.model": b"tokenizer"}
    payloads.update(
        {f"student/assets/{task}/norm_stats.json": b"{}" for task in c.TASKS}
    )
    records = {}
    for name, content in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records[name] = {"bytes": len(content), "sha256": c.digest(path)}
    (package / "resources.json").write_text(json.dumps({"files": records}))
    reference = json.loads((c.HERE / "teacher_reference_source.json").read_text())
    expected_tasks = copy.deepcopy(reference["tasks"])
    (package / "teacher_reference_source.json").write_text(json.dumps(reference))
    for name in ("formal100.json", "dev2.json"):
        (package / name).write_bytes((c.HERE / name).read_bytes())
    monkeypatch.setattr(c, "HERE", package)
    c.prepare(machine)
    result = json.loads(
        (Path(machine["output_root"]) / "prepared/teacher_reference.json").read_text()
    )
    assert result["tasks"] == expected_tasks
    assert result["teacher_checkpoints"] == {
        task: str((root / "teachers" / task).resolve()) for task in c.TASKS
    }
    assert result["training_signature"]["model"]["opd_task_norm_stats"] == {
        task: str(root / "student/assets" / task / "norm_stats.json")
        for task in c.TASKS
    }
    c.prepare(machine)
    (root / "student/assets" / c.TASKS[0] / "norm_stats.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        c.prepare(machine)


def test_pending_inventory_plans_all80saved_policy_points(machine):
    e.summarize(machine)
    import csv

    rows = list(
        csv.DictReader(
            (
                Path(machine["output_root"]) / "deliverables/checkpoint_results.csv"
            ).open()
        )
    )
    assert len(rows) == 80
    assert sum(row["formal_evaluation_planned"] == "True" for row in rows) == 80
    assert {row["evaluation_state"] for row in rows} == {"PENDING"}
    assert {row["checkpoint_ready"] for row in rows} == {"False"}
    assert {row["training_seed"] for row in rows} == {"1", "2"}


def test_plan_bootstraps_without_any_site_packages(machine, tmp_path):
    import subprocess

    path = tmp_path / "machine.json"
    path.write_text(json.dumps(machine))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "toolkits.robotwin_mopd.new4.experiment",
            "--machine",
            str(path),
            "plan",
        ],
        cwd=c.REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert len(plan["jobs"]) == 8
    assert plan["evaluation"] == "AFTER_EACH_TRAINING"
    assert plan["evaluation_rounds"] == list(range(5, 51, 5))
    assert plan["formal_episodes"] == 32000
    assert plan["optimizer_updates"] == 1100


def test_training_gate_rejects_unvalidated_or_changed_resources(machine, monkeypatch):
    output = Path(machine["output_root"]) / "prepared"
    output.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        e.require_deployment(machine)
    monkeypatch.setattr(e, "source_identity", lambda: {"source": "same"})
    monkeypatch.setattr(c, "check_runtime", lambda _: {"runtime": "same"})
    proof = {
        "status": "TARGET_GPU_VALIDATION_PASS",
        "source_identity": {"source": "same"},
        "runtime": {"runtime": "same"},
        "microbatch": 16,
        "evaluation_smoke": {"state": "COMPLETE"},
        "formal_pool_sha256": json.loads((c.HERE / "formal100.json").read_text())[
            "pool_content_sha256"
        ],
    }
    (output / "gpu_validation.json").write_text(json.dumps(proof))
    path = Path(machine["resource_root"]) / "small.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    receipt = {
        "manifest_sha256": c.digest(c.HERE / "resources.json"),
        "files": {
            "small.bin": {
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        },
    }
    (output / "resource_verification.json").write_text(json.dumps(receipt))
    e.require_deployment(machine)
    path.write_bytes(b"modified-and-not-verified")
    with pytest.raises(ValueError, match="resource changed"):
        e.require_deployment(machine)
