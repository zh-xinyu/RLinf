# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Formal recipe budgets, scheduler constraints and result integrity."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from toolkits.robotwin_mopd.formal import experiment as formal
from toolkits.robotwin_mopd.resources import cpu_budget


@pytest.fixture
def machine(tmp_path):
    return {
        "runtime_root": str(tmp_path / "runtime"),
        "resource_root": str(tmp_path / "resources"),
        "output_root": str(tmp_path / "results"),
    }


def test_slurm_cpu_limit():
    assert cpu_budget(None, {"SLURM_CPUS_PER_TASK": "32"}, 128) == 32
    assert cpu_budget(None, {}, 128) == 48
    assert cpu_budget(None, {}, 16) == 16
    with pytest.raises(ValueError):
        cpu_budget(48, {"SLURM_CPUS_PER_TASK": "32"}, 128)
    with pytest.raises(ValueError):
        cpu_budget(0, {}, 32)


@pytest.mark.parametrize(
    "method,reuse,adaptive",
    [("m0", 1, False), ("m1", 2, False), ("m2", 2, True), ("m3", 1, True)],
)
def test_matched_budget_and_refresh_contract(machine, method, reuse, adaptive):
    command = formal.train_command(machine, method, 0)
    values = dict(v.split("=", 1) for v in command if "=" in v)
    assert values["actor.global_batch_size"] == "256"
    assert values["actor.micro_batch_size"] == "16"
    assert values["env.train.total_num_envs"] == "32"
    assert values["actor.optim.lr"] == "1e-5"
    assert values["actor.optim.lr_warmup_steps"] == "4"
    assert int(values["runner.max_steps"]) == 100 * reuse
    assert values["runner.save_interval"] == "-1"
    assert json.loads(values["+runner.save_steps"]) == [
        r * reuse for r in [5, 10, 15, 20, *range(30, 101, 10)]
    ]
    assert values["algorithm.sync_suite_loss_weights"] == str(adaptive).lower()
    assert values["runner.rollout_allocation.mode"] == (
        "adaptive" if adaptive else "uniform"
    )


def test_every_checkpoint_gets_100_episodes(machine):
    jobs = [
        job
        for target in ("student", "teachers", "m0", "m1", "m2", "m3")
        for job in formal.eval_jobs(machine, target, 0)
    ]
    assert len(jobs) == 200
    assert len({job["output"] for job in jobs}) == 200
    for job in jobs:
        command = job["command"]
        assert command[command.index("--episodes") + 1] == "100"
    assert {j["round"] for j in formal.eval_jobs(machine, "m1", 0)} == {
        5,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    }
    with pytest.raises(ValueError):
        formal.train_command(machine, "m0", 3)


def test_prepare_materializes_external_tokenizer_link(machine, tmp_path, monkeypatch):
    """OpenPI's resolved-path cache guard must not follow a staged external link."""
    source = tmp_path / "original_tokenizer.model"
    source.write_bytes(b"verified tokenizer")
    staged = (
        Path(machine["resource_root"])
        / "openpi-data/big_vision/paligemma_tokenizer.model"
    )
    staged.parent.mkdir(parents=True)
    staged.symlink_to(source)
    monkeypatch.setattr(formal, "verify_resources", lambda machine: {"verified": True})
    formal.prepare(machine)
    copied = (
        Path(machine["output_root"])
        / "prepared/openpi-data/big_vision/paligemma_tokenizer.model"
    )
    assert copied.read_bytes() == source.read_bytes()
    assert not copied.is_symlink()
    formal.prepare(machine)
    copied.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="tokenizer differs"):
        formal.prepare(machine)


@pytest.mark.parametrize("target", ["student", "m3"])
def test_summary_rejects_wrong_checkpoint_and_reseed(
    machine, monkeypatch, tmp_path, target
):
    pool = {
        "pool_content_sha256": "frozen",
        "tasks": {
            t: [{"scene_seed": i, "instruction": "fixed"} for i in range(100)]
            for t in formal.TASKS
        },
    }
    (tmp_path / "formal100.json").write_text(json.dumps(pool))
    monkeypatch.setattr(formal, "HERE", tmp_path)
    job = formal.eval_jobs(machine, target, 0)[0]
    path = Path(job["output"])
    path.parent.mkdir(parents=True)
    result = {
        "evaluation_pool": {
            "pool_content_sha256": "frozen",
            "selected_scene_seeds": list(range(100)),
        },
        "requested_episodes": 100,
        "completed_episodes": 100,
        "task": job["task"],
        "episodes": [
            {"actual_reset_seed": i, "instruction": "fixed", "success_once": True}
            for i in range(100)
        ],
        "checkpoint": job["checkpoint"],
        "successes": 100,
        "chunk_length": 25,
        "denoising_steps": 10,
        "task_horizon": 400,
    }
    path.write_text(json.dumps(result))
    formal.summarize(machine)
    result["chunk_length"] = 50
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="fixed-horizon"):
        formal.summarize(machine)
    result["chunk_length"] = 25
    result["episodes"][0]["actual_reset_seed"] = 10000
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="seed/instruction"):
        formal.summarize(machine)
    result["episodes"][0]["actual_reset_seed"] = 0
    result["checkpoint"] = "/wrong/checkpoint"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="checkpoint"):
        formal.summarize(machine)


def test_cli_plan_and_summary_include_complete_factorial_matrix(machine, tmp_path):
    """The public plan and result table must expose both factors at both levels."""
    config = {
        **machine,
        "pace_reference_root": str(tmp_path / "pace"),
        "wandb_mode": "disabled",
        "profile": "blackwell-cu128-candidate",
    }
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(config))
    command = [
        sys.executable,
        str(formal.HERE / "experiment.py"),
        "--machine",
        str(path),
    ]
    plan = json.loads(subprocess.check_output([*command, "plan"], text=True))
    assert set(plan["training"]) == {"m0", "m1", "m2", "m3"}
    assert len(plan["evaluation"]) == 200
    m3_jobs = formal.eval_jobs(machine, "m3", 0)
    assert {Path(job["checkpoint"]).name for job in m3_jobs} == {
        "global_step_5",
        "global_step_10",
        "global_step_15",
        "global_step_20",
        "global_step_30",
        "global_step_40",
        "global_step_50",
        "global_step_60",
        "global_step_70",
        "global_step_80",
        "global_step_90",
        "global_step_100",
    }
    subprocess.run([*command, "summarize"], check=True)
    table = Path(machine["output_root"]) / "deliverables/checkpoint_results.csv"
    with table.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 50
    assert {int(row["rollout_round"]) for row in rows if row["experiment"] == "m3"} == {
        5,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    }
    assert all(row["status"] == "PENDING" and not row["task_macro_sr"] for row in rows)


@pytest.mark.parametrize("micro", [4, 8, 16, 32])
def test_hardware_microbatch_keeps_global_recipe_and_changes_smoke_id(machine, micro):
    machine["actor_micro_batch_size"] = micro
    for method in formal.METHODS:
        command = formal.train_command(machine, method, 0, smoke=True)
        values = dict(v.split("=", 1) for v in command if "=" in v)
        assert values["actor.global_batch_size"] == "256"
        assert values["actor.micro_batch_size"] == str(micro)
        assert values["actor.optim.lr"] == "1e-5"
        assert command[command.index("--run-id") + 1].endswith(f"_micro{micro}_smoke")


@pytest.mark.parametrize("micro", [0, -1, 3, 128, True, "16"])
def test_invalid_hardware_microbatch_is_rejected(machine, micro):
    machine["actor_micro_batch_size"] = micro
    with pytest.raises(ValueError, match="actor_micro_batch_size"):
        formal.train_command(machine, "m0", 0)


def test_formal_arms_cannot_silently_change_common_microbatch(machine):
    formal.freeze_training_settings(machine)
    formal.freeze_training_settings(machine)
    with pytest.raises(ValueError, match="Formal settings changed"):
        formal.freeze_training_settings({**machine, "actor_micro_batch_size": 8})


def test_sft500_identity_is_explicit(machine):
    assert formal.RECIPE["initialization"] == "Joint-SFT500"
    for method in formal.METHODS:
        command = formal.train_command(machine, method, 0)
        assert "sft500" in formal.run_id(method, 0)
        assert "runner.logger.wandb_group=robotwin_formal_sft500_100ep" in command


def test_checkpoint_paths_match_runner_and_optional_storage(machine, tmp_path):
    for method in formal.METHODS:
        command = formal.train_command(machine, method, 0)
        output = Path(command[command.index("--output-root") + 1])
        name = formal.run_id(method, 0)
        for job in formal.eval_jobs(machine, method, 0):
            step = job["round"] * formal.RECIPE["methods"][method]["reuse"]
            assert (
                Path(job["checkpoint"])
                == output / name / name / "checkpoints" / f"global_step_{step}"
            )
        assert "runner.logger.project_name=FORMAL_robotwin_pi05" in command
    machine["training_output_roots"] = {"m2": str(tmp_path / "local_disk")}
    command = formal.train_command(machine, "m2", 0)
    assert command[command.index("--output-root") + 1] == str(tmp_path / "local_disk")
    assert formal.checkpoint_path(machine, "m2", 0, 20).is_relative_to(
        tmp_path / "local_disk"
    )
    assert not formal.checkpoint_path(machine, "m0", 0, 10).is_relative_to(
        tmp_path / "local_disk"
    )


def test_runtime_pool_binding_preserves_scenes_and_protocol(machine, tmp_path):
    from toolkits.robotwin_mopd.feasible_pool import pool_content_sha256

    directory = tmp_path / "runtime_pools"
    directory.mkdir()
    machine["runtime_pool_dir"] = str(directory)
    for name in ("formal100.json", "dev20.json"):
        pool = json.loads((formal.HERE / name).read_text())
        pool["protocol"]["runtime_profile"]["sapien_content_fingerprint"]["sha256"] = (
            "a" * 64
        )
        pool["pool_content_sha256"] = pool_content_sha256(pool)
        (directory / name).write_text(json.dumps(pool))
    assert formal.evaluation_pool_path(machine) == directory / "formal100.json"
    assert (
        formal.evaluation_pool_path(machine, development=True)
        == directory / "dev20.json"
    )
    formal.freeze_training_settings(machine)
    for name in ("formal100.json", "dev20.json"):
        pool = json.loads((directory / name).read_text())
        pool["protocol"]["runtime_profile"]["sapien_content_fingerprint"]["sha256"] = (
            "b" * 64
        )
        pool["pool_content_sha256"] = pool_content_sha256(pool)
        (directory / name).write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="Formal settings changed"):
        formal.freeze_training_settings(machine)
    pool["tasks"][formal.TASKS[0]][0]["instruction"] = "changed instruction"
    pool["pool_content_sha256"] = pool_content_sha256(pool)
    (directory / "dev20.json").write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="changed frozen scenes or protocol"):
        formal.evaluation_pool_path(machine, development=True)


def test_runtime_pool_binding_rejects_relative_directory(machine):
    machine["runtime_pool_dir"] = "relative"
    with pytest.raises(ValueError, match="must be absolute"):
        formal.evaluation_pool_path(machine)
