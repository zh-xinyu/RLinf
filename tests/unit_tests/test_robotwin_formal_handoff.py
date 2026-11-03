# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Formal recipe budgets, scheduler constraints and result integrity."""

import json
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
    "method,reuse,adaptive", [("m0", 1, False), ("m1", 2, False), ("m2", 2, True)]
)
def test_matched_budget_and_refresh_contract(machine, method, reuse, adaptive):
    command = formal.train_command(machine, method, 0)
    values = dict(v.split("=", 1) for v in command if "=" in v)
    assert values["actor.global_batch_size"] == "128"
    assert values["actor.micro_batch_size"] == "4"
    assert values["env.train.total_num_envs"] == "32"
    assert values["actor.optim.lr"] == "1e-5"
    assert values["actor.optim.lr_warmup_steps"] == "4"
    assert int(values["runner.max_steps"]) == 20 * reuse
    assert int(values["runner.save_interval"]) == 5 * reuse
    assert values["algorithm.sync_suite_loss_weights"] == str(adaptive).lower()
    assert values["runner.rollout_allocation.mode"] == (
        "adaptive" if adaptive else "uniform"
    )


def test_every_checkpoint_gets_100_episodes(machine):
    jobs = [
        job
        for target in ("student", "teachers", "m0", "m1", "m2")
        for job in formal.eval_jobs(machine, target, 0)
    ]
    assert len(jobs) == 56
    assert len({job["output"] for job in jobs}) == 56
    for job in jobs:
        command = job["command"]
        assert command[command.index("--episodes") + 1] == "100"
    assert {j["round"] for j in formal.eval_jobs(machine, "m1", 0)} == {5, 10, 15, 20}
    with pytest.raises(ValueError):
        formal.train_command(machine, "m0", 1)


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


def test_summary_rejects_wrong_checkpoint_and_reseed(machine, monkeypatch, tmp_path):
    pool = {
        "pool_content_sha256": "frozen",
        "tasks": {
            t: [{"scene_seed": i, "instruction": "fixed"} for i in range(100)]
            for t in formal.TASKS
        },
    }
    (tmp_path / "formal100.json").write_text(json.dumps(pool))
    monkeypatch.setattr(formal, "HERE", tmp_path)
    job = formal.eval_jobs(machine, "student", 0)[0]
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
