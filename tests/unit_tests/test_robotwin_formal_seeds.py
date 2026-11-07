# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Independent training repeats preserve the formal scientific recipe."""

import json
from pathlib import Path

import pytest

from toolkits.robotwin_mopd.formal import experiment as formal


def machine(tmp_path, seed):
    return {
        "runtime_root": str(tmp_path / "runtime"),
        "resource_root": str(tmp_path / "resources"),
        "output_root": str(tmp_path / f"seed{seed}"),
        "training_seed": seed,
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_training_sources_and_evaluation_isolation(tmp_path, seed):
    cfg = machine(tmp_path, seed)
    expected = formal.seed_settings(seed)
    commands = [formal.train_command(cfg, method, seed) for method in formal.METHODS]
    for command in commands:
        overrides = dict(v.split("=", 1) for v in command if "=" in v)
        assert all(overrides[k] == str(v) for k, v in expected.items())
        assert f"seed{seed}" in command[command.index("--run-id") + 1]
        assert overrides["env.train.total_num_envs"] == "32"
        assert overrides["actor.global_batch_size"] == "256"
    # Include all 3200 possible task episodes and the reset retry window.
    for task_id, task in enumerate(formal.TASKS):
        start = expected["runner.rollout_allocation.seed"] + task_id * 1000003
        train = set(range(start, start + 3200 + 19))
        for pool_name in ("formal100.json", "dev20.json"):
            pool = json.loads((formal.HERE / pool_name).read_text())
            assert not train.intersection(
                row["scene_seed"] for row in pool["tasks"][task]
            )
        for other in (0, 1, 2):
            if other == seed:
                continue
            other_start = (
                formal.seed_settings(other)["runner.rollout_allocation.seed"]
                + task_id * 1000003
            )
            assert not train.intersection(range(other_start, other_start + 3200 + 19))


def test_seed_zero_unchanged_and_invalid_rejected():
    assert formal.seed_settings(0) == {
        "actor.seed": 0,
        "env.train.seed": 42,
        "runner.rollout_allocation.seed": 42,
        "+rollout.seed": 1000000000,
    }
    for bad in (-1, 3, True, 1.0, "1"):
        with pytest.raises(ValueError):
            formal.seed_settings(bad)


def test_seed_specific_smoke_evaluation_and_summary(tmp_path, monkeypatch):
    cfg = machine(tmp_path, 1)
    assert "seed1" in formal.smoke_run_id(cfg)
    jobs = formal.eval_jobs(cfg, "m2", 1)
    assert len(jobs) == 48 and all("seed1" in j["checkpoint"] for j in jobs)
    assert all("seed1" in j["output"] for j in jobs)
    jobs0 = formal.eval_jobs(machine(tmp_path, 0), "m2", 0)
    assert {j["output"] for j in jobs}.isdisjoint(j["output"] for j in jobs0)
    for a, b in zip(jobs, jobs0):
        for arg in ("--evaluation-pool", "--episodes"):
            assert (
                a["command"][a["command"].index(arg) + 1]
                == b["command"][b["command"].index(arg) + 1]
            )
    formal.summarize(cfg)
    import csv

    with (Path(cfg["output_root"]) / "deliverables/checkpoint_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 50
    assert all(
        row["training_seed"] == "1"
        for row in rows
        if row["experiment"] in formal.METHODS
    )
    formal.freeze_training_settings(cfg)
    changed = dict(cfg, training_seed=2)
    with pytest.raises(ValueError, match="settings changed"):
        formal.freeze_training_settings(changed)


def test_requested_queue_order():
    script = (formal.HERE / "slurm.sh").read_text()
    assert "for method in m0 m2 m1 m3; do" in script
