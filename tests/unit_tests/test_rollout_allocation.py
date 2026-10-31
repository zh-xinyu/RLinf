# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Check allocation, task coverage and physical accounting without a simulator."""

import json
from collections import Counter

import pytest
from omegaconf import OmegaConf

from rlinf.utils.rollout_allocation import RolloutAllocator, bounded_quotas


def allocator(mode="adaptive", **overrides):
    tasks = [
        {"task_id": i, "task": f"task{i}", "suite": s, "horizon": 17}
        for i, s in enumerate(["a", "a", "b", "b", "c", "c"])
    ]
    config = {
        "mode": mode,
        "warmup_rounds": 2,
        "window_rounds": 2,
        "interval_rounds": 1,
        "min_task_episodes": 2,
        "max_quota_change": 8,
        "rho": 0.5,
        "epsilon": 0.01,
        "seed": 0,
        "static_quotas": {"a": 16, "b": 40, "c": 40},
    }
    config.update(overrides)
    return RolloutAllocator(
        tasks, 8, 6, 2, OmegaConf.create(config), {"a": 0.9, "b": 0.9, "c": 0.9}
    )


def outcomes(plan):
    return [
        dict(
            row,
            success_once=row["suite"] == "a",
            simulation_steps=3 if row["suite"] == "a" else row["horizon"],
        )
        for row in plan
    ]


def test_adaptive_changes_acquisition_but_preserves_all_local_suites():
    a = allocator()
    seen = Counter()
    for round_index in range(5):
        plan = a.plan()
        assert len(plan) == 96
        counts = Counter(r["suite"] for r in plan)
        assert list(counts.values()) == a.quotas
        for worker in range(8):
            for epoch in range(2):
                cell = [
                    r for r in plan if r["worker"] == worker and r["epoch"] == epoch
                ]
                assert len(cell) == 6
                assert {r["suite"] for r in cell} == {"a", "b", "c"}
        if round_index < 2:
            assert a.quotas == [32, 32, 32]
        else:
            assert a.quotas[0] < a.quotas[1]
        for row in plan:
            seen[row["task"]] += 1
        a.observe(outcomes(plan))
    assert a.rounds == 5 and a.trajectories == 480
    for suite in a.suites:
        n = [seen[t["task"]] for t in a.pools[suite]]
        assert max(n) - min(n) <= 1
    json.dumps(a.state_dict())  # Nested DictConfig must not leak into checkpoints.


def test_zero_gaps_returns_uniform():
    a = allocator()
    for _ in range(2):
        a.observe(
            [dict(row, success_once=True, simulation_steps=1) for row in a.plan()]
        )
    a.plan()
    assert a.quotas == [32, 32, 32]


def test_coverage_failure_holds_quota_instead_of_dropping_tasks():
    a = allocator(min_task_episodes=10000)
    for _ in range(2):
        a.observe(outcomes(a.plan()))
    a.plan()
    assert a.quotas == [32, 32, 32]
    assert a.last_diagnostics["reason"] == "insufficient_task_coverage"


def test_static_and_adaptive_share_task_local_reset_sequence():
    uniform, static = allocator("uniform"), allocator("static")
    maps = []
    for a in (uniform, static):
        seeds = {}
        for _ in range(4):
            plan = a.plan()
            for row in plan:
                seeds[(row["task"], row["occurrence"])] = row["reset_seed"]
            a.observe(outcomes(plan))
        maps.append(seeds)
    for key in maps[0].keys() & maps[1].keys():
        assert maps[0][key] == maps[1][key]
    assert static.quotas == [16, 40, 40]


def test_task_macro_not_episode_micro_estimate():
    a = allocator()
    # Deliberately uneven episode counts: task0=1/10, task1=10/10.
    a.rounds = 2
    rows = []
    for task in a.tasks:
        n = 20 if task["task"] == "task0" else 10
        wins = 2 if task["task"] == "task0" else 10
        rows += [{"task": task["task"], "success_once": i < wins} for i in range(n)]
    a.history.append(rows)
    a.plan()
    assert a.last_diagnostics["success"]["a"] == pytest.approx(0.55)


@pytest.mark.parametrize("fault", ["duplicate", "identity", "premature", "zero_steps"])
def test_bad_physical_records_fail_closed(fault):
    a = allocator()
    records = outcomes(a.plan())
    if fault == "duplicate":
        records[-1] = records[0]
    elif fault == "identity":
        records[0]["reset_seed"] += 1
    elif fault == "premature":
        records[0].update(success_once=False, simulation_steps=1)
    else:
        records[0]["simulation_steps"] = 0
    with pytest.raises(ValueError):
        a.observe(records)
    assert a.rounds == 0


def test_round_is_counted_once_and_steps_are_not_multiplied_by_reuse():
    a = allocator()
    plan = a.plan()
    with pytest.raises(RuntimeError):
        a.plan()
    records = outcomes(plan)
    expected = sum(r["simulation_steps"] for r in records)
    a.observe(records)
    assert a.simulation_steps == expected
    assert a.trajectories == 96
    with pytest.raises(ValueError):
        a.observe(records)
    assert a.simulation_steps == expected


def test_bounded_projection_handles_extreme_and_fractional_targets():
    counts = bounded_quotas([0, 1, 0], 96, [32, 32, 32], 16, 8)
    assert sum(counts) == 96 and counts[1] == 40
    assert all(24 <= n <= 40 for n in counts)
    with pytest.raises(ValueError):
        bounded_quotas([0.5, 0.5], 10, [5, 5], 6, 2)


def test_real_wrapper_records_do_not_count_frozen_padding():
    import numpy as np
    from test_robocasa365_lifecycle import episode_env

    env = episode_env(horizons=(2, 5), success_steps=(99, 1))
    env._allocation_episode_plan = [{"task": "a"}, {"task": "b"}]
    env.chunk_step(np.zeros((2, 8, 1)))
    records = env.rollout_episode_records()
    assert [r["simulation_steps"] for r in records] == [2, 1]
    assert [r["success_once"] for r in records] == [False, True]


def test_window_discards_old_outcomes_and_logs_signed_progress():
    a = allocator()
    for index in range(4):
        plan = a.plan()
        a.observe(
            [
                dict(
                    row,
                    success_once=index < 2,
                    simulation_steps=1 if index < 2 else row["horizon"],
                )
                for row in plan
            ]
        )
    a.plan()
    assert a.last_diagnostics["success"] == dict.fromkeys(a.suites, 0.0)
    assert a.last_diagnostics["signed_progress"] == dict.fromkeys(a.suites, -1.0)


def reference_fixture(tmp_path):
    from rlinf.utils.rollout_allocation import training_signature

    cfg = OmegaConf.create(
        {
            "env": {"train": {"env_type": "robocasa365", "ignore_terminations": False}},
            "actor": {
                "model": {
                    "model_type": "gr00t",
                    "precision": "bf16",
                    "denoising_steps": 4,
                    "action_dim": 7,
                }
            },
            "rollout": {
                "model": {"precision": "bf16"},
                "expert_models": {
                    "a": {
                        "model_path": str(tmp_path / "teacher"),
                        "action_head_precision": "fp32",
                    }
                },
            },
        }
    )
    tasks = [
        {"task": "one", "suite": "a", "horizon": 100},
        {"task": "two", "suite": "a", "horizon": 200},
    ]
    data = {
        "schema_version": 1,
        "status": "COMPLETE",
        "training_signature": training_signature(cfg),
        "teacher_checkpoints": {"a": str(tmp_path / "teacher")},
        "tasks": [
            dict(tasks[0], episodes=20, successes=10),
            dict(tasks[1], episodes=10, successes=10),
        ],
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(data))
    return cfg, tasks, path, data


def test_teacher_reference_uses_task_macro_and_effective_rollout_model(tmp_path):
    from rlinf.utils.rollout_allocation import (
        load_teacher_reference,
        training_signature,
    )

    cfg, tasks, path, _ = reference_fixture(tmp_path)
    assert training_signature(cfg)["model"]["denoising_steps"] == 4
    assert load_teacher_reference(str(path), cfg, tasks) == {"a": 0.75}
    cfg.actor.model.denoising_steps = 8
    with pytest.raises(ValueError, match="protocol"):
        load_teacher_reference(str(path), cfg, tasks)


@pytest.mark.parametrize(
    "fault", ["incomplete", "horizon", "checkpoint", "counts", "precision"]
)
def test_teacher_reference_rejects_protocol_mismatch(tmp_path, fault):
    from rlinf.utils.rollout_allocation import load_teacher_reference

    cfg, tasks, path, data = reference_fixture(tmp_path)
    if fault == "incomplete":
        data["status"] = "SMOKE"
    elif fault == "horizon":
        data["tasks"][0]["horizon"] += 1
    elif fault == "checkpoint":
        data["teacher_checkpoints"]["a"] += "-other"
    elif fault == "counts":
        data["tasks"][0]["successes"] = 100
    else:
        cfg.rollout.expert_models.a.action_head_precision = "bf16"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_teacher_reference(str(path), cfg, tasks)


@pytest.mark.parametrize("recovery", [0, 9])
def test_runner_reconciles_physical_counts_and_logs_monotonic_updates(
    tmp_path, recovery
):
    from types import SimpleNamespace

    from rlinf.runners.embodied_runner import EmbodiedRunner

    a = allocator("uniform")
    records = outcomes(a.plan())
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.rollout_allocator = a
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "recovery_interactions": {
                    "rollout_rounds": recovery,
                    "trajectories": recovery * 96,
                    "simulation_steps": recovery * 100,
                }
            }
        }
    )
    runner.env = SimpleNamespace(
        get_rollout_allocation_records=lambda: SimpleNamespace(wait=lambda: [records])
    )
    logged = []
    runner.metric_logger = SimpleNamespace(
        log_path=str(tmp_path), log=lambda row, step: logged.append((step, row))
    )
    runner.global_step = 0
    runner.updates_per_rollout = 2
    runner.cumulative_env_steps = 0
    runner.teacher_calibration_budget = {
        "trajectories": 1000,
        "simulation_steps": 12345,
        "wall_seconds": 10,
    }
    runner._record_allocation_round()
    assert not logged  # Physical metrics must not advance W&B before update 1.
    metrics = {
        "refresh/optimizer_updates": 2,
        "refresh/real_env_steps_this_rollout": a.simulation_steps,
        "refresh/update_1/actor/velocity_mse": 1.0,
        "refresh/update_2/actor/velocity_mse": 0.9,
    }
    runner._log_optimizer_updates([metrics])
    assert [step for step, _ in logged] == [1, 2]
    for _, row in logged:
        assert row["interaction/rollout_trajectories"] == 96
        assert row["interaction/rollout_rounds"] == 1
        assert row["interaction/simulation_steps"] == a.simulation_steps
        assert (
            row["interaction/total_simulation_steps"]
            == a.simulation_steps + 12345 + recovery * 100
        )
        assert row["interaction/total_trajectories"] == 1096 + recovery * 96
        assert row["interaction/total_training_rollout_rounds"] == 1 + recovery
    assert "allocation/quota/a" not in logged[0][1]
    assert logged[1][1]["allocation/quota/a"] == 32
    assert logged[1][1]["calibration/trajectories"] == 1000
    metrics["refresh/real_env_steps_this_rollout"] += 1
    with pytest.raises(RuntimeError, match="accounting mismatch"):
        runner._log_optimizer_updates([metrics])


def test_round_boundary_restore_preserves_next_plan_and_gap():
    uninterrupted = allocator()
    for _ in range(4):
        uninterrupted.observe(outcomes(uninterrupted.plan()))
    state = json.loads(json.dumps(uninterrupted.state_dict()))
    restored = allocator()
    restored.load_state_dict(state)
    assert restored.plan() == uninterrupted.plan()
    assert restored.last_diagnostics == uninterrupted.last_diagnostics
    assert restored.simulation_steps == uninterrupted.simulation_steps


@pytest.mark.parametrize("fault", ["layout", "reference", "tasks", "history", "config"])
def test_round_boundary_restore_rejects_incompatible_state(fault):
    a = allocator()
    for _ in range(3):
        a.observe(outcomes(a.plan()))
    state = json.loads(json.dumps(a.state_dict()))
    if fault == "layout":
        state["history"][0][0]["worker"] = 99
    elif fault == "reference":
        state["reference"]["a"] = 0.1
    elif fault == "tasks":
        state["history"][0][0]["task_id"] = 99
    elif fault == "history":
        state["history"].pop()
    else:
        state["config"]["rho"] = 0.75
    with pytest.raises(ValueError):
        allocator().load_state_dict(state)


@pytest.mark.parametrize("probe", [False, True])
def test_runner_restores_full_round_state_without_restarting_warmup(tmp_path, probe):
    from types import SimpleNamespace

    from rlinf.runners.embodied_runner import EmbodiedRunner

    a = allocator("uniform")
    a.reference = None
    for _ in range(2 if probe else 3):
        a.observe(outcomes(a.plan()))
    updates = a.rounds * 2
    checkpoint = tmp_path / "global_step_6"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "runner_state.json").write_text(
        json.dumps(
            {
                "step_unit": "optimizer_update",
                "optimizer_updates": updates,
                "rollout_count": a.rounds,
                "env_steps": a.simulation_steps,
            }
        )
    )
    (checkpoint / "rollout_allocation_state.json").write_text(
        json.dumps(a.state_dict())
    )
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    cfg = dict(a.config, enabled=True)
    # Checkpoint config must match the resolved enabled setting as in production.
    saved = a.state_dict()
    saved["config"] = cfg
    (checkpoint / "rollout_allocation_state.json").write_text(json.dumps(saved))
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "rollout_allocation": dict(cfg, mode="static") if probe else cfg,
                "resume_dir": str(checkpoint),
                "resume_mode": "allocation_probe"
                if probe
                else "adaptive_round_boundary",
            }
        }
    )
    runner.reward = None
    runner.optimizer_update_steps = True
    runner.updates_per_rollout = 2
    runner.max_steps = 10
    runner.metric_logger = SimpleNamespace(log_path=str(tmp_path))
    runner.logger = SimpleNamespace(info=lambda message: None)
    handle = SimpleNamespace(wait=lambda: None)
    loaded = []
    runner.rollout = SimpleNamespace(init_worker=lambda: handle)
    runner.env = SimpleNamespace(
        init_worker=lambda: handle,
        describe_rollout_allocation=lambda: SimpleNamespace(
            wait=lambda: [
                {"worker": w, "tasks": a.tasks, "envs": 6, "epochs": 2}
                for w in range(8)
            ]
        ),
    )
    runner.actor = SimpleNamespace(
        init_worker=lambda: handle,
        load_checkpoint=lambda path: loaded.append(path) or handle,
        restore_opd_update_counters=lambda step: SimpleNamespace(
            wait=lambda: [{"optimizer_updates": step}]
        ),
    )
    runner.init_workers()
    assert runner.global_step == updates
    assert runner.cumulative_env_steps == a.simulation_steps
    assert loaded == [str(checkpoint / "actor")]
    if probe:
        assert Counter(r["suite"] for r in runner.rollout_allocator.plan()) == {
            "a": 16,
            "b": 40,
            "c": 40,
        }
    else:
        assert runner.rollout_allocator.plan() == a.plan()
    assert (
        json.loads((tmp_path / "resume_metadata.json").read_text())["exact_replay"]
        is False
    )


@pytest.mark.parametrize("mode", ["uniform", "static", "adaptive"])
def test_allocation_probe_preserves_history_and_task_reset_sequences(mode):
    source = allocator()
    for _ in range(2):
        source.observe(outcomes(source.plan()))
    state = json.loads(json.dumps(source.state_dict()))
    target = allocator(mode, static_quotas={"a": 24, "b": 36, "c": 36})
    if mode != "adaptive":
        target.reference = None
    target.load_state_dict(state, allocation_probe=True)
    assert target.cursors == source.cursors
    assert target.occurrences == source.occurrences
    assert target.history == source.history
    assert target.simulation_steps == source.simulation_steps
    plan = target.plan()
    if mode != "adaptive":
        expected = [32, 32, 32] if mode == "uniform" else [24, 36, 36]
        assert Counter(r["suite"] for r in plan) == dict(zip(target.suites, expected))
    original = source.plan()
    seeds = {(r["task"], r["occurrence"]): r["reset_seed"] for r in original}
    for row in plan:
        key = row["task"], row["occurrence"]
        if key in seeds:
            assert row["reset_seed"] == seeds[key]


@pytest.mark.parametrize(
    "fault", ["implicit", "late", "seed", "window", "rho", "history"]
)
def test_allocation_probe_rejects_uncontrolled_changes(fault):
    source = allocator()
    for _ in range(3 if fault == "late" else 2):
        source.observe(outcomes(source.plan()))
    state = json.loads(json.dumps(source.state_dict()))
    overrides = {}
    if fault in {"seed", "rho"}:
        overrides[fault] = 0.75 if fault == "rho" else 99
    elif fault == "window":
        overrides["window_rounds"] = 1
    elif fault == "history":
        state["history"][0][0]["suite"] = "b"
    target = allocator("static", **overrides)
    with pytest.raises(ValueError):
        target.load_state_dict(state, allocation_probe=fault != "implicit")
