# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Gap/query control must preserve LIBERO's native inputs and sample objective."""

import inspect
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_analytic_local_kl_loss
from rlinf.algorithms.native_microbatch import native_microbatch_plan
from rlinf.algorithms.suite_distillation import (
    finalize_suite_statistics,
    suite_query_quotas,
)
from rlinf.config import validate_suite_distillation_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.utils.rollout_allocation import RolloutAllocator, bounded_quotas
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor as Actor
from toolkits.libero_gr00t import launch_adaptive_query as launcher
from toolkits.libero_gr00t import launch_production as production

NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
QUOTAS = dict(zip(NAMES, [4, 28, 20, 12]))


def make_allocator(**overrides):
    tasks = [
        {"task": f"{suite}/{j}", "suite": suite, "task_id": i * 10 + j, "horizon": 17}
        for i, suite in enumerate(NAMES)
        for j in range(10)
    ]
    config = {
        "mode": "adaptive",
        "seed": 0,
        "warmup_rounds": 10,
        "window_rounds": 10,
        "interval_rounds": 10,
        "min_task_episodes": 1,
        "rho": 1.0,
        "epsilon": 0.0,
        "priority_power": 2.0,
        "max_quota_change": 6,
    }
    config.update(overrides)
    return RolloutAllocator(tasks, 4, 16, 1, config, dict.fromkeys(NAMES, 1.0))


@pytest.mark.parametrize("all_success", [False, True])
def test_squared_gap_replay_preserves_forty_task_coverage_and_round_timing(all_success):
    allocator = make_allocator()
    previous = [16] * 4
    decisions = []
    for round_ in range(1, 61):
        plan = allocator.plan()
        assert Counter(row["suite"] for row in plan) == dict(
            zip(NAMES, allocator.quotas)
        )
        for rank in range(4):
            assert {row["suite"] for row in plan if row["worker"] == rank} == set(NAMES)
        diagnostics = allocator.last_diagnostics
        if diagnostics["decision"]:
            decisions.append(round_)
            assert min(diagnostics["task_counts"].values()) >= 4
            gaps = diagnostics["gap"]
            total = sum(value**2 for value in gaps.values())
            probabilities = [gaps[s] ** 2 / total if total else 0.25 for s in NAMES]
            assert list(diagnostics["probabilities"].values()) == pytest.approx(
                probabilities
            )
            assert allocator.quotas == bounded_quotas(probabilities, 64, previous, 4, 6)
            if all_success:
                assert allocator.quotas == [16] * 4
        else:
            assert allocator.quotas == previous
        previous = allocator.quotas.copy()
        threshold = dict(zip(NAMES, [2, 4, 6, 8]))
        allocator.observe(
            [
                dict(
                    row,
                    success_once=all_success
                    or row["occurrence"] % 10 < threshold[row["suite"]],
                    simulation_steps=17,
                )
                for row in plan
            ]
        )
    assert decisions == [11, 21, 31, 41, 51]
    assert allocator.trajectories == 60 * 64
    restored = make_allocator()
    restored.load_state_dict(allocator.state_dict())
    assert restored.plan() == allocator.plan()


@pytest.mark.parametrize("power", [0, -1, float("nan"), float("inf")])
def test_invalid_priority_power_rejected(power):
    with pytest.raises(ValueError, match="priority_power"):
        make_allocator(priority_power=power)


def test_query_integerization_preserves_rank_budget_and_coverage():
    assert suite_query_quotas(QUOTAS, NAMES, 128) == [8, 56, 40, 24]
    assert suite_query_quotas(None, NAMES, 128) == [32] * 4
    assert suite_query_quotas(None, NAMES, 130) == [33, 33, 32, 32]
    assert sum(suite_query_quotas(QUOTAS, NAMES, 129)) == 129
    with pytest.raises(ValueError, match="at least one"):
        suite_query_quotas(dict(zip(NAMES, [0.001, 1, 1, 1])), NAMES, 128)


@pytest.mark.parametrize("synchronized", [False, True])
def test_real_sampler_obeys_counts_and_records_replacement(synchronized):
    ids = torch.repeat_interleave(torch.arange(4), 10)
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {
                    "suite_balanced_sampling": True,
                    "suite_query_weights": None if synchronized else QUOTAS,
                    "sync_suite_query_weights": synchronized,
                    "log_observation_pool_metrics": True,
                },
                "actor": {"global_batch_size": 512, "seed": 0},
            }
        ),
        rollout_batch={"forward_inputs": {"teacher_ids": ids.reshape(1, -1)}},
        _world_size=4,
        _rank=0,
        _round_suite_query_weights=QUOTAS,
        _teacher_names=lambda: NAMES,
        log_info=lambda _: None,
    )
    indices = Actor._suite_balanced_sample(worker, len(ids), 1, len(ids))
    assert torch.bincount(ids[indices], minlength=4).tolist() == [8, 56, 40, 24]
    diagnostics = worker._observation_pool_diagnostics
    assert [diagnostics[f"actor/query_draws/{s}"] for s in NAMES] == [8, 56, 40, 24]
    assert diagnostics[f"actor/query_distinct_source_rows/{NAMES[1]}"] == 10
    assert diagnostics[f"actor/replacement_fraction/{NAMES[1]}"] == pytest.approx(
        1 - 10 / 56
    )
    worker.rollout_batch["forward_inputs"]["teacher_ids"] = torch.zeros(
        1, 40, dtype=torch.long
    )
    with pytest.raises(ValueError, match="every configured teacher"):
        Actor._suite_balanced_sample(worker, 40, 1, 40)


def test_unequal_queries_native_padding_and_prefix_loss_match_global_objective():
    torch.manual_seed(19)
    counts = [8, 56, 40, 24]
    ids = torch.repeat_interleave(torch.arange(4), torch.tensor(counts))
    prefixes = torch.tensor([5, 5, 5, 10])[ids]
    x = torch.randn(128, 10, 7)
    teacher = torch.randn_like(x)
    times = torch.linspace(0.1, 4.0, 128)
    theta = torch.tensor(0.3, requires_grad=True)
    # Independent oracle: each genuine query contributes one prefix-normalized loss.
    reference = sum(
        ((theta * x[i, :k] - teacher[i, :k]).square() * times[i]).mean()
        for i, k in enumerate(prefixes.tolist())
    ) / len(ids)
    expected_gradient = torch.autograd.grad(reference, theta)[0]
    plan = native_microbatch_plan(torch.arange(128) % 7 + 5, 2, minimum_slots=75)
    loss = 0
    for slot in plan:
        valid = torch.arange(2) < slot.valid_rows
        part, _ = compute_analytic_local_kl_loss(
            theta * x[slot.indices],
            teacher[slot.indices],
            times[slot.indices],
            loss_mask=valid[:, None],
            action_prefix_lengths=prefixes[slot.indices],
        )
        loss = loss + part * slot.valid_rows / 128
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(torch.autograd.grad(loss, theta)[0], expected_gradient)


def test_macro_metric_does_not_follow_query_composition():
    stats = torch.tensor([[8, 16, 8], [112, 224, 56], [400, 800, 40], [96, 192, 24]])
    metrics = finalize_suite_statistics(stats, NAMES, weighted=True)
    assert metrics["actor/macro_velocity_mse"] == 4.25
    assert metrics["actor/macro_analytic_kl"] == 8.5
    stats[0] = 0
    assert "actor/macro_velocity_mse" not in finalize_suite_statistics(
        stats, NAMES, True
    )


@pytest.mark.parametrize("repeats", [1, 2])
def test_first_teacher_and_training_receive_same_actual_quotas(tmp_path, repeats):
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"runner": {}, "algorithm": {"sync_suite_query_weights": True}}
    )
    runner.rollout_allocator = SimpleNamespace(
        suites=NAMES, quotas=list(QUOTAS.values()), plan=lambda: []
    )
    runner.global_step = 0
    runner.max_steps = repeats
    runner.updates_per_rollout = repeats
    runner.optimizer_update_steps = True
    runner.weight_sync_interval = 1
    runner.overlap_env_bootstrap = False
    runner.reward = None
    runner.timer = lambda *args, **kwargs: nullcontext()
    runner._should_profile_step = lambda _: False
    runner.env_channel = runner.rollout_channel = runner.actor_channel = None
    runner.reward_channel = None
    runner.update_rollout_weights = lambda: None
    events = []
    runner._record_allocation_round = lambda: events.append("observe")
    handle = SimpleNamespace(wait=lambda: None)

    class StopAfterDispatch(Exception):
        pass

    def prepare(**kwargs):
        events.append(("teacher", kwargs))
        return handle

    def train(**kwargs):
        events.append(("train", kwargs))
        raise StopAfterDispatch

    runner.actor = SimpleNamespace(
        set_global_step=lambda _: None,
        recv_rollout_trajectories=lambda **_: handle,
        compute_advantages_and_returns=prepare,
        run_training=train,
    )
    runner.rollout = SimpleNamespace(
        set_global_step=lambda _: None, generate=lambda **_: handle
    )
    runner.env = SimpleNamespace(
        set_rollout_allocation=lambda *args: handle, interact=lambda **kwargs: handle
    )
    with pytest.raises(StopAfterDispatch):
        runner.run()
    assert events == [
        "observe",
        ("teacher", {"suite_query_weights": QUOTAS}),
        ("train", {"suite_query_weights": QUOTAS}),
    ]


def test_real_first_and_refresh_teacher_calls_share_quotas_and_refresh_noise(
    monkeypatch,
):
    from test_flow_refresh import bind, make_worker

    from rlinf.workers.actor import fsdp_actor_worker as module

    worker, pool = make_worker(monkeypatch)
    worker.cfg.algorithm.sync_suite_query_weights = True
    worker.cfg.algorithm.flow_refresh_updates = 2
    worker.cfg.algorithm.native_text_microbatches = True
    worker.cfg.actor.micro_batch_size = 2
    worker.cfg.runner = {"step_unit": "optimizer_update"}
    pool["forward_inputs"]["eagle_attention_mask"] = torch.tensor(
        [[[True, True, False], [True, True, True], [True, False, False]]] * 3
    )
    bind(worker, "_native_microbatch_plan")
    monkeypatch.setattr(module, "all_reduce_int", lambda value, op: value + 2)
    weights = {"a": 1, "b": 3, "c": 2}
    inspect.unwrap(Actor.compute_advantages_and_returns)(
        worker, suite_query_weights=weights
    )
    worker.optimizer_steps = 0
    seen = []

    def update():
        query = worker.teacher_queries[-1]
        seen.append(torch.bincount(query["teacher_ids"].reshape(-1)).tolist())
        worker.optimizer_steps += 1
        worker.lr_scheduler.last_epoch += 1
        worker.model.weight.data.add_(1)
        return {"actor/velocity_mse": 1.0}

    worker._run_training_once = update
    inspect.unwrap(Actor.run_training)(worker, suite_query_weights=weights)
    assert seen == [[2, 6, 4]] * 2
    assert len(worker.teacher_queries) == 2
    assert worker._sample_draw_count == worker._timestep_draw_count == 2
    assert set(worker.model.calls) == {2.0}  # Updated student generated the refresh.
    refreshed = worker.teacher_queries[-1]
    assert refreshed["_nft_x_t"].unique().numel() == 12
    assert pool["forward_inputs"]["chains"].count_nonzero() == 0
    worker.rollout_batch = pool
    with pytest.raises(ValueError, match="changed after the first teacher"):
        inspect.unwrap(Actor.run_training)(
            worker, suite_query_weights={"a": 2, "b": 2, "c": 2}
        )


@pytest.mark.parametrize("arm", ["m0", "m1", "m2", "m3"])
def test_new_recipe_changes_only_method_not_tested_training_parameters(
    tmp_path, monkeypatch, arm
):
    monkeypatch.setenv("OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    paths = {name: tmp_path / name for name in ["student", *NAMES]}
    repeats = 1 if arm in {"m0", "m3"} else 2
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / arm,
        gpus=[0, 1, 2, 3],
        config_name=f"libero_mopd_adaptive_query_{arm}",
        updates=100 * repeats,
        save_every_updates=10 * repeats,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        seed=0,
    )
    new = production.build_config(args, paths)
    args.config_name = f"libero_mopd_formal_t64_{arm}"
    args.save_every_updates = 25 * repeats  # Historical recipes remain unchanged.
    old = production.build_config(args, paths)
    adaptive = arm in {"m2", "m3"}
    assert new.algorithm.sync_suite_query_weights == adaptive
    assert new.algorithm.sync_suite_loss_weights is False
    assert new.algorithm.suite_loss_weights is None
    control = new.runner.rollout_allocation
    assert (
        control.warmup_rounds,
        control.window_rounds,
        control.interval_rounds,
        control.min_task_episodes,
        control.max_quota_change,
    ) == (10, 10, 10, 1, 6)
    assert new.runner.max_steps // repeats == 100
    assert new.runner.save_interval // repeats == 10
    forwarded = launcher.query_arguments(["--arm", arm, "--dry-run"])
    assert forwarded[forwarded.index("--save-every-updates") + 1] == str(10 * repeats)
    before, after = [OmegaConf.to_container(c, resolve=True) for c in (old, new)]
    allowed = {
        ("runner", "save_interval"),
        ("runner", "logger", "experiment_name"),
        ("runner", "logger", "wandb_group"),
        ("runner", "logger", "wandb_run_name"),
        ("runner", "rollout_allocation", "rho"),
        ("runner", "rollout_allocation", "epsilon"),
        ("runner", "rollout_allocation", "priority_power"),
        ("runner", "rollout_allocation", "warmup_rounds"),
        ("runner", "rollout_allocation", "window_rounds"),
        ("runner", "rollout_allocation", "min_task_episodes"),
        ("algorithm", "sync_suite_loss_weights"),
        ("algorithm", "suite_loss_weights"),
        ("algorithm", "sync_suite_query_weights"),
        ("algorithm", "suite_query_weights"),
    }
    for path in allowed:
        for cfg in (before, after):
            node = cfg
            for part in path[:-1]:
                node = node[part]
            node.pop(path[-1], None)
    assert before == after


def test_dedicated_launcher_defaults_to_t64_and_rejects_recipe_overrides():
    argv = launcher.query_arguments(["--arm", "m2", "--dry-run"])
    assert argv[argv.index("--config-name") + 1] == "libero_mopd_adaptive_query_m2"
    assert argv[argv.index("--updates") + 1] == "200"
    with pytest.raises(ValueError, match="trajectory-budget64"):
        launcher.query_arguments(["--arm", "m2", "--trajectory-budget", "32"])
    with pytest.raises(ValueError, match="override"):
        launcher.query_arguments(["--arm", "m1", "--global-batch-size", "256"])


@pytest.mark.parametrize("field", ["suite_loss_weights", "sync_suite_loss_weights"])
def test_validation_rejects_double_weighting(field):
    cfg = OmegaConf.create(
        {
            "algorithm": dict(
                sync_suite_query_weights=True,
                log_suite_velocity_metrics=True,
                suite_balanced_sampling=True,
                reward_type="chunk_level",
                loss_type="analytic_local_kl",
                **{field: QUOTAS if field == "suite_loss_weights" else True},
            ),
            "runner": {"rollout_allocation": {"enabled": True}},
            "rollout": {"expert_models": dict.fromkeys(NAMES, {})},
            "actor": {"training_backend": "fsdp", "global_batch_size": 512},
        }
    )
    with pytest.raises(ValueError, match="exclusive"):
        validate_suite_distillation_cfg(cfg)


@pytest.mark.parametrize("coverage", [1, 2, 3, 4])
def test_coverage_gate_does_not_change_valid_fixed_layout_plans(coverage):
    a = make_allocator(min_task_episodes=coverage)
    reference = make_allocator(min_task_episodes=1)
    for round_ in range(1, 51):
        plan = a.plan()
        assert plan == reference.plan()
        assert a.last_diagnostics["reason"] != "insufficient_task_coverage"
        records = [
            dict(row, success_once=row["suite"] == NAMES[-1], simulation_steps=17)
            for row in plan
        ]
        a.observe(records)
        reference.observe(records)
    assert a.quotas[-1] == 4


def test_uniform_and_adaptive_share_first_ten_physical_rounds():
    adaptive = make_allocator()
    uniform = make_allocator(mode="uniform")
    for _ in range(10):
        plan = adaptive.plan()
        assert plan == uniform.plan()
        records = [
            dict(row, success_once=row["suite"] != NAMES[0], simulation_steps=17)
            for row in plan
        ]
        adaptive.observe(records)
        uniform.observe(records)
    adaptive.plan()
    uniform.plan()
    assert adaptive.last_diagnostics["decision"]
    assert adaptive.quotas != uniform.quotas
    assert max(abs(a - b) for a, b in zip(adaptive.quotas, uniform.quotas)) <= 6
