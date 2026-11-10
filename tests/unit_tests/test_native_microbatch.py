# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check exact sample coverage and masked loss/gradient across native buckets."""

import pytest
import torch

from rlinf.algorithms.native_microbatch import (
    native_microbatch_plan,
    select_native_microbatch,
)


def test_buckets_cover_draws_and_keep_native_lengths():
    lengths = torch.tensor([5, 8, 5, 5, 10, 8, 5, 8, 8, 5])
    plans = [
        native_microbatch_plan(lengths, 4),
        native_microbatch_plan(torch.arange(1, 11), 4),
    ]
    slots = max(map(len, plans))
    plan = native_microbatch_plan(lengths, 4, slots)
    genuine = []
    for slot in plan:
        assert len(slot.indices) == 4
        assert lengths[slot.indices].unique().numel() == 1
        genuine.extend(slot.indices[: slot.valid_rows].tolist())
    assert sorted(genuine) == list(range(10))
    assert len(plan) == slots
    assert any(slot.valid_rows == 0 for slot in plan)


@pytest.mark.parametrize("weighted", [False, True])
def test_masked_bucket_loss_and_gradient_equal_full_objective(weighted):
    from rlinf.algorithms.losses import compute_analytic_local_kl_loss

    torch.manual_seed(13)
    rows = 10
    x = torch.randn(rows, 2, 3)
    target = torch.randn_like(x)
    tau_weights = torch.linspace(0.1, 3, rows)
    coefficients = torch.linspace(0.5, 1.5, rows) if weighted else None
    w = torch.tensor(0.3, requires_grad=True)
    direct, _ = compute_analytic_local_kl_loss(
        w * x, target, tau_weights, sample_weights=coefficients
    )
    direct.backward()
    gradient = w.grad.clone()
    w.grad = None
    plan = native_microbatch_plan(torch.tensor([5, 8, 5, 5, 10, 8, 5, 8, 8, 5]), 4, 10)
    total = 0.0
    data = {"x": x, "target": target, "tau": tau_weights, "id": torch.arange(rows)}
    for slot in plan:
        batch = select_native_microbatch(data, slot)
        mask = torch.zeros(4, 2, dtype=torch.bool)
        mask[: slot.valid_rows] = True
        loss, _ = compute_analytic_local_kl_loss(
            w * batch["x"],
            batch["target"],
            batch["tau"],
            loss_mask=mask,
            sample_weights=coefficients[slot.indices] if weighted else None,
        )
        scaled = loss * slot.valid_rows / rows
        scaled.backward()
        total += scaled.detach().item()
    assert total == pytest.approx(direct.item(), rel=2e-6)
    torch.testing.assert_close(w.grad, gradient)


def test_launcher_native_microbatch_is_opt_in(tmp_path):
    from types import SimpleNamespace

    from test_libero_production_launcher import fixture_paths

    from toolkits.libero_gr00t import launch_production as launcher

    python, models = fixture_paths(tmp_path)
    paths = {
        "student": models / "student",
        **{"libero_" + key: models / key for key in ["spatial", "object", "goal"]},
        "libero_10": models / "long",
    }
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=2,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
        global_batch_size=512,
    )
    original = launcher.build_config(args, paths)
    assert original.actor.micro_batch_size == 1
    assert "native_text_microbatches" not in original.algorithm
    args.native_microbatch_size = 4
    bucketed = launcher.build_config(args, paths)
    assert bucketed.actor.micro_batch_size == 4
    assert bucketed.algorithm.native_text_microbatches
    assert bucketed.actor.global_batch_size == original.actor.global_batch_size
    assert bucketed.actor.optim.lr == original.actor.optim.lr
    assert bucketed.actor.model == original.actor.model
    args.global_batch_size = 48
    with pytest.raises(ValueError, match="Global batch"):
        launcher.build_config(args, paths)


@pytest.mark.parametrize("trajectories", [32, 64, 96])
def test_parallel_trajectory_override_keeps_learning_recipe(tmp_path, trajectories):
    from types import SimpleNamespace

    from test_libero_production_launcher import fixture_paths

    from rlinf.utils.rollout_allocation import RolloutAllocator
    from toolkits.libero_gr00t import launch_production as launcher

    python, models = fixture_paths(tmp_path)
    paths = {
        "student": models / "student",
        **{"libero_" + key: models / key for key in ["spatial", "object", "goal"]},
        "libero_10": models / "long",
    }
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=60,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
        global_batch_size=512,
        native_microbatch_size=4,
        learning_rate=3e-5,
        trajectories_per_round=trajectories,
    )
    cfg = launcher.build_config(args, paths)
    assert cfg.env.train.total_num_envs == trajectories
    assert cfg.env.train.rollout_epoch == 1
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.micro_batch_size == 4
    assert cfg.actor.optim.lr == 3e-5
    assert cfg.algorithm.flow_refresh_updates == 2
    tasks = [
        {"suite": suite, "task": suite + str(i), "task_id": j * 10 + i}
        for j, suite in enumerate(paths)
        if suite != "student"
        for i in range(10)
    ]
    # Routing labels match the four actual suite names, independent of task IDs.
    allocator = RolloutAllocator(
        tasks, 4, trajectories // 4, 1, cfg.runner.rollout_allocation
    )
    plan = allocator.plan()
    assert len(plan) == trajectories
    assert all(
        sum(row["worker"] == rank for row in plan) == trajectories // 4
        for rank in range(4)
    )
    assert all(
        sum(row["suite"] == suite for row in plan) == trajectories // 4
        for suite in paths
        if suite != "student"
    )
    args.trajectories_per_round = 24
    with pytest.raises(ValueError, match="trajectories-per-round"):
        launcher.build_config(args, paths)


def test_launch_headroom_and_memory_logging_are_explicit(tmp_path):
    from types import SimpleNamespace

    from test_libero_production_launcher import fixture_paths

    from toolkits.libero_gr00t import launch_production as launcher

    python, models = fixture_paths(tmp_path)
    paths = {
        "student": models / "student",
        **{"libero_" + key: models / key for key in ["spatial", "object", "goal"]},
        "libero_10": models / "long",
    }
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=40,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
        global_batch_size=512,
        native_microbatch_size=2,
        learning_rate=3e-5,
        minimum_free_gib=26,
        log_cuda_memory=True,
    )
    cfg = launcher.build_config(args, paths)
    assert cfg.algorithm.log_cuda_memory
    assert cfg.actor.micro_batch_size == 2 and cfg.actor.global_batch_size == 512
    assert cfg.actor.optim.lr == 3e-5
    for invalid in [0, -1, float("nan"), float("inf")]:
        args.minimum_free_gib = invalid
        with pytest.raises(ValueError, match="minimum-free-gib"):
            launcher.build_config(args, paths)
