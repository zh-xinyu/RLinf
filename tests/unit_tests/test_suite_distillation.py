# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Suite diagnostics and grouped accumulation preserve the velocity objective."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.algorithms.losses import (
    compute_analytic_local_kl_loss,
    compute_velocity_distill_loss,
)
from rlinf.algorithms.suite_distillation import (
    SUITE_STATS_KEY,
    finalize_suite_statistics,
    group_sample_indices,
    suite_velocity_statistics,
)
from rlinf.workers.actor import fsdp_actor_worker as actor_module


@pytest.mark.parametrize("weighted", [False, True])
def test_statistics_use_valid_element_sums_and_do_not_change_gradients(weighted):
    torch.manual_seed(13)
    student = torch.randn(7, 4, 2, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    ids = torch.tensor([0, 1, 1, 2, 0, 2, 1])
    mask = torch.rand(7, 4, 1) > 0.4
    weights = torch.linspace(0.1, 1.7, 7)
    fn = compute_analytic_local_kl_loss if weighted else compute_velocity_distill_loss
    kwargs = {"kl_weight": weights} if weighted else {}
    loss, metrics = fn(
        student, teacher, loss_mask=mask, teacher_ids=ids, num_teachers=4, **kwargs
    )
    reference, _ = fn(student, teacher, loss_mask=mask, **kwargs)
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(loss, student, retain_graph=True)[0],
        torch.autograd.grad(reference, student)[0],
        rtol=0,
        atol=0,
    )
    assert teacher.grad is None
    stats = metrics[SUITE_STATS_KEY]
    assert not stats.requires_grad and stats.shape == (4, 3)
    full_mask = mask.expand_as(student)
    error = (student.detach() - teacher.detach()).square()
    weighted_error = error * weights[:, None, None] if weighted else error
    for s in range(4):
        expected = torch.stack(
            (
                (error[ids == s] * full_mask[ids == s]).sum(),
                (weighted_error[ids == s] * full_mask[ids == s]).sum(),
                full_mask[ids == s].sum(),
            )
        )
        torch.testing.assert_close(stats[s], expected.double())
    means = finalize_suite_statistics(stats, ["a", "b", "c", "empty"], weighted)
    assert means["actor/valid_velocity_elements/empty"] == 0
    assert "actor/velocity_mse/empty" not in means
    assert ("actor/analytic_kl/a" in means) == weighted


def test_grouping_preserves_selected_duplicates_and_within_suite_order():
    ids = torch.tensor([2, 0, 1, 2, 0])
    chosen = torch.tensor([3, 1, 4, 2, 1, 0])
    grouped = group_sample_indices(chosen, ids, 3)
    assert grouped.tolist() == [1, 4, 1, 2, 3, 0]
    with pytest.raises(ValueError):
        group_sample_indices(chosen, ids, 2)


class TinyVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.suite_batches = []

    def forward(self, forward_type, forward_inputs, nft_inputs, compute_values):
        self.suite_batches.append(forward_inputs["teacher_ids"].clone())
        return {"v_theta": self.linear(nft_inputs["x_t"])}


def run_actor_update(monkeypatch, order, counts):
    """Exercise the real actor loop and its actual loss/backward path on CPU."""
    monkeypatch.setattr(actor_module, "clear_memory", lambda: None)
    monkeypatch.setattr(actor_module, "all_reduce_dict", lambda value, op: value)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    torch.manual_seed(19)
    model = TinyVelocityModel()
    n = sum(counts)
    ids = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    batch = {
        "forward_inputs": {"teacher_ids": ids[None]},
        "prev_logprobs": torch.zeros(1, n, 1),
        "nft_x_t": torch.randn(1, n, 4, 3),
        "nft_timesteps": torch.rand(1, n),
        "nft_kl_weight": torch.rand(1, n),
        "teacher_velocities": torch.randn(1, n, 4, 2),
        "loss_mask": torch.ones(1, n, 1, dtype=torch.bool),
        "loss_mask_sum": torch.ones(1, n, 1, dtype=torch.long),
    }
    initial_loss, _ = compute_analytic_local_kl_loss(
        model.linear(batch["nft_x_t"][0]),
        batch["teacher_velocities"][0],
        batch["nft_kl_weight"][0],
    )
    expected_grads = torch.autograd.grad(initial_loss, tuple(model.parameters()))
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "suite_batch_order": order,
                "log_suite_velocity_metrics": True,
                "loss_type": "analytic_local_kl",
                "update_epoch": 1,
            },
            "actor": {"global_batch_size": n * 8, "micro_batch_size": 32, "seed": 0},
            "runner": {"step_unit": "optimizer_update"},
        }
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    sync_flags, gradients = [], []
    worker = SimpleNamespace(
        cfg=cfg,
        model=model,
        rollout_batch=batch,
        _sample_idx=torch.randperm(n),
        _rank=0,
        _world_size=8,
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        device=torch.device("cpu"),
        amp_context=nullcontext(),
        gradient_accumulation=n // 32,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        torch_platform=SimpleNamespace(empty_cache=lambda: None),
        grad_scaler=SimpleNamespace(scale=lambda x: x),
        _teacher_names=lambda: ["a", "b", "c"],
    )

    def before(model, is_last_micro_batch):
        sync_flags.append(is_last_micro_batch)
        return nullcontext()

    def step():
        gradients.append([p.grad.clone() for p in model.parameters()])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return norm, [3e-5]

    worker.before_micro_batch = before
    worker.optimizer_step = step
    for name in ["_run_training_once", "train_micro_batch"]:
        setattr(
            worker,
            name,
            MethodType(getattr(actor_module.EmbodiedFSDPActor, name), worker),
        )
    metrics = worker._run_training_once()
    assert len(gradients) == 1 and scheduler.last_epoch == 1
    assert sync_flags == [False] * (n // 32 - 1) + [True]
    assert len(model.suite_batches) == n // 32
    for got, expected in zip(gradients[0], expected_grads):
        torch.testing.assert_close(got, expected, rtol=2e-6, atol=1e-7)
    assert worker._suite_loss_stats is None
    return model, optimizer, metrics


@pytest.mark.parametrize("counts", [(96, 96, 96), (86, 85, 85)])
def test_actor_grouped_and_mixed_have_same_gradients_adam_update_and_metrics(
    monkeypatch, counts
):
    mixed, mixed_optim, mm = run_actor_update(monkeypatch, "mixed", counts)
    grouped, grouped_optim, gm = run_actor_update(monkeypatch, "grouped", counts)
    for a, b in zip(mixed.parameters(), grouped.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)
    for a, b in zip(mixed_optim.state.values(), grouped_optim.state.values()):
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=2e-6, atol=1e-7)
    for key in mm:
        assert gm[key] == pytest.approx(mm[key], rel=2e-6, abs=1e-7)
    if counts == (96, 96, 96):
        assert [x.unique().item() for x in grouped.suite_batches] == [0] * 3 + [
            1
        ] * 3 + [2] * 3
        assert gm["actor/analytic_kl"] == pytest.approx(
            sum(gm[f"actor/analytic_kl/{s}"] for s in ["a", "b", "c"]) / 3, rel=2e-6
        )
    # Unequal historical quotas keep sample-count weighting, not forced 1/3.
    expected = sum(
        gm[f"actor/analytic_kl/{s}"] * count
        for s, count in zip(["a", "b", "c"], counts)
    ) / sum(counts)
    assert gm["actor/analytic_kl"] == pytest.approx(expected, rel=2e-6)


def _distributed_stats_worker(rank, rendezvous, result_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    # Unequal counts and missing suites on each rank expose average-of-means bugs.
    ids = torch.tensor([0, 0, 1]) if rank == 0 else torch.tensor([1, 2])
    errors = torch.full((len(ids), 2, 1), float(rank + 1))
    stats = suite_velocity_statistics(errors, errors * 3, None, ids, 3)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if rank == 0:
        torch.save(stats, result_path)
    dist.destroy_process_group()


def test_distributed_sums_handle_unequal_rank_counts(tmp_path):
    mp.spawn(
        _distributed_stats_worker,
        args=(str(tmp_path / "init"), str(tmp_path / "stats.pt")),
        nprocs=2,
        join=True,
    )
    stats = torch.load(tmp_path / "stats.pt", weights_only=True)
    metrics = finalize_suite_statistics(stats, ["a", "b", "c"], True)
    assert [metrics[f"actor/velocity_mse/{s}"] for s in ["a", "b", "c"]] == [
        1.0,
        1.5,
        2.0,
    ]
    assert [metrics[f"actor/valid_velocity_elements/{s}"] for s in ["a", "b", "c"]] == [
        4.0,
        4.0,
        2.0,
    ]
