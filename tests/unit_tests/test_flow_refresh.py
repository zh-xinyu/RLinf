# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU checks for observation reuse, current-flow labels and update accounting."""

import inspect
import json
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.workers.actor import fsdp_actor_worker as module

Actor = module.EmbodiedFSDPActor


def bind(worker, name, unwrap=False):
    method = getattr(Actor, name)
    if unwrap:
        method = inspect.unwrap(method)
    setattr(worker, name, MethodType(method, worker))


class FlowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, forward_type, forward_inputs):
        assert forward_type == module.ForwardType.FLOW_REFRESH
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append(float(self.weight))
        size = len(forward_inputs["teacher_ids"])
        noise = torch.randn(size, 1, 1, 1)
        return {"chains": noise + self.weight * torch.arange(5).view(1, 5, 1, 1)}


def make_worker(monkeypatch):
    monkeypatch.setattr(module, "clear_memory", lambda: None)
    monkeypatch.setattr(module, "all_reduce_dict", lambda data, op: data)
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "flow_refresh_updates": 4,
                "suite_balanced_sampling": True,
                "loss_type": "analytic_local_kl",
                "adv_type": "none",
            },
            "actor": {
                "seed": 0,
                "global_batch_size": 12,
                "micro_batch_size": 4,
                "model": {"model_type": "gr00t"},
            },
            "rollout": {"expert_models": {"a": {}, "b": {}, "c": {}}},
        }
    )
    mask = torch.ones(3, 3, 1, dtype=torch.bool)
    mask[-1] = False
    pool = {
        "forward_inputs": {
            "teacher_ids": torch.arange(3).repeat(3, 1),
            "observation_id": torch.arange(9).reshape(3, 3),
            "chains": torch.zeros(3, 3, 5, 1, 1),
        },
        "prev_logprobs": torch.zeros(3, 3, 1),
        "loss_mask": mask,
        "loss_mask_sum": torch.ones_like(mask, dtype=torch.long),
        "dones": torch.zeros(4, 3, 2, dtype=torch.bool),
    }
    worker = SimpleNamespace(
        cfg=cfg,
        rollout_batch=pool,
        _world_size=1,
        _rank=0,
        version=0,
        enable_offload=False,
        is_weight_offloaded=False,
        model=FlowModel(),
        device=torch.device("cpu"),
        amp_context=nullcontext(),
        log_info=lambda message: None,
        _teacher_names=lambda: ["a", "b", "c"],
        _velocity_distill_flow_spec=lambda: (4, True, torch.arange(4)),
        _flow_sde_transition_spec=lambda: (0.5, None, None, None),
        lr_scheduler=SimpleNamespace(last_epoch=0),
    )
    for name in ("_suite_balanced_sample", "_prepare_flow_refresh_batch"):
        bind(worker, name)
    bind(worker, "compute_teacher_velocities", unwrap=True)
    worker.teacher_queries = []

    def teacher_pass(inputs, size):
        assert size == 12
        worker.teacher_queries.append(inputs)
        return inputs["_nft_x_t"] + inputs["teacher_ids"][:, None, None]

    worker._teacher_velocity_pass = teacher_pass
    return worker, pool


def test_refresh_resamples_full_pool_and_keeps_duplicate_flow_labels(monkeypatch):
    worker, pool = make_worker(monkeypatch)
    torch.manual_seed(17)
    worker._prepare_flow_refresh_batch()
    batch = worker.rollout_batch
    ids = batch["forward_inputs"]["observation_id"].flatten()
    assert set(ids.tolist()) == set(range(6))  # Excludes invalid last time row.
    assert ids.unique().numel() < ids.numel()  # Shortfall causes duplicate draws.
    assert torch.bincount(
        batch["forward_inputs"]["teacher_ids"].flatten()
    ).tolist() == [4, 4, 4]
    x_t = batch["nft_x_t"][0]
    torch.testing.assert_close(
        batch["teacher_velocities"][0],
        x_t + batch["forward_inputs"]["teacher_ids"][0, :, None, None],
    )
    for value in ids.unique():
        rows = ids == value
        if rows.sum() > 1:
            assert x_t[rows].unique().numel() == rows.sum()
    assert not x_t.requires_grad
    assert pool["forward_inputs"]["chains"].count_nonzero() == 0
    assert worker._sample_draw_count == 1  # No second observation subsample.
    worker.rollout_batch = pool
    worker.model.weight.data.fill_(2)
    worker._prepare_flow_refresh_batch()
    assert worker.model.calls == [1.0] * 3 + [2.0] * 3
    assert worker._sample_draw_count == worker._timestep_draw_count == 2
    assert not torch.equal(
        ids, worker.rollout_batch["forward_inputs"]["observation_id"].flatten()
    )


def test_r4_uses_original_first_then_three_current_student_refreshes(monkeypatch):
    worker, pool = make_worker(monkeypatch)
    seen = []

    def update():
        seen.append(worker.rollout_batch)
        worker.lr_scheduler.last_epoch += 1
        worker.model.weight.data.add_(1)
        worker.rollout_batch = {"sampled_only": True}
        return {"loss": float(worker.lr_scheduler.last_epoch)}

    worker._run_training_once = update
    metrics = inspect.unwrap(Actor.run_training)(worker)
    assert seen[0] is pool and len(seen) == 4
    assert worker.model.calls == [2.0] * 3 + [3.0] * 3 + [4.0] * 3
    assert metrics["refresh/optimizer_updates"] == 4
    assert metrics["refresh/physical_rollouts"] == 1
    assert metrics["refresh/real_env_steps_this_rollout"] == 18
    assert metrics["loss"] == 2.5
    assert metrics["refresh/update_4/loss"] == 4
    assert len(worker.teacher_queries) == 3


def test_r1_preserves_original_training_path(monkeypatch):
    worker, pool = make_worker(monkeypatch)
    worker.cfg.algorithm.flow_refresh_updates = 1
    worker._run_training_once = lambda: {"base": 1}
    assert inspect.unwrap(Actor.run_training)(worker) == {"base": 1}
    assert worker.rollout_batch is pool and not worker.model.calls


def test_controlled_refresh_does_not_consume_training_rng(monkeypatch):
    worker, _ = make_worker(monkeypatch)
    worker.cfg.runner = {"step_unit": "optimizer_update"}
    before = torch.random.get_rng_state().clone()
    worker._prepare_flow_refresh_batch()
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_controlled_refresh_accepts_integer_cuda_device(monkeypatch):
    worker, _ = make_worker(monkeypatch)
    worker.cfg.runner = {"step_unit": "optimizer_update"}
    worker.device = 0
    forked, seeds = [], []
    monkeypatch.setattr(module, "put_tensor_device", lambda data, device: data)
    monkeypatch.setattr(
        torch.random,
        "fork_rng",
        lambda devices: (forked.append(devices), nullcontext())[1],
    )
    monkeypatch.setattr(
        torch.cuda, "default_generators", (SimpleNamespace(manual_seed=seeds.append),)
    )
    worker._prepare_flow_refresh_batch()
    assert forked == [[0]]
    assert seeds == [104729]


def test_r2_records_two_distinct_updates(monkeypatch):
    worker, _ = make_worker(monkeypatch)
    worker.cfg.runner = {"step_unit": "optimizer_update"}
    worker.cfg.algorithm.flow_refresh_updates = 2
    worker.optimizer_steps = 0

    def update():
        worker.optimizer_steps += 1
        worker.lr_scheduler.last_epoch += 1
        worker.model.weight.data.add_(1)
        return {"actor/velocity_mse": 1.0 / worker.optimizer_steps}

    worker._run_training_once = update
    metrics = inspect.unwrap(Actor.run_training)(worker)
    assert worker.optimizer_steps == metrics["refresh/optimizer_updates"] == 2
    assert metrics["refresh/update_1/actor/velocity_mse"] == 1.0
    assert metrics["refresh/update_2/actor/velocity_mse"] == 0.5
    assert worker.model.calls == [2.0] * 3


@pytest.mark.parametrize("repeats", [1, 2, 4])
def test_runner_counts_rollouts_and_logs_each_optimizer_update(tmp_path, repeats):
    from rlinf.runners.embodied_runner import EmbodiedRunner

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {}})
    runner.global_step = 0
    runner.max_steps = 200
    runner.updates_per_rollout = repeats
    runner.optimizer_update_steps = True
    runner.cumulative_env_steps = 0
    runner.weight_sync_interval = 1
    runner.overlap_env_bootstrap = False
    runner.reward = None
    runner.timer = lambda *args, **kwargs: nullcontext()
    runner._should_profile_step = lambda step: False
    runner.env_channel = runner.rollout_channel = runner.actor_channel = None
    runner.reward_channel = None
    actor_versions, rollout_versions, collected, logged, saved = [], [], [], [], []
    handle = SimpleNamespace(wait=lambda: None)
    runner.update_rollout_weights = lambda: None
    runner._finish_run = lambda: None
    runner._log_step_metrics = lambda **kwargs: None

    def checkpoint(step):
        assert step == runner.global_step
        if step % 20 == 0:
            saved.append(step)
        return {}

    runner._maybe_eval_and_checkpoint = checkpoint
    runner.metric_logger = SimpleNamespace(
        log_path=str(tmp_path), log=lambda row, step: logged.append((step, row))
    )

    def interact(**kwargs):
        collected.append(runner.global_step)
        return handle

    def train():
        end = runner.global_step + repeats
        return SimpleNamespace(
            wait=lambda: [
                {
                    "refresh/optimizer_updates": end,
                    "refresh/real_env_steps_this_rollout": 48,
                    **{
                        f"refresh/update_{i}/actor/velocity_mse{suffix}": 1.0 / i
                        for i in range(1, repeats + 1)
                        for suffix in (
                            "",
                            "/atomic_seen",
                            "/composite_seen",
                            "/composite_unseen",
                        )
                    },
                }
            ]
        )

    runner.actor = SimpleNamespace(
        set_global_step=actor_versions.append,
        recv_rollout_trajectories=lambda **kwargs: handle,
        compute_advantages_and_returns=lambda: handle,
        run_training=train,
    )
    runner.rollout = SimpleNamespace(
        set_global_step=rollout_versions.append, generate=lambda **kwargs: handle
    )
    runner.env = SimpleNamespace(interact=interact)
    runner.run()
    assert runner.global_step == 200
    assert len(collected) == 200 // repeats
    assert actor_versions == list(range(200 // repeats))
    assert rollout_versions == list(range(0, 200, repeats))
    assert saved == list(range(20, 201, 20))
    assert [step for step, _ in logged] == list(range(1, 201))
    assert [row["train/actor/velocity_mse"] for _, row in logged] == [
        1.0 / i for i in range(1, repeats + 1)
    ] * (200 // repeats)
    assert runner.cumulative_env_steps == 48 * (200 // repeats)
    rows = [
        json.loads(line)
        for line in (tmp_path / "optimizer_updates.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["rollout_count"] == 200 // repeats
    for i, row in enumerate(rows):
        assert ("train/new_rollout_velocity_mse" in row) == (i % repeats == 0)
        assert row["train/refresh/is_refreshed"] == int(i % repeats > 0)
        for suite in ("atomic_seen", "composite_seen", "composite_unseen"):
            assert row[f"train/actor/velocity_mse/{suite}"] == pytest.approx(
                1 / (i % repeats + 1)
            )
            assert (f"train/new_rollout_velocity_mse/{suite}" in row) == (
                i % repeats == 0
            )


def test_controlled_base_updates_once_without_refresh(monkeypatch):
    worker, pool = make_worker(monkeypatch)
    worker.cfg.runner = {"step_unit": "optimizer_update"}
    worker.cfg.algorithm.flow_refresh_updates = 1
    worker.optimizer_steps = 0

    def update():
        assert worker.rollout_batch is pool
        worker.optimizer_steps += 1
        worker.lr_scheduler.last_epoch += 1
        return {"actor/velocity_mse": 0.5}

    worker._run_training_once = update
    metrics = inspect.unwrap(Actor.run_training)(worker)
    assert worker.optimizer_steps == metrics["refresh/optimizer_updates"] == 1
    assert not worker.model.calls and not worker.teacher_queries


def test_resume_counters_keep_optimizer_and_scheduler_progress():
    worker = SimpleNamespace(
        cfg=OmegaConf.create({"runner": {"resume_mode": "adaptive_round_boundary"}}),
        lr_scheduler=SimpleNamespace(last_epoch=40),
        optimizer=SimpleNamespace(
            state={"parameter": {"step": 40}}, param_groups=[{"lr": 3e-5}]
        ),
        _rank=0,
        enable_offload=False,
        log_info=lambda message: None,
    )
    result = inspect.unwrap(Actor.restore_opd_update_counters)(worker, 40)
    assert result["optimizer_updates"] == 40
    assert result["scheduler_last_epoch"] == 40
    assert worker._sample_draw_count == 40
    assert result["lr"] == 3e-5
    with pytest.raises(ValueError, match="mismatched"):
        inspect.unwrap(Actor.restore_opd_update_counters)(worker, 56)
