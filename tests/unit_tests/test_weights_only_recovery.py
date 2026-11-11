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
"""Verify explicit recovery without silently claiming to restore Adam moments."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.utils.weights_only_recovery import (
    restore_fresh_optimizer_schedule,
    validate_weights_only_initialization,
)


def make_optimizer():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / 4, 1.0)
    )
    return parameter, optimizer, scheduler


def test_continues_lr_and_updates_without_repeating_warmup():
    parameter, optimizer, scheduler = make_optimizer()
    restore_fresh_optimizer_schedule(optimizer, scheduler, 70, 4)
    assert not optimizer.state
    assert scheduler.last_epoch == 70
    assert scheduler.state_dict()["_step_count"] == 71
    assert optimizer.param_groups[0]["lr"] == 1e-5
    parameter.sum().backward()
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 71
    assert scheduler.state_dict()["_step_count"] == 72
    assert optimizer.param_groups[0]["lr"] == 1e-5
    # Adam's bias-correction counter reflects newly initialized moments, not 71.
    assert optimizer.state[parameter]["step"].item() == 1


def test_refuses_to_overwrite_existing_optimizer_state():
    parameter, optimizer, scheduler = make_optimizer()
    parameter.sum().backward()
    optimizer.step()
    with pytest.raises(ValueError, match="fresh optimizer"):
        restore_fresh_optimizer_schedule(optimizer, scheduler, 70, 4)


def test_rejects_warmup_or_nonconstant_schedule():
    _, optimizer, scheduler = make_optimizer()
    with pytest.raises(ValueError, match="warmup"):
        restore_fresh_optimizer_schedule(optimizer, scheduler, 2, 4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    with pytest.raises(ValueError, match="constant"):
        restore_fresh_optimizer_schedule(optimizer, scheduler, 70, 4)


def test_model_must_actually_load_checkpoint_weights(tmp_path):
    source = tmp_path / "source"
    weight = source / "actor/model_state_dict/full_weights.pt"
    weight.parent.mkdir(parents=True)
    torch.save({"weight": torch.ones(2)}, weight)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "actor").symlink_to(source / "actor", target_is_directory=True)
    state = {"weights_only": True}
    validate_weights_only_initialization(str(source), str(bundle), state)
    wrong = tmp_path / "wrong"
    (wrong / "actor/model_state_dict").mkdir(parents=True)
    torch.save(
        {"weight": torch.zeros(2)}, wrong / "actor/model_state_dict/full_weights.pt"
    )
    with pytest.raises(ValueError, match="initialization"):
        validate_weights_only_initialization(str(source), str(wrong), state)
    with pytest.raises(ValueError, match="exported"):
        validate_weights_only_initialization(str(source), str(bundle), {})


def test_actual_actor_recovery_entrypoint_preserves_global_count():
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    _, optimizer, scheduler = make_optimizer()
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "runner": {"resume_mode": "weights_only_round_boundary"},
                "actor": {"optim": {"lr_scheduler": "constant", "lr_warmup_steps": 4}},
            }
        ),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        _rank=0,
        enable_offload=False,
        log_info=lambda message: None,
    )
    state = EmbodiedFSDPActor.restore_opd_update_counters(worker, 70)
    assert state["optimizer_state_entries"] == 0
    assert state["optimizer_moments_restored"] is False
    assert state["scheduler_last_epoch"] == worker.optimizer_steps == 70
    assert worker._sample_draw_count == 70
    assert state["lr"] == 1e-5
    worker.cfg.runner.resume_mode = "adaptive_round_boundary"
    with pytest.raises(ValueError, match="missing or mismatched"):
        EmbodiedFSDPActor.restore_opd_update_counters(worker, 70)


def test_discarded_interactions_are_counted_and_incomplete_cost_is_explicit():
    from rlinf.runners.embodied_runner import EmbodiedRunner

    runner = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "runner": {
                    "recovery_interactions": {
                        "rollout_rounds": 4,
                        "trajectories": 128,
                        "simulation_steps": 43115,
                        "cost_complete": False,
                    }
                }
            }
        ),
        rollout_allocator=SimpleNamespace(
            rounds=36, trajectories=1152, simulation_steps=440000
        ),
        teacher_calibration_budget={"trajectories": 80, "simulation_steps": 10000},
    )
    result = EmbodiedRunner._allocation_interaction_metrics(runner)
    assert result["interaction/recovery_cost_complete"] == 0
    assert result["interaction/rollout_rounds"] == 36
    assert result["interaction/total_training_rollout_rounds"] == 40
    assert result["interaction/total_trajectories"] == 1152 + 128 + 80
    assert result["interaction/total_simulation_steps"] == 440000 + 43115 + 10000
