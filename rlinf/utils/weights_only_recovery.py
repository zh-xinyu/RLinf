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
"""Explicit, non-exact continuation when a checkpoint has no Adam state."""

from pathlib import Path

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def validate_weights_only_initialization(
    checkpoint: str, model_path: str, state: dict
) -> None:
    """Require model construction to load the requested checkpoint's weights."""
    source = Path(checkpoint) / "actor/model_state_dict/full_weights.pt"
    candidates = [
        Path(model_path) / "actor/model_state_dict/full_weights.pt",
        Path(model_path) / "model_state_dict/full_weights.pt",
    ]
    if not state.get("weights_only", False) or not source.is_file():
        raise ValueError("Weights-only continuation requires an exported checkpoint.")
    if not any(path.is_file() and path.samefile(source) for path in candidates):
        raise ValueError(
            "Model initialization must load the resume checkpoint weights."
        )


def restore_fresh_optimizer_schedule(
    optimizer: Optimizer,
    scheduler: LRScheduler,
    expected_updates: int,
    warmup_steps: int,
) -> None:
    """Continue constant LR after warmup, explicitly leaving Adam moments empty.

    This restores only scheduler counters. Model/actor/rollout RNG and optimizer
    moments are not recovered, so callers must label this as non-exact recovery.
    """
    if expected_updates <= 0 or expected_updates < warmup_steps:
        raise ValueError("Weights-only recovery must be beyond the initial warmup.")
    if optimizer.state:
        raise ValueError("Weights-only recovery expects a fresh optimizer.")
    rates = list(scheduler.base_lrs)
    if len(rates) != len(optimizer.param_groups):
        raise ValueError("Optimizer and scheduler parameter groups differ.")
    # The caller restricts this path to the constant scheduler. Verify the actual
    # schedule at the boundary too, instead of replaying optimizer-less step().
    if not hasattr(scheduler, "lr_lambdas") or any(
        law(expected_updates) != 1.0 for law in scheduler.lr_lambdas
    ):
        raise ValueError("Weights-only recovery supports constant post-warmup LR.")
    saved = scheduler.state_dict()
    saved.update(
        last_epoch=expected_updates,
        _step_count=expected_updates + 1,
        _last_lr=rates,
    )
    scheduler.load_state_dict(saved)
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate
