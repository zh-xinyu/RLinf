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
"""Tensor alignment helpers for sampled flow-transition OPD."""

import math
from typing import Optional

import torch

from rlinf.algorithms.losses import flow_sde_transition_coeffs


def sample_flow_timestep_indices(
    flat_batch_size: int,
    num_steps: int,
    *,
    seed: int,
    rank: int,
    draw_count: int,
) -> torch.Tensor:
    """Draw reproducible per-sample timesteps without repeating every update."""
    assert flat_batch_size > 0, "Flow timestep sampling needs a non-empty batch."
    assert num_steps > 0, "Flow timestep sampling needs at least one denoise step."
    assert draw_count > 0, "draw_count must advance once per supervision batch."
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(rank) + int(draw_count) * 104729)
    return torch.randint(
        0, num_steps, (flat_batch_size,), generator=generator, dtype=torch.long
    )


def count_flow_timesteps_by_suite(
    step_idx: torch.Tensor,
    num_steps: int,
    *,
    teacher_ids: Optional[torch.Tensor] = None,
    sample_idx: Optional[torch.Tensor] = None,
    num_teachers: Optional[int] = None,
) -> torch.Tensor:
    """Count the exact sampled training distribution over suite and timestep.

    Duplicate indices from suite-balanced sampling are retained because they also occur
    multiple times in the gradient batch.
    """
    steps = step_idx.reshape(-1).cpu().long()
    assert steps.numel() > 0, "Flow timestep counts need at least one sample."
    assert int(steps.min()) >= 0 and int(steps.max()) < num_steps, (
        f"flow timestep indices must be in [0, {num_steps}), got "
        f"[{int(steps.min())}, {int(steps.max())}]."
    )

    if teacher_ids is None:
        ids = torch.zeros_like(steps)
        suite_count = 1
    else:
        assert num_teachers is not None and num_teachers > 0, (
            "num_teachers is required when teacher_ids are provided."
        )
        ids = teacher_ids.reshape(-1).cpu().long()
        assert ids.numel() == steps.numel(), (
            f"teacher_ids has {ids.numel()} entries for {steps.numel()} timesteps."
        )
        assert int(ids.min()) >= 0 and int(ids.max()) < num_teachers, (
            f"teacher ids must be in [0, {num_teachers}), got "
            f"[{int(ids.min())}, {int(ids.max())}]."
        )
        suite_count = num_teachers

    if sample_idx is not None:
        rows = sample_idx.reshape(-1).cpu().long()
        assert rows.numel() > 0, "sample_idx must not select an empty training batch."
        assert int(rows.min()) >= 0 and int(rows.max()) < steps.numel(), (
            f"sample_idx must be in [0, {steps.numel()}), got "
            f"[{int(rows.min())}, {int(rows.max())}]."
        )
        steps = steps[rows]
        ids = ids[rows]

    encoded = ids * num_steps + steps
    return torch.bincount(encoded, minlength=suite_count * num_steps).reshape(
        suite_count, num_steps
    )


def build_flat_valid_transition_mask(
    flat_batch_size: int,
    time_dim: int,
    batch_dim: int,
    loss_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return valid rows in a flattened ``[time, batch]`` rollout.

    ``loss_mask`` describes transitions while flow chains describe states, so the mask's
    time axis can be one shorter. Any trailing state has no transition after it and is
    therefore invalid for distillation.
    """
    assert flat_batch_size == time_dim * batch_dim, (
        f"flat batch size {flat_batch_size} does not match "
        f"time_dim * batch_dim ({time_dim} * {batch_dim})."
    )
    valid = torch.ones(flat_batch_size, dtype=torch.bool)
    if loss_mask is None:
        return valid

    assert loss_mask.shape[1] == batch_dim, (
        f"loss_mask batch dim {loss_mask.shape[1]} does not match the rollout's "
        f"{batch_dim}; only the time axis may differ."
    )
    usable = min(loss_mask.shape[0], time_dim)
    flat_mask = loss_mask[:usable].reshape(usable * batch_dim, -1).any(dim=-1).cpu()
    valid[: usable * batch_dim] = flat_mask
    valid[usable * batch_dim :] = False
    return valid


def teacher_suite_presence(
    teacher_ids: torch.Tensor, num_teachers: int
) -> torch.Tensor:
    """Return a Boolean presence vector for routed teacher indices."""
    ids = teacher_ids.reshape(-1).cpu().long()
    assert num_teachers > 0, "At least one routed teacher must be configured."
    assert ids.numel() > 0, "A routed rollout must contain teacher ids."
    assert int(ids.min()) >= 0 and int(ids.max()) < num_teachers, (
        f"teacher ids must be in [0, {num_teachers}), got "
        f"[{int(ids.min())}, {int(ids.max())}]."
    )
    present = torch.zeros(num_teachers, dtype=torch.bool)
    present[torch.unique(ids)] = True
    return present


def select_flow_opd_logprobs(
    prev_logprobs: torch.Tensor,
    step_idx: torch.Tensor,
    *,
    joint_logprob: bool,
) -> torch.Tensor:
    """Select the rollout log-prob of each sample's stochastic denoise step."""
    assert prev_logprobs.dim() >= 5, (
        "flow OPD expects prev_logprobs shaped "
        "[time, batch, denoise_step, action_chunk, action_dim]."
    )
    time_dim, batch_dim = prev_logprobs.shape[:2]
    flat_prev = prev_logprobs.reshape(-1, *prev_logprobs.shape[2:])
    assert step_idx.numel() == flat_prev.shape[0], (
        f"step_idx has {step_idx.numel()} entries for {flat_prev.shape[0]} samples."
    )

    offset = 1 if joint_logprob else 0
    gather_idx = step_idx.reshape(-1).to(device=flat_prev.device, dtype=torch.long)
    gather_idx = gather_idx + offset
    assert int(gather_idx.min()) >= 0, (
        f"denoise step {int(gather_idx.min())} is negative."
    )
    assert int(gather_idx.max()) < flat_prev.shape[1], (
        f"denoise step {int(gather_idx.max())} is out of range for prev_logprobs "
        f"with {flat_prev.shape[1]} scored steps."
    )

    rows = torch.arange(flat_prev.shape[0], device=flat_prev.device)
    picked = flat_prev[rows, gather_idx]
    return picked.reshape(time_dim, batch_dim, *picked.shape[1:])


def select_opd_metric_rows(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prev_logprobs: torch.Tensor,
    *,
    sample_idx: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exactly the rows contributing to the OPD training metric.

    Suite-balanced sampling may contain duplicate indices. They are deliberately kept so
    this diagnostic has the same weighting as the actual gradient.
    """
    assert teacher_logprobs.shape == prev_logprobs.shape, (
        f"teacher_logprobs {teacher_logprobs.shape} must match "
        f"prev_logprobs {prev_logprobs.shape}."
    )
    time_dim, batch_dim = teacher_logprobs.shape[:2]
    flat_size = time_dim * batch_dim
    ids = teacher_ids.reshape(-1).cpu().long()
    assert ids.numel() == flat_size, (
        f"teacher_ids has {ids.numel()} entries for {flat_size} samples."
    )
    reward = (teacher_logprobs.float() - prev_logprobs.float()).reshape(flat_size, -1)

    if sample_idx is not None:
        rows = sample_idx.reshape(-1).cpu().long()
    else:
        valid = build_flat_valid_transition_mask(
            flat_size, time_dim, batch_dim, loss_mask
        )
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
    return ids[rows], reward[rows]


def flow_teacher_transition_logprobs(
    *,
    flat_chains: torch.Tensor,
    x_cur: torch.Tensor,
    step_idx: torch.Tensor,
    teacher_velocities: torch.Tensor,
    num_steps: int,
    noise_level: float,
    action_chunk: int,
    action_env_dim: int,
    c_sign: float,
    time_dim: int,
    batch_dim: int,
) -> torch.Tensor:
    """Score the student's sampled Flow-SDE transition under the teacher."""
    flat_size = time_dim * batch_dim
    assert flat_chains.shape[0] == flat_size
    assert x_cur.shape[0] == flat_size
    assert teacher_velocities.shape[0] == flat_size
    assert step_idx.numel() == flat_size

    w0, w1, c, sigma = flow_sde_transition_coeffs(
        num_steps=num_steps, noise_level=noise_level
    )
    assert torch.all(sigma > 0), (
        "Sampled flow OPD needs a stochastic transition; sigma must be positive."
    )
    device = x_cur.device
    step_idx = step_idx.reshape(-1).to(device=device, dtype=torch.long)
    scale = (w0 + w1).to(device=device)[step_idx].float()[:, None, None]
    c_k = (c_sign * c).to(device=device)[step_idx].float()[:, None, None]
    sigma_k = sigma.to(device=device)[step_idx].float()[:, None, None]

    rows = torch.arange(flat_size, device=flat_chains.device)
    chain_steps = step_idx.to(flat_chains.device)
    x_next = flat_chains[rows, chain_steps + 1].float()
    x_next = x_next[:, :action_chunk, :action_env_dim]
    mu_teacher = (
        x_cur.float()[:, :action_chunk, :action_env_dim] * scale
        + teacher_velocities.float()[:, :action_chunk, :action_env_dim] * c_k
    )
    log_probs = (
        -torch.log(sigma_k)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((x_next - mu_teacher) / sigma_k) ** 2
    ).contiguous()
    return log_probs.reshape(time_dim, batch_dim, *log_probs.shape[1:])
