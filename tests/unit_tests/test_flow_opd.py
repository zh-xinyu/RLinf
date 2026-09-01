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
"""Tests for sampled flow-transition OPD tensor alignment."""

import pytest
import torch

from rlinf.algorithms.flow_opd import (
    build_flat_valid_transition_mask,
    flow_teacher_transition_logprobs,
    select_flow_opd_logprobs,
    select_opd_metric_rows,
)


@pytest.mark.parametrize("joint_logprob", [False, True])
def test_select_flow_opd_logprobs_picks_sampled_step(joint_logprob):
    step_count = 5 if joint_logprob else 4
    prev = torch.arange(2 * 2 * step_count * 3 * 2).reshape(2, 2, step_count, 3, 2)
    step_idx = torch.tensor([0, 1, 2, 3])

    selected = select_flow_opd_logprobs(
        prev, step_idx, joint_logprob=joint_logprob
    )

    flat = prev.reshape(4, step_count, 3, 2)
    offset = int(joint_logprob)
    expected = torch.stack(
        [flat[row, step + offset] for row, step in enumerate(step_idx)]
    ).reshape(2, 2, 3, 2)
    torch.testing.assert_close(selected, expected)


def test_build_flat_valid_transition_mask_drops_trailing_state():
    loss_mask = torch.tensor(
        [
            [[True], [False]],
            [[True], [True]],
        ]
    )
    valid = build_flat_valid_transition_mask(
        flat_batch_size=6,
        time_dim=3,
        batch_dim=2,
        loss_mask=loss_mask,
    )
    torch.testing.assert_close(
        valid, torch.tensor([True, False, True, True, False, False])
    )


def test_select_opd_metric_rows_matches_sampled_training_distribution():
    teacher_ids = torch.tensor([[0, 1], [0, 1]])
    teacher = torch.tensor([[[[1.0]], [[10.0]]], [[[3.0]], [[20.0]]]])
    student = torch.zeros_like(teacher)
    sample_idx = torch.tensor([3, 0, 3])

    ids, reward = select_opd_metric_rows(
        teacher_ids,
        teacher,
        student,
        sample_idx=sample_idx,
        loss_mask=None,
    )

    torch.testing.assert_close(ids, torch.tensor([1, 0, 1]))
    torch.testing.assert_close(reward[:, 0], torch.tensor([20.0, 1.0, 20.0]))


def test_select_opd_metric_rows_uses_valid_mask_without_sampling():
    teacher_ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
    teacher = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1)
    loss_mask = torch.tensor([[[True], [False]], [[False], [True]]])

    ids, reward = select_opd_metric_rows(
        teacher_ids,
        teacher,
        torch.zeros_like(teacher),
        sample_idx=None,
        loss_mask=loss_mask,
    )

    torch.testing.assert_close(ids, torch.tensor([0, 1]))
    torch.testing.assert_close(reward[:, 0], torch.tensor([0.0, 3.0]))


def test_flow_teacher_transition_logprobs_slices_gr00t_action_dim():
    time_dim, batch_dim = 2, 2
    flat_size = time_dim * batch_dim
    flat_chains = torch.randn(flat_size, 5, 16, 32)
    x_cur = torch.randn(flat_size, 16, 32)
    teacher_velocities = torch.randn(flat_size, 16, 12)
    step_idx = torch.tensor([0, 1, 2, 3])

    logprobs = flow_teacher_transition_logprobs(
        flat_chains=flat_chains,
        x_cur=x_cur,
        step_idx=step_idx,
        teacher_velocities=teacher_velocities,
        num_steps=4,
        noise_level=0.5,
        action_chunk=16,
        action_env_dim=12,
        c_sign=-1.0,
        time_dim=time_dim,
        batch_dim=batch_dim,
    )

    assert logprobs.shape == (2, 2, 16, 12)
    assert torch.isfinite(logprobs).all()
