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
    count_flow_timesteps_by_suite,
    flow_teacher_transition_logprobs,
    map_target_flow_state_to_source,
    sample_flow_timestep_indices,
    select_flow_opd_logprobs,
    select_opd_metric_rows,
    teacher_suite_presence,
    transport_source_velocity_to_target,
)


def test_sample_flow_timestep_indices_advance_per_supervision_batch():
    first = sample_flow_timestep_indices(128, 4, seed=7, rank=2, draw_count=1)
    repeated = sample_flow_timestep_indices(128, 4, seed=7, rank=2, draw_count=1)
    second = sample_flow_timestep_indices(128, 4, seed=7, rank=2, draw_count=2)

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) < 4


def test_count_flow_timesteps_by_suite_keeps_balanced_sample_duplicates():
    counts = count_flow_timesteps_by_suite(
        torch.tensor([0, 1, 2, 3, 1, 0]),
        4,
        teacher_ids=torch.tensor([0, 0, 1, 1, 1, 0]),
        sample_idx=torch.tensor([4, 0, 4, 3]),
        num_teachers=2,
    )

    torch.testing.assert_close(
        counts,
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 2, 0, 1],
            ]
        ),
    )


@pytest.mark.parametrize("joint_logprob", [False, True])
def test_select_flow_opd_logprobs_picks_sampled_step(joint_logprob):
    step_count = 5 if joint_logprob else 4
    prev = torch.arange(2 * 2 * step_count * 3 * 2).reshape(2, 2, step_count, 3, 2)
    step_idx = torch.tensor([0, 1, 2, 3])

    selected = select_flow_opd_logprobs(prev, step_idx, joint_logprob=joint_logprob)

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


def test_teacher_suite_presence_preserves_missing_configured_suite():
    local = teacher_suite_presence(torch.tensor([0, 0, 2]), num_teachers=3)
    other_rank = teacher_suite_presence(torch.tensor([1, 1]), num_teachers=3)

    torch.testing.assert_close(local, torch.tensor([True, False, True]))
    torch.testing.assert_close(local | other_rank, torch.tensor([True, True, True]))


def test_flow_endpoint_transport_is_identity_for_matching_normalizers():
    x_t = torch.randn(4, 3, 5)
    velocity = torch.randn_like(x_t)
    steps = torch.tensor([0, 1, 2, 3])
    scale = torch.ones(5)
    shift = torch.zeros(5)

    source_x = map_target_flow_state_to_source(x_t, steps, 4, scale, shift)
    target_velocity = transport_source_velocity_to_target(
        source_x, velocity, steps, 4, scale, shift
    )

    torch.testing.assert_close(source_x, x_t)
    torch.testing.assert_close(target_velocity, velocity)


def test_flow_endpoint_transport_preserves_prior_and_recovers_source_state():
    source_x = torch.randn(4, 2, 3)
    steps = torch.tensor([0, 1, 2, 3])
    scale = torch.tensor([0.5, 2.0, 1.25])
    shift = torch.tensor([0.3, -0.2, 0.7])
    t = (steps.float() / 4).reshape(-1, 1, 1)
    scale_t = 1 + t * (scale.reshape(1, 1, -1) - 1)
    target_x = scale_t * source_x + t * shift.reshape(1, 1, -1)

    recovered = map_target_flow_state_to_source(target_x, steps, 4, scale, shift)

    torch.testing.assert_close(recovered, source_x)
    # The time-zero map must be identity so both policies retain N(0, I) as their prior.
    torch.testing.assert_close(target_x[0], source_x[0])


def test_flow_velocity_transport_matches_finite_difference():
    source_x = torch.tensor([[[0.2, -0.4]]], dtype=torch.float64)
    source_velocity = torch.tensor([[[0.7, 0.1]]], dtype=torch.float64)
    scale = torch.tensor([1.8, 0.6], dtype=torch.float64)
    shift = torch.tensor([-0.3, 0.25], dtype=torch.float64)
    step = torch.tensor([2])
    num_steps = 4

    transported = transport_source_velocity_to_target(
        source_x, source_velocity, step, num_steps, scale, shift
    )

    t = step.item() / num_steps
    eps = 1e-4

    def target_path(time):
        source_at_time = source_x + (time - t) * source_velocity
        scale_at_time = 1 + time * (scale - 1)
        return scale_at_time.reshape(1, 1, -1) * source_at_time + time * shift.reshape(
            1, 1, -1
        )

    finite_difference = (target_path(t + eps) - target_path(t - eps)) / (2 * eps)
    torch.testing.assert_close(
        transported.double(), finite_difference, rtol=1e-4, atol=1e-4
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
