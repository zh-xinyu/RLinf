# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Physical observation/action semantics of the few-shot screening adapter."""

import numpy as np
import pytest
import torch

pytest.importorskip("gr00t")

from toolkits.libero_gr00t.fewshot import convert_action, convert_observation


def test_grouped_observation_preserves_axis_angle_and_both_gripper_joints():
    raw = torch.tensor([[0.1, 0.2, 0.3, 2.8, -0.4, 0.2, 0.04, -0.03]])
    obs = {
        "states": raw,
        "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(1, 8, 8, 3, dtype=torch.uint8),
        "task_descriptions": ["pick up the cup"],
    }
    result = convert_observation(obs)
    np.testing.assert_array_equal(result["state.eef_rot_absolute"], raw[:, None, 3:6])
    np.testing.assert_array_equal(result["state.gripper_close"], raw[:, None, 6:8])
    assert result["video.front_view"].shape == (1, 1, 8, 8, 3)
    assert result["video.left_wrist_view"].all()


def test_action_keeps_controller_deltas_and_converts_gripper_only():
    position = np.array([[[0.2, -0.1, 0.0], [0.3, 0.4, 0.5]]])
    rotation = np.array([[[0.1, 0.0, -0.1], [0.2, 0.3, 0.4]]])
    result = convert_action(
        {
            "action.eef_pos_delta": position,
            "action.eef_rot_delta": rotation,
            "action.gripper_close": np.array([[[0.0], [1.0]]]),
        },
        chunk_size=2,
    )
    np.testing.assert_array_equal(result[..., :3], position)
    np.testing.assert_array_equal(result[..., 3:6], rotation)
    np.testing.assert_array_equal(result[..., 6], [[1.0, -1.0]])
