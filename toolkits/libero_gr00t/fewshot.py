# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation-only adapter for the public RLobot-jun few-shot checkpoint.

Transform reference: jun981015/gr00t-bigenlight, commit
75a306e8d97fcca101e708e821d02ca2b17d323e,
n15/gr00t/experiment/data_config.py:LiberoDataConfig.
The source is from the checkpoint publisher; exact training-code provenance
is not included in the checkpoint release.
"""

import numpy as np
from gr00t.data.transform.state_action import StateActionTransform

from rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config import (
    LiberoFrankaDataConfig,
)
from rlinf.models.embodiment.gr00t.simulation_io import normalize_gripper_action


class FewshotLiberoDataConfig(LiberoFrankaDataConfig):
    """Preserve grouped fields and convert absolute axis-angle state to 6D."""

    video_keys = ["video.front_view", "video.left_wrist_view"]
    state_keys = [
        "state.eef_pos_absolute",
        "state.eef_rot_absolute",
        "state.gripper_close",
    ]
    action_keys = [
        "action.eef_pos_delta",
        "action.eef_rot_delta",
        "action.gripper_close",
    ]

    def transform(self):
        transforms = super().transform()
        for transform in transforms.transforms:
            if isinstance(transform, StateActionTransform) and (
                "state.eef_rot_absolute" in transform.apply_to
            ):
                transform.target_rotations = {"state.eef_rot_absolute": "rotation_6d"}
        return transforms


def convert_observation(env_obs: dict) -> dict:
    """Map physical LIBERO observations to the checkpoint's named fields."""
    state = env_obs["states"].unsqueeze(1).cpu().numpy()
    return {
        "video.front_view": env_obs["main_images"].unsqueeze(1).cpu().numpy(),
        "video.left_wrist_view": env_obs["wrist_images"].unsqueeze(1).cpu().numpy(),
        "state.eef_pos_absolute": state[..., :3],
        "state.eef_rot_absolute": state[..., 3:6],
        "state.gripper_close": state[..., 6:8],
        "annotation.human.action.task_description": env_obs["task_descriptions"],
    }


def convert_action(action_chunk: dict, chunk_size: int = 1) -> np.ndarray:
    """Concatenate controller commands and decode the [0, 1] gripper."""
    action = np.concatenate(
        [action_chunk[k][:, :chunk_size] for k in FewshotLiberoDataConfig.action_keys],
        axis=-1,
    )
    if action.shape[-1] != 7:
        raise ValueError("Expected seven physical LIBERO action dimensions")
    return normalize_gripper_action(action, binarize=True)
