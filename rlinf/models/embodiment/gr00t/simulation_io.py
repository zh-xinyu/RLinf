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

import numpy as np
import torch
import torch.nn.functional as F


def convert_libero_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}

    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.image"] = env_obs["main_images"].unsqueeze(1).numpy()
    groot_obs["video.wrist_image"] = env_obs["wrist_images"].unsqueeze(1).numpy()
    # [B, 8] -> [B, T(1), 8]
    groot_obs["state.x"] = env_obs["states"].unsqueeze(1)[:, :, 0:1].numpy()
    groot_obs["state.y"] = env_obs["states"].unsqueeze(1)[:, :, 1:2].numpy()
    groot_obs["state.z"] = env_obs["states"].unsqueeze(1)[:, :, 2:3].numpy()
    groot_obs["state.roll"] = env_obs["states"].unsqueeze(1)[:, :, 3:4].numpy()
    groot_obs["state.pitch"] = env_obs["states"].unsqueeze(1)[:, :, 4:5].numpy()
    groot_obs["state.yaw"] = env_obs["states"].unsqueeze(1)[:, :, 5:6].numpy()
    groot_obs["state.gripper"] = env_obs["states"].unsqueeze(1)[:, :, 6:].numpy()
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]

    return groot_obs


def convert_maniskill_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}
    # video
    # TODO(lx): If we have a dataset on maniskill, resize can be avoided.
    # But now we have to resize images to libero data version.
    env_obs["main_images"] = cut_and_resize_images(
        env_obs["main_images"],
        env_obs["main_images"].shape[-3],  # H
        256,
    )
    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).numpy()
    # state
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    else:
        # gr00t_1_7 pad the state to input dimension
        # create state of [B, T, C]
        groot_obs["state.left_arm"] = np.zeros((env_obs["main_images"].shape[0], 1, 7))
    # annotation
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action_n1d5(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T N1.5 action chunk to Libero format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d6(
    action_chunk: dict[str, np.array],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.6 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")

    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d7(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.7 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")
    # gripper conversion is handled by the shared
    # ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.

    if action_array.shape[-1] != 7:
        raise ValueError(f"Expected 7-dim action, got {action_array.shape[-1]}")
    return action_array


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Convert GR00T action chunk to Maniskill format."""
    return action_chunk["action.left_arm"][:, :chunk_size]


def convert_to_isaaclab_stack_cube_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Isaaclab Stack Cube format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array[..., -1] = np.sign(action_array[..., -1])
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


# RoboCasa365 state layout, in the order the GR00T N1.5 checkpoints expect. The env's
# `observation.state_layout` must produce exactly this concatenation (16 dims total).
ROBOCASA_STATE_SLICES = (
    ("state.base_position", 0, 3),
    ("state.base_rotation", 3, 7),
    ("state.end_effector_position_relative", 7, 10),
    ("state.end_effector_rotation_relative", 10, 14),
    ("state.gripper_qpos", 14, 16),
)


def convert_robocasa_obs_to_gr00t_format(env_obs):
    """Convert a RoboCasa365 observation to the GR00T N1.5 modality dict.

    Three cameras, mapped from the env's generic image slots:
    ``main_images`` -> agentview_left, ``wrist_images`` -> eye_in_hand,
    ``extra_view_images[0]`` -> agentview_right.
    """
    groot_obs = {}
    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.robot0_agentview_left"] = env_obs["main_images"].unsqueeze(1).numpy()
    groot_obs["video.robot0_eye_in_hand"] = env_obs["wrist_images"].unsqueeze(1).numpy()
    # (concat order is the data config's video_keys, not this insertion order)
    extra = env_obs.get("extra_view_images", None)
    assert extra is not None, (
        "RoboCasa365 GR00T needs a third camera (robot0_agentview_right); configure it "
        "as the env's extra view."
    )
    if extra.ndim == 5:
        # [B, V, H, W, C] -> take the first extra view
        extra = extra[:, 0]
    groot_obs["video.robot0_agentview_right"] = extra.unsqueeze(1).numpy()

    states = env_obs["states"]
    assert states.shape[-1] == 16, (
        f"RoboCasa365 GR00T expects a 16-d state, got {states.shape[-1]}. Set the env's "
        "observation.state_layout to [base_pos, base_quat, base_to_eef_pos, "
        "base_to_eef_quat, gripper_qpos]."
    )
    states = states.unsqueeze(1)  # [B, T(1), 16]
    for key, start, stop in ROBOCASA_STATE_SLICES:
        groot_obs[key] = states[:, :, start:stop].numpy()

    # Must match RoboCasaPandaOmronDataConfig.language_keys.
    groot_obs["annotation.human.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_robocasa_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Concatenate GR00T action groups into the 12-d vector RoboCasa expects.

    Order is ``[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]``,
    which is what the env config's ``base_mode_index: 11`` indexes into.
    """
    parts = [
        action_chunk["action.end_effector_position"],
        action_chunk["action.end_effector_rotation"],
        action_chunk["action.gripper_close"],
        action_chunk["action.base_motion"],
        action_chunk["action.control_mode"],
    ]
    parts = [np.asarray(part)[:, :chunk_size] for part in parts]
    return np.concatenate(parts, axis=-1)


OBS_CONVERSION = {
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
    "isaaclab_stack_cube": convert_libero_obs_to_gr00t_format,
    "robocasa365": convert_robocasa_obs_to_gr00t_format,
}

ACTION_CONVERSION_N1D5 = {
    "libero": convert_to_libero_action_n1d5,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
    "robocasa365": convert_to_robocasa_action,
}

ACTION_CONVERSION_N1D6 = {
    "libero": convert_to_libero_action_n1d6,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}

ACTION_CONVERSION_N1D7 = {
    "libero": convert_to_libero_action_n1d7,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """Cut and resize the images to the crop size."""
    images_nchw = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    original_width = images_nchw.shape[-1]  # W
    start = (original_width - crop_size) // 2
    end = start + crop_size

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images_nchw[:, :, :, start:end]  # [B, C, H, crop_W]

    # Resize: interpolate to target_size x target_size
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )  # [B, C, target_size, target_size]

    # Convert back to NHWC
    resized_nhwc = resized_tensor.permute(
        0, 2, 3, 1
    ).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    return resized_nhwc


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
