# Copyright 2025 The RLinf Authors.
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


"""
Here we define the Modality config not present in the gr00t package.
The modality config of gr00t is in gr00t/experiment/data_config.py
"""

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class ManiskillWidowXDataConfig(BaseDataConfig):
    """
    Modality config for Maniskill WidowX.
    """

    video_keys = ["video.ego_view"]
    state_keys = ["state.left_arm"]
    action_keys = ["action.left_arm"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=dict.fromkeys(self.action_keys, "min_max"),
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class LiberoFrankaDataConfig(BaseDataConfig):
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:
        if action_norm == "min_max":
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=dict.fromkeys(self.action_keys, "min_max"),
            )
        else:
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "mean_std",
                    "action.y": "mean_std",
                    "action.z": "mean_std",
                    "action.roll": "mean_std",
                    "action.pitch": "mean_std",
                    "action.yaw": "mean_std",
                    "action.gripper": "min_max",
                },
            )
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=dict.fromkeys(self.state_keys, "min_max"),
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            action_transform,
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboCasaPandaOmronDataConfig(BaseDataConfig):
    """Modality config for RoboCasa365 with the Panda + Omron mobile base.

    Ported from ``gr00t.experiment.data_config.SinglePandaGripperDataConfig``, which ships
    with Isaac-GR00T n1.5-release and is the config the released RoboCasa365 checkpoints
    were trained with. Only the video key *names* differ: RoboCasa365 records
    ``robot0_agentview_left / robot0_agentview_right / robot0_eye_in_hand`` where the
    upstream config says ``left_view / right_view / wrist_view``. Everything else -- key
    order, rotation targets, normalization modes -- is copied verbatim, because each of
    them silently changes what the model is fed:

    - the **state concat order** puts the end effector first and the base last;
    - the two quaternion state keys are converted to ``rotation_6d`` (so the state the
      encoder sees is 20-d, not the 16-d raw layout in ``metadata.json``);
    - ``gripper_close`` and ``control_mode`` are ``binary``, not ``min_max``.

    Actions get normalization but no rotation target, so ``action.end_effector_rotation``
    stays axis-angle and the emitted action is 12-d.
    """

    # Order and names taken from robocasa's own GR00T wrapper
    # (robocasa/wrappers/gym_wrapper.py: camera_names, 256x256), which is what the
    # validated official evaluation feeds the model. metadata.json lists the cameras in a
    # different order; that order is the dataset's, not the concat order.
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    # RoboCasa365 uses "annotation.human.task_description" -- NOT the
    # "annotation.human.action.task_description" that NVIDIA's SinglePandaGripper config
    # (and LIBERO/ManiSkill here) use. Getting this wrong does not raise: the model simply
    # receives no task instruction and acts blindly, which is what a 0/16 specialist looks
    # like. Source: robocasa-benchmark/Isaac-GR00T PandaOmronDataConfig.
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
