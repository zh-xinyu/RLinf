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
"""Check GR00T student/teacher state and action normalization transport."""

import copy
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import open_dict

from toolkits.mopd.probe_precision_init import compose_mopd_config


def _dummy_robocasa_observation(batch_size: int = 2) -> dict:
    """Build valid raw state/image inputs without starting RoboCasa."""
    states = torch.zeros(batch_size, 16, dtype=torch.float32)
    states[:, 0:3] = torch.tensor([2.5, -1.5, 0.705])
    states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    states[:, 7:10] = torch.tensor([0.25, -0.05, 0.45])
    states[:, 10:14] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    states[:, 14:16] = torch.tensor([0.02, 0.02])
    images = torch.zeros(batch_size, 256, 256, 3, dtype=torch.uint8)
    return {
        "states": states.to(torch.bfloat16).float(),
        "main_images": images.clone(),
        "wrist_images": images.clone(),
        "extra_view_images": images.clone(),
        "task_descriptions": ["turn on the microwave"] * batch_size,
    }


def _full_transformed_state(model, env_obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    observations = model.obs_convert_fn(copy.deepcopy(env_obs))
    for key, value in observations.items():
        if not isinstance(value, np.ndarray):
            observations[key] = np.array(value)
    transformed = model.apply_transforms(observations)
    return transformed["state"].float(), transformed["state_mask"].bool()


def main() -> None:
    """Load real checkpoints and validate the state/endpoint adapters."""
    os.environ.setdefault(
        "EMBODIED_PATH",
        str(Path(__file__).resolve().parents[2] / "examples" / "embodiment"),
    )
    cfg = compose_mopd_config("robocasa365_mopd_gr00t")
    from rlinf.models import get_model

    student = get_model(cfg.actor.model)
    student.eval()
    env_obs = _dummy_robocasa_observation()
    full_state, full_mask = _full_transformed_state(student, env_obs)
    adapted_state, adapted_mask = student.opd_normalize_robocasa_state(
        env_obs["states"]
    )
    torch.testing.assert_close(adapted_state, full_state, rtol=0, atol=1e-6)
    assert torch.equal(adapted_mask, full_mask)

    identity_scale, identity_shift = student.opd_action_endpoint_affine_to(
        student.metadata
    )
    torch.testing.assert_close(identity_scale, torch.ones_like(identity_scale))
    torch.testing.assert_close(identity_shift, torch.zeros_like(identity_shift))
    student_metadata = student.metadata
    del student

    teacher_cfg = copy.deepcopy(cfg.actor.model)
    with open_dict(teacher_cfg):
        for key, value in cfg.rollout.expert_models.atomic_seen.items():
            teacher_cfg[key] = value
    teacher = get_model(teacher_cfg)
    teacher.eval()
    teacher_state, teacher_mask = teacher.opd_normalize_robocasa_state(
        env_obs["states"]
    )
    assert torch.equal(teacher_mask, full_mask)
    endpoint_scale, endpoint_shift = teacher.opd_action_endpoint_affine_to(
        student_metadata
    )

    state_delta = (teacher_state - full_state).abs()
    valid_state_delta = state_delta[full_mask]
    valid_scale = endpoint_scale[: teacher.valid_action_dim]
    valid_shift = endpoint_shift[: teacher.valid_action_dim]
    assert float(valid_state_delta.max()) > 0
    torch.testing.assert_close(valid_scale, torch.ones_like(valid_scale))
    torch.testing.assert_close(valid_shift, torch.zeros_like(valid_shift))

    print("student state adapter matches full transform: yes", flush=True)
    print(
        "teacher-vs-student normalized state: "
        f"mean_abs={float(valid_state_delta.mean()):.6f}, "
        f"max_abs={float(valid_state_delta.max()):.6f}",
        flush=True,
    )
    print(
        "teacher->student action endpoint affine (identity for released checkpoints): "
        f"scale=[{float(valid_scale.min()):.6f}, {float(valid_scale.max()):.6f}], "
        f"max_abs_shift={float(valid_shift.abs().max()):.6f}",
        flush=True,
    )
    print("NORMALIZATION_TRANSPORT_OK", flush=True)


if __name__ == "__main__":
    main()
