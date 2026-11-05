# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Cross-checkpoint LIBERO physical actions and state representations."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("gr00t")
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.state_action import StateActionTransform

from rlinf.algorithms.flow_opd import (
    map_target_flow_state_to_source,
    transport_source_velocity_to_target,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    GR00T_N1_5_ForRLActionPrediction as Policy,
)
from rlinf.models.embodiment.gr00t.simulation_io import OBS_CONVERSION


def metadata(grouped, stretch=1.0):
    """Make distinct valid metadata without requiring downloaded checkpoints."""
    scalar = [(k, 1) for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]
    actions = (
        [("eef_pos_delta", 3), ("eef_rot_delta", 3), ("gripper_close", 1)]
        if grouped
        else scalar
    )
    states = (
        [("eef_pos_absolute", 3), ("eef_rot_absolute", 3), ("gripper_close", 2)]
        if grouped
        else scalar[:-1] + [("gripper", 2)]
    )
    out = {
        "embodiment_tag": "new_embodiment",
        "statistics": {},
        "modalities": {"video": {}},
    }
    for kind, fields in (("state", states), ("action", actions)):
        out["statistics"][kind], out["modalities"][kind] = {}, {}
        offset = 0
        for name, size in fields:
            low = [-stretch * (offset + i + 1) for i in range(size)]
            high = [stretch * (offset + i + 2) for i in range(size)]
            if kind == "action" and "gripper" in name:
                low, high = [0.0], [1.0]
            out["statistics"][kind][name] = {
                "min": low,
                "max": high,
                "mean": [0.0] * size,
                "std": [1.0] * size,
                "q01": low,
                "q99": high,
            }
            out["modalities"][kind][name] = {
                "shape": [size],
                "continuous": True,
                "rotation_type": "axis_angle"
                if kind == "state" and name == "eef_rot_absolute"
                else None,
                "absolute": not (kind == "action" and name.endswith("delta")),
            }
            offset += size
    return DatasetMetadata.model_validate(out)


def model(grouped, stretch=1.0):
    meta = metadata(grouped, stretch)
    transforms, config = [], {}
    for kind in ("state", "action"):
        keys = [f"{kind}.{name}" for name in getattr(meta.modalities, kind)]
        transform = StateActionTransform(
            apply_to=keys,
            normalization_modes=dict.fromkeys(keys, "min_max"),
            target_rotations={"state.eef_rot_absolute": "rotation_6d"}
            if grouped and kind == "state"
            else {},
        )
        transform.set_metadata(meta)
        transforms.append(transform)
        config[kind] = SimpleNamespace(modality_keys=keys)
    return SimpleNamespace(
        metadata=meta,
        obs_convert_fn=OBS_CONVERSION["libero_grouped" if grouped else "libero"],
        _modality_transform=SimpleNamespace(transforms=transforms),
        _modality_config=config,
        config=SimpleNamespace(action_head_cfg={"action_dim": 32, "max_state_dim": 64}),
        valid_action_dim=7,
    )


def decode(model, normalized):
    transform = model._modality_transform.transforms[1]
    data, start = {}, 0
    for key in model._modality_config["action"].modality_keys:
        size = model.metadata.modalities.action[key.split(".")[1]].shape[0]
        data[key] = normalized[..., start : start + size]
        start += size
    data = transform.unapply(data)
    return torch.cat(
        [data[k] for k in model._modality_config["action"].modality_keys], -1
    )


@pytest.mark.parametrize(
    "source_grouped,target_grouped",
    [(False, True), (True, False), (True, True), (False, False)],
)
def test_endpoint_map_preserves_decoded_controller_commands(
    source_grouped, target_grouped
):
    source, target = model(source_grouped, 0.7), model(target_grouped, 1.3)
    scale, shift = Policy.opd_action_endpoint_affine_to(
        source,
        target.metadata,
        target_action_transform=target._modality_transform.transforms[1],
        target_action_keys=target._modality_config["action"].modality_keys,
    )
    source_x = torch.randn(3, 16, 32, generator=torch.Generator().manual_seed(8))
    target_x = scale * source_x + shift
    torch.testing.assert_close(
        decode(source, source_x), decode(target, target_x), atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(target_x[..., 7:], source_x[..., 7:])
    torch.testing.assert_close(target_x[..., 6], source_x[..., 6])


def test_transport_preserves_prior_and_matches_continuous_path_derivative():
    source, target = model(False, 0.7), model(True, 1.3)
    scale, shift = Policy.opd_action_endpoint_affine_to(source, target.metadata)
    x0, velocity = torch.randn(2, 16, 32), torch.randn(2, 16, 32)
    steps = torch.tensor([0, 2])
    t = (steps / 4).reshape(-1, 1, 1)
    source_x = x0 + t * velocity

    def path(time):
        return (1 + time * (scale - 1)) * (x0 + time * velocity) + time * shift

    target_x = path(t)
    recovered = map_target_flow_state_to_source(target_x, steps, 4, scale, shift)
    torch.testing.assert_close(recovered, source_x)
    torch.testing.assert_close(target_x[0], x0[0])
    actual = transport_source_velocity_to_target(
        source_x, velocity, steps, 4, scale, shift
    )
    eps = 1e-3
    expected = (path(t + eps) - path(t - eps)) / (2 * eps)
    torch.testing.assert_close(actual, expected, atol=4e-4, rtol=4e-4)


@pytest.mark.parametrize("grouped,valid_dim", [(False, 8), (True, 11)])
def test_state_adapter_matches_checkpoint_transform_and_preserves_raw(
    grouped, valid_dim
):
    policy = model(grouped, 0.8)
    raw = torch.tensor([[0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 0.04, -0.03]])
    original = raw.clone()
    actual, mask = Policy.opd_normalize_state(policy, raw)
    obs = {
        "states": raw,
        "main_images": torch.zeros(1, 4, 4, 3),
        "wrist_images": torch.zeros(1, 4, 4, 3),
        "task_descriptions": ["pick up the cup"],
    }
    converted = policy.obs_convert_fn(obs)
    state = {
        k: torch.from_numpy(v).clone()
        for k, v in converted.items()
        if k.startswith("state.")
    }
    full = policy._modality_transform.transforms[0](state)
    expected = torch.cat(
        [full[k] for k in policy._modality_config["state"].modality_keys], -1
    )
    torch.testing.assert_close(actual[..., :valid_dim], expected)
    torch.testing.assert_close(raw, original)
    assert int(mask.sum()) == valid_dim
    assert not mask[..., valid_dim:].any() and not actual[..., valid_dim:].any()
    if grouped:
        torch.testing.assert_close(
            actual[..., 3:9], torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]])
        )
    other, _ = Policy.opd_normalize_state(model(grouped, 1.5), raw)
    assert not torch.equal(actual[..., :3], other[..., :3])


def test_unsupported_target_transform_and_permuted_concat_fail():
    source, target = model(False), model(True)
    transform = target._modality_transform.transforms[1]
    transform.normalization_modes["action.eef_pos_delta"] = "mean_std"
    with pytest.raises(ValueError, match="min_max"):
        Policy.opd_action_endpoint_affine_to(
            source, target.metadata, target_action_transform=transform
        )
    transform.normalization_modes["action.eef_pos_delta"] = "min_max"
    transform.target_rotations["action.eef_rot_delta"] = "axis_angle"
    with pytest.raises(ValueError, match="rotations"):
        Policy.opd_action_endpoint_affine_to(
            source, target.metadata, target_action_transform=transform
        )
    with pytest.raises(ValueError, match="canonical"):
        Policy.opd_action_endpoint_affine_to(
            source,
            target.metadata,
            target_action_keys=list(
                reversed(target._modality_config["action"].modality_keys)
            ),
        )
    source._modality_config["action"].modality_keys.reverse()
    with pytest.raises(ValueError, match="canonical"):
        Policy.opd_action_endpoint_affine_to(source, target.metadata)


def test_gripper_convention_and_bad_ranges_fail():
    source, target = model(False), model(True)
    target.metadata.statistics.action["gripper_close"].min[0] = -1
    with pytest.raises(ValueError, match="gripper"):
        Policy.opd_action_endpoint_affine_to(source, target.metadata)
    target = model(True)
    target.metadata.statistics.action["eef_pos_delta"].max[0] = (
        target.metadata.statistics.action["eef_pos_delta"].min[0]
    )
    with pytest.raises(ValueError, match="nondegenerate"):
        Policy.opd_action_endpoint_affine_to(source, target.metadata)
