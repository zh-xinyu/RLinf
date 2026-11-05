# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Canonical fixed batches are explicit and preserve legacy B1 token layouts."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("gr00t")
from rlinf.models.embodiment.gr00t.gr00t_n1d5 import validate_opd_padding_config
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    GR00T_N1_5_ForRLActionPrediction as Policy,
)
from rlinf.models.embodiment.gr00t.simulation_io import OBS_CONVERSION


@pytest.mark.parametrize("fixed", [None, False])
def test_canonical_mode_requires_explicit_fixed_policy(fixed):
    with pytest.raises(ValueError, match="requires"):
        validate_opd_padding_config(
            OmegaConf.create(
                {"opd_fixed_text_padding": fixed, "opd_canonical_fixed_padding": True}
            )
        )


def test_default_remains_disabled_and_invalid_flag_rejected():
    assert validate_opd_padding_config(OmegaConf.create({})) is False
    assert (
        validate_opd_padding_config(
            OmegaConf.create(
                {"opd_fixed_text_padding": True, "opd_canonical_fixed_padding": True}
            )
        )
        is True
    )
    with pytest.raises(ValueError, match="boolean"):
        validate_opd_padding_config(
            OmegaConf.create({"opd_canonical_fixed_padding": "yes"})
        )


class StopBeforeBackbone(Exception):
    pass


def mixed_inputs():
    return {
        "eagle_input_ids": torch.tensor([[0, 7, 8], [4, 5, 6]]),
        "eagle_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        "state": torch.zeros(2, 1, 64),
        "state_mask": torch.ones(2, 1, 64),
        "eagle_pixel_values": torch.zeros(2, 2, 3, 2, 2),
        "eagle_image_sizes": torch.ones(2, 2, 2),
        "embodiment_id": torch.full((2,), 31),
    }


@pytest.mark.parametrize("refresh", [False, True])
def test_nft_and_refresh_canonicalize_before_backbone(refresh):
    captured = {}

    def prepare(inputs):
        captured.update(inputs)
        raise StopBeforeBackbone

    policy = SimpleNamespace(
        opd_fixed_text_padding=True,
        opd_canonical_fixed_padding=True,
        padding_value=570,
        prepare_input=prepare,
    )
    with pytest.raises(StopBeforeBackbone):
        Policy.nft_forward(policy, mixed_inputs(), refresh_flow=refresh)
    assert captured["eagle_input_ids"].shape == (2, 570)
    assert captured["eagle_input_ids"][:, :3].tolist() == [[7, 8, 0], [4, 5, 6]]


def test_real_predict_preprocessing_canonicalizes_mixed_prompts_only_when_opted_in():
    captured = {}

    class FakePolicy:
        predict_action_batch = Policy.predict_action_batch
        _check_state_is_batched = Policy._check_state_is_batched
        obs_convert_fn = staticmethod(OBS_CONVERSION["libero_grouped"])
        opd_fixed_text_padding = True
        opd_canonical_fixed_padding = True
        _opd_state_adapter_validated = True
        padding_value = 570

        def apply_transforms(self, obs):
            assert obs["state.eef_pos_absolute"].shape[0] == 2
            return mixed_inputs()

        def _get_rl_action(self, inputs, mode):
            captured.update(inputs)
            raise StopBeforeBackbone

    raw = {
        "main_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 8),
        "task_descriptions": ["a", "longer"],
    }
    policy = FakePolicy()
    with pytest.raises(StopBeforeBackbone):
        policy.predict_action_batch(raw, mode="eval")
    assert captured["eagle_input_ids"][:, :3].tolist() == [[7, 8, 0], [4, 5, 6]]
    policy.opd_canonical_fixed_padding = False
    with pytest.raises(StopBeforeBackbone):
        policy.predict_action_batch(raw, mode="eval")
    assert captured["eagle_input_ids"][:, :3].tolist() == [[0, 7, 8], [4, 5, 6]]


def test_single_prompt_tokens_match_existing_fixed_padding_exactly():
    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        restore_opd_text_padding,
    )

    value = {
        "eagle_input_ids": torch.tensor([[7, 8, 9]]),
        "eagle_attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    canonical = restore_opd_text_padding(value, fixed_length=570)
    for key in value:
        assert torch.equal(
            canonical[key], torch.nn.functional.pad(value[key], (0, 567))
        )


def test_diagnostic_noise_injection_is_exact_and_restored():
    from toolkits.libero_gr00t.fixed_padding_parity import inject_initial_noise

    original = torch.randn
    noise = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4)
    with inject_initial_noise(noise, policy_call=False):
        actual = torch.randn(size=(1, 2, 4), dtype=torch.float32)
        assert torch.equal(actual, noise.float())
        assert torch.randn(2).shape == (2,)
    assert torch.randn is original
    with pytest.raises(RuntimeError, match="exactly one"):
        with inject_initial_noise(noise, policy_call=False):
            pass
    assert torch.randn is original


def test_diagnostic_rejects_duplicate_noise_and_lossy_cast():
    from toolkits.libero_gr00t.fixed_padding_parity import inject_initial_noise

    with pytest.raises(RuntimeError, match="Multiple"):
        with inject_initial_noise(torch.ones(1, 2, 4), policy_call=False):
            torch.randn(1, 2, 4)
            torch.randn(1, 2, 4)
    with pytest.raises(ValueError, match="cast"):
        with inject_initial_noise(torch.full((1, 2, 4), 1.001), policy_call=False):
            torch.randn(1, 2, 4, dtype=torch.bfloat16)


def test_preprocessing_uses_exact_ids_and_deterministic_repeats_are_separate():
    from toolkits.libero_gr00t.fixed_padding_parity import comparison_result

    expected = torch.tensor([200000.0])
    actual = expected + 1
    ids = comparison_result(
        "preprocessing/eagle_input_ids", actual, expected, 1e-5, 1e-4, True
    )
    assert not ids["close"] and ids["exact_required"]
    repeat = comparison_result("nft_b1_repeat", actual, expected, 1e-5, 1e-4, True)
    batch = comparison_result("nft_b4_vs_b1", actual, expected, 1e-5, 1e-4, True)
    assert not repeat["close"] and repeat["category"] == "within_mode_repeat"
    assert batch["close"] and batch["category"] == "batch_forward"
    assert batch["rtol"] == 1e-4 and not batch["exact"]


def test_comparison_rejects_equal_infinite_outputs():
    from toolkits.libero_gr00t.fixed_padding_parity import comparison_result

    values = torch.tensor([float("inf")])
    result = comparison_result("nft_b4_vs_b1", values, values, 1e-5, 1e-4, True)
    assert not result["finite"] and not result["close"]
    assert result["max_abs"] is None


def test_diagnostic_rejects_same_shape_noise_from_an_unrelated_callsite():
    from toolkits.libero_gr00t.fixed_padding_parity import inject_initial_noise

    original = torch.randn
    with pytest.raises(RuntimeError, match="outside GR00T"):
        with inject_initial_noise(torch.ones(1, 2, 4)):
            torch.randn(1, 2, 4)
    assert torch.randn is original


def test_head_precision_control_is_local_reversible_and_casts_boundaries():
    from toolkits.libero_gr00t.fixed_padding_parity import fp32_head_control

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            self.seen = []

        @staticmethod
        def _cast_batch_feature_floats(feature, dtype):
            return {
                key: value.to(dtype) if value.is_floating_point() else value
                for key, value in feature.items()
            }

        def forward(self, backbone_output, action_input):
            self.seen.append(
                (
                    backbone_output["x"].dtype,
                    action_input["x"].dtype,
                    torch.is_autocast_enabled("cpu"),
                )
            )
            return self.linear(backbone_output["x"]) + action_input["x"]

        def get_rl_action(self, backbone_output, action_input):
            return self.forward(backbone_output, action_input)

    head = Head()
    original_forward = head.forward
    features = {"x": torch.ones(1, 2, dtype=torch.bfloat16)}
    prior_tf32 = torch.backends.cuda.matmul.allow_tf32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with fp32_head_control(head):
            out = head(backbone_output=features, action_input=features)
            rollout = head.get_rl_action(features, features)
            assert out.dtype == rollout.dtype == torch.float32
            assert head.seen == [(torch.float32, torch.float32, False)] * 2
            assert torch.is_autocast_enabled("cpu")
    assert head.forward == original_forward
    assert torch.backends.cuda.matmul.allow_tf32 == prior_tf32
    assert features["x"].dtype == torch.bfloat16
    with pytest.raises(ValueError, match="FP32"):
        with fp32_head_control(head.bfloat16()):
            pass
