# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""GR00T native prompts survive ragged rollout storage and teacher adaptation."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("gr00t")
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    GR00T_N1_5_ForRLActionPrediction as Policy,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    restore_opd_text_padding,
)


def collected():
    """Processor left padding followed by collection right padding."""
    return {
        "eagle_input_ids": torch.tensor([[0, 7, 8, 9, 0, 0], [4, 5, 6, 0, 0, 0]]),
        "eagle_attention_mask": torch.tensor([[0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]]),
        "state": torch.zeros(2, 1, 64),
    }


def test_native_recovery_matches_individual_unpadded_prompts():
    inputs = collected()
    result = restore_opd_text_padding(inputs)
    assert result["eagle_input_ids"].tolist() == [[7, 8, 9], [4, 5, 6]]
    assert result["eagle_attention_mask"].tolist() == [[1, 1, 1], [1, 1, 1]]
    assert inputs["eagle_input_ids"].shape == (2, 6)
    assert result["state"] is inputs["state"]


def test_teacher_padding_handles_mixed_lengths_without_mutating_student():
    inputs = collected()
    inputs["eagle_attention_mask"][1, 2] = 0
    result = restore_opd_text_padding(inputs, fixed_length=570)
    assert result["eagle_input_ids"].shape == (2, 570)
    assert result["eagle_input_ids"][:, :4].tolist() == [[7, 8, 9, 0], [4, 5, 0, 0]]
    assert result["eagle_attention_mask"].sum(-1).tolist() == [3, 2]
    assert torch.count_nonzero(result["eagle_input_ids"][:, 3:]) == 0
    assert inputs["eagle_input_ids"][1, 2] == 6
    with pytest.raises(ValueError, match="equal true text lengths"):
        restore_opd_text_padding(inputs)
    for row in range(2):
        micro = {k: v[row : row + 1] for k, v in inputs.items()}
        assert restore_opd_text_padding(micro)["eagle_input_ids"].shape[-1] == 3 - row


def test_invalid_mask_empty_and_overlength_fail_explicitly():
    inputs = collected()
    with pytest.raises(ValueError, match="exceeds"):
        restore_opd_text_padding(inputs, fixed_length=2)
    inputs["eagle_attention_mask"][0] = 0
    with pytest.raises(ValueError, match="unmasked"):
        restore_opd_text_padding(inputs)
    inputs["eagle_attention_mask"][0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        restore_opd_text_padding(inputs)


@pytest.mark.parametrize("refresh_flow", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_nft_and_refresh_repair_only_explicit_native_mode(refresh_flow, native):
    inputs = collected()
    inputs.update(
        state_mask=torch.ones(2, 1, 64),
        eagle_pixel_values=torch.zeros(2, 2, 3, 2, 2),
        eagle_image_sizes=torch.ones(2, 2, 2),
        embodiment_id=torch.full((2,), 31),
    )
    captured = {}

    class StopBeforeBackbone(Exception):
        pass

    def prepare_input(value):
        captured.update(value)
        raise StopBeforeBackbone

    policy = SimpleNamespace(prepare_input=prepare_input)
    if native:
        policy.opd_fixed_text_padding = False
    with pytest.raises(StopBeforeBackbone):
        Policy.nft_forward(policy, inputs, refresh_flow=refresh_flow)
    expected = [[7, 8, 9], [4, 5, 6]] if native else inputs["eagle_input_ids"].tolist()
    assert captured["eagle_input_ids"].tolist() == expected


def test_native_rollout_batch_matches_b1_inference_and_retains_training_payload():
    from rlinf.models.embodiment.gr00t.simulation_io import OBS_CONVERSION

    class FakePolicy:
        predict_action_batch = Policy.predict_action_batch
        _check_state_is_batched = Policy._check_state_is_batched
        obs_convert_fn = staticmethod(OBS_CONVERSION["libero_grouped"])
        opd_fixed_text_padding = False
        _opd_state_adapter_validated = True
        output_action_chunks = 2

        def __init__(self):
            self.seen_lengths = []

        def apply_transforms(self, obs):
            # Assert preprocessing itself receives one observation and native text.
            assert obs["state.eef_pos_absolute"].shape[0] == 1
            length = len(obs["annotation.human.action.task_description"][0])
            ids = torch.arange(1, length + 1).unsqueeze(0)
            return {
                "state": torch.tensor(obs["state.eef_pos_absolute"]),
                "state_mask": torch.ones(1, 1, 3, dtype=torch.bool),
                "eagle_input_ids": ids,
                "eagle_attention_mask": torch.ones_like(ids),
                "eagle_pixel_values": torch.zeros(1, 2, 3, 2, 2),
                "eagle_image_sizes": torch.full((1, 2, 2), 2),
                "embodiment_id": torch.tensor([31]),
            }

        def _get_rl_action(self, inputs, mode):
            length = inputs["eagle_input_ids"].shape[-1]
            self.seen_lengths.append(length)
            actions = torch.full((1, 2, 7), float(length))
            return actions, {
                "prev_logprobs": actions.clone(),
                "prev_values": None,
                "forward_inputs": {
                    **inputs,
                    "chains": actions.unsqueeze(1).repeat(1, 5, 1, 1),
                    "denoise_inds": torch.tensor([[2]]),
                },
            }

        def _get_unnormalized_action(self, actions):
            return {"action": actions.numpy()}

        @staticmethod
        def action_convert_fn(actions, chunk_size):
            return actions["action"][:, :chunk_size]

    obs = {
        "states": torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10,
        "main_images": torch.zeros(3, 2, 2, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(3, 2, 2, 3, dtype=torch.uint8),
        "task_descriptions": ["ab", "cdef", "ghi"],
    }
    model = FakePolicy()
    actions, result = model.predict_action_batch(obs, mode="eval")
    assert model.seen_lengths == [2, 4, 3]
    assert result["prev_values"] is None
    assert result["forward_inputs"]["eagle_input_ids"].shape == (3, 4)
    assert result["forward_inputs"]["eagle_attention_mask"].sum(-1).tolist() == [
        2,
        4,
        3,
    ]
    for row in range(3):
        single_obs = {k: v[row : row + 1] for k, v in obs.items()}
        expected_action, expected = FakePolicy().predict_action_batch(
            single_obs, mode="eval"
        )
        torch.testing.assert_close(actions[row : row + 1], expected_action)
        stored = {k: v[row : row + 1] for k, v in result["forward_inputs"].items()}
        trimmed = restore_opd_text_padding(stored)
        assert trimmed.keys() == expected["forward_inputs"].keys()
        for key, value in trimmed.items():
            torch.testing.assert_close(value, expected["forward_inputs"][key])
        torch.testing.assert_close(
            result["prev_logprobs"][row : row + 1], expected["prev_logprobs"]
        )
