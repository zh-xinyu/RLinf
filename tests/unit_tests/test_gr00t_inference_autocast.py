# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Rollout compute precision must not rely on the worker entering autocast."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("gr00t")

from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    GR00T_N1_5_ForRLActionPrediction,
)


class FP32Head(torch.nn.Linear):
    def get_rl_action(self, backbone, action, mode):
        output = self(backbone)
        return {}, {
            "actions": output,
            "chains": output[:, None],
            "denoise_inds": torch.tensor([0]),
            "prev_logprobs": None,
            "prev_values": None,
        }


def test_opt_in_rollout_autocast_matches_external_context_and_keeps_master_dtype():
    head = FP32Head(2, 2)
    model = SimpleNamespace(
        device=torch.device("cpu"),
        prepare_input=lambda inputs: (inputs["state"], None),
        backbone=torch.nn.Identity(),
        action_head=head,
        validate_data=lambda *args, **kwargs: None,
        image_nums=2,
    )
    inputs = {
        "state": torch.tensor([[0.125, 0.75]], dtype=torch.bfloat16),
        "eagle_pixel_values": torch.zeros(2, 3, 2, 2),
        "eagle_image_sizes": torch.zeros(2, 2, dtype=torch.long),
    }
    forward = GR00T_N1_5_ForRLActionPrediction._get_rl_action
    with pytest.raises(RuntimeError, match="dtype|same type"):
        forward(model, inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = forward(model, inputs)
    model.inference_autocast_dtype = torch.bfloat16
    actual, _ = forward(model, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert head.weight.dtype == torch.float32
    assert not torch.is_autocast_enabled("cpu")
