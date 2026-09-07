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
"""Tests for GR00T N1.5 nested action-head compute dtype alignment."""

from types import SimpleNamespace

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

pytest.importorskip("gr00t")

from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    FlowMatchingActionHeadForRLActionPrediction,
)


def test_cast_batch_feature_floats_preserves_nonfloating_tensors():
    feature = BatchFeature(
        data={
            "state": torch.tensor([[0.1001, 0.2002]], dtype=torch.float32),
            "embodiment_id": torch.tensor([31], dtype=torch.int64),
            "label": "robocasa",
        }
    )

    cast = FlowMatchingActionHeadForRLActionPrediction._cast_batch_feature_floats(
        feature, torch.bfloat16
    )

    assert cast["state"].dtype == torch.bfloat16
    assert cast["embodiment_id"].dtype == torch.int64
    assert cast["label"] == "robocasa"
    assert feature["state"].dtype == torch.float32


def test_raw_state_uses_each_checkpoint_transform_and_preserves_input(monkeypatch):
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import gr00t_action_model as module

    class StateTransform:
        def __init__(self, mean):
            self.mean = mean
            self.apply_to = [key for key, _, _ in module.ROBOCASA_STATE_SLICES]

        def __call__(self, states):
            return {key: value - self.mean for key, value in states.items()}

    monkeypatch.setattr(module, "StateActionTransform", StateTransform)
    raw = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    original = raw.clone()
    outputs = []
    for mean in (1.0, 3.0):
        transform = StateTransform(mean)
        model = SimpleNamespace(
            obs_convert_fn=module.OBS_CONVERSION["robocasa365"],
            _modality_transform=SimpleNamespace(transforms=[transform]),
            _modality_config={
                "state": SimpleNamespace(modality_keys=transform.apply_to)
            },
            config=SimpleNamespace(action_head_cfg={"max_state_dim": 64}),
        )
        state, mask = (
            module.GR00T_N1_5_ForRLActionPrediction.opd_normalize_robocasa_state(
                model, raw
            )
        )
        torch.testing.assert_close(state[:, :, :16], (raw - mean).unsqueeze(1))
        assert mask[:, :, :16].all() and not mask[:, :, 16:].any()
        assert not state[:, :, 16:].any()
        outputs.append(state)
    assert not torch.equal(outputs[0], outputs[1])
    torch.testing.assert_close(raw, original)
