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
"""Tests for reusable MOPD diagnostic helpers."""

import pytest
import torch
from torch import nn

from toolkits.mopd.probe_gradient_geometry import gradient_geometry
from toolkits.mopd.probe_precision_init import validate_model_precision


class _MixedToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2).to(torch.bfloat16)
        self.backbone.requires_grad_(False)
        self.action_head = nn.Linear(2, 2).to(torch.float32)


def test_validate_model_precision_accepts_bf16_backbone_fp32_head():
    model = _MixedToyModel()

    trainable = validate_model_precision(model)

    assert trainable == list(model.action_head.parameters())


def test_gradient_geometry_reports_conflict_and_cancellation():
    gradients = {
        "a": [torch.tensor([1.0, 0.0])],
        "b": [torch.tensor([0.0, 1.0])],
        "c": [torch.tensor([-1.0, 0.0])],
    }

    geometry = gradient_geometry(gradients)

    assert geometry["suite_grad_norms"] == {"a": 1.0, "b": 1.0, "c": 1.0}
    assert geometry["pairwise"]["a__b"]["cosine"] == pytest.approx(0.0)
    assert geometry["pairwise"]["a__c"]["cosine"] == pytest.approx(-1.0)
    assert geometry["pairwise"]["b__c"]["cosine"] == pytest.approx(0.0)
    assert geometry["sum_gradient_norm"] == pytest.approx(1.0)
    assert geometry["cancellation_ratio"] == pytest.approx(1.0 / 3.0)
