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
