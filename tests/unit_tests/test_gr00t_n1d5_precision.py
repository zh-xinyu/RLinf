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
"""Tests for GR00T N1.5 mixed checkpoint precision restoration."""

import json

import torch
from safetensors.torch import save_file
from torch import nn

from rlinf.models.embodiment.gr00t.gr00t_n1d5 import _restore_action_head_fp32


class _ToyActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(2))


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.action_head = _ToyActionHead()


def test_restore_action_head_fp32_from_sharded_checkpoint(tmp_path):
    expected_weight = torch.tensor([[0.1001, 0.2002, 0.3003], [0.4004, 0.5005, 0.6006]])
    expected_bias = torch.tensor([0.7007, 0.8008])
    expected_scale = torch.tensor([0.9009, 1.001])
    save_file(
        {"action_head.projection.weight": expected_weight},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "action_head.projection.bias": expected_bias,
            "action_head.scale": expected_scale,
        },
        tmp_path / "model-00002-of-00002.safetensors",
    )
    index = {
        "weight_map": {
            "action_head.projection.weight": "model-00001-of-00002.safetensors",
            "action_head.projection.bias": "model-00002-of-00002.safetensors",
            "action_head.scale": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    model = _ToyModel().to(torch.bfloat16)
    restored = _restore_action_head_fp32(model, tmp_path)

    assert restored == 3
    assert model.backbone.weight.dtype == torch.bfloat16
    assert model.action_head.projection.weight.dtype == torch.float32
    assert model.action_head.scale.dtype == torch.float32
    torch.testing.assert_close(
        model.action_head.projection.weight, expected_weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.action_head.projection.bias, expected_bias, rtol=0, atol=0
    )
    torch.testing.assert_close(model.action_head.scale, expected_scale, rtol=0, atol=0)
