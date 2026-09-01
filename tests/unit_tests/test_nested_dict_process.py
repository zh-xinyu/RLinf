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
"""Tests for nested rollout-batch tensor processing."""

import pytest
import torch

from rlinf.utils.nested_dict_process import stack_list_of_dict_tensor


def test_stack_list_pads_only_ragged_eagle_prompt_fields():
    batches = [
        {
            "eagle_input_ids": torch.tensor([[11, 12, 13], [21, 22, 23]]),
            "eagle_attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "states": torch.ones(2, 4),
        },
        {
            "eagle_input_ids": torch.tensor(
                [[31, 32, 33, 34], [41, 42, 43, 44]]
            ),
            "eagle_attention_mask": torch.ones(2, 4, dtype=torch.bool),
            "states": torch.zeros(2, 4),
        },
    ]

    stacked = stack_list_of_dict_tensor(batches)

    assert stacked["eagle_input_ids"].shape == (2, 2, 4)
    assert stacked["eagle_attention_mask"].shape == (2, 2, 4)
    torch.testing.assert_close(
        stacked["eagle_input_ids"][0, :, -1], torch.tensor([0, 0])
    )
    torch.testing.assert_close(
        stacked["eagle_attention_mask"][0, :, -1],
        torch.tensor([False, False]),
    )
    assert stacked["states"].shape == (2, 2, 4)


def test_stack_list_still_rejects_other_ragged_tensors():
    batches = [
        {"states": torch.ones(2, 3)},
        {"states": torch.ones(2, 4)},
    ]

    with pytest.raises(RuntimeError, match="stack expects each tensor"):
        stack_list_of_dict_tensor(batches)
