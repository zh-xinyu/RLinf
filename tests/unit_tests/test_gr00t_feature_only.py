# Copyright 2026 The RLinf Authors.
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
"""Check real Qwen3 hidden-state/gradient equivalence without simulator or GPU."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rlinf.models.embodiment.gr00t.gr00t_n1d5.memory_utils import (
    enable_feature_only_backbone,
)


@pytest.mark.parametrize("training", [False, True])
def test_feature_only_preserves_features_gradients_and_checkpoint(training):
    torch.manual_seed(7)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=True,
    )
    lm = Qwen3ForCausalLM(cfg).train(training)
    wrapper = SimpleNamespace(
        backbone=SimpleNamespace(eagle_model=SimpleNamespace(language_model=lm))
    )
    inputs = torch.randint(0, 64, (2, 9))
    mask = torch.ones_like(inputs)
    mask[0, -2:] = 0
    original = lm(input_ids=inputs, attention_mask=mask, output_hidden_states=True)
    expected = [h.detach().clone() for h in original.hidden_states]
    original.hidden_states[-1].square().mean().backward()
    grads = {
        k: p.grad.clone() if p.grad is not None else None
        for k, p in lm.named_parameters()
    }
    keys = set(lm.state_dict())
    parameter_ids = {k: id(p) for k, p in lm.named_parameters()}
    lm.zero_grad(set_to_none=True)
    enable_feature_only_backbone(wrapper)
    enable_feature_only_backbone(wrapper)
    calls = []
    hook = lm.lm_head.register_forward_hook(lambda *args: calls.append(1))
    actual = lm(input_ids=inputs, attention_mask=mask, output_hidden_states=True)
    assert actual.logits is None and not calls
    for a, b in zip(actual.hidden_states, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    actual.hidden_states[-1].square().mean().backward()
    for name, p in lm.named_parameters():
        if grads[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)
    assert keys == set(lm.state_dict())
    assert parameter_ids == {k: id(p) for k, p in lm.named_parameters()}
    assert lm.lm_head.weight is lm.model.embed_tokens.weight
    with pytest.raises(ValueError, match="language loss"):
        lm(input_ids=inputs, labels=inputs)
    hook.remove()
