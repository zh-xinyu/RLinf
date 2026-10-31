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
"""Avoid unused language logits in the GR00T N1.5 feature backbone."""

from types import MethodType

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def _forward_without_logits(self, *args, **kwargs) -> CausalLMOutputWithPast:
    """Preserve decoder features and autograd while omitting the vocabulary head."""
    if kwargs.pop("labels", None) is not None:
        raise ValueError("GR00T feature-only backbone does not support language loss.")
    if kwargs.pop("logits_to_keep", 0) != 0:
        raise ValueError("GR00T feature-only backbone does not produce logits.")
    if kwargs.get("return_dict") is False:
        raise ValueError("GR00T feature-only backbone requires dictionary outputs.")
    kwargs["return_dict"] = True
    outputs = self.model(*args, **kwargs)
    return CausalLMOutputWithPast(
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def enable_feature_only_backbone(model: torch.nn.Module) -> None:
    """Skip Qwen3 vocabulary projection without changing parameters or features.

    Eagle consumes the selected decoder hidden state, never token logits. Retain
    the full hidden-state tuple and its original layer/normalization semantics.
    The LM head stays registered (including tied embeddings) for checkpoint and
    FSDP compatibility. This switch is specific to GR00T's Qwen3 feature path.
    """
    language_model = model.backbone.eagle_model.language_model
    if language_model.config.model_type != "qwen3":
        raise ValueError("Feature-only GR00T currently requires a Qwen3 backbone.")
    if hasattr(language_model, "_opd_original_forward"):
        return
    language_model._opd_original_forward = language_model.forward
    language_model.forward = MethodType(_forward_without_logits, language_model)
