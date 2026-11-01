# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real tiny Gemma attention verifies joint NFT values and observation gradients."""

from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("openpi")
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # noqa: E402
from transformers import GemmaConfig, GemmaModel  # noqa: E402

from rlinf.models.embodiment.openpi.openpi_action_model import (  # noqa: E402
    OpenPi0ForRLActionPrediction as Policy,
)


def tiny_policy():
    """Use actual patched Gemma attention and pi0.5 suffix embeddings on CPU."""
    policy = Policy.__new__(Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(action_horizon=3, action_chunk=3)
    policy.pi05 = True
    policy.gradient_checkpointing_enabled = False
    policy.action_in_proj = torch.nn.Linear(32, 32)
    policy.action_out_proj = torch.nn.Linear(32, 32)
    policy.time_mlp_in = torch.nn.Linear(32, 32)
    policy.time_mlp_out = torch.nn.Linear(32, 32)
    policy.prefix_proj = torch.nn.Linear(3, 32)

    def gemma(adaptive):
        cfg = GemmaConfig(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=4,
            hidden_activation="gelu_pytorch_tanh",
            use_adarms=adaptive,
            adarms_cond_dim=32 if adaptive else None,
            attention_dropout=0.0,
        )
        cfg._attn_implementation = "eager"
        return GemmaModel(cfg)

    backbone = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    torch.nn.Module.__init__(backbone)
    backbone.paligemma = torch.nn.Module()
    backbone.paligemma.model = torch.nn.Module()
    backbone.paligemma.model.language_model = gemma(False)
    backbone.paligemma.language_model = backbone.paligemma.model.language_model
    backbone.paligemma.config = SimpleNamespace(
        text_config=backbone.paligemma.language_model.config
    )
    backbone.gemma_expert = torch.nn.Module()
    backbone.gemma_expert.model = gemma(True)
    policy.paligemma_with_expert = backbone

    def embed_prefix(self, images, image_masks, tokens, token_masks):
        embeddings = self.prefix_proj(images[0])
        return embeddings, image_masks[0], torch.zeros_like(image_masks[0])

    policy.embed_prefix = MethodType(embed_prefix, policy)
    return policy


def inputs():
    return (
        [torch.randn(2, 4, 3)],
        [torch.tensor([[True, True, True, False], [True, True, True, True]])],
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1, dtype=torch.bool),
        torch.randn(2, 32),
        torch.randn(2, 3, 32),
        torch.tensor([0.2, 0.8]),
    )


def cached_velocity(policy, batch):
    _, prefix_mask, cache = policy._build_prefix_cache(*batch[:4])
    return policy.get_velocity(batch[4], batch[5], batch[6], prefix_mask, cache)[0]


def test_joint_and_cached_velocity_and_gradients_match():
    torch.manual_seed(3)
    policy = tiny_policy().eval()
    batch = inputs()
    cached = cached_velocity(policy, batch)
    cached.square().mean().backward()
    cached_grads = {
        key: value.grad.detach().clone()
        for key, value in policy.named_parameters()
        if value.grad is not None
    }
    policy.zero_grad(set_to_none=True)
    joint, _ = policy.get_joint_velocity(*batch)
    torch.testing.assert_close(joint, cached, rtol=2e-5, atol=2e-6)
    joint.square().mean().backward()
    for key, value in policy.named_parameters():
        if key in cached_grads:
            assert value.grad is not None, key
            torch.testing.assert_close(
                value.grad, cached_grads[key], rtol=3e-4, atol=2e-6
            )
    assert policy.prefix_proj.weight.grad.abs().sum() > 0


def test_joint_checkpointed_training_preserves_values_and_finite_gradients():
    torch.manual_seed(9)
    policy = tiny_policy().eval()
    batch = inputs()
    reference = cached_velocity(policy, batch).detach()
    policy.train()
    policy.gradient_checkpointing_enabled = True
    policy.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
    policy.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
    output, _ = policy.get_joint_velocity(*batch)
    torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-6)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert policy.prefix_proj.weight.grad.abs().sum() > 0
    before = policy.prefix_proj.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, policy.prefix_proj.weight)


def test_nft_routes_joint_training_and_cached_evaluation(monkeypatch):
    policy = tiny_policy()
    policy.config.nft_joint_forward = True
    policy.config.add_value_head = False
    batch = inputs()
    observation = {"image": {}, "image_mask": {}, "state": batch[4]}
    monkeypatch.setattr(policy, "input_transform", lambda obs, transpose: obs)
    monkeypatch.setattr(policy, "_preprocess_observation", lambda obs, train: batch[:5])
    joint = policy.get_joint_velocity
    cached = policy.get_velocity
    calls = []

    def record_joint(*args):
        calls.append("joint")
        return joint(*args)

    def record_cached(*args):
        calls.append("cached")
        return cached(*args)

    monkeypatch.setattr(policy, "get_joint_velocity", record_joint)
    monkeypatch.setattr(policy, "get_velocity", record_cached)
    kwargs = {"nft_inputs": {"x_t": batch[5], "timesteps": batch[6]}}
    training = policy.nft_forward(observation, **kwargs)
    assert training["x_t"] is batch[5]
    assert training["timesteps"] is batch[6]
    policy.eval()
    evaluation = policy.nft_forward(observation, **kwargs)
    assert calls == ["joint", "cached"]
    torch.testing.assert_close(
        training["v_theta"], evaluation["v_theta"], rtol=2e-5, atol=2e-6
    )
