# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Actor-only FP32 masters preserve small updates and inference dtype contracts."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.workers.actor import fsdp_actor_worker as actor_module


class MixedPrecisionModel(torch.nn.Module):
    """Small policy with tied trainable weights and frozen inference state."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.02], dtype=torch.bfloat16))
        self.tied_weight = self.weight
        self.fp32 = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.frozen = torch.nn.Parameter(
            torch.tensor([0.3], dtype=torch.bfloat16), requires_grad=False
        )
        self.register_buffer("running", torch.tensor([0.4], dtype=torch.bfloat16))
        self.register_buffer("indices", torch.tensor([2], dtype=torch.long))


def actor_model(monkeypatch, model, enabled=None):
    """Exercise the real provider without starting Ray or allocating a GPU."""
    cfg = OmegaConf.create({"actor": {"model": {}}, "runner": {}})
    if enabled is not None:
        cfg.actor.cast_trainable_params_to_fp32 = enabled
    worker = SimpleNamespace(cfg=cfg, log_info=lambda _: None)
    monkeypatch.setattr(actor_module, "get_model", lambda _: model)
    return actor_module.EmbodiedFSDPActor.model_provider_func(worker)


def test_actor_master_cast_is_opt_in(monkeypatch):
    model = MixedPrecisionModel()
    original = model.weight
    output = actor_model(monkeypatch, model)
    assert output is model
    assert model.weight is original
    assert model.weight.dtype == torch.bfloat16


def test_actor_cast_preserves_frozen_buffers_and_ties(monkeypatch):
    model = MixedPrecisionModel()
    original_parameter = model.weight
    expected = model.weight.detach().float().clone()
    frozen_ptr, buffer_ptr = model.frozen.data_ptr(), model.running.data_ptr()
    fp32_ptr = model.fp32.data_ptr()
    model.weight.grad = torch.ones_like(model.weight)

    output = actor_model(monkeypatch, model, enabled=True)

    assert output is model
    assert model.weight is original_parameter is model.tied_weight
    assert model.weight.requires_grad
    assert model.weight.dtype == model.weight.grad.dtype == torch.float32
    torch.testing.assert_close(model.weight, expected)
    assert model.fp32.data_ptr() == fp32_ptr
    assert model.frozen.dtype == torch.bfloat16
    assert not model.frozen.requires_grad
    assert model.frozen.data_ptr() == frozen_ptr
    assert model.running.dtype == torch.bfloat16
    assert model.running.data_ptr() == buffer_ptr
    assert model.indices.dtype == torch.long


def test_small_adamw_update_survives_in_fp32_master(monkeypatch):
    model = actor_model(monkeypatch, MixedPrecisionModel(), enabled=True)
    before = model.weight.detach().clone()
    optimizer = torch.optim.AdamW([model.weight], lr=2.5e-5, weight_decay=0.0)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()

    torch.testing.assert_close(before - model.weight, torch.full_like(before, 2.5e-5))
    assert optimizer.state[model.weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[model.weight]["exp_avg_sq"].dtype == torch.float32

    # A BF16 optimizer-facing parameter loses the same update at this magnitude.
    low_precision = torch.nn.Parameter(before.to(torch.bfloat16))
    low_optimizer = torch.optim.AdamW([low_precision], lr=2.5e-5, weight_decay=0.0)
    low_precision.grad = torch.ones_like(low_precision)
    low_optimizer.step()
    torch.testing.assert_close(low_precision.float(), before, rtol=0, atol=0)
