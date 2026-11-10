# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""The actor must drop GPU views/cache before completing rollout weight sync."""

import asyncio
import inspect
import weakref
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.utils import cuda_cache
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


@pytest.mark.parametrize("retention", [0, 20])
def test_handoff_finishes_transfer_then_drops_views_and_releases(
    monkeypatch, retention
):
    events, views = [], []

    def state_dict():
        tensor = torch.ones(3)
        views.append(weakref.ref(tensor))
        return {"parameter": tensor}

    async def sync(state, send_func, version):
        assert version == 2 and views[0]() is state["parameter"]
        events.append("synchronized")

    def offload(grad):
        assert grad and views[0]() is not None
        events.append("offloaded")
        worker.is_weight_offloaded = True

    def release():
        assert views[0]() is None
        events.append("released")

    monkeypatch.setattr(cuda_cache, "release_cuda_cache_to_floor", release)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 12 * 2**30)
    worker = SimpleNamespace(
        cfg=OmegaConf.create({"actor": {"cuda_cache_retention_gib": retention}}),
        enable_offload=True,
        is_optimizer_offloaded=True,
        is_weight_offloaded=False,
        get_rollout_state_dict=state_dict,
        weight_syncer=SimpleNamespace(sender_initialized=lambda: True, sync=sync),
        get_rollout_sync_version=lambda: 2,
        offload_param_and_grad=offload,
        log_info=lambda message: None,
    )
    asyncio.run(inspect.unwrap(EmbodiedFSDPActor.sync_model_to_rollout)(worker))
    expected = ["synchronized", "offloaded"] + (["released"] if retention else [])
    assert events == expected
