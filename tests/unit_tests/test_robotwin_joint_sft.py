# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the isolated four-task joint demonstration warmstart."""

import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from toolkits.robotwin_mopd.joint_sft import (
    cast_trainable_fp32,
    demo_forward,
    initialize_wandb,
    make_flow_targets,
    memory_metrics,
    suite_totals,
    totals_metrics,
    wandb_payload,
)
from toolkits.robotwin_mopd.joint_sft_data import (
    RankBatchIndices,
    action_chunk,
    collate_demo_rows,
    decode_rgb_jpeg,
    validate_action_alignment,
)


def test_official_next_state_alignment_and_terminal_chunk():
    states = np.arange(4 * 14, dtype=np.float32).reshape(4, 14)
    actions = np.concatenate([states[1:], states[-1:]])
    validate_action_alignment(states, actions)
    chunk = action_chunk(actions, 2, horizon=50)
    assert chunk.shape == (50, 14)
    np.testing.assert_array_equal(chunk, np.repeat(states[-1:], 50, axis=0))
    with pytest.raises(ValueError, match="next-state"):
        validate_action_alignment(states, states)


def test_marked_rgb_decode_matches_official_opencv_convention():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :] = [220, 25, 60]
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    raw = encoded.tobytes()
    # Same COM payload as official XPolicyLab.encode_image_bit.
    segment = b"\xff\xfe\x00\x0aXPL-RGB1"
    marked = raw[:2] + segment + raw[2:]
    expected = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(marked, np.uint8), 1), cv2.COLOR_BGR2RGB
    )
    np.testing.assert_array_equal(decode_rgb_jpeg(marked), expected)
    with pytest.raises(ValueError, match="marked"):
        decode_rgb_jpeg(raw)


def test_non_jpeg_is_rejected_even_with_correct_pixels():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    with pytest.raises(ValueError, match="marked"):
        decode_rgb_jpeg(buffer.getvalue())


def test_global_batches_balanced_and_rank_disjoint():
    batches = [list(RankBatchIndices(128, 4, rank, 2)) for rank in range(4)]
    for step in range(2):
        all_rows = [
            row for rows in batches for row in rows[step * 32 : (step + 1) * 32]
        ]
        assert sorted(all_rows) == list(range(step * 128, (step + 1) * 128))
        assert np.bincount(np.array(all_rows) % 4).tolist() == [32] * 4


def test_conditional_flow_target_endpoints_and_sign():
    actions = torch.ones((3, 50, 32)) * 2
    noise = torch.ones_like(actions) * -3
    x_t, target = make_flow_targets(actions, noise, torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(x_t[:, 0, 0], torch.tensor([2.0, -0.5, -3.0]))
    torch.testing.assert_close(target, noise - actions)
    with pytest.raises(ValueError, match="shapes"):
        make_flow_targets(actions, noise[:, :25], torch.zeros(3))


def test_demo_forward_full_horizon_gradient_and_aug_flag():
    weight = torch.nn.Parameter(torch.tensor(0.25))
    seen = {}

    def preprocess(observation, *, train):
        seen["augment"] = train
        return [observation], [], None, None, None

    def velocity(images, masks, tokens, token_masks, state, x_t, timesteps):
        return weight * x_t, None

    model = SimpleNamespace(
        _preprocess_observation=preprocess, get_joint_velocity=velocity
    )
    actions = torch.ones((2, 50, 32))
    noise = torch.zeros_like(actions)
    times = torch.tensor([0.2, 0.7])
    output = demo_forward(
        model,
        observation=None,
        actions=actions,
        noise=noise,
        timesteps=times,
        augment_images=False,
    )
    expected = (weight * (1 - times[:, None, None]) + 1).square().flatten(1).mean(1)
    torch.testing.assert_close(output["loss_by_sample"], expected)
    output["loss_by_sample"].mean().backward()
    assert weight.grad is not None and weight.grad.abs() > 0
    assert seen["augment"] is False


def test_fp32_cast_preserves_tied_and_frozen_contract():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    model.alias = model.weight
    model.frozen = torch.nn.Parameter(
        torch.ones(4, dtype=torch.bfloat16), requires_grad=False
    )
    model.register_buffer("cache", torch.ones(4, dtype=torch.bfloat16))
    identity = id(model.weight)
    cast_trainable_fp32(model)
    assert id(model.weight) == identity and model.alias is model.weight
    assert model.weight.dtype == torch.float32
    assert model.frozen.dtype == model.cache.dtype == torch.bfloat16


def test_suite_metrics_use_counts_not_mean_of_microbatch_means():
    values = torch.tensor([1.0, 3.0, 7.0])
    losses = dict.fromkeys(
        ("loss_by_sample", "physical_loss_by_sample", "padded_loss_by_sample"), values
    )
    totals = suite_totals(losses, torch.tensor([0, 0, 2]))
    metrics = totals_metrics(totals)
    assert metrics["adjust_bottle/loss"] == 2
    assert metrics["place_container_plate/loss"] == 7
    assert metrics["loss"] == pytest.approx(11 / 3)


def test_collation_preserves_task_identity_and_prompt():
    rows = [
        {
            "actions": np.zeros((50, 14), np.float32),
            "prompt": text,
            "opd_task_ids": index,
        }
        for index, text in enumerate(("a", "b"))
    ]
    batch = collate_demo_rows(rows)
    assert batch["actions"].shape == (2, 50, 14)
    assert batch["prompt"] == ["a", "b"]
    assert batch["opd_task_ids"].tolist() == [0, 1]


def test_memory_peak_uses_maximum_across_ranks():
    report = memory_metrics([[30.1, 46.0], [35.5, 45.0], [32.0, 47.0]])
    assert report["peak_allocated_gb"] == 35.5
    assert report["peak_reserved_gb"] == 47.0
    assert report["peak_memory_by_rank"][1]["allocated_gb"] == 35.5


def test_wandb_is_explicit_rank_zero_only():
    assert initialize_wandb(SimpleNamespace(wandb_project=None), rank=0) is None
    # Must return before importing/initializing wandb or accessing other options.
    assert initialize_wandb(SimpleNamespace(wandb_project="test"), rank=1) is None
    payload = wandb_payload(
        "validation",
        50,
        {"step": 999, "loss": 0.2, "peak_memory_by_rank": [{"rank": 0}]},
    )
    assert payload == {"optimizer_step": 50, "validation/loss": 0.2}
