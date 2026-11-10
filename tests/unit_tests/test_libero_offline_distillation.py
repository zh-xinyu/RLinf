# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Budget balance and gradient reduction checks for fixed-observation distillation."""

import numpy as np
import pytest
import torch

from rlinf.algorithms.losses import compute_analytic_local_kl_loss
from toolkits.libero_gr00t.collect_teacher_trajectories import (
    read_attempts,
    save_episode,
    task_collection_complete,
)
from toolkits.libero_gr00t.offline_distillation import bucket_rows
from toolkits.libero_gr00t.offline_observations import sample_frame_indices


def test_task_sampling_does_not_follow_trajectory_lengths():
    ids = np.concatenate([np.full(10 + 100 * i, i) for i in range(10)])
    indices = sample_frame_indices(ids, 10000, 42)
    counts = np.bincount(ids[indices], minlength=10)
    assert np.all(np.abs(counts - 1000) < 150)
    np.testing.assert_array_equal(indices, sample_frame_indices(ids, 10000, 42))
    with pytest.raises(ValueError):
        sample_frame_indices(ids[ids != 9], 16, 0)


def test_native_buckets_preserve_rows_without_policy_padding():
    rows = [
        {
            "eagle_input_ids": torch.ones(1, length, dtype=torch.long),
            "eagle_attention_mask": torch.ones(1, length, dtype=torch.long),
            "index": i,
        }
        for i, length in enumerate([4, 8, 4, 4, 8, 12, 12])
    ]
    groups = bucket_rows(rows, 2)
    assert sorted(row["index"] for group in groups for row in group) == list(range(7))
    assert all(len(group) <= 2 for group in groups)
    assert all(
        len({r["eagle_input_ids"].shape[-1] for r in group}) == 1 for group in groups
    )


def test_native_bucket_rejects_padding():
    with pytest.raises(ValueError):
        bucket_rows(
            [
                {
                    "eagle_input_ids": torch.ones(1, 4),
                    "eagle_attention_mask": torch.tensor([[1, 1, 0, 0]]),
                }
            ],
            2,
        )


def test_mixed_prefix_microbatch_gradients_equal_suite_means():
    scale = torch.tensor(0.4, requires_grad=True)
    generator = torch.Generator().manual_seed(123)
    inputs = [torch.randn(6, 10, 7, generator=generator) for _ in range(4)]
    targets = [torch.randn(6, 10, 7, generator=generator) for _ in range(4)]
    prefixes = [5, 5, 5, 10]
    weights = torch.linspace(0.5, 2.0, 6)
    reference = (
        sum(
            compute_analytic_local_kl_loss(scale * x[:, :k], y[:, :k], weights)[0]
            for x, y, k in zip(inputs, targets, prefixes, strict=True)
        )
        / 4
    )
    reference.backward()
    expected = scale.grad.clone()
    scale.grad = None
    total = 0
    # Uneven2/3/1-sized native buckets must preserve the global sample average.
    for x, y, k in zip(inputs, targets, prefixes, strict=True):
        for begin, end in [(0, 2), (2, 5), (5, 6)]:
            loss, _ = compute_analytic_local_kl_loss(
                scale * x[begin:end, :k], y[begin:end, :k], weights[begin:end]
            )
            weighted = loss * ((end - begin) / 24)
            weighted.backward()
            total += float(weighted.detach())
    torch.testing.assert_close(scale.grad, expected)
    assert total == pytest.approx(float(reference.detach()), rel=1e-6)


def test_bounded_success_minimum_is_explicit():
    assert task_collection_complete(2, target=5, minimum=1)
    assert not task_collection_complete(0, target=5, minimum=1)
    assert not task_collection_complete(2, target=5, minimum=5)
    with pytest.raises(ValueError):
        task_collection_complete(2, target=5, minimum=0)


def test_successful_episodes_survive_partial_suite(tmp_path):
    import json

    row = {"task_id": 3, "initial_state_id": 7, "accepted": True}
    path = save_episode(
        tmp_path / "episodes",
        row,
        [np.zeros((2, 4, 4, 3), dtype=np.uint8)],
        [np.zeros(8, dtype=np.float32)],
        [np.zeros((5, 7), dtype=np.float32)],
        [5],
    )
    row["episode_file"] = str(path.relative_to(tmp_path))
    (tmp_path / "attempts.jsonl").write_text(json.dumps(row) + "\n")
    assert read_attempts(tmp_path) == [row]
    with np.load(path, allow_pickle=False) as data:
        assert data["images"].shape == (1, 2, 4, 4, 3)
        assert int(data["execution_counts"].sum()) == 5
    with pytest.raises(FileExistsError):
        save_episode(tmp_path / "episodes", row, [0], [0], [0], [0])
    path.unlink()
    with pytest.raises(ValueError):
        read_attempts(tmp_path)
