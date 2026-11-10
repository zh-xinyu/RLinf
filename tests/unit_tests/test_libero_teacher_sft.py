# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Guard behavior-cloning action alignment, episode boundaries and terminal masks."""

from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.libero_gr00t.teacher_sft_data import (
    apply_label_transforms,
    executed_action_targets,
)


def actions_from_ids(ids):
    return np.broadcast_to(
        np.asarray(ids, dtype=np.float32)[..., None], (*np.shape(ids), 7)
    ).copy()


def test_labels_follow_executed_replans_and_mask_terminal_padding():
    actions = actions_from_ids([[0, 1, 2], [3, 4, 5], [6, 700, 800]])
    target, valid = executed_action_targets(actions, np.array([3, 3, 1]), horizon=5)
    np.testing.assert_array_equal(
        target[:, :, 0], [[0, 1, 2, 3, 4], [3, 4, 5, 6, 6], [6, 6, 6, 6, 6]]
    )
    np.testing.assert_array_equal(
        valid, [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]]
    )
    assert not np.isin(target, [700, 800]).any()


def test_separate_episodes_never_supply_each_others_future_actions():
    a, am = executed_action_targets(
        actions_from_ids([[11, 12]]), np.array([1]), horizon=4
    )
    b, bm = executed_action_targets(
        actions_from_ids([[91, 92]]), np.array([2]), horizon=4
    )
    np.testing.assert_array_equal(a[0, :, 0], [11, 11, 11, 11])
    np.testing.assert_array_equal(b[0, :, 0], [91, 92, 92, 92])
    assert am.sum() == 1 and bm.sum() == 2


@pytest.mark.parametrize(
    "counts",
    [np.array([0, 2]), np.array([3, 2]), np.array([1, 2]), np.array([2.0, 2.0])],
)
def test_invalid_execution_counts_fail(counts):
    with pytest.raises(ValueError):
        executed_action_targets(actions_from_ids([[1, 2], [3, 4]]), counts)


def test_nonfinite_actions_fail():
    actions = actions_from_ids([[1, 2]])
    actions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        executed_action_targets(actions, np.array([2]))


@pytest.mark.parametrize("raises", [False, True])
def test_action_packing_does_not_enable_image_augmentation_and_restores_mode(raises):
    image = SimpleNamespace(training=False)
    packer = SimpleNamespace(
        training=False, language_dropout_prob=0, eagle_processor=object()
    )

    def transform(data):
        assert packer.training and not image.training
        if raises:
            raise RuntimeError("Transform failed")
        return {"action_mask": True}

    model = SimpleNamespace(
        _modality_transform=SimpleNamespace(transforms=[image, packer]),
        apply_transforms=transform,
    )
    if raises:
        with pytest.raises(RuntimeError):
            apply_label_transforms(model, {})
    else:
        assert apply_label_transforms(model, {})["action_mask"]
    assert not packer.training and not image.training


def test_bf16_checkpoint_distribution_restores_the_same_native_fp32_beta_law():
    import torch

    from toolkits.libero_gr00t.teacher_sft import restore_native_time_sampler

    head = SimpleNamespace(
        config=SimpleNamespace(noise_beta_alpha=1.5, noise_beta_beta=1.0),
        beta_dist=torch.distributions.Beta(
            torch.tensor(1.5, dtype=torch.bfloat16),
            torch.tensor(1.0, dtype=torch.bfloat16),
        ),
    )
    restore_native_time_sampler(head)
    torch.manual_seed(123)
    observed = head.beta_dist.sample((32,))
    torch.manual_seed(123)
    expected = torch.distributions.Beta(
        torch.tensor(1.5, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)
    ).sample((32,))
    assert observed.dtype == torch.float32
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
