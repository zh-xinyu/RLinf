# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate mapping of real FSDP flat gradients back to selected head parameters."""

from types import SimpleNamespace

import pytest
import torch

from toolkits.libero_gr00t.fsdp_parity import selected_flat_gradients, tensor_difference


def test_padding_and_wrapper_names_do_not_corrupt_selected_gradients():
    flat = SimpleNamespace(
        grad=torch.tensor([1.0, 2.0, 99.0, 99.0, 3.0, 4.0, 5.0, 6.0]),
        _numels_with_padding=(2, 2, 4),
        _is_padding_mask=(False, True, False),
        _fqns=("bias", "inner.weight"),
        _shapes=((2,), (2, 2)),
    )
    result = selected_flat_gradients(
        "_fsdp_wrapped_module.action_head", flat, {"action_head.inner.weight"}
    )
    assert set(result) == {"action_head.inner.weight"}
    torch.testing.assert_close(
        result["action_head.inner.weight"], torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    )


def test_sharded_gradients_are_rejected_instead_of_claiming_world_one_parity():
    flat = SimpleNamespace(grad=torch.ones(2), _numels_with_padding=(4,))
    with pytest.raises(ValueError, match="world-one"):
        selected_flat_gradients("", flat, set())


def test_difference_reports_failed_gradient_and_nonzero_reference():
    report = tensor_difference(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]), 1e-5, 1e-4
    )
    assert not report["close"] and report["max_abs"] == 1
    assert report["reference_nonzero_elements"] == 2
