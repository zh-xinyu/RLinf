# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Sample ordering and additive diagnostics for routed velocity distillation."""

import math
from collections.abc import Mapping
from numbers import Real

import torch

SUITE_STATS_KEY = "_suite_velocity_stats"


def normalized_suite_loss_weights(configured: Mapping, names: list[str]) -> list[float]:
    """Return positive suite coefficients with mean one in routing order."""
    if not isinstance(configured, Mapping) or set(configured) != set(names):
        raise ValueError("suite_loss_weights must name every routed teacher exactly.")
    values = [configured[name] for name in names]
    if not values or any(
        isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
        for v in values
    ):
        raise ValueError("suite_loss_weights must be finite positive numbers.")
    mean = sum(values) / len(values)
    return [v / mean for v in values]


def sample_weighted_velocity_loss(
    error: torch.Tensor, mask: torch.Tensor | None, sample_weights: torch.Tensor
) -> torch.Tensor:
    """Weight the numerator only; never normalize within a suite microbatch.

    The actor supplies mean-one coefficients and equal suite counts per update.
    Dividing by a microbatch's weight sum would cancel homogeneous-suite weights.
    ``error`` already includes the original flow-time weighting, if applicable.
    """
    if sample_weights.ndim != 1 or len(sample_weights) != len(error):
        raise ValueError("Expected one suite loss coefficient per sample.")
    weights = sample_weights.detach().to(error).reshape((-1,) + (1,) * (error.ndim - 1))
    weighted = error * weights
    if mask is None:
        return weighted.mean()
    mask = mask.expand_as(error)
    return (weighted * mask).sum() / mask.sum().clamp_min(1.0)


def group_sample_indices(
    sample_indices: torch.Tensor, teacher_ids: torch.Tensor, num_teachers: int
) -> torch.Tensor:
    """Group already selected rows stably, preserving duplicates and sample weights."""
    ids = teacher_ids.reshape(-1).cpu()[sample_indices.cpu()]
    if ids.numel() == 0 or ids.min() < 0 or ids.max() >= num_teachers:
        raise ValueError("Selected teacher IDs do not match the routed teachers.")
    order = torch.argsort(ids, stable=True).to(sample_indices.device)
    return sample_indices[order]


@torch.no_grad()
def suite_velocity_statistics(
    squared_error: torch.Tensor,
    weighted_error: torch.Tensor,
    mask: torch.Tensor | None,
    teacher_ids: torch.Tensor,
    num_teachers: int,
) -> torch.Tensor:
    """Return detached [suite, (MSE sum, weighted sum, valid elements)] on device.

    The mask, when present, must already broadcast to the velocity error shape.
    Empty suites contribute zeros; callers sum statistics before dividing, across
    both microbatches and distributed ranks. No host transfer occurs here.
    """
    ids = teacher_ids.reshape(-1).to(device=squared_error.device, dtype=torch.long)
    if ids.numel() != squared_error.shape[0] or num_teachers < 1:
        raise ValueError("Suite statistics require one teacher ID per sample.")
    if mask is None:
        counts = squared_error.new_full(
            (ids.numel(),), math.prod(squared_error.shape[1:])
        )
        error = squared_error
        weighted = weighted_error
    else:
        mask = mask.expand_as(squared_error)
        counts = mask.flatten(1).sum(1)
        error = squared_error * mask
        weighted = weighted_error * mask
    rows = torch.stack(
        (error.flatten(1).sum(1), weighted.flatten(1).sum(1), counts), dim=1
    ).to(torch.float64)
    stats = rows.new_zeros((num_teachers, 3))
    stats.index_add_(0, ids, rows)
    return stats


def finalize_suite_statistics(
    stats: torch.Tensor, names: list[str], weighted: bool
) -> dict[str, float]:
    """Format globally summed statistics with one device-to-host transfer."""
    if stats.shape != (len(names), 3):
        raise ValueError("Suite names and additive statistics shape differ.")
    result = {}
    for name, (error_sum, weighted_sum, count) in zip(names, stats.cpu().tolist()):
        result[f"actor/valid_velocity_elements/{name}"] = count
        # Absence is not zero loss. Skip means for globally empty suites.
        if count > 0:
            result[f"actor/velocity_mse/{name}"] = error_sum / count
            if weighted:
                result[f"actor/analytic_kl/{name}"] = weighted_sum / count
    return result
