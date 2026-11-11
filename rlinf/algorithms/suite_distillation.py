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


def prefix_normalized_velocity_loss(
    error: torch.Tensor,
    mask: torch.Tensor | None,
    prefix_lengths: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average executed-prefix error per genuine observation before sample weighting.

    Episode/text-padding validity remains separate from the action prefix. This
    avoids assigning double suite weight to ten-action samples versus five-action
    samples merely because their valid action-coordinate counts differ.
    """
    if error.ndim != 3 or prefix_lengths.shape != (error.shape[0],):
        raise ValueError("Expected [batch,action,dim] errors and one prefix per row")
    lengths = prefix_lengths.detach().to(device=error.device)
    if torch.is_floating_point(lengths) or bool(
        ((lengths < 1) | (lengths > error.shape[1])).any()
    ):
        raise ValueError(
            "Executed prefixes must be positive integers within action width"
        )
    valid = (
        torch.arange(error.shape[1], device=error.device)[None, :, None]
        < lengths[:, None, None]
    ).expand_as(error)
    if mask is not None:
        episode = mask
        while episode.ndim < error.ndim:
            episode = episode.unsqueeze(-1)
        valid = valid & episode.expand_as(error).bool()
    counts = valid.flatten(1).sum(1)
    values = (error * valid).flatten(1).sum(1) / counts.clamp_min(1)
    if sample_weights is not None:
        if sample_weights.shape != values.shape:
            raise ValueError("Need one suite loss coefficient per observation")
        values = values * sample_weights.detach().to(values)
    return values.sum() / (counts > 0).sum().clamp_min(1), valid.to(error.dtype)


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


def suite_query_quotas(
    configured: Mapping | None, names: list[str], total: int
) -> list[int]:
    """Allocate a fixed query budget by largest remainder, preserving coverage.

    The default is the existing equal quota with ties resolved in routing order.
    Explicit proportions affect sampling only; callers must not also apply suite
    loss coefficients. Reject budgets that would leave a suite without queries.
    """
    if not names or total < len(names):
        raise ValueError("The query budget must cover every routed teacher.")
    if configured is None:
        return [
            total // len(names) + int(i < total % len(names)) for i in range(len(names))
        ]
    try:
        weights = normalized_suite_loss_weights(configured, names)
    except ValueError as error:
        raise ValueError(
            str(error).replace("suite_loss_weights", "suite_query_weights")
        ) from error
    targets = [total * weight / sum(weights) for weight in weights]
    quotas = [math.floor(value) for value in targets]
    order = sorted(range(len(names)), key=lambda i: (-(targets[i] - quotas[i]), i))
    for i in order[: total - sum(quotas)]:
        quotas[i] += 1
    if min(quotas) < 1:
        raise ValueError(
            "suite_query_weights must leave at least one query per suite per rank."
        )
    return quotas


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
    if names and all(f"actor/velocity_mse/{name}" in result for name in names):
        result["actor/macro_velocity_mse"] = sum(
            result[f"actor/velocity_mse/{name}"] for name in names
        ) / len(names)
        if weighted:
            result["actor/macro_analytic_kl"] = sum(
                result[f"actor/analytic_kl/{name}"] for name in names
            ) / len(names)
    return result
