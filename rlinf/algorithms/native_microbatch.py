# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native GR00T length buckets without adding padded text to the policy."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeMicrobatch:
    """Fixed-size input rows and the number of genuine selected samples."""

    indices: torch.Tensor
    valid_rows: int


def native_microbatch_plan(
    lengths: torch.Tensor, microbatch: int, minimum_slots: int = 0
) -> list[NativeMicrobatch]:
    """Cover every selected row once, including repeated observation draws.

    Every slot has one true text length and exactly ``microbatch`` input rows.
    Duplicate inputs fill remainders only; their losses must be masked out.
    ``minimum_slots`` synchronizes the number of FSDP calls across ranks.
    """
    if lengths.ndim != 1 or lengths.numel() == 0 or microbatch < 1:
        raise ValueError("Expected nonempty token lengths and a positive microbatch")
    values = lengths.detach().cpu().tolist()
    if any(int(value) != value or value <= 0 for value in values):
        raise ValueError("True text lengths must be positive integers")
    buckets = {}
    for row, length in enumerate(values):
        buckets.setdefault(length, []).append(row)
    plan = []
    for rows in buckets.values():
        for offset in range(0, len(rows), microbatch):
            selected = rows[offset : offset + microbatch]
            padded = selected + [selected[0]] * (microbatch - len(selected))
            plan.append(NativeMicrobatch(torch.tensor(padded), len(selected)))
    while len(plan) < minimum_slots:
        plan.append(NativeMicrobatch(torch.zeros(microbatch, dtype=torch.long), 0))
    return plan


def select_native_microbatch(batch: dict, slot: NativeMicrobatch) -> dict:
    """Select flat rows recursively without modifying the observation pool."""
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = select_native_microbatch(value, slot)
        elif isinstance(value, torch.Tensor):
            result[key] = value.index_select(0, slot.indices.to(value.device))
        else:
            raise TypeError(f"Native microbatch expects tensor leaves: {key}")
    return result
