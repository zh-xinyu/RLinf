# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared, dependency-free contract for the six Offline50 LIBERO runs."""

from __future__ import annotations

ROUNDS = 80
CURVE_EPISODES = 20
CURVE_ROUNDS = (0, 10, 20, 30, 40, 60, 80)
INTERVALS = (5, 10)
REUSE = (1, 2)


def config_name(method: str, repeats: int, interval: int | None = None) -> str:
    """Select a distinct tracked YAML without mutating shared configuration."""
    if repeats not in REUSE:
        raise ValueError("Offline50 experiments support R1/R2")
    if method == "uniform" and interval is None:
        return f"libero_mopd_uniform_offline50_r{repeats}"
    if method == "adaptive" and interval in INTERVALS:
        return f"libero_mopd_adaptive_release_offline50_i{interval}_r{repeats}"
    raise ValueError("Adaptive requires interval5/10; Uniform has no interval")


CONFIG_NAMES = tuple(config_name("uniform", r) for r in REUSE) + tuple(
    config_name("adaptive", r, i) for i in INTERVALS for r in REUSE
)
