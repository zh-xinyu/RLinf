# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared, dependency-free contract for the six Offline50 LIBERO runs."""

from __future__ import annotations

ROUNDS = 80
CURVE_EPISODES = 20
CURVE_ROUNDS = (0, 10, 20, 30, 40, 60, 80)
INTERVALS = (5, 10)
RHO_VALUES = (0.5, 1.0)
ADAPTIVE_INTERVAL = 5
REUSE = (1, 2)


def rho_tag(rho: float) -> str:
    """Use stable filesystem/CSV labels for the two declared strengths."""
    if rho not in RHO_VALUES:
        raise ValueError("Adaptive strength must be rho0.5 or rho1")
    return "0p5" if rho == 0.5 else "1"


def config_name(
    method: str, repeats: int, interval: int | None = None, *, rho: float | None = None
) -> str:
    """Select a distinct tracked YAML without mutating shared configuration."""
    if repeats not in REUSE:
        raise ValueError("Offline50 experiments support R1/R2")
    if method == "uniform" and interval is None and rho is None:
        return f"libero_mopd_uniform_offline50_r{repeats}"
    if (
        method == "adaptive"
        and rho is not None
        and interval in (None, ADAPTIVE_INTERVAL)
    ):
        return f"libero_mopd_adaptive_release_offline50_rho{rho_tag(rho)}_r{repeats}"
    if method == "adaptive" and interval in INTERVALS and rho is None:
        return f"libero_mopd_adaptive_release_offline50_i{interval}_r{repeats}"
    raise ValueError("Adaptive requires rho0.5/1; interval-only presets are historical")


CONFIG_NAMES = (
    tuple(config_name("uniform", r) for r in REUSE)
    + tuple(config_name("adaptive", r, i) for i in INTERVALS for r in REUSE)
    + tuple(config_name("adaptive", r, rho=rho) for rho in RHO_VALUES for r in REUSE)
)
