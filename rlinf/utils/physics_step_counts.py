# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exact, session-local simulator counters, distinct from valid loss masks."""

from collections.abc import Iterable, Mapping
from numbers import Integral

PHYSICS_COUNT_KEYS = (
    "policy_simulation_steps",
    "settling_simulation_steps",
    "total_simulation_steps",
    "reset_episodes",
)


def sum_physics_step_counts(rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Validate and sum counters across stages or ranks without averaging.

    Counters start at environment creation and are not checkpoint-restored. A
    missing source is an error: silently treating it as zero would undercount
    physical interaction when an enabled environment lacks instrumentation.
    """
    totals = dict.fromkeys(PHYSICS_COUNT_KEYS, 0)
    sources = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(PHYSICS_COUNT_KEYS):
            raise ValueError("Every environment/rank must expose all physics counters")
        for key in PHYSICS_COUNT_KEYS:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"Physics counter {key} must be a nonnegative integer")
        if row["total_simulation_steps"] != (
            row["policy_simulation_steps"] + row["settling_simulation_steps"]
        ):
            raise ValueError("Physics total must equal policy plus settling steps")
        for key in PHYSICS_COUNT_KEYS:
            totals[key] += int(row[key])
        sources += 1
    if not sources:
        raise ValueError(
            "Physics logging requires at least one instrumented environment"
        )
    return totals


def physics_session_metrics(counts: Mapping[str, int]) -> dict[str, int]:
    """Name metrics explicitly as process-session counts, not resumed lifetime totals."""
    checked = sum_physics_step_counts([counts])
    labels = {
        "policy_simulation_steps": "policy_steps",
        "settling_simulation_steps": "settling_steps",
        "total_simulation_steps": "total_steps",
        "reset_episodes": "reset_episodes",
    }
    return {
        f"interaction/physics_{labels[key]}_session": value
        for key, value in checked.items()
    }
