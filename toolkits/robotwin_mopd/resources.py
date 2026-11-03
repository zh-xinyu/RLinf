# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Resource limits shared by local and Slurm experiment launchers."""


def cpu_budget(requested: int | None, environ: dict, affinity: int) -> int:
    """Respect scheduler and affinity limits without changing the local default."""
    allocation = int(environ.get("SLURM_CPUS_PER_TASK", affinity))
    limit = min(allocation, affinity)
    value = requested if requested is not None else min(48, limit)
    if not 1 <= value <= limit:
        raise ValueError(
            f"CPU request {value} exceeds allocation/affinity limit {limit}"
        )
    return value
