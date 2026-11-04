# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Explicit launch resources, separate from the frozen scientific recipe."""

from __future__ import annotations

import os
import re
import subprocess

PROFILES = (
    "gr00t-n15-robocasa365-20261019",
    "gr00t-n15-robocasa365-blackwell-cu128",
)


def gpu_ids(env: dict | None = None, require_allocation: bool = False) -> list[int]:
    """Select only allocated devices; never replace Slurm visibility with all GPUs."""
    env = os.environ if env is None else env
    count = int(env.get("OPD_NUM_GPUS", "8"))
    if count not in (4, 8):
        raise ValueError("Formal execution supports four or eight GPUs")
    slurm = env.get("OPD_SCHEDULER", "local") == "slurm"
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if slurm and require_allocation:
        if not env.get("SLURM_JOB_ID") or not visible:
            raise ValueError(
                "Run inside a Slurm GPU allocation with CUDA_VISIBLE_DEVICES"
            )
        if int(env.get("SLURM_JOB_NUM_NODES", "1")) != 1:
            raise ValueError("This recipe requires one Slurm node")
        if int(env.get("SLURM_CPUS_PER_TASK", "0")) < int(
            env.get("OPD_RAY_CPUS", "32")
        ):
            raise ValueError("Slurm cpus-per-task is smaller than ray_cpus")
    if visible is None or (not visible and not require_allocation):
        # CPU-only planning can happen on a login node; launch rechecks allocation.
        return list(range(count))
    ids = []
    for token in visible.split(","):
        token = token.strip()
        if token.isdigit():
            ids.append(int(token))
        elif re.fullmatch(r"GPU-[a-fA-F0-9-]+", token):
            # RLinf/EGL require integer device IDs; resolve only this allocated UUID.
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index",
                    "--format=csv,noheader,nounits",
                    f"--id={token}",
                ],
                text=True,
                env=dict(env),
            ).strip()
            if not output.isdigit():
                raise ValueError(f"Cannot resolve allocated GPU UUID: {token}")
            ids.append(int(output))
        else:
            raise ValueError(
                "Use whole NVIDIA GPUs; empty/MIG GPU assignments are unsupported"
            )
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"Expected {count} distinct allocated GPUs, got {ids}")
    return sorted(ids)


def resource_overrides() -> list[str]:
    """Return explicit placement and optional dedicated-node cache overrides."""
    # Quote the value to stop Hydra interpreting commas as a sweep.
    devices = ",".join(map(str, gpu_ids()))
    overrides = [
        "~cluster.component_placement",
        "+cluster.component_placement={"
        + ",".join(f"{role}:'{devices}'" for role in ("actor", "env", "rollout"))
        + "}",
    ]
    if os.environ.get("OPD_RESOURCE_MODE", "shared") == "dedicated":
        overrides += [
            "actor.cuda_cache_retention_gib=0",
            "actor.cuda_cache_preallocate_gib=0",
        ]
    return overrides
