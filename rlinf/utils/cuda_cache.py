# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""Opt-in retention of a dedicated actor process's existing CUDA cache."""

import math

import torch


def retain_cuda_cache(
    max_reserved_gib: float, preallocate_gib: float = 0, headroom_gib: float = 8
) -> None:
    """Skip explicit cache releases below a bound in this worker process only.

    Optional preallocation warms this process's reusable allocator cache, then
    drops all temporary tensors. It does not consume RNG or hold a live ballast.
    Above the bound, explicit empty_cache calls release cache before refilling
    the requested floor. Live tensors and allocator OOM reclamation are unchanged.
    This is not a GPU lock: other processes, CUDA contexts and fragmentation can
    still cause OOM. Enable only with headroom for colocated rollout/env workers.
    """
    if not math.isfinite(max_reserved_gib) or max_reserved_gib <= 0:
        raise ValueError("CUDA cache retention must be finite and positive.")
    if (
        not math.isfinite(preallocate_gib)
        or not 0 <= preallocate_gib <= max_reserved_gib
    ):
        raise ValueError("CUDA cache preallocation must be between zero and retention.")
    if not math.isfinite(headroom_gib) or headroom_gib < 0:
        raise ValueError("CUDA cache headroom must be finite and nonnegative.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA cache retention requires a CUDA actor.")
    if getattr(torch.cuda.empty_cache, "_rlinf_cache_retention", False):
        raise RuntimeError("CUDA cache retention is already configured.")
    original = torch.cuda.empty_cache
    limit = int(max_reserved_gib * 1024**3)
    floor = int(preallocate_gib * 1024**3)
    headroom = int(headroom_gib * 1024**3)

    def warm_cache() -> None:
        missing = floor - torch.cuda.memory_reserved()
        if missing <= 0:
            return
        free, _ = torch.cuda.mem_get_info()
        if free < missing + headroom:
            raise RuntimeError(
                "Insufficient free GPU memory for actor cache preallocation "
                "and colocated-worker headroom; refusing to reserve memory. "
                f"device={torch.cuda.current_device()}, free_gib={free / 1024**3:.3f}, "
                f"reserved_gib={torch.cuda.memory_reserved() / 1024**3:.3f}, "
                f"missing_gib={missing / 1024**3:.3f}, headroom_gib={headroom_gib}"
            )
        blocks = []
        try:
            while torch.cuda.memory_reserved() < floor:
                # Small adjacent allocator blocks can be reused by model forwards;
                # keep them live until the cache reaches the requested size.
                blocks.append(
                    torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")
                )
            torch.cuda.synchronize()
        finally:
            blocks.clear()

    def bounded_empty_cache() -> None:
        if torch.cuda.memory_reserved() > limit:
            original()
        warm_cache()

    warm_cache()
    bounded_empty_cache._rlinf_cache_retention = True
    torch.cuda.empty_cache = bounded_empty_cache
