# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""Opt-in retention of a dedicated actor process's existing CUDA cache."""

import math

import torch


def retain_cuda_cache(max_reserved_gib: float) -> None:
    """Skip explicit cache releases below a bound in this worker process only.

    No memory is preallocated. Live tensors and allocator OOM reclamation are
    unaffected. Above the bound, explicit empty_cache calls work as usual.
    This is not a GPU lock: other processes, CUDA contexts and fragmentation can
    still cause OOM. Enable only with headroom for colocated rollout/env workers.
    """
    if not math.isfinite(max_reserved_gib) or max_reserved_gib <= 0:
        raise ValueError("CUDA cache retention must be finite and positive.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA cache retention requires a CUDA actor.")
    if getattr(torch.cuda.empty_cache, "_rlinf_cache_retention", False):
        raise RuntimeError("CUDA cache retention is already configured.")
    original = torch.cuda.empty_cache
    limit = int(max_reserved_gib * 1024**3)

    def bounded_empty_cache() -> None:
        if torch.cuda.memory_reserved() > limit:
            original()

    bounded_empty_cache._rlinf_cache_retention = True
    torch.cuda.empty_cache = bounded_empty_cache
