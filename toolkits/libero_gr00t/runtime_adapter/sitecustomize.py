# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Process-scoped EGL/backend selection for the pinned LIBERO runtime."""

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path


class _NodeEGLFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if os.environ.get("OPD_LIBERO_AUTOMAP_EGL") != "1":
            return None
        files = {
            "robosuite.utils.binding_utils": "binding_utils.py",
            "robosuite.renderers.context.egl_context": "egl_context.py",
        }
        if fullname in files:
            return importlib.util.spec_from_file_location(
                fullname, Path(__file__).parent / files[fullname]
            )
        return None


sys.meta_path.insert(0, _NodeEGLFinder())

# PyTorch2.7.1 SM120 FP32 efficient SDPA fails the retained FP64 reference check.
if os.environ.get("OPD_BLACKWELL_SDPA_SAFE") == "1":
    import torch

    torch.backends.cuda.enable_mem_efficient_sdp(False)
