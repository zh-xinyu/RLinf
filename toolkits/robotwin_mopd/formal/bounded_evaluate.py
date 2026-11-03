# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small deployment smoke with a bounded PyTorch allocator on a shared GPU.

The allocator cap does not cover Vulkan/CUDA context allocations. Leave extra
free device memory; this helper is not the formal evaluation entrypoint.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=0.35)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.fraction <= 0.5:
        parser.error("Shared-GPU smoke requires 0 < fraction <= 0.5")
    if torch.cuda.device_count() != 1:
        parser.error("Select exactly one visible GPU")
    torch.cuda.set_per_process_memory_fraction(args.fraction, device=0)
    arguments = args.evaluator_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    sys.argv = ["evaluate.py", *arguments]
    from toolkits.robotwin_mopd.evaluate import main as evaluate

    try:
        evaluate()
    finally:
        args.memory_report.parent.mkdir(parents=True, exist_ok=True)
        args.memory_report.write_text(
            json.dumps(
                {
                    "allocator_fraction": args.fraction,
                    "device_total_bytes": torch.cuda.get_device_properties(
                        0
                    ).total_memory,
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "max_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "scope": "PyTorch only; excludes Vulkan and CUDA context memory",
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
