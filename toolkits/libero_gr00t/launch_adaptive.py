# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch the owner-tuned Adaptive presets with the shared fixed T64 budgets."""

import sys

from toolkits.libero_gr00t import launch_production as production
from toolkits.libero_gr00t.launch_uniform import uniform_arguments


def adaptive_arguments(argv: list[str]) -> list[str]:
    """Reuse resource/numerical guards while keeping Adaptive YAMLs independent."""
    forwarded = uniform_arguments(argv)
    index = forwarded.index("--config-name") + 1
    forwarded[index] = forwarded[index].replace(
        "libero_mopd_uniform_r", "libero_mopd_adaptive_release_r"
    )
    return forwarded


def main(argv: list[str] | None = None) -> None:
    """Run only when explicitly selected, using a bound teacher reference."""
    forwarded = adaptive_arguments(list(sys.argv[1:] if argv is None else argv))
    original = sys.argv
    try:
        sys.argv = [str(__file__), *forwarded]
        production.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
