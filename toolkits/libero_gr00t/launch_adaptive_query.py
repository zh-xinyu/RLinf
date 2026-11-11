# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch gap-squared/query-synchronized arms with the tested LIBERO T64 recipe."""

import sys

from toolkits.libero_gr00t import launch_formal as historical
from toolkits.libero_gr00t import launch_production as production

# Preserve 100 physical rounds, with the newly agreed ten-round save/eval grid.
ARMS = {"m0": (100, 10), "m1": (200, 20), "m2": (200, 20), "m3": (100, 10)}


def query_arguments(argv: list[str]) -> list[str]:
    """Reuse the fixed budgets and safe resource flags of the formal launcher."""
    options = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    if "--trajectory-budget" not in options:
        argv = [*argv, "--trajectory-budget", "64"]
    forwarded = historical.formal_arguments(argv)
    index = forwarded.index("--config-name") + 1
    prefix = "libero_mopd_formal_t64_"
    if not forwarded[index].startswith(prefix):
        raise ValueError("The adaptive-query recipe requires trajectory-budget64")
    arm = forwarded[index].removeprefix(prefix)
    forwarded[index] = f"libero_mopd_adaptive_query_{arm}"
    updates, save = ARMS[arm]
    forwarded[forwarded.index("--updates") + 1] = str(updates)
    forwarded[forwarded.index("--save-every-updates") + 1] = str(save)
    return forwarded


def main(argv: list[str] | None = None) -> None:
    """Keep owned-Ray cleanup, explicit checkpoints and plan-only defaults."""
    forwarded = query_arguments(list(sys.argv[1:] if argv is None else argv))
    original = sys.argv
    try:
        sys.argv = [str(__file__), *forwarded]
        production.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
