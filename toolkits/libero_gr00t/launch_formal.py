# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch the fixed T32/T64,100-round LIBERO recipes through the owned path."""

import argparse
import sys

from toolkits.libero_gr00t import launch_production as production

ARMS = {"m0": (100, 25), "m1": (200, 50), "m2": (200, 50), "m3": (100, 25)}
PASSTHROUGH = {
    "--python",
    "--models",
    "--student",
    "--spatial-teacher",
    "--output",
    "--gpus",
    "--seed",
    "--cpus",
    "--timeout-seconds",
    "--resume-checkpoint",
    "--minimum-free-gib",
    "--actor-cache-retention-gib",
    "--actor-cache-preallocate-gib",
    "--log-cuda-memory",
    "--native-rollout-batch-size",
    "--skip-finished-rollout",
    "--run",
    "--dry-run",
}


def formal_arguments(argv: list[str]) -> list[str]:
    """Reject numerical overrides that would silently change the shared recipe."""
    options = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise ValueError("Duplicate formal launch options are not accepted")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--trajectory-budget", type=int, choices=[32, 64], default=32)
    selected, remaining = parser.parse_known_args(argv)
    for token in remaining:
        if token.startswith("--") and token.split("=", 1)[0] not in PASSTHROUGH:
            raise ValueError(f"Unsupported formal recipe override: {token}")
    updates, save = ARMS[selected.arm]
    return remaining + [
        "--config-name",
        f"libero_mopd_formal_{'t64_' if selected.trajectory_budget == 64 else ''}{selected.arm}",
        "--run-purpose",
        "formal",
        "--updates",
        str(updates),
        "--save-every-updates",
        str(save),
        "--rollout-steps",
        "480",
        "--execution-protocol",
        "official-mixed",
    ]


def main(argv: list[str] | None = None) -> None:
    """Reuse checkpoint checks, private Ray execution and owner-scoped cleanup."""
    forwarded = formal_arguments(list(sys.argv[1:] if argv is None else argv))
    original = sys.argv
    try:
        sys.argv = [str(__file__), *forwarded]
        production.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
