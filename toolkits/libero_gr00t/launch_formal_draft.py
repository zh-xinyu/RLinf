# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""UNFROZEN four-arm draft adapter; reuse production launcher without file edits.

Register exactly one selected draft config through the existing CONFIG_NAME in
this process, then call the unchanged production.main ownership/execution path.
Default remains dry-run. This entrypoint does not establish experiment readiness.
"""

import argparse
import sys

from toolkits.libero_gr00t import launch_production as production

ARMS = {"m0": (150, 25), "m1": (300, 50), "m2": (300, 50), "m3": (150, 25)}
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
    "--student-padding-profile",
    "--run",
    "--dry-run",
}


def draft_arguments(argv: list[str]) -> tuple[str, list[str]]:
    """Select one arm and reject overrides that could silently alter the matrix."""
    options = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise ValueError("Duplicate options are not accepted by the draft adapter")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", required=True, choices=ARMS)
    selected, remaining = parser.parse_known_args(argv)
    for token in remaining:
        if token.startswith("--") and token.split("=", 1)[0] not in PASSTHROUGH:
            raise ValueError(f"Unsupported draft override: {token}")
    padding_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    padding_parser.add_argument(
        "--student-padding-profile", choices=["native"], default="native"
    )
    padding_parser.parse_known_args(remaining)
    updates, save = ARMS[selected.arm]
    name = f"libero_mopd_formal_draft_{selected.arm}"
    return name, remaining + [
        "--config-name",
        name,
        "--updates",
        str(updates),
        "--save-every-updates",
        str(save),
        "--rollout-steps",
        "520",
    ]


def main(argv: list[str] | None = None) -> None:
    """Delegate the real CLI after a temporary process-local config registration."""
    name, forwarded = draft_arguments(list(sys.argv[1:] if argv is None else argv))
    original_name, original_argv = production.CONFIG_NAME, sys.argv
    try:
        production.CONFIG_NAME = name
        sys.argv = [str(__file__), *forwarded]
        production.main()
    finally:
        production.CONFIG_NAME = original_name
        sys.argv = original_argv


if __name__ == "__main__":
    main()
