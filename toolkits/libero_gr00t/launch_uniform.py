# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch the fixed Uniform-R1/R2/R4, T64, 100-round LIBERO controls."""

import argparse
import sys

from toolkits.libero_gr00t import launch_adaptive_query as query
from toolkits.libero_gr00t import launch_production as production

REUSE_UPDATES = (1, 2, 4)


def uniform_arguments(argv: list[str]) -> list[str]:
    """Preserve strict recipe validation while selecting a uniform reuse budget."""
    options = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise ValueError("Duplicate uniform launch options are not accepted")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--reuse-updates", type=int, choices=REUSE_UPDATES, required=True
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        choices=(2, 4, 8),
        help="Per-rank training microbatch; omit for YAML default4, freeze after node acceptance",
    )
    selected, remaining = parser.parse_known_args(argv)
    # Prevent a forwarded arm from replacing the uniform control selection.
    if "--arm" in options:
        raise ValueError("Uniform launch selects R with --reuse-updates, not --arm")
    repeats = selected.reuse_updates
    forwarded = query.query_arguments(
        ["--arm", "m0" if repeats == 1 else "m1", *remaining]
    )
    forwarded[forwarded.index("--config-name") + 1] = f"libero_mopd_uniform_r{repeats}"
    forwarded[forwarded.index("--updates") + 1] = str(100 * repeats)
    forwarded[forwarded.index("--save-every-updates") + 1] = str(10 * repeats)
    if selected.microbatch_size is not None:
        forwarded += ["--native-microbatch-size", str(selected.microbatch_size)]
    return forwarded


def main(argv: list[str] | None = None) -> None:
    """Use owned-Ray execution with plan-only defaults and explicit model paths."""
    forwarded = uniform_arguments(list(sys.argv[1:] if argv is None else argv))
    original = sys.argv
    try:
        sys.argv = [str(__file__), *forwarded]
        production.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
