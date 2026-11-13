# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Train a server's two assigned Offline50 runs, then automatically evaluate both."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from toolkits.libero_gr00t.run_offline50 import REPO, effective_settings, label

# Each four-GPU server collects160 rounds and performs240 optimizer updates.
ASSIGNMENTS = {
    1: (("uniform", 1, None), ("adaptive", 2, 1.0)),
    2: (("uniform", 2, None), ("adaptive", 1, 0.5)),
    3: (("adaptive", 1, 1.0), ("adaptive", 2, 0.5)),
}


def main(argv: list[str] | None = None) -> None:
    """Keep runtime/logging in the existing machine JSON; never reinstall tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--server", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Default prints two plans; --run trains both, then evaluates both",
    )
    args = parser.parse_args(argv)
    measurements = []
    for method, r, rho in ASSIGNMENTS[args.server]:
        selected = argparse.Namespace(
            **vars(args),
            method=method,
            r=r,
            rho=rho,
            interval=5 if rho is not None else None,
            seed=0,
        )
        settings = effective_settings(selected)
        run = Path(settings["results"]) / label(selected)
        command = [
            sys.executable,
            "-m",
            "toolkits.libero_gr00t.run_offline50",
            "--machine",
            str(args.machine.absolute()),
            "--method",
            method,
            "--r",
            str(r),
            "--seed",
            "0",
        ]
        if args.student is not None:
            command += ["--student", str(args.student.absolute())]
        if rho is not None:
            command += ["--rho", str(rho)]
        if args.run:
            complete = (run / "status").is_file() and (
                run / "status"
            ).read_text().strip() == "OK"
            if complete:
                print(f"Training already completed; retaining {run}", flush=True)
            else:
                if run.exists():
                    raise FileExistsError(
                        f"Existing incomplete training must be recovered separately; not overwritten: {run}"
                    )
                train = [*command, "--train-only", "--run"]
                print(shlex.join(train), flush=True)
                subprocess.run(train, cwd=REPO, check=True)
            measurements.append([*command, "--evaluate-all", "--run"])
        else:
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=REPO, check=True)
    # Final checkpoints are now available for both runs before lengthy evaluation.
    for command in measurements:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
