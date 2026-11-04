# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Submit the four training jobs and 25 evaluations as a serial Slurm chain."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from toolkits.robocasa_mopd.experiment import (  # noqa: E402
    ARMS,
    ROUNDS,
    bind_outputs,
    load_machine,
)


def stages() -> list[list[str]]:
    """Keep the training and independent-evaluation budgets explicit."""
    return (
        [["train", arm] for arm in ARMS]
        + [["eval", "joint-sft-full"]]
        + [["eval", arm, "--round", str(r)] for arm in ARMS for r in ROUNDS]
    )


def submission(args, machine: dict, stage: list[str], previous: str) -> list[str]:
    """Build one dependency-protected submission without invoking a shell."""
    command = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        f"--dependency=afterok:{previous}",
        "--kill-on-invalid-dep=yes",
        f"--gpus={machine['num_gpus']}",
        f"--cpus-per-task={machine['ray_cpus']}",
        f"--time={args.time_limit}",
        f"--job-name=rc-{'-'.join(stage).replace('--', '')}",
    ]
    if args.account:
        command.append(f"--account={args.account}")
    return command + [
        str(REPO / "toolkits/robocasa_mopd/slurm.sbatch"),
        str(args.machine.resolve()),
        *stage,
    ]


def main() -> None:
    """Preview or submit a chain; preserve every accepted job ID if submission fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--account")
    parser.add_argument("--smoke-job-id", required=True)
    parser.add_argument("--time-limit", default="14-00:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+", args.smoke_job_id):
        parser.error(
            "--smoke-job-id must be the numeric job ID from the smoke submission"
        )
    machine = load_machine(args.machine)
    if machine["scheduler"] != "slurm":
        parser.error("Use a scheduler=slurm machine JSON")
    previous = args.smoke_job_id
    records = []
    record_file = None
    if not args.dry_run:
        bind_outputs(machine)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record_file = REPO / "outputs/submissions" / f"{stamp}.json"
        record_file.parent.mkdir(parents=True, exist_ok=True)
    for index, stage in enumerate(stages()):
        command = submission(args, machine, stage, previous)
        print(json.dumps(command), flush=True)
        if args.dry_run:
            previous = f"PREVIOUS_JOB_{index + 1}"
            continue
        response = subprocess.check_output(command, cwd=REPO, text=True).strip()
        job_id = response.split(";", 1)[0]
        records.append(
            {"stage": stage, "command": command, "sbatch_response": response}
        )
        record_file.write_text(json.dumps(records, indent=2) + "\n")
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch response; inspect {record_file}")
        previous = job_id
    if record_file:
        print(f"Submitted {len(records)} serial jobs. Job IDs: {record_file}")


if __name__ == "__main__":
    main()
