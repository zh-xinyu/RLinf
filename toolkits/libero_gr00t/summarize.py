# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Summarize completed paired screening without treating it as formal evaluation."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = []
    for path in args.inputs:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows[-1].get("kind") != "complete":
            raise ValueError(f"Incomplete screening: {path}")
        protocol = rows[0]
        expected = {
            (task, trial)
            for task in protocol["tasks"]
            for trial in range(protocol["trials"])
        }
        episodes = [r for r in rows if r.get("kind") == "episode"]
        for role in ["student", "teacher"]:
            selected = [r for r in episodes if r["role"] == role]
            if (
                len(selected) != len(expected)
                or {(r["task_id"], r["trial"]) for r in selected} != expected
            ):
                raise ValueError(f"Missing or duplicated trials: {path}: {role}")
            if not all(r["complete_episode"] for r in selected):
                raise ValueError(
                    "A partial training segment is not an evaluation episode"
                )
            successes = sum(r["success_once"] for r in selected)
            output.append(
                {
                    "suite": protocol["suite"],
                    "role": role,
                    "successes": successes,
                    "episodes": len(selected),
                    "success_rate": successes / len(selected),
                    "simulation_steps": sum(r["steps"] for r in selected),
                    "reset_s": sum(r["reset_s"] for r in selected),
                    "simulation_s": sum(r["simulation_s"] for r in selected),
                    "inference_s": sum(r["inference_s"] for r in selected),
                    "source": str(path.resolve()),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
