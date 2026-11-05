# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Summarize complete suite evaluations, keeping unfinished pilots explicit."""

import argparse
import csv
import json
from pathlib import Path

NAMES = ("spatial", "object", "goal", "long")


def summarize(directory: Path) -> dict:
    """Export raw update rows and evaluations without pooling incomplete suites."""
    suites = {}
    for name in NAMES:
        rows = []
        lines = (directory / f"{name}.jsonl").read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A live writer may not have finished its final line.
                if index != len(lines) - 1:
                    raise
        suites[name] = rows
    evaluations = []
    for name, rows in suites.items():
        for row in rows:
            if row["kind"] == "evaluation":
                evaluations.append(
                    {
                        "suite": name,
                        "round": row["round"],
                        "optimizer_step": row["optimizer_step"],
                        "successes": row["successes"],
                        "episodes": row["episodes"],
                        "success_rate": row["success_rate"],
                    }
                )
    joint = []
    for round_index in sorted({row["round"] for row in evaluations}):
        rows = [row for row in evaluations if row["round"] == round_index]
        if {row["suite"] for row in rows} != set(NAMES) or len(rows) != 4:
            continue
        joint.append(
            {
                "round": round_index,
                "optimizer_step": rows[0]["optimizer_step"],
                "successes": sum(row["successes"] for row in rows),
                "episodes": sum(row["episodes"] for row in rows),
                "equal_suite_success_rate": sum(row["success_rate"] for row in rows)
                / 4,
            }
        )
    updates = [row for row in suites["spatial"] if row["kind"] == "update"]
    fields = [
        "optimizer_step",
        "round",
        "reuse_index",
        "joint_velocity_mse",
        "joint_analytic_kl",
        "lr",
        "grad_norm",
    ]
    with (directory / "training_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in updates)
    if evaluations:
        with (directory / "evaluation_summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(evaluations[0]))
            writer.writeheader()
            writer.writerows(evaluations)
    completed = {
        name: next((row for row in reversed(rows) if row["kind"] == "complete"), None)
        for name, rows in suites.items()
    }
    summary = {
        "complete": all(completed.values()),
        "joint_evaluations": joint,
        "last_logged_update": updates[-1]["optimizer_step"] if updates else 0,
        "per_suite_completion": completed,
        "formal_performance_evidence": False,
        "limitations": "One held-out initial state per task; shared GPUs; feasibility runner.",
    }
    if summary["complete"]:
        summary["training_interaction"] = {
            key.removeprefix("local_"): sum(row[key] for row in completed.values())
            for key in (
                "local_simulation_steps",
                "local_settling_steps",
                "local_trajectories",
            )
        }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))


if __name__ == "__main__":
    main()
