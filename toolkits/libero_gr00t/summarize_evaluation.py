# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Audit one four-suite evaluation's raw worker logs before reporting success rates."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EPISODE = re.compile(
    r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)"
)


def audit(logs: list[Path], episodes_per_task: int = 50) -> tuple[dict, list[dict]]:
    """Require exact task/trial coverage; collapse only identical log duplicates.

    Inputs must belong to one checkpoint evaluation. Logs alone cannot establish
    checkpoint identity, correct horizons, or noise/reset settings: retain the
    resolved configuration and checkpoint manifest with the report.
    """
    if not 1 <= episodes_per_task <= 50:
        raise ValueError("episodes_per_task must be in [1, 50]")
    expected = {
        (task, trial) for task in range(40) for trial in range(episodes_per_task)
    }
    records, sources, duplicate_count = {}, [], 0
    for path in logs:
        data = path.read_bytes()
        sources.append(
            {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
        )
        for match in EPISODE.finditer(data.decode(errors="replace")):
            task, trial = int(match[1]), int(match[2])
            key, success = (task, trial), match[3] == "True"
            if key not in expected:
                raise ValueError(f"Unexpected task/trial pair: {key}")
            if key in records:
                if records[key] != success:
                    raise ValueError(f"Conflicting repeated task/trial result: {key}")
                duplicate_count += 1
            records[key] = success
    missing = sorted(expected - records.keys())
    rows = [
        {
            "suite": SUITES[task // 10],
            "task_id": task,
            "suite_task_id": task % 10,
            "trial_id": trial,
            "success": int(success),
        }
        for (task, trial), success in sorted(records.items())
    ]
    report = {
        "complete": not missing,
        "episodes": len(records),
        "expected_episodes": len(expected),
        "episodes_per_task": episodes_per_task,
        "missing_pairs": missing,
        "identical_log_duplicates": duplicate_count,
        "source_logs": sources,
        "suite_success_rates": None,
        "macro_success_rate": None,
        "protocol_note": "Coverage audit only; retain checkpoint identity and resolved config.",
    }
    if not missing:
        rates = {
            suite: sum(row["success"] for row in rows if row["suite"] == suite)
            / (10 * episodes_per_task)
            for suite in SUITES
        }
        report.update(
            suite_success_rates=rates, macro_success_rate=sum(rates.values()) / 4
        )
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        required=True,
        help="Raw worker logs from ONE checkpoint evaluation; do not combine runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    report, rows = audit(args.logs, args.episodes_per_task)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["suite", "task_id", "suite_task_id", "trial_id", "success"],
        )
        writer.writeheader()
        writer.writerows(rows)
    if not report["complete"]:
        raise SystemExit(
            f"Incomplete evaluation: {report['episodes']}/{report['expected_episodes']}; SR withheld"
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("episodes", "suite_success_rates", "macro_success_rate")
            }
        )
    )


if __name__ == "__main__":
    main()
