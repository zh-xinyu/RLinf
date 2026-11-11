# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Merge complete independent LIBERO evaluations into shared result CSVs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = (240, 240, 320, 480)
EXECUTION = (5, 5, 5, 10)
COMMON = ["arm", "seed", "round", "optimizer_update"]
TAIL = [
    "training_trajectories",
    "training_simulation_steps",
    "training_wall_seconds",
    "checkpoint_identity",
    "raw_evaluation_path",
]
TASK_FIELDS = (
    COMMON + ["suite", "task_id", "successes", "episodes", "success_rate"] + TAIL
)
SUMMARY_FIELDS = (
    COMMON
    + [
        "spatial_sr",
        "object_sr",
        "goal_sr",
        "long_sr",
        "macro_sr",
        "episodes",
    ]
    + TAIL
)
IDENTITY_KEYS = (
    "base_checkpoint_identity",
    "metadata_sha256",
    "execution_lengths",
    "horizons",
    "rng_protocol",
    "seed",
    "inference_model_config",
    "source_sha256",
)


def audited_counts(output: Path) -> tuple[dict, list[int]]:
    """Require every official task/state pair before exporting a checkpoint."""
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    if (
        manifest.get("mode") != "run"
        or manifest.get("tasks") != list(range(40))
        or manifest.get("trials") != list(range(50))
        or manifest.get("seed") != 1234
        or manifest.get("execution_lengths") != dict(zip(SUITES, EXECUTION))
        or manifest.get("horizons") != dict(zip(SUITES, HORIZONS))
        or summary.get("complete") is not True
        or summary.get("episodes") != 2000
        or summary.get("expected_episodes") != 2000
        or summary.get("missing_pairs") != []
    ):
        raise ValueError("Unified CSVs require the complete official 40x50 protocol")
    counts = [0] * 40
    seen = set()
    for task in range(40):
        directory = output / f"task_{task:02d}"
        if (directory / "status").read_text().strip() != "OK":
            raise ValueError(f"Incomplete evaluation task: {task}")
        for row in json.loads((directory / "episodes.json").read_text()):
            pair = row["task_id"], row["trial_id"]
            suite = task // 10
            if (
                pair[0] != task
                or type(pair[1]) is not int
                or pair[1] not in range(50)
                or pair in seen
                or row["suite"] != SUITES[suite]
                or type(row["success_once"]) is not bool
                or row["horizon"] != HORIZONS[suite]
                or not 1 <= row["simulation_steps"] <= HORIZONS[suite]
                or (
                    not row["success_once"]
                    and row["simulation_steps"] != HORIZONS[suite]
                )
            ):
                raise ValueError(f"Invalid or duplicate raw evaluation pair: {pair}")
            seen.add(pair)
            counts[task] += int(row["success_once"])
    if len(seen) != 2000:
        raise ValueError("Raw task/state coverage is incomplete")
    rates = {
        suite: sum(counts[i * 10 : i * 10 + 10]) / 500 for i, suite in enumerate(SUITES)
    }
    if any(
        not math.isclose(summary["suite_success_rates"][s], value, abs_tol=1e-12)
        for s, value in rates.items()
    ) or not math.isclose(
        summary["macro_success_rate"], sum(rates.values()) / 4, abs_tol=1e-12
    ):
        raise ValueError("Evaluation summary disagrees with raw episodes")
    for key in IDENTITY_KEYS:
        if key not in manifest:
            raise ValueError(f"Missing evaluation identity: {key}")
    return manifest, counts


def merge_csv(path: Path, fields: list[str], keys: list[str], new: list[dict]) -> None:
    """Upsert audited rows under the caller's lock, retaining other experiments."""
    records = {}
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fields:
                raise ValueError(f"Existing CSV schema differs: {path}")
            for row in reader:
                key = tuple(row[k] for k in keys)
                if key in records:
                    raise ValueError(f"Duplicate existing result key in {path}: {key}")
                records[key] = row
    for row in new:
        row = {key: str(row.get(key, "")) for key in fields}
        key = tuple(row[k] for k in keys)
        old = records.get(key)
        if old and old["checkpoint_identity"] not in ("", row["checkpoint_identity"]):
            raise ValueError(
                f"Checkpoint changed for existing result {key}; use a separate results root"
            )
        records[key] = row
    temporary = path.with_suffix(".csv.partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(
            records, key=lambda k: tuple(int(v) if v.isdigit() else v for v in k)
        ):
            writer.writerow(records[key])
    temporary.replace(path)


def record_evaluation(
    results: Path,
    output: Path,
    repeats: int,
    seed: int,
    round_: int,
    training_run: Path | None = None,
) -> None:
    """Atomically merge task/macro results from one audited checkpoint evaluation."""
    if (
        repeats not in (1, 2, 4)
        or seed not in (0, 1, 2)
        or round_ not in range(0, 101, 10)
    ):
        raise ValueError(
            "Expected Uniform R1/R2/R4, seed0/1/2 and a ten-round checkpoint"
        )
    manifest, counts = audited_counts(output)
    weights = manifest.get("weights")
    if round_ == 0:
        if weights is not None or manifest.get("weights_sha256") is not None:
            raise ValueError("Round0 must evaluate the base HF student")
        checkpoint = manifest["base_checkpoint_identity"]["content_sha256"]
    else:
        if (
            not weights
            or not manifest.get("weights_sha256")
            or f"global_step_{repeats * round_}" not in Path(weights).parts
        ):
            raise ValueError("Weights do not match the requested physical round and R")
        checkpoint = manifest["weights_sha256"]
    row = {
        "arm": "INITIAL" if round_ == 0 else f"Uniform-R{repeats}",
        "seed": "" if round_ == 0 else seed,
        "round": round_,
        "optimizer_update": round_ * repeats,
        "training_trajectories": 64 * round_,
        "checkpoint_identity": checkpoint,
        "raw_evaluation_path": str(output.absolute()),
    }
    if round_ == 0:
        row.update(training_simulation_steps=0, training_wall_seconds=0)
    elif training_run is not None:
        history = training_run / "libero_uniform/rollout_allocation.jsonl"
        if history.exists():
            matching = [
                json.loads(line)
                for line in history.read_text().splitlines()
                if line.strip()
            ]
            matching = [state for state in matching if state["rounds"] == round_]
            if matching:
                row.update(
                    training_trajectories=matching[-1]["trajectories"],
                    training_simulation_steps=matching[-1]["simulation_steps"],
                )
    tasks = [
        {
            **row,
            "suite": SUITES[task // 10],
            "task_id": task,
            "successes": successes,
            "episodes": 50,
            "success_rate": successes / 50,
        }
        for task, successes in enumerate(counts)
    ]
    rates = [sum(counts[i * 10 : i * 10 + 10]) / 500 for i in range(4)]
    summary = {
        **row,
        **dict(zip(("spatial_sr", "object_sr", "goal_sr", "long_sr"), rates)),
        "macro_sr": sum(rates) / 4,
        "episodes": 2000,
    }
    identity = {key: manifest[key] for key in IDENTITY_KEYS}
    results.mkdir(parents=True, exist_ok=True)
    with (results / ".results_csv.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        identity_path = results / "results_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "Results root already contains another initialization or evaluation protocol"
            )
        # Export independently atomically; a retry upserts both files if interrupted.
        merge_csv(
            results / "checkpoint_results.csv",
            TASK_FIELDS,
            COMMON[:3] + ["task_id"],
            tasks,
        )
        merge_csv(
            results / "checkpoint_summary.csv", SUMMARY_FIELDS, COMMON[:3], [summary]
        )
        temporary = identity_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(identity, indent=2) + "\n")
        temporary.replace(identity_path)


def main() -> None:
    """Rebuild one checkpoint's CSV rows without launching a GPU evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--r", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--round", type=int, choices=range(0, 101, 10), required=True)
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args()
    record_evaluation(
        args.results, args.evaluation, args.r, args.seed, args.round, args.training_run
    )


if __name__ == "__main__":
    main()
