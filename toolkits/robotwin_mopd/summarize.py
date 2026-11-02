# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Summarize pilot JSONL metrics without querying W&B or loading TensorFlow."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    """Read completed JSONL records, tolerating only a partial trailing write."""
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def metric_counts(rows: list[dict]) -> tuple[dict, list[float]]:
    """Separate absent MSE values from invalid values without claiming finiteness."""
    counts = {"finite": 0, "missing": 0, "invalid": 0}
    finite_values = []
    for row in rows:
        value = row.get("train/actor/velocity_mse")
        if value is None:
            counts["missing"] += 1
        elif type(value) in (int, float) and math.isfinite(value):
            counts["finite"] += 1
            finite_values.append(value)
        else:
            counts["invalid"] += 1
    return counts, finite_values


def mse_windows(updates: list[dict], window_rounds: int) -> dict:
    """Use observed rollout IDs; no assumption about updates per physical round."""
    if window_rounds < 1:
        raise ValueError("window_rounds must be positive")
    grouped = defaultdict(list)
    unassigned = 0
    for row in updates:
        rollout = row.get("rollout_count")
        if type(rollout) is int and rollout > 0:
            grouped[rollout].append(row)
        else:
            unassigned += 1
    round_ids = sorted(grouped)
    all_counts, _ = metric_counts(updates)
    result = {
        "mse_window_rounds_requested": window_rounds,
        "updates_without_valid_rollout_count": unassigned,
        "loss_metric_counts": all_counts,
        "loss_all_finite": False
        if all_counts["invalid"]
        else (True if updates and not all_counts["missing"] else None),
    }
    for label, selected in (
        ("first", round_ids[:window_rounds]),
        ("last", round_ids[-window_rounds:]),
    ):
        rows = [row for rollout in selected for row in grouped[rollout]]
        counts, values = metric_counts(rows)
        result[f"mse_{label}_window"] = (
            statistics.mean(values) if values and not counts["invalid"] else None
        )
        result[f"mse_{label}_window_details"] = {
            "rollout_counts": selected,
            "round_count": len(selected),
            "update_count": len(rows),
            "update_steps": [row.get("update_step") for row in rows],
            "metric_counts": counts,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window-rounds", type=int, default=5)
    args = parser.parse_args()
    if args.window_rounds < 1:
        parser.error("window-rounds must be positive")
    updates = read_rows(args.run_dir / "optimizer_updates.jsonl")
    episodes = read_rows(args.run_dir / "rollout_episodes.jsonl")
    latest_round = max((r["round"] for r in episodes), default=-1)
    recent = [r for r in episodes if r["round"] > latest_round - args.window_rounds]
    tasks = defaultdict(list)
    for row in recent:
        tasks[row["task"]].append(bool(row["success_once"]))
    sr = {name: sum(values) / len(values) for name, values in tasks.items()}
    result = {
        "run_dir": str(args.run_dir.resolve()),
        "logged_updates": len(updates),
        "last_update": max((r["update_step"] for r in updates), default=0),
        "physical_rounds": len({r["round"] for r in episodes}),
        "trajectories": len(episodes),
        "executed_control_targets": sum(r["simulation_steps"] for r in episodes),
        **mse_windows(updates, args.window_rounds),
        "recent_training_sr": sr,
        "recent_training_macro_sr": statistics.mean(sr.values()) if sr else None,
        "recent_episode_counts": {name: len(values) for name, values in tasks.items()},
        "note": "Training rollout SR, not independent evaluation. Control targets exclude terminal padding.",
    }
    (args.run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
