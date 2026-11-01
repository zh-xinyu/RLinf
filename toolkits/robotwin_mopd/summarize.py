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
    losses = [r.get("train/actor/velocity_mse") for r in updates]
    losses = [x for x in losses if x is not None]
    n = min(len(losses), 2 * args.window_rounds)
    sr = {name: sum(values) / len(values) for name, values in tasks.items()}
    result = {
        "run_dir": str(args.run_dir.resolve()),
        "logged_updates": len(updates),
        "last_update": max((r["update_step"] for r in updates), default=0),
        "physical_rounds": len({r["round"] for r in episodes}),
        "trajectories": len(episodes),
        "executed_control_targets": sum(r["simulation_steps"] for r in episodes),
        "loss_all_finite": all(math.isfinite(x) for x in losses) if losses else None,
        "mse_first_window": statistics.mean(losses[:n]) if n else None,
        "mse_last_window": statistics.mean(losses[-n:]) if n else None,
        "recent_training_sr": sr,
        "recent_training_macro_sr": statistics.mean(sr.values()) if sr else None,
        "recent_episode_counts": {name: len(values) for name, values in tasks.items()},
        "note": "Training rollout SR, not independent evaluation. Control targets exclude terminal padding.",
    }
    (args.run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
