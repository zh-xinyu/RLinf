# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Build the 25-row result table from local evaluation counts and training logs."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

from toolkits.robocasa_mopd import experiment

FIELDS = [
    "experiment",
    "initialization",
    "training_seed",
    "rollout_round",
    "optimizer_updates",
    "episodes_per_task",
    "num_tasks",
    "atomic_seen_sr",
    "composite_seen_sr",
    "composite_unseen_sr",
    "suite_macro_sr",
    "task_macro_sr",
    "expected_training_trajectories",
    "training_trajectories",
    "training_simulation_steps",
    "training_round_seconds",
    "checkpoint_id",
    "weights_sha256",
    "eval_directory",
    "wandb_url",
    "status",
    "notes",
]
SUITES = ("atomic_seen", "composite_seen", "composite_unseen")


def template_rows() -> list[dict]:
    """Create empty measured fields and explicit expected budgets, not fake results."""
    rows = []
    for arm, rounds in [
        ("joint-sft-full", [0]),
        *[(arm, experiment.ROUNDS) for arm in experiment.ARMS],
    ]:
        for r in rounds:
            repeats = 0 if arm == "joint-sft-full" else experiment.ARMS[arm][1]
            rows.append(
                {
                    "experiment": arm.upper(),
                    "initialization": experiment.INITIALIZATION,
                    "training_seed": "" if arm == "joint-sft-full" else 0,
                    "rollout_round": r,
                    "optimizer_updates": r * repeats,
                    "episodes_per_task": 50,
                    "num_tasks": 50,
                    "expected_training_trajectories": r * 96,
                    "checkpoint_id": "Joint-SFT-Full"
                    if arm == "joint-sft-full"
                    else f"{arm.upper()}/U{r * repeats}",
                    "eval_directory": f"outputs/eval/{experiment.eval_id(arm, r)}",
                    "status": "PENDING",
                }
            )
    return rows


def evaluation_scores(directory: Path, expected_weights: Path | None) -> dict:
    """Recompute scores from all task counts and reject wrong budgets or checkpoint IDs."""
    plan = json.loads((directory / "plan.json").read_text())
    manifest = json.loads(
        (experiment.REPO / "evaluations/robocasa365/target50_manifest.json").read_text()
    )
    tasks = {
        (suite, task): horizon
        for suite, entries in manifest.items()
        for task, horizon in entries.items()
    }
    if plan["mode"] != "formal" or plan["episodes_per_task"] != 50 or plan["seed"] != 7:
        raise ValueError("Expected formal evaluation, 50 episodes/task, seed7")
    if plan["protocol"] != "rlinf-robocasa365-target50-native-v1":
        raise ValueError("Unexpected evaluation protocol")
    inputs = json.loads(
        (experiment.REPO / "docs/opd/collaboration/checkpoints.json").read_text()
    )["checkpoints"][experiment.INITIALIZATION]
    expected_base = {
        name: record["sha256"]
        for name, record in inputs.items()
        if name.endswith((".safetensors", ".safetensors.index.json"))
    }
    if (
        plan.get("base_model_sha256") != expected_base
        or plan.get("base_config_sha256") != inputs["config.json"]["sha256"]
        or plan.get("base_metadata_sha256")
        != inputs["experiment_cfg/metadata.json"]["sha256"]
    ):
        raise ValueError("Evaluation initialization does not match Joint-SFT-Full")
    if {(t["suite"], t["task"]): t["horizon"] for t in plan["tasks"]} != tasks or len(
        plan["tasks"]
    ) != 50:
        raise ValueError("Evaluation task manifest mismatch")
    if expected_weights is None:
        if plan["weights"] is not None:
            raise ValueError(
                "Joint-SFT-Full evaluation must not have a trained-weight override"
            )
    else:
        # Compare the stable path suffix; result directories may move between machines.
        if (
            not plan["weights"]
            or Path(plan["weights"]).parts[-6:] != expected_weights.parts[-6:]
        ):
            raise ValueError("Evaluation points to a different experiment/checkpoint")
    counts = dict.fromkeys(SUITES, 0)
    seen = set()
    with (directory / "per_task_results.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = row["suite"], row["task"]
            if key not in tasks or key in seen or int(row["horizon"]) != tasks[key]:
                raise ValueError("Missing, repeated, or changed task")
            n, successes = int(row["episodes"]), int(row["success_once_count"])
            if (
                n != 50
                or not 0 <= successes <= n
                or not math.isclose(
                    float(row["success_once"]), successes / n, abs_tol=1e-6
                )
            ):
                raise ValueError("Invalid episode/success counts")
            seen.add(key)
            counts[key[0]] += successes
    if seen != set(tasks):
        raise ValueError("Expected exactly all50 tasks")
    rates = {suite: counts[suite] / (len(manifest[suite]) * 50) for suite in SUITES}
    summary = json.loads((directory / "summary.json").read_text())
    overall = sum(counts.values()) / 2500
    if (
        summary["tasks"] != 50
        or summary["episodes"] != 2500
        or not math.isclose(summary["success_once"], overall, abs_tol=1e-6)
    ):
        raise ValueError("Summary disagrees with per-task counts")
    for suite in SUITES:
        if not math.isclose(
            summary["suites"][suite]["success_once"], rates[suite], abs_tol=1e-6
        ):
            raise ValueError("Suite summary disagrees with per-task counts")
    return {
        **{s + "_sr": rates[s] for s in SUITES},
        "suite_macro_sr": sum(rates.values()) / 3,
        "task_macro_sr": overall,
        "weights_sha256": plan.get("weights_sha256")
        or json.dumps(plan.get("base_model_sha256", {}), sort_keys=True),
    }


def training_rows(arm: str) -> dict[int, dict]:
    """Read update-level interaction counters, rejecting mixed or duplicate histories."""
    path = (
        experiment.REPO
        / "outputs/train"
        / experiment.ARMS[arm][0]
        / "optimizer_updates.jsonl"
    )
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            update = int(row["update_step"])
            if update in rows:
                raise ValueError(f"Duplicate update {update}: {path}")
            rows[update] = row
    return rows


def round_times(arm: str) -> dict[int, float]:
    """Read one timer per completed round, excluding startup and independent evaluation."""
    folder = experiment.REPO / "outputs/train" / experiment.ARMS[arm][0] / "tensorboard"
    if not list(folder.glob("events.out.tfevents.*")):
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(folder), size_guidance={"scalars": 0}).Reload()
    if "time/rollout_round" not in events.Tags()["scalars"]:
        return {}
    values = {}
    for event in events.Scalars("time/rollout_round"):
        if event.step in values or not math.isfinite(event.value) or event.value < 0:
            raise ValueError("Invalid or repeated rollout time")
        values[event.step] = event.value
    return values


def collect() -> list[dict]:
    """Collect available results; leave missing measurements empty and explain status."""
    rows = template_rows()
    histories, times = {}, {}
    for row in rows:
        arm = row["experiment"].lower()
        directory = experiment.REPO / row["eval_directory"]
        notes = []
        try:
            if directory.exists():
                status = directory / "status"
                if status.is_file() and status.read_text().strip() == "OK":
                    expected = (
                        None
                        if arm == "joint-sft-full"
                        else experiment.weights_path(arm, row["rollout_round"])
                    )
                    row.update(evaluation_scores(directory, expected))
                    row["status"] = "EVAL_COMPLETE"
                else:
                    row["status"] = "INCOMPLETE"
            if arm == "joint-sft-full":
                row.update(
                    training_trajectories=0,
                    training_simulation_steps=0,
                    training_round_seconds=0,
                )
                if row["status"] == "EVAL_COMPLETE":
                    row["status"] = "COMPLETE"
            else:
                if arm not in histories:
                    histories[arm] = training_rows(arm)
                    times[arm] = round_times(arm)
                update = row["optimizer_updates"]
                record = histories[arm].get(update)
                if record:
                    if (
                        record["interaction/rollout_rounds"] != row["rollout_round"]
                        or record["interaction/rollout_trajectories"]
                        != row["expected_training_trajectories"]
                    ):
                        raise ValueError("Training update/round/trajectory mismatch")
                    row.update(
                        training_trajectories=record[
                            "interaction/rollout_trajectories"
                        ],
                        training_simulation_steps=record[
                            "interaction/simulation_steps"
                        ],
                    )
                else:
                    notes.append("Training counters unavailable")
                repeats = experiment.ARMS[arm][1]
                steps = range(repeats, update + 1, repeats)
                if all(step in times[arm] for step in steps):
                    row["training_round_seconds"] = sum(
                        times[arm][step] for step in steps
                    )
                else:
                    notes.append("Round timers unavailable; provide TensorBoard logs")
                log = (
                    experiment.REPO
                    / "outputs/launches"
                    / experiment.ARMS[arm][0]
                    / "driver.log"
                )
                if log.exists():
                    match = re.search(
                        r"https://(?:wandb\.ai|forge\.coreweave\.com)/[^\s/]+/[^\s/]+/runs/[A-Za-z0-9_-]+",
                        log.read_text(errors="replace"),
                    )
                    if match:
                        row["wandb_url"] = match.group()
                if (
                    row["status"] == "EVAL_COMPLETE"
                    and record
                    and "training_round_seconds" in row
                ):
                    row["status"] = "COMPLETE"
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row["status"] = "INVALID"
            notes.append(str(exc))
        row["notes"] = "; ".join(notes)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a derived table; source evaluation and training files remain untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate a blank template or collect local results without requiring checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    rows = template_rows() if args.template else collect()
    write_csv(args.output, rows)
    print(
        f"Wrote {len(rows)} rows to {args.output}; {sum(r['status'] == 'COMPLETE' for r in rows)} complete"
    )
    if any(row["status"] == "INVALID" for row in rows):
        raise SystemExit("Invalid results detected; inspect status/notes in the CSV")


if __name__ == "__main__":
    main()
