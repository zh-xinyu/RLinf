# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bind completed native teacher evaluations to an adaptive pilot reference."""

import argparse
import json
import math
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.rollout_allocation import load_teacher_reference, training_signature


def calibration_wall_seconds(eval_dir: Path, results: list[dict], explicit=None):
    """Use recorded elapsed time, never sum parallel episode durations as wall time."""
    if explicit is not None:
        seconds, source = float(explicit), "explicit_elapsed_seconds"
    elif all(
        r.get("started_unix") is not None and r.get("finished_unix") is not None
        for r in results
    ):
        seconds = max(r["finished_unix"] for r in results) - min(
            r["started_unix"] for r in results
        )
        source = "recorded_evaluation_timestamps"
    elif (eval_dir / "teacher_pilot_launch.json").is_file():
        start = (eval_dir / "teacher_pilot_launch.json").stat().st_mtime
        end = max(
            (eval_dir / f"teacher_pilot_{r['task']}.json").stat().st_mtime
            for r in results
        )
        seconds, source = (
            end - start,
            "approximate_launch_manifest_to_last_result_mtime",
        )
    else:
        raise ValueError("No elapsed calibration timestamps; supply --wall-seconds")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid calibration elapsed time")
    return seconds, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-episodes", type=int, default=5)
    parser.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("OPD_RUN_DIR", str(args.eval_dir))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05")
    OmegaConf.resolve(cfg)
    rows = []
    results = []
    paths = {}
    for task, expert in cfg.rollout.expert_models.items():
        data = json.loads((args.eval_dir / f"teacher_pilot_{task}.json").read_text())
        results.append(data)
        n = data["completed_episodes"]
        if n != data["requested_episodes"] or n < args.min_episodes:
            raise ValueError(f"Incomplete calibration: {task}, n={n}")
        path = str(Path(expert.model_path).resolve())
        if str(Path(data["checkpoint"]).resolve()) != path or data["task"] != task:
            raise ValueError(f"Wrong teacher: {task}")
        if (
            data["chunk_length"] != cfg.actor.model.num_action_chunks
            or data["denoising_steps"] != cfg.actor.model.num_steps
        ):
            raise ValueError(f"Wrong flow/control protocol: {task}")
        if data["normalization"] != OmegaConf.to_container(
            cfg.actor.model.opd_task_norm_stats, resolve=True
        ):
            raise ValueError(f"Wrong normalization: {task}")
        horizon = int(cfg.env.train.task_horizons[task])
        if data["task_horizon"] != horizon:
            raise ValueError(f"Wrong horizon: {task}")
        control_steps = data.get("simulation_steps")
        if (
            data.get("simulation_step_unit") != "executed_control_targets"
            or type(control_steps) is not int
            or not 0 <= control_steps <= n * horizon
            or control_steps != sum(row["simulation_steps"] for row in data["episodes"])
        ):
            raise ValueError(f"Invalid calibration interaction budget: {task}")
        paths[task] = path
        rows.append(
            {
                "task": task,
                "suite": task,
                "horizon": horizon,
                "episodes": n,
                "successes": data["successes"],
                "simulation_steps": control_steps,
            }
        )
    wall_seconds, wall_source = calibration_wall_seconds(
        args.eval_dir, results, args.wall_seconds
    )
    reference = {
        "schema_version": 1,
        "status": "COMPLETE",
        "training_signature": training_signature(cfg),
        "teacher_checkpoints": paths,
        "tasks": rows,
        "purpose": "pilot calibration; not formal evaluation",
        "simulation_step_unit": "executed_control_targets",
        "wall_seconds": wall_seconds,
        "wall_seconds_source": wall_source,
        "episode_wall_seconds_sum": sum(
            ep["wall_seconds"] for r in results for ep in r["episodes"]
        ),
        "source_eval_dir": str(args.eval_dir.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2) + "\n")
    print(load_teacher_reference(str(args.output), cfg, rows))


if __name__ == "__main__":
    main()
