# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bind completed native teacher evaluations to an adaptive pilot reference."""

import argparse
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.rollout_allocation import load_teacher_reference, training_signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-episodes", type=int, default=5)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("OPD_RUN_DIR", str(args.eval_dir))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05")
    OmegaConf.resolve(cfg)
    rows = []
    paths = {}
    for task, expert in cfg.rollout.expert_models.items():
        data = json.loads((args.eval_dir / f"teacher_pilot_{task}.json").read_text())
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
        paths[task] = path
        rows.append(
            {
                "task": task,
                "suite": task,
                "horizon": horizon,
                "episodes": n,
                "successes": data["successes"],
            }
        )
    reference = {
        "schema_version": 1,
        "status": "COMPLETE",
        "training_signature": training_signature(cfg),
        "teacher_checkpoints": paths,
        "tasks": rows,
        "purpose": "pilot calibration; not formal evaluation",
        "source_eval_dir": str(args.eval_dir.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2) + "\n")
    print(load_teacher_reference(str(args.output), cfg, rows))


if __name__ == "__main__":
    main()
