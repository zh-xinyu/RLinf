# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check formal protocol and calibration against the active runtime without GPUs."""

import argparse
import json
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.rollout_allocation import load_teacher_reference
from toolkits.robotwin_mopd import author_pool
from toolkits.robotwin_mopd.formal.experiment import (
    RECIPE,
    REPO,
    TASKS,
    evaluation_pool_path,
    read_machine,
    seed_settings,
    train_command,
    training_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, type=Path)
    args = parser.parse_args()
    machine = read_machine(args.machine)
    expected = "2.7.1" if machine["profile"] == "blackwell-cu128-candidate" else "2.6.0"
    if torch.__version__.split("+")[0] != expected:
        raise ValueError(f"Expected Torch {expected}; got {torch.__version__}")
    cuda = "12.8" if expected == "2.7.1" else "12.4"
    if torch.version.cuda != cuda:
        raise ValueError(f"Expected CUDA wheel {cuda}; got {torch.version.cuda}")
    os.environ["OPD_RUN_DIR"] = str(Path(machine["output_root"]) / "preflight")
    command = train_command(machine, "m2", training_seed(machine))
    overrides = command[command.index("4") + 1 :]
    # run.py supplies physical placement at launch; no hardware scheduling here.
    with initialize_config_dir(
        version_base="1.1", config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05", overrides=overrides)
    OmegaConf.resolve(cfg)
    pool = author_pool.load_pool(evaluation_pool_path(machine))
    if pool["protocol"]["episodes_per_task"] != 100:
        raise ValueError("Formal pool must contain 100 episodes per task")
    author_pool.validate_pool_protocol(
        pool,
        task_config=OmegaConf.to_container(cfg.env.eval.task_config, resolve=True),
        task_horizons=OmegaConf.to_container(cfg.env.eval.task_horizons, resolve=True),
        robotwin_root=Path(os.environ["ROBOTWIN_PATH"]),
    )
    reference = Path(machine["output_root"]) / "prepared/teacher_reference.json"
    metadata = json.loads(reference.read_text())
    rates = load_teacher_reference(str(reference), cfg, metadata["tasks"])
    # Conservative upper bound: all training episodes assigned to each task,
    # including every possible native reset retry (0..19).
    max_task_episodes = RECIPE["rounds"] * int(cfg.env.train.total_num_envs)
    base_seed = seed_settings(training_seed(machine))["runner.rollout_allocation.seed"]
    for task_id, task in enumerate(TASKS):
        training_seeds = set(
            range(
                base_seed + task_id * 1000003,
                base_seed + task_id * 1000003 + max_task_episodes + 19,
            )
        )
        evaluation_seeds = {row["scene_seed"] for row in pool["tasks"][task]}
        if training_seeds & evaluation_seeds:
            raise ValueError(f"Train/eval seed overlap for {task}")
    output = Path(machine["output_root"]) / "prepared/preflight.json"
    output.write_text(
        json.dumps(
            {
                "status": "CPU_PREFLIGHT_PASS",
                "training_seed": training_seed(machine),
                "seed_settings": seed_settings(training_seed(machine)),
                "gpu_validation": "PENDING",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "teacher_reference_rates": rates,
                "training_trajectory_bound_per_task": max_task_episodes,
                "pool_content_sha256": pool["pool_content_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
