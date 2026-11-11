# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Select a frozen Uniform experiment using a local machine JSON and R/seed."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIRED = {"python", "models", "spatial_teacher", "student", "results"}
OPTIONAL = {
    "gpus",
    "cpus",
    "microbatch_size",
    "wandb_mode",
    "timeout_seconds",
    "actor_cache_retention_gib",
    "actor_cache_preallocate_gib",
}


def machine_config(path: Path) -> dict:
    """Read a path-only/resource configuration without importing ML packages."""
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Machine JSON must be an object")
    if REQUIRED - config.keys() or config.keys() - REQUIRED - OPTIONAL:
        raise ValueError(
            f"Missing machine keys: {sorted(REQUIRED - config.keys())}; "
            f"unknown keys: {sorted(config.keys() - REQUIRED - OPTIONAL)}"
        )
    for key in REQUIRED:
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise ValueError(f"Machine {key} must be an absolute path")
    config = {
        "gpus": [0, 1, 2, 3],
        "cpus": 32,
        "microbatch_size": 4,
        "wandb_mode": "offline",
        "timeout_seconds": 604800,
        "actor_cache_retention_gib": 20,
        "actor_cache_preallocate_gib": 12,
        **config,
    }
    gpus = config["gpus"]
    if (
        not isinstance(gpus, list)
        or len(gpus) != 4
        or any(type(g) is not int or g < 0 for g in gpus)
        or len(set(gpus)) != 4
    ):
        raise ValueError("gpus must be four distinct nonnegative visible GPU IDs")
    for key in ("cpus", "timeout_seconds"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["microbatch_size"]) is not int or config["microbatch_size"] not in (
        2,
        4,
        8,
    ):
        raise ValueError("microbatch_size must be 2, 4 or 8")
    if config["wandb_mode"] not in ("offline", "online", "disabled"):
        raise ValueError("wandb_mode must be offline, online or disabled")
    for key in ("actor_cache_retention_gib", "actor_cache_preallocate_gib"):
        if type(config[key]) not in (int, float) or not 0 <= config[key] < 1000:
            raise ValueError(f"{key} must be a finite nonnegative GiB amount")
    if config["actor_cache_preallocate_gib"] > config["actor_cache_retention_gib"]:
        raise ValueError("Cache preallocation must not exceed retention")
    return config


def launch_spec(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Derive output and logging labels from the same R/seed passed to training."""
    config = machine_config(args.machine)
    label = f"Uniform-R{args.r}_seed{args.seed}"
    if args.eval_round is not None:
        return evaluation_spec(args, config, label), dict(os.environ)
    output = args.output or Path(config["results"]) / (
        label if args.run else label + "_plan"
    )
    command = [
        config["python"],
        "-m",
        "toolkits.libero_gr00t.launch_uniform",
        "--reuse-updates",
        str(args.r),
        "--seed",
        str(args.seed),
        "--microbatch-size",
        str(config["microbatch_size"]),
        "--python",
        config["python"],
        "--models",
        config["models"],
        "--student",
        config["student"],
        "--spatial-teacher",
        config["spatial_teacher"],
        "--gpus",
        *map(str, config["gpus"]),
        "--cpus",
        str(config["cpus"]),
        "--timeout-seconds",
        str(config["timeout_seconds"]),
        "--actor-cache-retention-gib",
        str(config["actor_cache_retention_gib"]),
        "--actor-cache-preallocate-gib",
        str(config["actor_cache_preallocate_gib"]),
        "--skip-finished-rollout",
        "--log-cuda-memory",
        "--output",
        str(output.absolute()),
    ]
    if args.run:
        command.append("--run")
    env = dict(os.environ)
    env.update(
        WANDB_MODE=config["wandb_mode"],
        OPD_WANDB_GROUP=f"T64_uniform_seed{args.seed}",
        OPD_WANDB_RUN_NAME=f"Uniform-R{args.r}_T64_B512_seed{args.seed}",
    )
    return command, env


def evaluation_spec(args: argparse.Namespace, config: dict, label: str) -> list[str]:
    """Select shared initial or trained weights on the fixed ten-round grid."""
    root = Path(config["results"])
    run = root / label
    training_config = args.config or run / "resolved_config.yaml"
    if args.config is None and args.eval_round == 0 and not training_config.is_file():
        training_config = root / (label + "_plan") / "resolved_config.yaml"
    eval_label = (
        "initial_student"
        if args.eval_round == 0
        else f"{label}_round{args.eval_round:03d}"
    )
    output = args.output or root / "evaluations" / (
        eval_label if args.run else eval_label + "_plan"
    )
    command = [
        config["python"],
        "-m",
        "toolkits.libero_gr00t.launch_parallel_evaluation",
        "--python",
        config["python"],
        "--config",
        str(training_config.absolute()),
        "--student",
        config["student"],
        "--gpus",
        *map(str, config["gpus"]),
        "--envs-per-task",
        "5",
        "--cpus",
        str(max(2, config["cpus"] // 4)),
        "--seed",
        "1234",
        "--output",
        str(output.absolute()),
    ]
    if args.eval_round == 0:
        command.append("--baseline")
    else:
        weights = (
            training_config.parent
            / "libero_uniform/checkpoints"
            / f"global_step_{args.eval_round * args.r}"
            / "actor/model_state_dict/full_weights.pt"
        )
        command += ["--weights", str(weights)]
    if args.run:
        command.append("--run")
    return command


def main(argv: list[str] | None = None) -> None:
    """Run the selected runtime wrapper; no activation or shell exports needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--r", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, help="Optional new output directory")
    parser.add_argument(
        "--eval-round",
        type=int,
        choices=range(0, 101, 10),
        help="Evaluate this physical round instead of training; 0 is shared initialization",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Evaluation only: relocated resolved training config",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Start admitted experiment")
    mode.add_argument("--dry-run", action="store_true", help="Default: write plan only")
    args = parser.parse_args(argv)
    if args.config is not None and args.eval_round is None:
        parser.error("--config is only supported with --eval-round")
    command, env = launch_spec(args)
    print(shlex.join(command), flush=True)
    subprocess.run(command, env=env, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
