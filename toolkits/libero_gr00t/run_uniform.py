# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Select a frozen Uniform experiment using a local machine JSON and R/seed."""

from __future__ import annotations

import argparse
import hashlib
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


def evaluate_saved_checkpoints(
    args: argparse.Namespace, config: dict, run: Path
) -> None:
    """Evaluate saved rounds sequentially after the owned training job has exited."""
    import fcntl

    if (run / "status").read_text().strip() != "OK":
        raise ValueError(
            "Automatic evaluation requires a successfully completed training run"
        )
    resolved = run / "resolved_config.yaml"
    for round_ in range(10, 101, 10):
        weights = (
            run
            / "libero_uniform/checkpoints"
            / f"global_step_{round_ * args.r}"
            / "actor/model_state_dict/full_weights.pt"
        )
        if not weights.is_file():
            raise FileNotFoundError(weights)
    identity = {
        "r": args.r,
        "seed": args.seed,
        "training_config_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }
    receipt = run / "post_training_evaluation.json"
    state = {"identity": identity, "status": "RUNNING", "rounds": {}}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["identity"] != identity:
            raise ValueError(
                "Post-training evaluation recipe changed; use the original resolved run"
            )
        state["rounds"] = previous["rounds"]

    def persist() -> None:
        temporary = receipt.with_suffix(".json.partial")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(receipt)

    label = f"Uniform-R{args.r}_seed{args.seed}"
    for round_ in range(0, 101, 10):
        selected = argparse.Namespace(**vars(args))
        selected.config = resolved
        selected.eval_round = round_
        selected.output = None
        command = evaluation_spec(selected, config, label)
        default_output = Path(command[command.index("--output") + 1])
        previous = state["rounds"].get(str(round_), {})
        output = Path(previous.get("output", str(default_output)))
        attempts = list(previous.get("failed_attempts", []))
        if previous.get("status") in ("FAILED", "RUNNING"):
            attempts.append(str(output))
            attempt = 2
            while output.exists():
                output = default_output.with_name(
                    default_output.name + f"_attempt{attempt}"
                )
                attempt += 1
        command[command.index("--output") + 1] = str(output)
        command.append("--reuse-complete")
        record = {
            "status": "RUNNING",
            "output": str(output),
            "optimizer_update": round_ * args.r,
            "failed_attempts": attempts,
        }
        state["rounds"][str(round_)] = record
        persist()
        try:
            # Separate jobs may share an initial result; serialize each output's
            # identity/coverage check and creation, without holding GPU tensors.
            lock = default_output.parent / ".locks" / (default_output.name + ".lock")
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                print(shlex.join(command), flush=True)
                subprocess.run(command, env=dict(os.environ), cwd=REPO, check=True)
            record["status"] = "COMPLETE"
            persist()
        except BaseException as error:
            record.update(status="FAILED", error=repr(error))
            state["status"] = "FAILED"
            persist()
            raise
    state["status"] = "COMPLETE"
    persist()


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
    followup = parser.add_mutually_exclusive_group()
    followup.add_argument(
        "--train-only",
        action="store_true",
        help="Skip default post-training evaluations",
    )
    followup.add_argument(
        "--evaluate-all",
        action="store_true",
        help="Evaluate/recheck all saved rounds without training",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Start admitted experiment")
    mode.add_argument("--dry-run", action="store_true", help="Default: write plan only")
    args = parser.parse_args(argv)
    if args.eval_round is not None and (args.train_only or args.evaluate_all):
        parser.error(
            "--eval-round cannot be combined with --train-only or --evaluate-all"
        )
    if args.config is not None and args.eval_round is None and not args.evaluate_all:
        parser.error("--config is only supported with --eval-round or --evaluate-all")
    if args.evaluate_all and (not args.run or args.output is not None):
        parser.error("--evaluate-all requires --run; use --config for a relocated run")
    config = machine_config(args.machine)
    if args.evaluate_all:
        run = (
            args.config.parent
            if args.config
            else Path(config["results"]) / f"Uniform-R{args.r}_seed{args.seed}"
        )
        evaluate_saved_checkpoints(args, config, run.absolute())
        return
    command, env = launch_spec(args)
    print(shlex.join(command), flush=True)
    subprocess.run(command, env=env, cwd=REPO, check=True)
    if args.run and args.eval_round is None and not args.train_only:
        run = Path(command[command.index("--output") + 1])
        evaluate_saved_checkpoints(args, config, run)


if __name__ == "__main__":
    main()
