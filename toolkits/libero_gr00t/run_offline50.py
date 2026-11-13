# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run the six 80-round Offline50 controls, then automatically evaluate/export."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from toolkits.libero_gr00t.collect_results import record_evaluation
from toolkits.libero_gr00t.offline50_recipe import (
    CURVE_EPISODES,
    CURVE_ROUNDS,
    ROUNDS,
    config_name,
)
from toolkits.libero_gr00t.run_uniform import (
    REPO,
    check_csv_directory,
    machine_config,
    prepare_adaptive_reference,
)


def write_json(path: Path, value: dict) -> None:
    """Publish a small receipt atomically without changing training files."""
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def effective_settings(args: argparse.Namespace) -> dict:
    """Read existing local settings unchanged; isolate only new outputs/student."""
    settings = machine_config(args.machine)
    if getattr(args, "student", None) is not None:
        settings["student"] = str(args.student.absolute())
    settings["results"] = str(Path(settings["results"]) / "offline50_80r")
    settings["csv_dir"] = str(Path(settings["csv_dir"]) / "offline50_80r")
    return settings


def bind_reference(args: argparse.Namespace, settings: dict) -> Path:
    """Bind with effective paths without editing the supplied machine JSON."""
    directory = Path(settings["results"]) / "effective_machines"
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(settings, sort_keys=True)
    machine = directory / (hashlib.sha256(text.encode()).hexdigest()[:20] + ".json")
    if not machine.exists():
        write_json(machine, settings)
    selected = argparse.Namespace(**vars(args))
    selected.machine = machine
    return prepare_adaptive_reference(selected, settings)


def verify_initializer(student: Path) -> str:
    """Require the restored full HF checkpoint by bytes, never by folder name."""
    manifest = json.loads(
        (REPO / "toolkits/libero_gr00t/configs/offline50_student.json").read_text()
    )
    expected = manifest["checkpoint_identity"]
    for record in expected["files"]:
        path = student / record["path"]
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            raise ValueError(
                f"Offline-Distilled-50 checkpoint file missing/wrong size: {path}"
            )
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
            actual = digest.hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Not the agreed Offline-Distilled-50 initializer: {path}")
    return expected["content_sha256"]


def label(args: argparse.Namespace) -> str:
    """Separate all six variants from old SFT100 and historical runs."""
    method = args.method.capitalize()
    if args.interval is not None:
        method += f"-I{args.interval}"
    return f"Offline50_80r_{method}-R{args.r}_seed{args.seed}"


def training_spec(args: argparse.Namespace, settings: dict) -> tuple[list[str], dict]:
    """Use existing owned-Ray machinery and unchanged runtime/model interfaces."""
    run = args.output or Path(settings["results"]) / (
        label(args) if args.run else label(args) + "_plan"
    )
    command = [
        settings["python"],
        "-m",
        "toolkits.libero_gr00t.launch_production",
        "--config-name",
        config_name(args.method, args.r, args.interval),
        "--run-purpose",
        "formal",
        "--execution-protocol",
        "official-mixed",
        "--python",
        settings["python"],
        "--models",
        settings["models"],
        "--student",
        settings["student"],
        "--spatial-teacher",
        settings["spatial_teacher"],
        "--gpus",
        *map(str, settings["gpus"]),
        "--cpus",
        str(settings["cpus"]),
        "--seed",
        str(args.seed),
        "--updates",
        str(ROUNDS * args.r),
        "--save-every-updates",
        str(10 * args.r),
        "--rollout-steps",
        "480",
        "--native-microbatch-size",
        str(settings["microbatch_size"]),
        "--actor-cache-retention-gib",
        str(settings["actor_cache_retention_gib"]),
        "--actor-cache-preallocate-gib",
        str(settings["actor_cache_preallocate_gib"]),
        "--timeout-seconds",
        str(settings["timeout_seconds"]),
        "--skip-finished-rollout",
        "--log-cuda-memory",
        "--output",
        str(run.absolute()),
    ]
    if args.run:
        command.append("--run")
    env = dict(os.environ)
    # Do not inherit stale W&B names/IDs from another shell experiment.
    env.pop("WANDB_RUN_ID", None)
    env.pop("OPD_WANDB_RUN_ID", None)
    env.update(
        WANDB_MODE=settings["wandb_mode"],
        OPD_WANDB_GROUP=f"offline50_80r_gapquery_seed{args.seed}",
        OPD_WANDB_RUN_NAME=label(args),
    )
    if getattr(args, "bound_teacher_reference", None) is not None:
        env["OPD_LIBERO_TEACHER_REFERENCE"] = str(args.bound_teacher_reference)
    return command, env


def contract(args: argparse.Namespace, identity: str, resolved: Path) -> dict:
    """Keep evaluation recovery tied to the exact training recipe."""
    return {
        "schema_version": 1,
        "method": args.method,
        "r": args.r,
        "interval": args.interval,
        "seed": args.seed,
        "rounds": ROUNDS,
        "initializer": identity,
        "training_config_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "curve_rounds": list(CURVE_ROUNDS),
        "curve_episodes_per_task": CURVE_EPISODES,
        "final_episodes_per_task": 50,
    }


def weights(run: Path, args: argparse.Namespace, round_: int) -> Path:
    """Use optimizer-update IDs for checkpoints and physical rounds for plots."""
    return (
        run
        / f"libero_{args.method}/checkpoints"
        / f"global_step_{round_ * args.r}"
        / "actor/model_state_dict/full_weights.pt"
    )


def compatible_training_contract(actual: dict, expected: dict) -> bool:
    """Allow the evaluation-only 10-to-20 upgrade without retraining a policy."""
    if actual == expected:
        return True
    return (
        actual.get("curve_episodes_per_task") == 10
        and expected["curve_episodes_per_task"] == 20
        and {**actual, "curve_episodes_per_task": 20} == expected
    )


def evaluation_spec(
    args: argparse.Namespace,
    settings: dict,
    run: Path,
    round_: int,
    episodes: int,
    output: Path,
) -> list[str]:
    """Run exact official states with episode-private noise and native B1 policy."""
    command = [
        settings["python"],
        "-m",
        "toolkits.libero_gr00t.launch_parallel_evaluation",
        "--python",
        settings["python"],
        "--config",
        str(run / "resolved_config.yaml"),
        "--student",
        settings["student"],
        "--gpus",
        *map(str, settings["gpus"]),
        "--cpus",
        str(max(2, settings["cpus"] // 4)),
        "--envs-per-task",
        "5",
        "--trials",
        *map(str, range(episodes)),
        "--seed",
        "1234",
        "--output",
        str(output),
        "--reuse-complete",
        "--resume-incomplete",
        "--run",
    ]
    command += (
        ["--baseline"]
        if round_ == 0
        else ["--weights", str(weights(run, args, round_))]
    )
    return command


def export_evaluation(
    args: argparse.Namespace,
    settings: dict,
    run: Path,
    output: Path,
    round_: int,
    episodes: int,
) -> None:
    """Separate curve CSVs from final CSVs; reuse full endpoints without physics."""
    common = {"method": args.method, "interval": args.interval}
    record_evaluation(
        Path(settings["csv_dir"]),
        output,
        args.r,
        args.seed,
        round_,
        run,
        episodes_per_task=episodes,
        **common,
    )
    if episodes == 50:
        record_evaluation(
            Path(settings["csv_dir"]),
            output,
            args.r,
            args.seed,
            round_,
            run,
            episodes_per_task=CURVE_EPISODES,
            **common,
        )


def evaluate(
    args: argparse.Namespace, settings: dict, run: Path, identity: str
) -> None:
    """Prioritize final results, then run intermediate curves with resumable tasks."""
    if (run / "status").read_text().strip() != "OK":
        raise ValueError(
            "Training has not completed successfully; no evaluation started"
        )
    expected = contract(args, identity, run / "resolved_config.yaml")
    if not compatible_training_contract(
        json.loads((run / "offline50_experiment.json").read_text()), expected
    ):
        raise ValueError(
            "Training recipe/variant changed; use the original run arguments"
        )
    selected = (
        [
            (
                args.eval_round,
                args.episodes_per_task
                or (50 if args.eval_round in (0, ROUNDS) else CURVE_EPISODES),
            )
        ]
        if args.eval_round is not None
        else [
            (0, 50),
            (ROUNDS, 50),
            *[(r, CURVE_EPISODES) for r in CURVE_ROUNDS if r not in (0, ROUNDS)],
        ]
    )
    for round_, _ in selected:
        if round_ and not weights(run, args, round_).is_file():
            raise FileNotFoundError(weights(run, args, round_))
    receipt = run / f"post_training_evaluation_n{CURVE_EPISODES}.json"
    state = {"identity": expected, "status": "RUNNING", "measurements": {}}
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["identity"] != expected:
            raise ValueError("Evaluation receipt differs from the frozen experiment")
        state["measurements"] = old["measurements"]
    # Prevent two recovery commands from dispatching duplicate jobs for one run.
    with (run / ".offline50_evaluation.lock").open("a") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX)
        # Reload after taking the lock: another invocation may have completed.
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous["identity"] != expected:
                raise ValueError("Evaluation recipe changed while waiting for lock")
            state["measurements"] = previous["measurements"]
        write_json(receipt, state)
        for round_, episodes in selected:
            key = f"{round_}:{episodes}"
            root = (
                Path(settings["csv_dir"]) / "shared_evaluations"
                if round_ == 0
                else run / "evaluations"
            )
            root.mkdir(parents=True, exist_ok=True)
            output = root / (
                f"initial_offline50_n{episodes}"
                if round_ == 0
                else f"round{round_:03d}_n{episodes}"
            )
            lock_path = root / (output.name + ".lock")
            record = {
                "status": "RUNNING",
                "output": str(output),
                "round": round_,
                "episodes_per_task": episodes,
            }
            state["measurements"][key] = record
            write_json(receipt, state)
            try:
                with lock_path.open("a") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    command = evaluation_spec(
                        args, settings, run, round_, episodes, output
                    )
                    print(shlex.join(command), flush=True)
                    subprocess.run(command, cwd=REPO, env=dict(os.environ), check=True)
                record["status"] = "EVALUATED"
                write_json(receipt, state)
                export_evaluation(args, settings, run, output, round_, episodes)
                record["status"] = "COMPLETE"
                write_json(receipt, state)
            except BaseException as error:
                record.update(
                    error=repr(error),
                    status="EVALUATED" if record["status"] == "EVALUATED" else "FAILED",
                )
                state["status"] = "FAILED"
                write_json(receipt, state)
                raise
        state["status"] = "COMPLETE" if args.eval_round is None else "SELECTED_COMPLETE"
        write_json(receipt, state)


def main(argv: list[str] | None = None) -> None:
    """Launch through the existing runtime without new environment requirements."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument(
        "--student",
        type=Path,
        help="Offline50 HF directory; overrides the path in memory, leaving the machine JSON unchanged",
    )
    parser.add_argument("--method", choices=("uniform", "adaptive"), default="uniform")
    parser.add_argument("--r", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--interval",
        type=int,
        choices=(5, 10),
        help="Required for Adaptive; rounds, not optimizer updates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New training output directory; existing outputs are never overwritten",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Evaluation recovery only: relocated resolved_config.yaml",
    )
    parser.add_argument("--eval-round", type=int, choices=CURVE_ROUNDS)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        choices=(CURVE_EPISODES, 50),
        help="Single-checkpoint evaluation only",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluate-all", action="store_true")
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Check the exact student bytes; no training/evaluation",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Train80 rounds, then automatically evaluate; default is plan-only",
    )
    args = parser.parse_args(argv)
    if (args.method == "adaptive") != (args.interval is not None):
        parser.error("Adaptive requires --interval 5 or 10; Uniform must omit it")
    if args.episodes_per_task is not None and args.eval_round is None:
        parser.error("--episodes-per-task requires --eval-round")
    if args.eval_round is not None and (
        args.train_only or args.evaluate_all or not args.run
    ):
        parser.error(
            "--eval-round requires --run and cannot combine with training/queue modes"
        )
    if args.config and not (args.evaluate_all or args.eval_round is not None):
        parser.error("--config is for evaluation recovery only")
    if args.evaluate_all and (not args.run or args.output):
        parser.error("--evaluate-all requires --run and no --output")
    settings = effective_settings(args)
    identity = verify_initializer(Path(settings["student"]))
    print(f"Verified Offline-Distilled-50: {identity}", flush=True)
    if args.verify_only:
        return
    if args.run:
        # The configured parent must already be mounted/prepared. No fallback.
        check_csv_directory(Path(settings["csv_dir"]).parent)
        Path(settings["csv_dir"]).mkdir(exist_ok=True)
        check_csv_directory(Path(settings["csv_dir"]))
    if args.evaluate_all or args.eval_round is not None:
        run = (
            args.config.parent
            if args.config
            else Path(settings["results"]) / label(args)
        )
        evaluate(args, settings, run.absolute(), identity)
        return
    if args.method == "adaptive":
        args.bound_teacher_reference = bind_reference(args, settings)
    command, env = training_spec(args, settings)
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)
    run = Path(command[command.index("--output") + 1])
    write_json(
        run / "offline50_experiment.json",
        contract(args, identity, run / "resolved_config.yaml"),
    )
    if args.run and not args.train_only:
        evaluate(args, settings, run, identity)


if __name__ == "__main__":
    main()
