# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded SFT100/200/500 evaluation followed by predeclared SFT500 MOPD.

No best-checkpoint selection: 100 and 200 are descriptive; only 500 can initialize
MOPD. A weak, explicitly exploratory viability screen precedes that fresh run.
The queue owns only children started through the existing scoped launchers.
"""

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

TASKS = ("adjust_bottle", "handover_mic", "place_container_plate", "stack_bowls_two")
SFT_RUN = "robotwin4_joint_sft_500_b128_micro4_pilot_20261101"
MOPD_RUN = "robotwin4_joint_sft500_r4_t32_b128_pilot_20261101"


class ReviewRequired(RuntimeError):
    """A stopped queue requiring review; never permission to evict another job."""


class BusyGPUs(ReviewRequired):
    """Selected GPUs are occupied; the queue must stop without eviction."""


def validate_report(report: dict) -> None:
    """Require exactly eight recorded episodes per task and consistent counts."""
    if set(report["tasks"]) != set(TASKS):
        raise ValueError("Evaluation task set differs from the declared four tasks")
    records = report["episodes"]
    if Counter(row["task"] for row in records) != Counter(dict.fromkeys(TASKS, 8)):
        raise ValueError("Evaluation requires 32 actual records, eight per task")
    for task in TASKS:
        rows = [row for row in records if row["task"] == task]
        if any(not isinstance(row["success_once"], bool) for row in rows):
            raise ValueError("Episode success must be boolean")
        if any(type(row["actual_reset_seed"]) is not int for row in rows):
            raise ValueError("Actual reset seeds must be recorded as integers")
        successes = sum(row["success_once"] for row in rows)
        summary = report["tasks"][task]
        if (
            summary["episodes"] != 8
            or summary["successes"] != successes
            or summary["success_rate"] != successes / 8
        ):
            raise ValueError("Task summary does not match episode records")


def scene_provenance(report: dict) -> dict:
    """Report actual reset retries without changing the historical episode protocol."""
    records = report["episodes"]
    return {
        "episode_counts": dict(Counter(row["task"] for row in records)),
        "unique_actual_scene_seeds_per_task": {
            task: len(
                {row["actual_reset_seed"] for row in records if row["task"] == task}
            )
            for task in TASKS
        },
        "duplicate_actual_scene_seeds_allowed": True,
        "sampling_note": "Eight episodes per task; not necessarily eight independent scenes. Native reset retries may duplicate actual scene seeds.",
    }


def viability_gate(report: dict) -> dict:
    """An exploratory screen, not evidence of statistical equivalence."""
    validate_report(report)
    successful_tasks = [
        task for task in TASKS if report["tasks"][task]["successes"] >= 1
    ]
    return {
        **scene_provenance(report),
        "passed": len(successful_tasks) >= 2,
        "successful_tasks": successful_tasks,
        "criterion": "All four tasks have 8 episodes; at least 1 success in at least 2 tasks",
        "interpretation": "Exploratory viability screen only; SFT500 was predeclared",
    }


def compare_reports(initial: dict, final: dict) -> dict:
    """Compare SR while reporting whether actual scene-seed sequences match."""
    validate_report(initial)
    validate_report(final)

    def scene(report):
        return [(row["task"], row["actual_reset_seed"]) for row in report["episodes"]]

    return {
        "initial_scene_provenance": scene_provenance(initial),
        "final_scene_provenance": scene_provenance(final),
        "matched_actual_scene_seed_sequence": scene(initial) == scene(final),
        "action_noise_fixed": False,
        "comparison_kind": "Matched scenes; action noise not fixed"
        if scene(initial) == scene(final)
        else "Unmatched scenes; descriptive comparison only",
        "tasks": {
            task: {
                "initial_successes": initial["tasks"][task]["successes"],
                "final_successes": final["tasks"][task]["successes"],
                "episodes_each": 8,
                "success_rate_change": final["tasks"][task]["success_rate"]
                - initial["tasks"][task]["success_rate"],
            }
            for task in TASKS
        },
    }


def mopd_command(base: Path, checkpoint: Path) -> list[str]:
    """Start fresh optimization from the fixed SFT500 model checkpoint."""
    return [
        sys.executable,
        "-u",
        "-m",
        "toolkits.robotwin_mopd.run",
        "--num-gpus",
        "8",
        "--run-id",
        MOPD_RUN,
        "--output-root",
        str(base / "train"),
        "runner.max_steps=80",
        "runner.save_interval=20",
        "runner.resume_dir=null",
        "runner.ckpt_path=null",
        "env.train.total_num_envs=32",
        "algorithm.flow_refresh_updates=4",
        "actor.global_batch_size=128",
        "actor.micro_batch_size=8",
        "actor.optim.lr=1e-5",
        "actor.optim.value_lr=1e-5",
        "actor.optim.lr_warmup_steps=4",
        "+actor.model.openpi.nft_full_horizon=true",
        "actor.model.openpi.nft_joint_forward=true",
        "actor.cast_trainable_params_to_fp32=true",
        "actor.model.require_fp32_trainable_params=true",
        "actor.fsdp_config.mixed_precision.param_dtype=null",
        "actor.fsdp_config.mixed_precision.reduce_dtype=null",
        "actor.fsdp_config.mixed_precision.buffer_dtype=null",
        "actor.fsdp_config.amp_autocast.enabled=true",
        "actor.fsdp_config.amp_autocast.precision=bf16",
        f"actor.model.model_path={json.dumps(str(checkpoint))}",
        "runner.logger.wandb_group=robotwin_joint_sft_mopd_pilots",
    ]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def exact_process_alive(identity: dict) -> bool:
    """Never mistake a reused PID for the original supervisor."""
    try:
        process = psutil.Process(identity["pid"])
        return (
            abs(process.create_time() - identity["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def upstream_state(identity: dict, state_path: Path) -> tuple[dict, bool]:
    state = json.loads(state_path.read_text())
    if state.get("pid") != identity["pid"]:
        raise ReviewRequired(f"Upstream identity changed: {state_path}")
    status = state["status"]
    if status == "FAILED" or "REVIEW" in status:
        raise ReviewRequired(f"Upstream stopped with {status}: {state_path}")
    alive = exact_process_alive(identity)
    if not alive and status != "COMPLETE":
        raise ReviewRequired(f"Upstream exited without COMPLETE: {state_path}")
    return state, alive


def require_idle(count: int, env: dict) -> None:
    """Reject both memory pressure and any selected compute-process ownership."""
    from toolkits.robocasa_mopd.owned_ray import check_gpus

    try:
        check_gpus(count, env)
    except RuntimeError as error:
        raise BusyGPUs(str(error)) from error
    visible = env["CUDA_VISIBLE_DEVICES"]
    uuids = set(
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={visible}",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
            env=env,
        ).splitlines()
    )
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        env=env,
    ).splitlines()
    if any(row.split(",")[0].strip() in uuids for row in rows):
        raise BusyGPUs(
            "Selected GPUs still have compute processes; no eviction attempted"
        )


def queue_files(base: Path, tag: str) -> tuple[Path, Path]:
    """Give each explicitly requested queue its own immutable launch identity."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", tag):
        raise ValueError(
            "Queue tag must contain only letters, numbers, underscores or hyphens"
        )
    setup = base / "setup"
    return (
        setup / f"joint_sft_followup_queue_{tag}.json",
        setup / f"joint_sft_followup_launch_{tag}.json",
    )


def wait_for_idle(
    count: int, env: dict, grace_seconds: float, deadline: float, on_wait=None
) -> None:
    """Optionally wait for GPU drain without relaxing ownership or memory checks."""
    end = min(time.monotonic() + grace_seconds, deadline)
    while True:
        try:
            require_idle(count, env)
            return
        except BusyGPUs as error:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise
            if on_wait is not None:
                on_wait(str(error), remaining)
            time.sleep(min(10, remaining))


def parse_args(argv=None):
    """Parse recovery identity and optional bounded GPU-drain grace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--queue-tag", default="20261101")
    parser.add_argument(
        "--upstream-launch",
        type=Path,
        help="Explicit eval-only recovery supervisor launch identity",
    )
    parser.add_argument(
        "--upstream-status",
        type=Path,
        help="Matching recovery supervisor atomic status JSON",
    )
    parser.add_argument(
        "--gpu-drain-grace-seconds",
        type=int,
        default=0,
        help="Explicit wait for strict GPU-idle checks, 0 (default) to 600 seconds",
    )
    args = parser.parse_args(argv)
    try:
        queue_files(args.base, args.queue_tag)
    except ValueError as error:
        parser.error(str(error))
    if (args.upstream_launch is None) != (args.upstream_status is None):
        parser.error(
            "--upstream-launch and --upstream-status must be supplied together"
        )
    if not 0 <= args.gpu_drain_grace_seconds <= 600:
        parser.error("--gpu-drain-grace-seconds must be between 0 and 600")
    return args


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    setup = base / "setup"
    statefile, identityfile = queue_files(base, args.queue_tag)
    upstream_launch = (
        args.upstream_launch or setup / "longer_opd_queue_launch_20261101.json"
    ).resolve()
    upstream_status_path = (
        args.upstream_status or setup / "longer_opd_queue_20261101.json"
    ).resolve()
    from toolkits.robotwin_mopd.run import run_command

    with open(f"/tmp/robotwin-joint-sft-followup-{os.getuid()}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if statefile.exists() or identityfile.exists():
            raise FileExistsError(
                "Queue identity/state already exists; review before any rerun"
            )
        identity = {"pid": os.getpid(), "create_time": psutil.Process().create_time()}
        atomic_json(identityfile, identity)
        state = dict(
            identity,
            started_unix=time.time(),
            predeclared_init_step=500,
            action_noise_fixed=False,
            queue_tag=args.queue_tag,
            upstream_launch=str(upstream_launch),
            upstream_status=str(upstream_status_path),
            gpu_drain_grace_seconds=args.gpu_drain_grace_seconds,
        )
        deadline = time.monotonic() + 10 * 3600

        def write(status, **kwargs):
            state.update(status=status, updated_unix=time.time(), **kwargs)
            atomic_json(statefile, state)

        def bounded_run(command, log, env):
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("Bounded followup reached its 10-hour deadline")
            signal.alarm(remaining)
            try:
                run_command(command, log, env)
            finally:
                signal.alarm(0)

        def wait_upstream(identity_path, status_path, checkpoint=None, step=None):
            upstream = json.loads(identity_path.read_text())
            while time.monotonic() < deadline:
                status, alive = upstream_state(upstream, status_path)
                if checkpoint is None:
                    if status["status"] == "COMPLETE" and not alive:
                        return
                else:
                    metadata = checkpoint / "metadata.json"
                    weights = checkpoint / "actor/model_state_dict/full_weights.pt"
                    if (
                        metadata.exists()
                        and weights.exists()
                        and weights.stat().st_size > 0
                    ):
                        try:
                            data = json.loads(metadata.read_text())
                        except json.JSONDecodeError:
                            data = {}
                        if (
                            data.get("step") == step
                            and data.get("checkpoint_kind")
                            == "full_weights_and_optimizer"
                        ):
                            return
                    if status["status"] == "COMPLETE" and not alive:
                        raise ReviewRequired(
                            f"SFT finished without valid checkpoint {step}"
                        )
                time.sleep(20)
            raise TimeoutError("Upstream wait exceeded queue's 10-hour deadline")

        def interrupted(signum, frame):
            if signum == signal.SIGALRM:
                raise TimeoutError(
                    "Queue's 10-hour deadline expired during owned child"
                )
            raise KeyboardInterrupt(signum)

        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
            signal.signal(sig, interrupted)
        env4 = dict(os.environ, CUDA_VISIBLE_DEVICES="4,5,6,7", USE_TF="0")
        env8 = dict(env4, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
        sft_identity = setup / "joint_sft/long_pilot_launch_20261101.json"
        sft_status = setup / "joint_sft/long_pilot_status_20261101.json"
        sft_root = base / "train" / SFT_RUN
        reports = {}

        def idle_before_launch(count, env):
            wait_for_idle(
                count,
                env,
                args.gpu_drain_grace_seconds,
                deadline,
                on_wait=lambda reason, remaining: write(
                    "WAITING_GPU_DRAIN",
                    gpu_drain_reason=reason,
                    gpu_drain_remaining_seconds=remaining,
                ),
            )

        def evaluate(checkpoint, name):
            idle_before_launch(4, env4)
            command = [
                sys.executable,
                "-u",
                str(setup / "launch_teacher_train_diagnostic.py"),
                name,
                "--trajectories",
                "16",
                "--rounds",
                "2",
                "--seed",
                "200000",
                "--student-checkpoint",
                str(checkpoint),
            ]
            write("EVALUATING", evaluation_name=name, evaluation_command=command)
            bounded_run(command, setup / f"{name}.queue.log", env4)
            path = base / "analysis" / name
            if not (path / "COMPLETE").exists():
                raise ReviewRequired(f"Evaluation did not publish COMPLETE: {name}")
            report = json.loads((path / "results.json").read_text())
            validate_report(report)
            return report

        try:
            write("WAITING_LONGER_OPD")
            wait_upstream(
                upstream_launch,
                upstream_status_path,
            )
            for step in (100, 200):
                write("WAITING_SFT_CHECKPOINT", checkpoint_step=step)
                checkpoint = sft_root / "checkpoints" / f"global_step_{step}"
                wait_upstream(sft_identity, sft_status, checkpoint, step)
                reports[str(step)] = evaluate(
                    checkpoint, f"joint_sft_step{step}_trainpath_t16_20261101"
                )
                upstream_state(json.loads(sft_identity.read_text()), sft_status)
            write("WAITING_SFT_COMPLETE")
            wait_upstream(sft_identity, sft_status)
            checkpoint = sft_root / "checkpoints/global_step_500"
            wait_upstream(sft_identity, sft_status, checkpoint, 500)
            initial = evaluate(checkpoint, "joint_sft_step500_trainpath_t16_20261101")
            reports["500"] = initial
            gate = viability_gate(initial)
            atomic_json(
                base / "analysis/joint_sft_evaluations_20261101.json",
                {"predeclared_init_step": 500, "gate": gate, "reports": reports},
            )
            if not gate["passed"]:
                write(
                    "REVIEW_REQUIRED",
                    gate=gate,
                    reason="SFT500 failed exploratory multi-task viability screen",
                )
                return
            idle_before_launch(8, env8)
            command = mopd_command(base, checkpoint)
            write("TRAINING_MOPD", gate=gate, command=command)
            bounded_run(command, setup / "joint_sft500_mopd_20261101.log", env8)
            run = base / "train" / MOPD_RUN
            if (run / "status").read_text().strip() != "COMPLETE":
                raise ReviewRequired("MOPD returned without COMPLETE")
            comparisons = {}
            for step in (40, 80):
                checkpoint = run / MOPD_RUN / "checkpoints" / f"global_step_{step}"
                if not (checkpoint / "actor/model_state_dict/full_weights.pt").exists():
                    raise ReviewRequired(f"Missing completed MOPD checkpoint {step}")
                final = evaluate(
                    checkpoint, f"joint_sft500_mopd_step{step}_trainpath_t16_20261101"
                )
                comparisons[str(step)] = compare_reports(initial, final)
            summary = {
                "predeclared_init_step": 500,
                "gate": gate,
                "sft_reports": reports,
                "mopd_comparisons": comparisons,
            }
            atomic_json(
                base / "analysis/joint_sft500_mopd_summary_20261101.json", summary
            )
            write("COMPLETE", comparisons=comparisons)
        except BusyGPUs as error:
            write("WAITING_GPU_REVIEW", reason=str(error), automatic_retry=False)
        except ReviewRequired as error:
            write("REVIEW_REQUIRED", reason=str(error))
        except BaseException as error:
            write("FAILED", error=repr(error))
            raise


if __name__ == "__main__":
    main()
