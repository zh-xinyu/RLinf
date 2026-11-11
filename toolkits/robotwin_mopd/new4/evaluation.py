# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-pool evaluation through production workers, one CSV row per checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable

from toolkits.robotwin_mopd.new4 import contracts as c


def pool_path(machine: dict, *, smoke: bool = False) -> Path:
    """Use the exact shipped pool; runtime path differences are provenance only."""
    name = "dev2.json" if smoke else "formal100.json"
    prepared = Path(machine["output_root"]) / "prepared/evaluation" / name
    source = c.HERE / name
    if prepared.is_file():
        if json.loads(prepared.read_text()) != json.loads(source.read_text()):
            raise ValueError("Prepared evaluation pool differs from the fixed release")
        return prepared
    return source


def task_directory(machine: dict, job: dict, round_: int, task: str) -> Path:
    """Keep every checkpoint/task result separate from training and setup smoke."""
    return (
        Path(machine["output_root"]) / "eval" / c.job_id(job) / f"round{round_}" / task
    )


def read_result(
    machine: dict,
    directory: Path,
    task: str,
    checkpoint: Path,
    *,
    episodes: int = 100,
    smoke: bool = False,
) -> dict | None:
    """Reject wrong checkpoints, scene IDs, actions protocols or partial counts."""
    path = directory / "summary.json"
    if not path.is_file():
        return None
    from omegaconf import OmegaConf

    from toolkits.robotwin_mopd.author_pool import (
        load_pool,
        runtime_identity,
        validate_scene_info,
    )

    pool = load_pool(pool_path(machine, smoke=smoke), task_names=c.TASKS)
    result = json.loads(path.read_text())
    expected = pool["tasks"][task][:episodes]
    if (
        result["task"] != task
        or Path(result["checkpoint"]).resolve() != checkpoint.resolve()
        or result["pool_content_sha256"] != pool["pool_content_sha256"]
        or result["pool_protocol"] != pool["protocol"]
        or result["requested_episodes"] != episodes
        or result["completed_episodes"] != episodes
        or len(result["episodes"]) != episodes
        or result.get("evaluation_chain")
        != "EmbodiedEvalRunner/EnvWorker/MultiStepRolloutWorker"
        or runtime_identity(result["verified_runtime"])
        != runtime_identity(pool["protocol"]["runtime_profile"])
    ):
        raise ValueError(f"Wrong/incomplete evaluation identity: {path}")
    successes = 0
    for index, (row, entry) in enumerate(
        zip(result["episodes"], expected, strict=True)
    ):
        if (
            row["episode"] != index
            or row["actual_reset_seed"] != entry["scene_seed"]
            or row["instruction"] != entry["instruction"]
            or row["task"] != task
            or type(row["success_once"]) is not bool
            or not 0 < row["simulation_steps"] <= c.RECIPE["task_horizons"][task]
        ):
            raise ValueError(f"Evaluation episode identity mismatch: {path}, {index}")
        validate_scene_info(entry["episode_info"], row["actual_scene_info"])
        successes += row["success_once"]
    if (
        result["successes"] != successes
        or result["success_rate"] != successes / episodes
    ):
        raise ValueError(f"Evaluation success counts differ: {path}")
    cfg = OmegaConf.load(directory / "resolved_config.yaml")
    if cfg.rollout.model.num_steps != 10 or cfg.rollout.model.num_action_chunks != 25:
        raise ValueError("Evaluation must retain ODE10 and executionH25")
    norms = cfg.rollout.model.opd_task_norm_stats
    if set(norms) != set(c.TASKS) or any(
        Path(norms[name]).resolve()
        != (
            Path(machine["resource_root"]) / "student/assets" / name / "norm_stats.json"
        ).resolve()
        for name in c.TASKS
    ):
        raise ValueError("Evaluation normalizers differ from the training contract")
    return {
        "success_rate": successes / episodes,
        "successes": successes,
        "episodes": episodes,
        "summary": str(path),
        "pool_content_sha256": pool["pool_content_sha256"],
    }


def checkpoint_results(machine: dict, job: dict, round_: int, checkpoint: Path) -> dict:
    """Return partial task progress without fabricating a macro success rate."""
    results = {
        task: read_result(
            machine, task_directory(machine, job, round_, task), task, checkpoint
        )
        for task in c.TASKS
    }
    completed = sum(value is not None for value in results.values())
    return {
        "tasks": results,
        "completed_tasks": completed,
        "state": "COMPLETE"
        if completed == 4
        else "PARTIAL"
        if completed
        else "PENDING",
        "macro_sr": sum(row["success_rate"] for row in results.values()) / 4
        if completed == 4
        else None,
    }


def worker_command(
    machine: dict,
    task: str,
    checkpoint: Path,
    directory: Path,
    gpu: int,
    *,
    smoke: bool = False,
) -> list[str]:
    """Retain production policy transforms and exact fixed-pool RNG behavior."""
    command = [
        str(Path(machine["runtime_root"]) / "venv/bin/python"),
        "-u",
        "-m",
        "toolkits.robotwin_mopd.worker_evaluate",
        "--task",
        task,
        "--tasks",
        *c.TASKS,
        "--pool",
        str(pool_path(machine, smoke=smoke)),
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        "2" if smoke else "100",
        "--gpu",
        str(gpu),
        "--output",
        str(directory),
        "--gpu-memory-fraction",
        str(machine.get("evaluation_gpu_memory_fraction", 0.32)),
    ]
    for name in c.TASKS:
        command += [
            "--norm-stats",
            f"{name}={Path(machine['resource_root']) / 'student/assets' / name / 'norm_stats.json'}",
        ]
    for value in c.overrides(machine, {"method": "uniform_r2", "seed": 2}):
        command += ["--override", value]
    return command


def run_task(
    machine: dict,
    task: str,
    checkpoint: Path,
    directory: Path,
    gpu: int,
    *,
    smoke: bool = False,
) -> None:
    """Own one single-GPU Ray cluster; never stop or overwrite another job."""
    from toolkits.robocasa_mopd.owned_ray import OWNER_KEY, stop_owned, wait_for_gpus

    if task not in c.TASKS or directory.exists():
        raise ValueError(f"Unknown task or existing partial output: {directory}")
    env = c.environment(machine)
    index = c.TASKS.index(task)
    port = machine["ray_port"] + 10 + index
    dashboard = machine["dashboard_port"] + 10 + index
    worker_first = machine["ray_port"] + 20000 + index * 128
    worker_last = worker_first + 127
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        RAY_ADDRESS=f"127.0.0.1:{port}",
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        RLINF_NODE_RANK="0",
        GPU_FREE_MB=str(machine.get("evaluation_minimum_gpu_free_mib", 30000)),
    )
    wait_for_gpus(1, env)
    directory.parent.mkdir(parents=True, exist_ok=True)
    log_path = directory.parent / f"{task}.log"
    if log_path.exists():
        raise FileExistsError(log_path)
    owner = "new4-eval-" + uuid.uuid4().hex
    with open(f"/tmp/opd-robotwin-{os.getuid()}-{port}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for value in (port, dashboard, *range(worker_first, worker_last + 1)):
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", value)) == 0:
                    raise RuntimeError(f"Evaluation port {value} is already in use")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Evaluation received signal {signum}")

        handlers = {
            value: signal.signal(value, interrupted)
            for value in (signal.SIGTERM, signal.SIGHUP)
        }
        try:
            with log_path.open("x") as log:
                subprocess.run(
                    [
                        str(Path(machine["runtime_root"]) / "venv/bin/ray"),
                        "start",
                        "--head",
                        f"--port={port}",
                        f"--min-worker-port={worker_first}",
                        f"--max-worker-port={worker_last}",
                        "--num-gpus=1",
                        f"--num-cpus={machine.get('evaluation_cpus_per_task', 8)}",
                        "--include-dashboard=true",
                        "--dashboard-host=127.0.0.1",
                        f"--dashboard-port={dashboard}",
                        "--temp-dir=" + tempfile.mkdtemp(prefix="new4-eval-ray-"),
                    ],
                    env={**env, OWNER_KEY: owner},
                    cwd=c.REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                process = subprocess.Popen(
                    worker_command(
                        machine, task, checkpoint, directory, gpu, smoke=smoke
                    ),
                    env=env,
                    cwd=c.REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    if process.wait():
                        raise RuntimeError(
                            f"Evaluation failed: {task}; preserve {log_path}"
                        )
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            if (
                read_result(
                    machine,
                    directory,
                    task,
                    checkpoint,
                    episodes=2 if smoke else 100,
                    smoke=smoke,
                )
                is None
            ):
                raise ValueError("Evaluation completed without a validated result")
        finally:
            try:
                stop_owned(owner)
            finally:
                for value, handler in handlers.items():
                    signal.signal(value, handler)


def evaluate_checkpoint(
    machine: dict,
    machine_path: Path,
    job: dict,
    round_: int,
    checkpoint: Path,
    *,
    on_complete: Callable[[], None] | None = None,
    smoke: bool = False,
) -> dict:
    """Evaluate four tasks concurrently, then publish exactly one checkpoint row."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(visible) != 4
        or len(set(visible)) != 4
        or not all(value.isdecimal() for value in visible)
    ):
        raise ValueError("Evaluation needs four allocated physical GPU IDs")
    children = []
    outputs = {}
    for task in c.TASKS:
        outputs[task] = (
            Path(machine["output_root"]) / "smoke/evaluation" / task
            if smoke
            else task_directory(machine, job, round_, task)
        )
    lock_path = next(iter(outputs.values())).parent / ".checkpoint.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Checkpoint evaluation received signal {signum}")

    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handlers = {
            value: signal.signal(value, interrupted)
            for value in (signal.SIGTERM, signal.SIGHUP)
        }
        try:
            for task, gpu in zip(c.TASKS, visible, strict=True):
                previous = read_result(
                    machine,
                    outputs[task],
                    task,
                    checkpoint,
                    episodes=2 if smoke else 100,
                    smoke=smoke,
                )
                if previous is not None:
                    continue
                if outputs[task].exists():
                    raise FileExistsError(
                        f"Partial evaluation preserved: {outputs[task]}; archive the failed task before retry"
                    )
                command = [
                    str(Path(machine["runtime_root"]) / "venv/bin/python"),
                    "-m",
                    "toolkits.robotwin_mopd.new4.evaluation",
                    "--machine",
                    str(machine_path),
                    "--task",
                    task,
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(outputs[task]),
                    "--gpu",
                    gpu,
                ]
                if smoke:
                    command.append("--smoke")
                children.append(
                    subprocess.Popen(
                        command,
                        env=c.environment(machine),
                        cwd=c.REPO,
                        start_new_session=True,
                    )
                )
            codes = [process.wait() for process in children]
            if any(codes):
                raise RuntimeError(
                    f"Checkpoint evaluation failed: {codes}; completed tasks and logs retained"
                )
            results = {
                task: read_result(
                    machine,
                    outputs[task],
                    task,
                    checkpoint,
                    episodes=2 if smoke else 100,
                    smoke=smoke,
                )
                for task in c.TASKS
            }
            if any(result is None for result in results.values()):
                raise ValueError("Evaluation is missing task results")
            record = {
                "state": "COMPLETE",
                "tasks": results,
                "macro_sr": sum(result["success_rate"] for result in results.values())
                / 4,
            }
            if on_complete:
                on_complete()
            return record
        finally:
            for process in children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in children:
                if process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

            for value, handler in handlers.items():
                signal.signal(value, handler)


def main() -> None:
    """Launch a single owned task evaluation selected by the parent driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--task", choices=c.TASKS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_task(
        c.read_machine(args.machine),
        args.task,
        args.checkpoint,
        args.output,
        args.gpu,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
