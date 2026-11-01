# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Wait for one identified pilot, then evaluate its final student and base extension.

This queue never starts training, stops other processes, or re-runs teachers.
Only a dead original launcher, COMPLETE status, and the exact final checkpoint
permit evaluations to start. Busy GPUs are left untouched.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

TASKS = ("adjust_bottle", "handover_mic", "place_container_plate", "stack_bowls_two")
REPO = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def same_process_alive(pid: int, created: float) -> bool:
    """Treat a reused PID as a different process, never as the original launcher."""
    try:
        process = psutil.Process(pid)
        return (
            abs(process.create_time() - created) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def training_gate(run: Path, identity: dict, checkpoint: Path) -> str:
    if same_process_alive(identity["pid"], identity["create_time"]):
        return "WAITING_TRAINING"
    status = (
        (run / "status").read_text().strip() if (run / "status").exists() else "MISSING"
    )
    if status != "COMPLETE":
        return f"BLOCKED_TRAINING_{status}"
    weights = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not weights.is_file() or weights.stat().st_size == 0:
        return "BLOCKED_MISSING_FINAL_CHECKPOINT"
    return "READY"


def reference_episodes(path: Path, expected: int, task: str) -> list[dict]:
    data = read_json(path)
    episodes = data["episodes"]
    if (
        data["task"] != task
        or data["completed_episodes"] != expected
        or len(episodes) != expected
    ):
        raise ValueError(
            f"Expected exactly {expected} complete {task} episodes: {path}"
        )
    seeds = [int(row["actual_reset_seed"]) for row in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Repeated actual reset seeds: {path}")
    return episodes


def build_jobs(references: Path, output: Path, checkpoint: Path) -> list[dict]:
    jobs = []
    for kind in ("student", "base_pilot_extension"):
        for task in TASKS:
            base = reference_episodes(references / f"base_pilot_{task}.json", 3, task)
            reference_episodes(references / f"teacher_pilot_{task}.json", 5, task)
            seed = (
                200000 if kind == "student" else int(base[-1]["actual_reset_seed"]) + 1
            )
            episodes = 5 if kind == "student" else 2
            result = output / f"{kind}_{task}.json"
            if result.exists():
                raise FileExistsError(f"Refusing to overwrite evaluation: {result}")
            command = [
                sys.executable,
                "-u",
                str(REPO / "toolkits/robotwin_mopd/evaluate.py"),
                "--task",
                task,
                "--episodes",
                str(episodes),
                "--seed",
                str(seed),
                "--checkpoint",
                str(checkpoint) if kind == "student" else "base",
                "--output",
                str(result),
            ]
            jobs.append(
                {
                    "id": f"{kind}_{task}",
                    "kind": kind,
                    "task": task,
                    "episodes": episodes,
                    "seed": seed,
                    "command": command,
                    "output": str(result),
                    "log": str(result.with_suffix(".log")),
                    "status": "QUEUED",
                }
            )
    return jobs


def idle_gpus(allowed: list[str]) -> list[str]:
    """Require near-idle memory/utilization and no compute process on a GPU."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(command, text=True, timeout=15).splitlines()
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    ).splitlines()
    busy = {row.split(",")[0].strip() for row in apps if row.strip()}
    result = []
    for row in rows:
        index, uuid, used, utilization = [value.strip() for value in row.split(",")]
        if (
            index in allowed
            and uuid not in busy
            and float(used) <= 512
            and float(utilization) <= 5
        ):
            result.append(index)
    return result


def summarize(references: Path, output: Path, jobs: list[dict]) -> None:
    """Retain episode counts and explicit seed matching, including partial results."""
    summary = {
        "note": "Pilot estimates only; teacher adjust_bottle uses a shifted seed set.",
        "tasks": {},
    }
    for task in TASKS:
        sources = {
            "student": [output / f"student_{task}.json"],
            "base": [
                references / f"base_pilot_{task}.json",
                output / f"base_pilot_extension_{task}.json",
            ],
            "teacher": [references / f"teacher_pilot_{task}.json"],
        }
        entry = {}
        for kind, paths in sources.items():
            episodes = [
                episode
                for path in paths
                if path.exists()
                for episode in read_json(path)["episodes"]
            ]
            seeds = [episode["actual_reset_seed"] for episode in episodes]
            successes = sum(bool(episode["success_once"]) for episode in episodes)
            entry[kind] = {
                "episodes": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes) if episodes else None,
                "actual_reset_seeds": seeds,
                "unique_seeds": len(set(seeds)),
                "sources": [str(path) for path in paths],
            }
        student = entry["student"]["actual_reset_seeds"]
        for comparison in ("base", "teacher"):
            other = entry[comparison]["actual_reset_seeds"]
            entry[f"student_vs_{comparison}_seeds"] = {
                "same_ordered_five_seeds": len(student) == len(other) == 5
                and student == other,
                "overlap": sorted(set(student) & set(other)),
                "student_only": sorted(set(student) - set(other)),
                "reference_only": sorted(set(other) - set(student)),
            }
        summary["tasks"][task] = entry
    summary["jobs"] = [
        {key: job.get(key) for key in ("id", "status", "pid", "returncode")}
        for job in jobs
    ]
    atomic_json(output / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--launcher-create-time", type=float, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_parallel < 1 or args.poll_seconds < 1:
        parser.error("Parallelism and polling interval must be positive")
    run, output, references = (
        path.resolve() for path in (args.run_dir, args.output, args.references)
    )
    runtime = Path(os.environ["ROBOTWIN_RUNTIME_ROOT"]).resolve()
    if Path(sys.prefix).resolve() != runtime / "venv":
        parser.error("Use the isolated RoboTwin runtime")
    identity = {"pid": args.launcher_pid, "create_time": args.launcher_create_time}
    # At arm time the launcher must still be the process explicitly authorized.
    if not same_process_alive(identity["pid"], identity["create_time"]):
        raise RuntimeError(
            "Original launcher is not alive; refusing to arm a new queue"
        )
    process = psutil.Process(args.launcher_pid)
    command = process.cmdline()
    if (
        process.uids().real != os.getuid()
        or "toolkits.robotwin_mopd.run" not in command
        or run.name not in command
    ):
        raise RuntimeError("Launcher identity/command does not match this pilot")
    checkpoint = run / run.name / "checkpoints/global_step_40"
    jobs = build_jobs(references, output, checkpoint)
    plan = {
        "launcher": identity,
        "launcher_command": command,
        "run_dir": str(run),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "references": str(references),
        "jobs": jobs,
        "status": "WAITING_TRAINING",
        "queue_pid": os.getpid(),
        "queue_create_time": psutil.Process().create_time(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_config_sha256": hashlib.sha256(
            (run / "resolved_config.yaml").read_bytes()
        ).hexdigest(),
        "gpu_policy": {
            "allowed": args.gpus,
            "max_parallel": args.max_parallel,
            "maximum_used_mib": 512,
            "maximum_utilization": 5,
            "no_existing_compute_processes": True,
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "queue.json").exists():
            raise FileExistsError(
                "This output already has a queue; refusing a duplicate"
            )
        atomic_json(output / "queue.json", plan)
        (output / "queue.pid").write_text(str(os.getpid()) + "\n")
        print(
            f"Armed for launcher {identity}; waiting without acquiring GPUs", flush=True
        )
        while True:
            gate = training_gate(run, identity, checkpoint)
            if gate == "READY":
                break
            plan["status"] = gate
            plan["last_checked"] = time.time()
            atomic_json(output / "queue.json", plan)
            if gate.startswith("BLOCKED"):
                print(gate, flush=True)
                return
            time.sleep(args.poll_seconds)
        print(
            "Training complete, original launcher exited, final checkpoint present",
            flush=True,
        )
        active = {}
        plan["status"] = "EVALUATING"
        while any(job["status"] in ("QUEUED", "RUNNING") for job in jobs):
            for job_id, (child, stream, gpu) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                job = next(job for job in jobs if job["id"] == job_id)
                job.update(
                    returncode=code,
                    finished_at=time.time(),
                    status="COMPLETE" if code == 0 else "FAILED",
                )
                if code == 0:
                    try:
                        reference_episodes(
                            Path(job["output"]), job["episodes"], job["task"]
                        )
                    except (ValueError, KeyError, OSError) as error:
                        job.update(status="FAILED", validation_error=str(error))
                del active[job_id]
                print(f"{job_id}: {job['status']} (exit {code})", flush=True)
            try:
                available = [
                    gpu
                    for gpu in idle_gpus(args.gpus.split(","))
                    if gpu not in {value[2] for value in active.values()}
                ]
                plan.pop("gpu_query_error", None)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                available = []
                plan["gpu_query_error"] = str(error)
            for gpu in available:
                if len(active) >= args.max_parallel:
                    break
                job = next((job for job in jobs if job["status"] == "QUEUED"), None)
                if job is None:
                    break
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    USE_TF="0",
                    OPD_RUN_DIR=str(output),
                )
                stream = Path(job["log"]).open("x")
                child = subprocess.Popen(
                    job["command"],
                    cwd=REPO,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job.update(
                    status="RUNNING",
                    pid=child.pid,
                    create_time=psutil.Process(child.pid).create_time(),
                    gpu=gpu,
                    started_at=time.time(),
                )
                active[job["id"]] = (child, stream, gpu)
                print(f"Started {job['id']} on GPU {gpu}, PID {child.pid}", flush=True)
            plan["last_checked"] = time.time()
            atomic_json(output / "queue.json", plan)
            summarize(references, output, jobs)
            if active or any(job["status"] == "QUEUED" for job in jobs):
                time.sleep(args.poll_seconds)
        plan["status"] = (
            "COMPLETE"
            if all(job["status"] == "COMPLETE" for job in jobs)
            else "PARTIAL_FAILURE"
        )
        plan["finished_at"] = time.time()
        atomic_json(output / "queue.json", plan)
        summarize(references, output, jobs)
        print(plan["status"], flush=True)


if __name__ == "__main__":
    main()
