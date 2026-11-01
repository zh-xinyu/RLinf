# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Resume-safe supplementary feasible/unseen evaluation, after the original queue.

No training is launched or interrupted. Each attempt has a distinct output path;
resuming adopts running processes or validated results instead of overwriting
partial evidence. Three policies share one immutable scene/instruction pool.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from toolkits.robotwin_mopd.feasible_pool import load_pool
from toolkits.robotwin_mopd.finish_pilot import (
    REPO,
    TASKS,
    atomic_json,
    idle_gpus,
    read_json,
    same_process_alive,
)


def upstream_gate(path: Path, identity: dict) -> str:
    if same_process_alive(identity["pid"], identity["create_time"]):
        return "WAITING_UNFILTERED_QUEUE"
    state = read_json(path)
    if state["status"] != "COMPLETE":
        return "BLOCKED_UPSTREAM_" + state["status"]
    if not (
        Path(state["checkpoint"]) / "actor/model_state_dict/full_weights.pt"
    ).is_file():
        return "BLOCKED_MISSING_STUDENT_CHECKPOINT"
    return "READY"


def build_jobs(output: Path, checkpoint: Path) -> list[dict]:
    jobs = [
        {
            "id": "build_pool",
            "kind": "pool",
            "status": "QUEUED",
            "attempts": [],
            "output": str(output / "evaluation_pool.json"),
        }
    ]
    for policy in ("teacher", "base", "student"):
        for task in TASKS:
            jobs.append(
                {
                    "id": f"{policy}_{task}",
                    "kind": "evaluation",
                    "policy": policy,
                    "task": task,
                    "checkpoint": str(checkpoint) if policy == "student" else policy,
                    "status": "QUEUED",
                    "attempts": [],
                    "output": str(output / f"{policy}_{task}.json"),
                }
            )
    return jobs


def validate_result(job: dict, path: Path, pool_path: Path) -> dict:
    if job["kind"] == "pool":
        pool = load_pool(path)
        if set(pool["protocol"]["task_names"]) != set(TASKS):
            raise ValueError("Pool does not cover exactly the four pilot tasks")
        if any(len(pool["tasks"][task]) < 5 for task in TASKS):
            raise ValueError("Pool has fewer than five accepted scenes per task")
        return pool
    pool = load_pool(pool_path)
    result = read_json(path)
    if result["task"] != job["task"] or result["completed_episodes"] != 5:
        raise ValueError("Evaluation is incomplete or has the wrong task")
    expected = pool["tasks"][job["task"]][:5]
    episodes = result["episodes"]
    if (
        len(episodes) != 5
        or result["evaluation_pool"]["pool_content_sha256"]
        != pool["pool_content_sha256"]
    ):
        raise ValueError("Evaluation did not use the common pool")
    for row, scene in zip(episodes, expected, strict=True):
        if (
            row["actual_reset_seed"] != scene["scene_seed"]
            or row["reset_seed"] != scene["scene_seed"]
        ):
            raise ValueError("Evaluation changed a fixed scene seed")
        if (
            row["instruction"] != scene["instruction"]
            or row["evaluation_pool_sha256"] != pool["pool_content_sha256"]
        ):
            raise ValueError("Evaluation changed a fixed prompt or pool")
    return result


def attempt_alive(attempt: dict) -> bool:
    """Recover a child even if the queue died between spawn and PID persistence."""
    if attempt.get("pid") is not None:
        return same_process_alive(attempt["pid"], attempt["create_time"])
    for process in psutil.process_iter(
        ["pid", "uids", "cmdline", "create_time", "status"]
    ):
        try:
            if (
                process.info["uids"].real == os.getuid()
                and process.info["cmdline"] == attempt["command"]
                and process.info["status"] != psutil.STATUS_ZOMBIE
            ):
                attempt.update(pid=process.pid, create_time=process.info["create_time"])
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def make_attempt(job: dict, output: Path, gpu: str) -> dict:
    directory = output / "attempts" / job["id"] / f"attempt_{len(job['attempts']) + 1}"
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / "result.json"
    if job["kind"] == "pool":
        command = [
            sys.executable,
            "-u",
            str(REPO / "toolkits/robotwin_mopd/feasible_pool.py"),
            "--output",
            str(result),
            "--episodes-per-task",
            "5",
        ]
    else:
        command = [
            sys.executable,
            "-u",
            str(REPO / "toolkits/robotwin_mopd/evaluate.py"),
            "--task",
            job["task"],
            "--episodes",
            "5",
            "--checkpoint",
            job["checkpoint"],
            "--output",
            str(result),
            "--evaluation-pool",
            str(output / "evaluation_pool.json"),
        ]
    return {
        "command": command,
        "gpu": gpu,
        "output": str(result),
        "log": str(directory / "run.log"),
        "started_at": time.time(),
        "status": "STARTING",
    }


def idle_pool_gpus(allowed: list[str]) -> list[str]:
    """Match the pool builder's stricter 256 MiB startup guard."""
    idle = idle_gpus(allowed)
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    ).splitlines()
    return [
        index.strip()
        for row in rows
        for index, used in [row.split(",")]
        if index.strip() in idle and float(used) <= 256
    ]


def summarize(output: Path, jobs: list[dict]) -> None:
    pool_path = output / "evaluation_pool.json"
    if not pool_path.exists():
        return
    pool = load_pool(pool_path)
    summary = {
        "protocol": pool["protocol"],
        "pool_content_sha256": pool["pool_content_sha256"],
        "note": "Supplementary feasible/unseen pilot; separate from unchanged seen/unfiltered training and first evaluation.",
        "tasks": {},
    }
    for task in TASKS:
        entry = {}
        pairs = []
        for policy in ("teacher", "base", "student"):
            job = next(
                j for j in jobs if j.get("task") == task and j.get("policy") == policy
            )
            path = Path(job["output"])
            if not path.exists() and job["attempts"]:
                path = Path(job["attempts"][-1]["output"])
            data = read_json(path) if path.exists() else {"episodes": []}
            episodes = data["episodes"]
            signatures = [
                [row["actual_reset_seed"], row["instruction"]] for row in episodes
            ]
            pairs.append(signatures)
            successes = sum(bool(row["success_once"]) for row in episodes)
            entry[policy] = {
                "episodes": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes) if episodes else None,
                "seed_prompt_pairs": signatures,
                "status": job["status"],
                "source": str(path),
            }
        entry["identical_five_scenes_and_prompts"] = (
            all(len(pair) == 5 for pair in pairs) and pairs[0] == pairs[1] == pairs[2]
        )
        summary["tasks"][task] = entry
    atomic_json(output / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream", type=Path, required=True, help="Original finish_pilot queue.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_parallel < 1 or args.max_attempts < 1 or args.poll_seconds < 1:
        parser.error("Parallelism, attempts, and polling interval must be positive")
    runtime = Path(os.environ["ROBOTWIN_RUNTIME_ROOT"]).resolve()
    if Path(sys.prefix).resolve() != runtime / "venv":
        parser.error("Use the isolated RoboTwin runtime")
    upstream, output = args.upstream.resolve(), args.output.resolve()
    previous = read_json(upstream)
    identity = {
        "pid": previous["queue_pid"],
        "create_time": previous["queue_create_time"],
    }
    checkpoint = Path(previous["checkpoint"])
    plan = {
        "status": "WAITING_UNFILTERED_QUEUE",
        "upstream": str(upstream),
        "upstream_identity": identity,
        "checkpoint": str(checkpoint),
        "jobs": build_jobs(output, checkpoint),
        "output": str(output),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "queue.json"
        if manifest.exists():
            if not args.resume:
                raise FileExistsError(
                    "Queue already exists; use --resume to retain attempts"
                )
            plan = read_json(manifest)
            if plan["upstream_identity"] != identity or plan["upstream"] != str(
                upstream
            ):
                raise ValueError("Cannot resume against a different upstream queue")
            if plan["status"] == "COMPLETE":
                print("Already complete; no jobs relaunched", flush=True)
                return
            for job in plan["jobs"]:
                if (
                    job["status"] == "FAILED"
                    and len(job["attempts"]) < args.max_attempts
                ):
                    job["status"] = "QUEUED"
        plan.update(
            queue_pid=os.getpid(),
            queue_create_time=psutil.Process().create_time(),
            max_attempts=args.max_attempts,
            max_parallel=args.max_parallel,
        )
        atomic_json(manifest, plan)
        (output / "queue.pid").write_text(str(os.getpid()) + "\n")
        print(f"Armed; waiting for unfiltered queue {identity} to finish", flush=True)
        while True:
            gate = upstream_gate(upstream, identity)
            if gate == "READY":
                break
            plan.update(status=gate, updated_at=time.time())
            atomic_json(manifest, plan)
            if gate.startswith("BLOCKED"):
                return
            time.sleep(args.poll_seconds)
        jobs = plan["jobs"]
        children = {}
        streams = {}
        pool_path = output / "evaluation_pool.json"
        try:
            while True:
                # Recover an interrupted queue without duplicating a still-live job.
                for job in jobs:
                    if job["status"] not in ("STARTING", "RUNNING"):
                        continue
                    attempt = job["attempts"][-1]
                    child = children.get(job["id"])
                    code = child.poll() if child is not None else None
                    if attempt_alive(attempt):
                        job["status"] = "RUNNING"
                        continue
                    stream = streams.pop(job["id"], None)
                    if stream is not None:
                        stream.close()
                    try:
                        if code is not None and code != 0:
                            raise RuntimeError(f"Subprocess exited {code}")
                        data = validate_result(job, Path(attempt["output"]), pool_path)
                        atomic_json(Path(job["output"]), data)
                        job["status"] = "COMPLETE"
                        attempt["status"] = "COMPLETE"
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        attempt.update(status="FAILED", error=str(error))
                        job["status"] = (
                            "QUEUED"
                            if len(job["attempts"]) < args.max_attempts
                            else "FAILED"
                        )
                    attempt.update(returncode=code, finished_at=time.time())
                    print(f"{job['id']}: {attempt['status']}", flush=True)
                if jobs[0]["status"] == "FAILED":
                    plan["status"] = "BLOCKED_POOL_BUILD_FAILED"
                    break
                pool_ready = jobs[0]["status"] == "COMPLETE"
                eligible = jobs[1:] if pool_ready else jobs[:1]
                active = [
                    job for job in jobs if job["status"] in ("STARTING", "RUNNING")
                ]
                if pool_ready and all(
                    job["status"] in ("COMPLETE", "FAILED") for job in jobs[1:]
                ):
                    plan["status"] = (
                        "COMPLETE"
                        if all(job["status"] == "COMPLETE" for job in jobs)
                        else "PARTIAL_FAILURE"
                    )
                    break
                plan["status"] = (
                    "EVALUATING_FIXED_POOL"
                    if pool_ready
                    else "BUILDING_POOL"
                    if active
                    else "WAITING_GPU_FOR_POOL"
                )
                try:
                    available = idle_pool_gpus(args.gpus.split(","))
                    plan.pop("gpu_query_error", None)
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    available = []
                    plan["gpu_query_error"] = str(error)
                busy = {job["attempts"][-1]["gpu"] for job in active}
                limit = args.max_parallel if pool_ready else 1
                for gpu in available:
                    if len(active) >= limit:
                        break
                    if gpu in busy:
                        continue
                    job = next(
                        (job for job in eligible if job["status"] == "QUEUED"), None
                    )
                    if job is None:
                        break
                    attempt = make_attempt(job, output, gpu)
                    job["attempts"].append(attempt)
                    job["status"] = "STARTING"
                    # Persist exact argv before spawn; resume can find an orphan by argv.
                    atomic_json(manifest, plan)
                    stream = Path(attempt["log"]).open("x")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=gpu,
                        USE_TF="0",
                        OPD_RUN_DIR=str(output),
                    )
                    child = subprocess.Popen(
                        attempt["command"],
                        cwd=REPO,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    attempt.update(
                        pid=child.pid,
                        create_time=psutil.Process(child.pid).create_time(),
                        status="RUNNING",
                    )
                    job["status"] = "RUNNING"
                    children[job["id"]] = child
                    streams[job["id"]] = stream
                    active.append(job)
                    print(
                        f"Started {job['id']} attempt{len(job['attempts'])}, GPU{gpu}, PID{child.pid}",
                        flush=True,
                    )
                    atomic_json(manifest, plan)
                plan["updated_at"] = time.time()
                atomic_json(manifest, plan)
                summarize(output, jobs)
                time.sleep(args.poll_seconds)
        except BaseException:
            plan.update(status="INTERRUPTED", updated_at=time.time())
            atomic_json(manifest, plan)
            # Never terminate a peer or discard a running evaluation on queue exit.
            raise
        plan["finished_at"] = time.time()
        atomic_json(manifest, plan)
        summarize(output, jobs)
        print(plan["status"], flush=True)


if __name__ == "__main__":
    main()
