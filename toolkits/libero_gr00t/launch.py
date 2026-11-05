# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch bounded suite-parallel evaluation or synchronized four-teacher MOPD."""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
NAMES = ["spatial", "object", "goal", "long"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--spatial-teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["eval", "mopd"], required=True)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=SUITES)
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--reuse", type=int, default=2)
    parser.add_argument(
        "--student-head-precision", choices=["bf16", "fp32"], default="bf16"
    )
    parser.add_argument("--rollout-chunks", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "mopd" and args.suites != SUITES:
        raise ValueError("Joint probe requires all four suites in canonical order")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {
        int(row.split(",")[0]): int(row.split(",")[1])
        for row in output.strip().splitlines()
    }
    gpus = args.gpus or sorted(free, key=free.get, reverse=True)[: len(args.suites)]
    if len(gpus) != len(args.suites) or len(set(gpus)) != len(gpus):
        raise ValueError("Need one distinct GPU per suite")
    if any(free[gpu] < 30 * 1024 for gpu in gpus):
        raise RuntimeError(f"Each selected GPU must have at least 30GiB free: {free}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    jobs = []
    manifest = []
    try:
        for rank, (gpu, suite) in enumerate(zip(gpus, args.suites)):
            name = NAMES[SUITES.index(suite)]
            teacher = args.spatial_teacher if name == "spatial" else args.models / name
            command = [
                "nice",
                "-n",
                "10",
                str(args.python.resolve()),
                "-u",
                "toolkits/libero_gr00t/probe.py",
                "--mode",
                args.mode,
                "--student-head-precision",
                args.student_head_precision,
                "--student",
                str((args.models / "student").resolve()),
                "--teacher",
                str(teacher.resolve()),
                "--suite",
                suite,
                "--output",
                str((args.output / f"{name}.jsonl").resolve()),
                "--rounds",
                str(args.rounds),
                "--reuse",
                str(args.reuse),
                "--rollout-chunks",
                str(args.rollout_chunks),
                "--tasks",
                *map(str, args.tasks),
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            if args.mode == "mopd":
                command += ["--coordinate-mode", "student_coordinate_sde"]
                env.update(
                    RANK=str(rank),
                    WORLD_SIZE=str(len(gpus)),
                    MASTER_ADDR="127.0.0.1",
                    MASTER_PORT=str(port),
                )
            else:
                for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
                    env.pop(key, None)
            stream = (args.output / f"{name}.log").open("w")
            process = subprocess.Popen(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            jobs.append((process, stream))
            manifest.append(
                {
                    "suite": suite,
                    "gpu": gpu,
                    "free_mib": free[gpu],
                    "pid": process.pid,
                    "command": command,
                }
            )
        (args.output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
        while any(p.poll() is None for p, _ in jobs):
            if any(p.poll() not in (None, 0) for p, _ in jobs):
                raise RuntimeError("A probe rank failed; see per-suite logs")
            time.sleep(2)
        if any(p.returncode != 0 for p, _ in jobs):
            raise RuntimeError("Probe failed")
    finally:
        for process, stream in jobs:
            if process.poll() is None:
                process.terminate()
        for process, stream in jobs:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


if __name__ == "__main__":
    main()
