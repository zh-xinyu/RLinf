# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch one pilot suite rank per GPU without changing shared Ray state."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from omegaconf import OmegaConf

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
NAMES = ["spatial", "object", "goal", "long"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--spatial-teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs=4, type=int)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if cfg["world_size"] != 4:
        raise ValueError("Four-suite pilot requires four ranks")
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {
        int(row.split(",")[0]): int(row.split(",")[1])
        for row in result.strip().splitlines()
    }
    gpus = args.gpus or sorted(free, key=free.get, reverse=True)[:4]
    if len(set(gpus)) != 4 or any(free[g] < 30 * 1024 for g in gpus):
        raise RuntimeError(f"Need four distinct GPUs with >=30GiB free: {free}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), args.output / "recipe.yaml")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"])
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    (args.output / "code_provenance.json").write_text(
        json.dumps(
            {
                "git_commit": revision,
                "working_tree_status": status,
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "clean_worktree": not bool(status.strip()),
            },
            indent=2,
        )
        + "\n"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    jobs = []
    manifest = []
    try:
        for rank, (gpu, suite, name) in enumerate(zip(gpus, SUITES, NAMES)):
            teacher = args.spatial_teacher if name == "spatial" else args.models / name
            command = [
                "nice",
                "-n",
                "10",
                str(args.python.resolve()),
                "-u",
                "toolkits/libero_gr00t/pilot.py",
                "--config",
                str((args.output / "recipe.yaml").resolve()),
                "--student",
                str(args.student.resolve()),
                "--teacher",
                str(teacher.resolve()),
                "--suite",
                suite,
                "--output",
                str((args.output / f"{name}.jsonl").resolve()),
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                RANK=str(rank),
                WORLD_SIZE="4",
                LOCAL_RANK="0",
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
            )
            stream = (args.output / f"{name}.log").open("w")
            process = subprocess.Popen(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            jobs.append((process, stream))
            manifest.append(
                {
                    "rank": rank,
                    "gpu": gpu,
                    "suite": suite,
                    "free_mib": free[gpu],
                    "pid": process.pid,
                    "command": command,
                }
            )
        (args.output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
        while any(p.poll() is None for p, _ in jobs):
            if any(p.poll() not in (None, 0) for p, _ in jobs):
                raise RuntimeError("Pilot rank failed; see logs")
            time.sleep(2)
        if any(p.returncode != 0 for p, _ in jobs):
            raise RuntimeError("Pilot rank failed")
    finally:
        for p, _ in jobs:
            if p.poll() is None:
                p.terminate()
        for p, stream in jobs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            stream.close()


if __name__ == "__main__":
    main()
