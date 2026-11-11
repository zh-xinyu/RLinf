# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Scope a private Ray cluster around the verified teacher training-rollout path."""

import fcntl
import json
import os
import signal
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path

from toolkits.robocasa_mopd.owned_ray import OWNER_KEY, stop_owned, wait_for_gpus
from toolkits.robotwin_mopd.new4 import contracts as c


def run(machine: dict, env: dict) -> None:
    visible = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(visible) != 4
        or len(set(visible)) != 4
        or not all(value.isdecimal() for value in visible)
    ):
        raise ValueError("Teacher smoke requires four explicit physical GPU IDs")
    output = Path(machine["output_root"]) / "smoke/teachers"
    if output.exists():
        raise FileExistsError(output)
    wait_for_gpus(4, env)
    port, dashboard = machine["ray_port"], machine["dashboard_port"]
    python = str(Path(machine["runtime_root"]) / "venv/bin/python")
    owner = "robotwin-new4-teacher-" + uuid.uuid4().hex
    job = {"method": "uniform_r2", "seed": 2}
    command = [
        python,
        "-u",
        "-m",
        "toolkits.robotwin_mopd.teacher_train_rollout",
        "--output",
        str(output),
        "--physical-gpus",
        ",".join(visible),
        "--trajectories",
        "32",
        "--rounds",
        "1",
        "--seed",
        "50000000",
        "--teacher-storage",
        "resident",
    ]
    for value in c.overrides(machine, job):
        command += ["--override", value]
    command += ["--override", "runner.logger.logger_backends=[]"]
    log_path = output.parent / "teacher_smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"/tmp/opd-robotwin-{os.getuid()}-{port}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in (port, dashboard):
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", item)) == 0:
                    raise RuntimeError(f"Requested Ray port {item} is occupied")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Teacher smoke received signal {signum}")

        handlers = {
            item: signal.signal(item, interrupted)
            for item in (signal.SIGTERM, signal.SIGHUP)
        }
        try:
            with log_path.open("x") as log:
                subprocess.run(
                    [
                        str(Path(machine["runtime_root"]) / "venv/bin/ray"),
                        "start",
                        "--head",
                        f"--port={port}",
                        "--num-gpus=4",
                        f"--num-cpus={machine.get('num_cpus', 32)}",
                        "--include-dashboard=true",
                        "--dashboard-host=127.0.0.1",
                        f"--dashboard-port={dashboard}",
                        "--temp-dir=" + tempfile.mkdtemp(prefix="new4-teacher-ray-"),
                    ],
                    env={**env, OWNER_KEY: owner},
                    cwd=c.REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=c.REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    if process.wait():
                        raise RuntimeError(
                            "Teacher deployment smoke failed; preserve logs"
                        )
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            result = json.loads((output / "results.json").read_text())
            if set(result["tasks"]) != set(c.TASKS) or any(
                row["episodes"] != 8 for row in result["tasks"].values()
            ):
                raise ValueError(
                    "Teacher smoke did not produce eight real episodes/task"
                )
            if not (output / "COMPLETE").is_file():
                raise ValueError("Teacher smoke did not finish")
            if any(row["successes"] < 1 for row in result["tasks"].values()):
                raise ValueError(
                    "A teacher achieved0/8; inspect deployment before formal training"
                )
        finally:
            try:
                stop_owned(owner)
            finally:
                for item, handler in handlers.items():
                    signal.signal(item, handler)
