# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Owner-scoped Ray cleanup for serial runs; never delete directories or stop peers.

Only ``ray start`` and its descendants receive OPD_RAY_PROCESS_OWNER. The queue,
driver launcher and unrelated jobs do not, even when they share a Unix account.
"""

import argparse
import os
import re
import socket
import subprocess

import psutil

OWNER_KEY = "OPD_RAY_PROCESS_OWNER"


def check_gpus(count: int, env: dict[str, str]) -> str:
    """Check selected GPUs using their real capacities, without reserving them.

    A positive GPU_FREE_MB requires that much free memory per GPU. Otherwise,
    refuse GPUs with more than 10% memory in use. Count/visibility mismatches
    are configuration errors, not conditions to retry by waiting.
    """
    if count < 1:
        raise ValueError("GPU count must be positive")
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        if not visible or not all(
            re.fullmatch(r"[0-9]+|GPU-[a-fA-F0-9-]+", value)
            for value in visible.split(",")
        ):
            raise ValueError("CUDA_VISIBLE_DEVICES must contain GPU indices or UUIDs")
        query.append(f"--id={visible}")
    output = subprocess.check_output(query, text=True, env=env)
    rows = [line.split(",") for line in output.splitlines() if line.strip()]
    if len(rows) != count:
        raise ValueError(
            f"Requested {count} GPUs, selected {len(rows)}; set CUDA_VISIBLE_DEVICES explicitly"
        )
    min_free = float(env.get("GPU_FREE_MB", "0"))
    if any(
        float(total) - float(used) < min_free
        if min_free > 0
        else float(used) > 0.1 * float(total)
        for _, used, total in rows
    ):
        raise RuntimeError("Selected GPUs are busy; refusing to start")
    return output


def owned_processes(owner: str) -> list[psutil.Process]:
    """Select live processes with both our UID and the exact launch-time marker."""
    selected = []
    for process in psutil.process_iter():
        try:
            if (
                process.pid != os.getpid()
                and process.uids().real == os.getuid()
                and process.status() != psutil.STATUS_ZOMBIE
                and process.environ().get(OWNER_KEY) == owner
            ):
                selected.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return selected


def stop_owned(owner: str) -> None:
    """Terminate only matching processes, rechecking for late children before exit."""
    for _ in range(3):
        processes = owned_processes(owner)
        if not processes:
            return
        print(f"Stopping {len(processes)} Ray processes owned by {owner}", flush=True)
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=10)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
    if owned_processes(owner):
        raise RuntimeError(f"Owned Ray processes survived cleanup: {owner}")


def require_free_ray() -> None:
    """Refuse other local Ray clusters or occupied default ports without touching them."""
    for process in psutil.process_iter(["pid", "name", "status"]):
        if (
            process.info["name"] in {"gcs_server", "raylet"}
            and process.info["status"] != psutil.STATUS_ZOMBIE
        ):
            raise RuntimeError(
                f"Existing Ray process; refusing to stop it: {process.pid}"
            )
    for port in (6379, 8265):
        with socket.socket() as connection:
            connection.settimeout(1)
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"Port {port} is occupied; refusing to stop its owner"
                )


def main() -> None:
    """Run a conservative preflight or owner-scoped cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "stop", "gpu-check"))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", args.owner):
        parser.error("owner must be a nonempty safe run identifier")
    if args.action == "gpu-check":
        try:
            print(check_gpus(args.gpus, dict(os.environ)), end="")
        except ValueError as exc:
            parser.error(str(exc))
        except RuntimeError as exc:
            print(str(exc), flush=True)
            raise SystemExit(3) from exc
    elif args.action == "stop":
        stop_owned(args.owner)
    else:
        require_free_ray()


if __name__ == "__main__":
    main()
