# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Isolated RoboTwin pilot launcher with owner-scoped Ray cleanup."""

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from toolkits.robocasa_mopd.owned_ray import (
    OWNER_KEY,
    require_free_ray,
    stop_owned,
    wait_for_gpus,
)

REPO = Path(__file__).resolve().parents[2]


def run_command(command, log, env):
    """Run one subprocess and terminate only its process group if interrupted."""
    with log.open("a") as stream:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main():
    """Record exact configuration/code and launch an isolated GPU pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    from toolkits.robotwin_mopd.resources import cpu_budget

    num_cpus = cpu_budget(args.num_cpus, os.environ, len(os.sched_getaffinity(0)))
    if args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    if args.num_gpus != 8 and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Set CUDA_VISIBLE_DEVICES explicitly for a GPU subset")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", args.run_id):
        parser.error("Invalid run ID")
    runtime = Path(os.environ["ROBOTWIN_RUNTIME_ROOT"]).resolve()
    if Path(sys.prefix).resolve() != runtime / "venv":
        parser.error("Use the isolated RoboTwin runtime")
    output = args.output_root.resolve() / args.run_id
    env = dict(os.environ)
    env.update(
        OPD_ROBOTWIN_PROFILE="pi05-four-task",
        OPD_RUN_DIR=str(output),
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        PYTHONNOUSERSITE="1",
        RLINF_NODE_RANK="0",
        RAY_ADDRESS="127.0.0.1:6379",
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
    )
    env.setdefault("USE_TF", "0")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("RAY_DASHBOARD_STARTUP_TIMEOUT_S", "300")
    env.setdefault("RAY_DASHBOARD_SUBPROCESS_MODULE_WAIT_READY_TIMEOUT", "180")
    env.pop(OWNER_KEY, None)
    os.environ.update(env)
    placement_overrides = []
    if args.num_gpus != 8:
        selected = env["CUDA_VISIBLE_DEVICES"].split(",")
        if len(selected) != args.num_gpus or not all(x.isdigit() for x in selected):
            parser.error("GPU subset requires exactly --num-gpus numeric physical IDs")
        if len(set(selected)) != len(selected):
            parser.error("GPU IDs must be distinct")
        # RLinf discovers physical devices independently of Ray/CUDA's mask.
        # Replace 'all' explicitly so a subset never schedules other GPUs.
        ranks = ",".join(selected)
        placement_overrides = [
            "~cluster.component_placement",
            f'+cluster.component_placement={{actor:"{ranks}",env:"{ranks}",rollout:"{ranks}"}}',
        ]
        if any("cluster.component_placement" in item for item in args.overrides):
            parser.error("Do not override component placement with --num-gpus subset")
    overrides = [
        f"runner.logger.experiment_name={args.run_id}",
        *placement_overrides,
        *args.overrides,
    ]
    if args.smoke:
        overrides += [
            "runner.max_steps=2",
            "runner.save_interval=2",
            "runner.logger.logger_backends=[tensorboard]",
            "env.train.total_num_envs=32",
        ]
        if any(
            item.lstrip("+").startswith("runner.save_steps=") for item in args.overrides
        ):
            overrides += ["++runner.save_steps=[2]", "runner.save_interval=-1"]
    with initialize_config_dir(
        version_base="1.1", config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05", overrides=overrides)
    OmegaConf.resolve(cfg)
    from rlinf.config import validate_flow_refresh_cfg, validate_rollout_allocation_cfg

    validate_flow_refresh_cfg(cfg)
    validate_rollout_allocation_cfg(cfg)
    command = [
        sys.executable,
        "-u",
        str(REPO / "examples/embodiment/train_embodied_agent.py"),
        "--config-name",
        "robotwin4_mopd_pi05",
        *overrides,
    ]
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return
    from rlinf.envs.robotwin.runtime_profile import validate_robotwin4_runtime

    runtime_profile = validate_robotwin4_runtime()
    if output.exists():
        raise FileExistsError(output)
    with open(f"/tmp/opd-single-job-{os.getuid()}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_free_ray()
        wait_for_gpus(args.num_gpus, env)
        output.mkdir(parents=True)
        (output / "runtime_profile.json").write_text(
            json.dumps(runtime_profile, indent=2) + "\n"
        )
        OmegaConf.save(cfg, output / "resolved_config.yaml")
        (output / "code.diff").write_bytes(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
        )
        # Also snapshot new untracked implementation files for exact pilot provenance.
        import tarfile

        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            paths = subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=REPO,
                text=True,
            ).splitlines()
            for path in paths:
                if path.startswith(
                    (
                        "rlinf/",
                        "toolkits/robotwin",
                        "examples/embodiment/",
                        "tests/unit_tests/",
                    )
                ):
                    archive.add(REPO / path, arcname=path)
        owner = f"robotwin-pilot-{uuid.uuid4().hex}"
        (output / "launch.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "host": socket.gethostname(),
                    "owner": owner,
                    "runtime": str(runtime),
                    "num_gpus": args.num_gpus,
                    "num_cpus": num_cpus,
                    "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                    "numa_policy": env.get("ROBOTWIN_NUMA_POLICY", "default"),
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                    ).strip(),
                },
                indent=2,
            )
        )
        ray_command = [
            str(runtime / "venv/bin/ray"),
            "start",
            "--head",
            "--port=6379",
            f"--num-cpus={num_cpus}",
            f"--num-gpus={args.num_gpus}",
            "--include-dashboard=true",
            "--dashboard-host=127.0.0.1",
            "--dashboard-port=8265",
            f"--temp-dir={tempfile.mkdtemp(prefix='robotwin-ray-', dir='/tmp')}",
        ]

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        (output / "status").write_text("RUNNING\n")
        try:
            run_command(ray_command, output / "ray.log", {**env, OWNER_KEY: owner})
            run_command(command, output / "driver.log", env)
            (output / "status").write_text("COMPLETE\n")
        except BaseException:
            (output / "status").write_text("FAILED\n")
            raise
        finally:
            stop_owned(owner)


if __name__ == "__main__":
    main()
