# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Plan/run a private four-rank teacher probe through env.train, without an actor."""

import argparse
import json
import math
import os
import signal
import tempfile
import uuid
from pathlib import Path

from omegaconf import OmegaConf

from toolkits.libero_gr00t.launch_production import (
    REPO,
    choose_ports,
    read_runtime_environment,
    run_logged,
)
from toolkits.libero_gr00t.teacher_reference import digest
from toolkits.libero_gr00t.teacher_training_probe import (
    configure_probe,
    select_reference,
)
from toolkits.robocasa_mopd.owned_ray import OWNER_KEY, check_gpus, stop_owned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Resolved native training recipe with teacher paths",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Completed teacher_reference.py capture directory",
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--trials", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--capture-chunks", type=int, nargs="*", default=[])
    parser.add_argument("--memory-gib", type=float, default=10)
    parser.add_argument("--minimum-free-gib", type=float, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument(
        "--run", action="store_true", help="Default writes a dry-run plan only"
    )
    args = parser.parse_args()
    for name in ("config", "reference", "python"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; no retry/overwrite")
    if (
        not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
        or not math.isfinite(args.minimum_free_gib)
        or args.minimum_free_gib < args.memory_gib + 2
        or args.cpus < 1
    ):
        raise ValueError(
            "Require positive timeout/cpus and at least allocator-cap +2GiB free"
        )
    source = OmegaConf.load(args.config)
    reference = select_reference(args.reference, source, args.tasks, args.trials)
    runtime = read_runtime_environment(args.python)
    cfg = configure_probe(
        source,
        args.output,
        args.python,
        args.gpus,
        reference,
        args.tasks,
        args.trials,
        args.memory_gib,
        args.capture_chunks,
        runtime,
    )
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml", resolve=True)
    (args.output / "selected_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    owner, ports = "opd-libero-teacher-" + uuid.uuid4().hex, choose_ports()
    ray_root = Path(tempfile.mkdtemp(prefix="opdlt-", dir="/tmp"))
    ray_dir = ray_root / "ray"
    ray_dir.mkdir()
    env = {**os.environ, **runtime}
    env.pop(OWNER_KEY, None)
    env.update(
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        RLINF_NODE_RANK="0",
        RAY_ADDRESS=f"127.0.0.1:{ports['port']}",
        RAY_TMPDIR=str(ray_root),
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        CUDA_VISIBLE_DEVICES=",".join(map(str, sorted(args.gpus))),
        GPU_FREE_MB=str(args.minimum_free_gib * 1024),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        RAY_USAGE_STATS_ENABLED="0",
    )
    ray_command = [
        str(args.python),
        "-m",
        "ray.scripts.scripts",
        "start",
        "--head",
        "--node-ip-address=127.0.0.1",
        f"--num-cpus={args.cpus}",
        "--num-gpus=4",
        "--object-store-memory=2147483648",
        "--include-dashboard=true",
        "--dashboard-host=127.0.0.1",
        "--disable-usage-stats",
        f"--temp-dir={ray_dir}",
        *[f"--{name}={port}" for name, port in ports.items()],
    ]
    driver_command = [
        str(args.python),
        "-m",
        "toolkits.libero_gr00t.teacher_training_probe",
        "--config",
        str(args.output / "resolved_config.yaml"),
    ]
    paths = [
        Path(__file__),
        REPO / "toolkits/libero_gr00t/teacher_training_probe.py",
        REPO / "toolkits/libero_gr00t/teacher_training_workers.py",
        REPO / "rlinf/workers/env/env_worker.py",
        REPO / "rlinf/workers/rollout/hf/huggingface_worker.py",
        REPO / "rlinf/envs/libero/libero_env.py",
        REPO / "rlinf/envs/action_utils.py",
    ]
    manifest = {
        "mode": "run" if args.run else "dry_run",
        "owner": owner,
        "physical_gpus": sorted(args.gpus),
        "ray_address": env["RAY_ADDRESS"],
        "ray_temp_dir": str(ray_dir),
        "ray_command": ray_command,
        "driver_command": driver_command,
        "runtime_env_vars": runtime,
        "source_sha256": {str(path.relative_to(REPO)): digest(path) for path in paths},
        "source_config": str(args.config),
        "source_config_sha256": digest(args.config),
        "reference": str(args.reference),
        "reference_manifest_sha256": digest(args.reference / "manifest.json"),
        "tasks": args.tasks,
        "trials": args.trials,
        "expected_episodes": 4 * len(args.tasks) * len(args.trials),
        "actor_launched": False,
        "optimizer_updates": 0,
        "memory_gib": args.memory_gib,
        "minimum_free_gib": args.minimum_free_gib,
        "timeout_seconds": args.timeout_seconds,
        "scope": "Actual EnvWorker.interact(env.train) and HF.generate. One native specialist/environment per rank; no mixed-suite actor-buffer validation.",
    }
    (args.output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    status = args.output / "status"
    if not args.run:
        status.write_text("DRY_RUN\n")
        return
    try:
        (args.output / "gpu_preflight.txt").write_text(check_gpus(4, env))
    except BaseException:
        status.write_text("PREFLIGHT_FAILED\n")
        raise
    ray_env = {**env, OWNER_KEY: owner}
    ray_env.pop("RAY_ADDRESS", None)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Teacher train probe interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    status.write_text("RUNNING\n")
    try:
        run_logged(ray_command, ray_env, args.output / "ray.log", timeout_seconds=120)
        address = (ray_dir / "ray_current_cluster").read_text().strip()
        if int(address.rsplit(":", 1)[1]) != ports["port"]:
            raise RuntimeError("Private Ray address differs from the planned port")
        env["RAY_ADDRESS"] = address
        run_logged(
            driver_command,
            env,
            args.output / "driver.log",
            timeout_seconds=args.timeout_seconds,
        )
        report = json.loads((args.output / "summary.json").read_text())
        if (
            report.get("complete") is not True
            or report.get("episodes") != manifest["expected_episodes"]
        ):
            raise ValueError("Incomplete actual-training-path episode coverage")
        status.write_text("OK\n")
    except BaseException:
        status.write_text("FAILED\n")
        raise
    finally:
        stop_owned(owner)


if __name__ == "__main__":
    main()
