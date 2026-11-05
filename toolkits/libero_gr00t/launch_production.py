# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an isolated production smoke; execute only with explicit --run.

GPU arguments are physical IDs, not indices into CUDA_VISIBLE_DEVICES. The
launcher never joins an existing Ray cluster or invokes global Ray cleanup.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_flow_refresh_cfg, validate_rollout_allocation_cfg
from rlinf.scheduler.cluster.config import ClusterConfig
from toolkits.robocasa_mopd.owned_ray import OWNER_KEY, check_gpus, stop_owned

REPO = Path(__file__).resolve().parents[2]
CONFIG_NAME = "libero_four_suite_mopd_refresh_r2_gr00t_pilot"
PORT_NAMES = (
    "port",
    "dashboard-port",
    "dashboard-agent-listen-port",
    "dashboard-agent-grpc-port",
    "runtime-env-agent-port",
    "ray-client-server-port",
    "metrics-export-port",
    "node-manager-port",
    "object-manager-port",
)


def choose_ports() -> dict[str, int]:
    """Choose distinct available local ports; a bind race fails rather than attaches."""
    sockets = []
    try:
        for _ in PORT_NAMES:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", 0))
        return dict(zip(PORT_NAMES, [s.getsockname()[1] for s in sockets], strict=True))
    finally:
        for sock in sockets:
            sock.close()


def build_config(args, checkpoint_paths: dict[str, Path]):
    """Compose a complete launch recipe without starting Cluster or loading models."""
    gpus = sorted(args.gpus)
    if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
        raise ValueError("Specify four distinct nonnegative physical GPU IDs")
    if args.updates < 2 or args.updates % 2:
        raise ValueError("Refresh-R2 smoke updates must be positive multiples of two")
    if args.rollout_steps < 8 or args.rollout_steps > 520 or args.rollout_steps % 8:
        raise ValueError("rollout-steps must be a multiple of eight in [8, 520]")
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name=CONFIG_NAME)
    with open_dict(cfg):
        cfg.actor.model.model_path = str(checkpoint_paths["student"])
        for suite, path in checkpoint_paths.items():
            if suite != "student":
                cfg.rollout.expert_models[suite].model_path = str(path)
        cfg.runner.logger.log_path = str(args.output)
        cfg.runner.logger.experiment_name = "libero_production_smoke"
        cfg.runner.max_steps = args.updates
        cfg.runner.save_interval = args.updates
        cfg.runner.val_check_interval = -1
        cfg.runner.overlap_env_bootstrap = False
        cfg.cluster.component_placement = {
            "actor,env,rollout": {
                "node_group": "libero_smoke",
                "placement": ",".join(map(str, gpus)),
            }
        }
        cfg.cluster.node_groups = [
            {
                "label": "libero_smoke",
                "node_ranks": "0",
                "env_configs": [
                    {
                        "node_ranks": "0",
                        "python_interpreter_path": str(args.python),
                    }
                ],
            }
        ]
        cfg.env.train.max_episode_steps = args.rollout_steps
        cfg.env.train.max_steps_per_rollout_epoch = args.rollout_steps
        cfg.env.train.task_suite_horizons = {
            suite: min(horizon, args.rollout_steps)
            for suite, horizon in cfg.env.train.task_suite_horizons.items()
        }
    OmegaConf.resolve(cfg)
    validate_flow_refresh_cfg(cfg)
    validate_rollout_allocation_cfg(cfg)
    ClusterConfig.from_dict_cfg(cfg.cluster)
    return cfg


def read_runtime_environment(python: Path) -> dict[str, str]:
    """Read only required runtime variables from the wrapper's env.sh setup."""
    keys = [
        "PYTHONPATH",
        "LIBERO_CONFIG_PATH",
        "MUJOCO_GL",
        "NUMBA_CACHE_DIR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
        "USE_TF",
        "JAX_PLATFORMS",
        "TOKENIZERS_PARALLELISM",
    ]
    command = (
        "import os,json; "
        f"print(json.dumps({{k:os.environ[k] for k in {keys!r} if k in os.environ}}))"
    )
    result = subprocess.check_output([str(python), "-c", command], text=True)
    values = json.loads(result)
    if not values.get("LIBERO_CONFIG_PATH") or not values.get("PYTHONPATH"):
        raise ValueError("Runtime wrapper must source LIBERO env.sh and PYTHONPATH")
    return values


def run_logged(command: list[str], env: dict[str, str], log: Path) -> None:
    """Terminate only this subprocess group on interruption."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--spatial-teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs=4, type=int, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--cpus", type=int, default=16)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Default; write plan only")
    args = parser.parse_args()
    args.python = args.python.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new output directory for each dry-run or run")
    if args.cpus < 1:
        raise ValueError("cpus must be positive")
    runtime_env_file = args.python.parent.parent / "env.sh"
    if not runtime_env_file.is_file():
        raise FileNotFoundError(
            f"Expected runtime env.sh beside bin/: {runtime_env_file}"
        )
    checkpoint_paths = {
        "student": args.student.resolve(strict=True),
        "libero_spatial": args.spatial_teacher.resolve(strict=True),
        "libero_object": (args.models / "object").resolve(strict=True),
        "libero_goal": (args.models / "goal").resolve(strict=True),
        "libero_10": (args.models / "long").resolve(strict=True),
    }
    for path in checkpoint_paths.values():
        for filename in ["config.json", "experiment_cfg/metadata.json"]:
            if not (path / filename).is_file():
                raise FileNotFoundError(path / filename)
    runtime_vars = read_runtime_environment(args.python)
    cfg = build_config(args, checkpoint_paths)
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml")
    owner = "opd-libero-" + uuid.uuid4().hex
    ray_root = Path(tempfile.mkdtemp(prefix="opdl-", dir="/tmp"))
    # Keep Ray's standard <RAY_TMPDIR>/ray address-file layout even for children
    # which reconnect without an explicit address; it remains private to this run.
    ray_dir = ray_root / "ray"
    ray_dir.mkdir()
    ports = choose_ports()
    env = {**os.environ, **runtime_vars}
    env.pop(OWNER_KEY, None)
    env.update(
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        RLINF_NODE_RANK="0",
        RAY_ADDRESS=f"127.0.0.1:{ports['port']}",
        RAY_TMPDIR=str(ray_root),
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        CUDA_VISIBLE_DEVICES=",".join(map(str, sorted(args.gpus))),
        GPU_FREE_MB=str(30 * 1024),
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
        str(REPO / "examples/embodiment/train_embodied_agent.py"),
        "--config-path",
        str(args.output),
        "--config-name",
        "resolved_config",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ]
    manifest = {
        "mode": "run" if args.run else "dry_run",
        "owner": owner,
        "physical_gpus": sorted(args.gpus),
        "ray_address": env["RAY_ADDRESS"],
        "ray_temp_dir": str(ray_dir),
        "runtime_env_file": str(runtime_env_file),
        "ray_command": ray_command,
        "driver_command": driver_command,
        "checkpoints": {k: str(v) for k, v in checkpoint_paths.items()},
        "optimizer_updates": args.updates,
        "save_interval": args.updates,
        "rollout_rounds": args.updates // 2,
        "rollout_steps": args.rollout_steps,
        "incomplete_episodes": args.rollout_steps < 520,
        "performance_evidence": False,
        "differences_from_standalone": [
            "Production Ray/FSDP, T16 rather than T4, all suites on each actor rank",
            "All official training initial states; no held-out evaluation in this smoke",
            "First update uses rollout flow; subsequent update refreshes flow",
            "Four CPU-cached teachers per actor, streamed to GPU one at a time",
            "Production checkpoint at final update; no standalone head-only save",
        ],
        "gpu_free_preflight": "30 GiB per selected card, checked only with --run",
    }
    manifest_path = args.output / "launch.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status = args.output / "status"
    if not args.run:
        status.write_text("DRY_RUN\n")
        print(f"Dry-run plan written to {args.output}; no Ray/GPU job started.")
        return
    # Check immediately before launch; this is not a memory reservation.
    try:
        (args.output / "gpu_preflight.txt").write_text(check_gpus(4, env))
    except BaseException:
        status.write_text("PREFLIGHT_FAILED\n")
        raise
    ray_env = {**env, OWNER_KEY: owner}
    ray_env.pop("RAY_ADDRESS", None)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    status.write_text("RUNNING\n")
    try:
        run_logged(ray_command, ray_env, args.output / "ray.log")
        # Require the address file created by this private head, never auto discovery.
        private_address = (ray_dir / "ray_current_cluster").read_text().strip()
        if int(private_address.rsplit(":", 1)[1]) != ports["port"]:
            raise RuntimeError("Private Ray head address differs from the launch plan")
        env["RAY_ADDRESS"] = private_address
        manifest["ray_address"] = private_address
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        run_logged(driver_command, env, args.output / "driver.log")
        status.write_text("OK\n")
    except BaseException:
        status.write_text("FAILED\n")
        raise
    finally:
        stop_owned(owner)


if __name__ == "__main__":
    main()
