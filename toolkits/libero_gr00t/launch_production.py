# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an isolated production smoke; execute only with explicit --run.

GPU arguments are physical IDs, not indices into CUDA_VISIBLE_DEVICES. The
launcher never joins an existing Ray cluster or invokes global Ray cleanup.
"""

import argparse
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_flow_refresh_cfg, validate_rollout_allocation_cfg
from rlinf.scheduler.cluster.config import ClusterConfig
from toolkits.libero_gr00t.padding_profile import (
    PADDING_PROFILES,
    apply_student_padding_profile,
    student_input_contract,
)
from toolkits.robocasa_mopd.owned_ray import OWNER_KEY, check_gpus, stop_owned

REPO = Path(__file__).resolve().parents[2]
CONFIG_NAME = "libero_four_suite_mopd_refresh_r2_gr00t_pilot"
ALLOCATION_SMOKE_OVERRIDES = {
    "warmup_rounds": 2,
    "window_rounds": 2,
    "interval_rounds": 1,
    "min_task_episodes": 1,
    "rho": 1.0,
}
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


def inspect_resume_dcp(path: Path, expected_updates: int) -> dict:
    """Check checkpoint metadata, shard completeness and the small scheduler record."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    metadata_path = path / ".metadata"
    metadata = FileSystemReader(path).read_metadata()
    keys = list(metadata.state_dict_metadata)
    for prefix in ("fsdp_checkpoint.model.", "fsdp_checkpoint.optimizers.state."):
        if not any(key.startswith(prefix) for key in keys):
            raise ValueError(f"Resume checkpoint is missing {prefix}")
    scheduler_key = "fsdp_checkpoint.lr_schedulers.0.last_epoch"
    scheduler_epoch = None
    sizes = {}
    for index, storage in metadata.storage_data.items():
        shard = (path / storage.relative_path).resolve()
        if shard.parent != path.resolve() or not shard.is_file():
            raise ValueError("Resume checkpoint has an unavailable shard")
        required = storage.offset + storage.length
        if storage.offset < 0 or storage.length <= 0 or shard.stat().st_size < required:
            raise ValueError("Resume checkpoint contains a truncated shard")
        sizes[shard.name] = shard.stat().st_size
        if index.fqn == scheduler_key:
            if storage.length > 1024 * 1024:
                raise ValueError("Unexpectedly large scheduler metadata record")
            with shard.open("rb") as stream:
                stream.seek(storage.offset)
                scheduler_epoch = torch.load(
                    io.BytesIO(stream.read(storage.length)), weights_only=True
                )
    if type(scheduler_epoch) is not int or scheduler_epoch != expected_updates:
        raise ValueError("Resume scheduler epoch differs from optimizer-update count")
    return {
        "scheduler_last_epoch": scheduler_epoch,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "shard_sizes": sizes,
    }


def validate_resume_checkpoint(checkpoint: Path, cfg) -> dict:
    """Require the same learning recipe at a completed quota-rollout boundary."""
    checkpoint = checkpoint.resolve(strict=True)
    # Checkpoints from this launcher have <run>/<experiment>/checkpoints/step.
    source_config = checkpoint.parents[2] / "resolved_config.yaml"
    source = OmegaConf.load(source_config)
    state_path = checkpoint / "runner_state.json"
    allocation_path = checkpoint / "rollout_allocation_state.json"
    state = json.loads(state_path.read_text())
    allocation = json.loads(allocation_path.read_text())
    step, reuse = state["optimizer_updates"], int(cfg.algorithm.flow_refresh_updates)
    if cfg.runner.max_steps > cfg.runner.max_epochs:
        raise ValueError("Total --updates exceeds runner.max_epochs")
    if (
        type(step) is not int
        or step <= 0
        or step % reuse
        or checkpoint.name != f"global_step_{step}"
        or state["step_unit"] != "optimizer_update"
        or state["rollout_count"] != step // reuse
        or allocation["rounds"] != state["rollout_count"]
        or allocation["simulation_steps"] != state["env_steps"]
        or step >= min(cfg.runner.max_steps, cfg.runner.max_epochs)
    ):
        raise ValueError(
            "Resume requires a complete round and a larger total --updates"
        )
    if (
        cfg.runner.step_unit != "optimizer_update"
        or not cfg.runner.rollout_allocation.enabled
        or cfg.runner.ckpt_path is not None
    ):
        raise ValueError(
            "Resume requires quota collection and no weights-only checkpoint"
        )
    # Ignore only output placement. Keep all numerical/model/teacher semantics.
    for key in (
        "actor",
        "algorithm",
        "rollout",
        "weight_syncer",
        "runner.step_unit",
        "runner.rollout_allocation",
        "runner.weight_sync_interval",
        "runner.use_training_pipeline",
        "runner.overlap_env_bootstrap",
    ):
        before, after = OmegaConf.select(source, key), OmegaConf.select(cfg, key)
        if OmegaConf.is_config(before):
            before = OmegaConf.to_container(before, resolve=True)
            after = OmegaConf.to_container(after, resolve=True)
        if key in {"actor", "rollout"}:
            # A historical missing flag has exactly the current boolean False
            # semantics. Normalize only this field, never True or false-like 0.
            for section in (before, after):
                models = [section.get("model", {})]
                if key == "rollout":
                    models.extend(section.get("expert_models", {}).values())
                for model in models:
                    if model.get("opd_canonical_fixed_padding") is False:
                        model.pop("opd_canonical_fixed_padding")
        if before != after:
            raise ValueError(f"Resume recipe changed at {key}")
    before = OmegaConf.to_container(source.env.train, resolve=True)
    after = OmegaConf.to_container(cfg.env.train, resolve=True)
    for env in (before, after):
        env.get("video_cfg", {}).pop("video_base_dir", None)
    if before != after:
        raise ValueError("Resume recipe changed at env.train")
    source_placement = source.cluster.component_placement["actor,env,rollout"].placement
    if len(set(str(source_placement).split(","))) != 4:
        raise ValueError("Resume requires the original four-rank layout")
    trajectories_per_round = int(
        cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch
    )
    if allocation["trajectories"] != allocation["rounds"] * trajectories_per_round:
        raise ValueError("Resume trajectory budget or counters changed")
    if allocation["config"] != OmegaConf.to_container(
        cfg.runner.rollout_allocation, resolve=True
    ):
        raise ValueError(
            "Resume allocation state differs from the configured controller"
        )
    dcp = inspect_resume_dcp(checkpoint / "actor/dcp_checkpoint", step)
    return {
        "source_checkpoint": str(checkpoint),
        "source_config": str(source_config),
        "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "runner_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "allocation_state_sha256": hashlib.sha256(
            allocation_path.read_bytes()
        ).hexdigest(),
        "starting_optimizer_update": step,
        "starting_rollout_round": state["rollout_count"],
        "additional_optimizer_updates": cfg.runner.max_steps - step,
        "additional_rollout_rounds": (cfg.runner.max_steps - step) // reuse,
        "exact_replay": False,
        "limitations": "Fresh simulator instances and restarted rollout RNG; physics session counters restart. Allocation policy-step totals and task cursors continue.",
        "dcp": dcp,
    }


def build_config(args, checkpoint_paths: dict[str, Path]):
    """Compose a complete launch recipe without starting Cluster or loading models."""
    gpus = sorted(args.gpus)
    if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
        raise ValueError("Specify four distinct nonnegative physical GPU IDs")
    if args.rollout_steps < 8 or args.rollout_steps > 520 or args.rollout_steps % 8:
        raise ValueError("rollout-steps must be a multiple of eight in [8, 520]")
    seed = getattr(args, "seed", None)
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
        raise ValueError("seed must be an integer in [0, 2147483647]")
    allocation_smoke = getattr(args, "allocation_smoke", False)
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if allocation_smoke and resume_checkpoint is not None:
        raise ValueError("allocation-smoke cannot be combined with resume-checkpoint")
    config_name = getattr(args, "config_name", CONFIG_NAME)
    padding_profile = getattr(args, "student_padding_profile", "native")
    if padding_profile not in PADDING_PROFILES:
        raise ValueError("Unknown student padding profile")
    if (
        padding_profile == "canonical-fixed570"
        and config_name != "libero_mopd_candidate_m1"
    ):
        raise ValueError("canonical-fixed570 is an experimental M1/R2 profile only")
    if allocation_smoke and config_name not in {
        "libero_mopd_candidate_m2",
        "libero_mopd_candidate_m3",
    }:
        raise ValueError("allocation-smoke is only valid for adaptive M2/M3")
    if allocation_smoke and args.rollout_steps != 520:
        raise ValueError("allocation-smoke requires full native horizons (520)")
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name=config_name)
    reuse = int(cfg.algorithm.flow_refresh_updates)
    if args.updates < reuse or args.updates % reuse:
        raise ValueError("Smoke updates must be positive multiples of reuse R")
    if allocation_smoke and args.updates // reuse < 3:
        raise ValueError("allocation-smoke requires at least three complete rounds")
    save_interval = getattr(args, "save_every_updates", None)
    if save_interval is None:
        save_interval = args.updates
    if save_interval < reuse or save_interval > args.updates or save_interval % reuse:
        raise ValueError(
            "Save interval must be a complete-round multiple within the run"
        )
    with open_dict(cfg):
        if seed is not None:
            cfg.actor.seed = seed
            cfg.rollout.seed = seed
            cfg.env.train.seed = seed
            # The legacy short smoke has no allocation block; adding only a
            # seed does not enable allocation. Candidate arms already have it.
            OmegaConf.update(
                cfg, "runner.rollout_allocation.seed", seed, force_add=True
            )
        if padding_profile == "canonical-fixed570":
            # This changes both input policy and update batch; it is not a
            # throughput-equivalent version of the native B64/micro1 run.
            cfg.actor.global_batch_size = 256
            cfg.actor.micro_batch_size = 4
            if (
                int(cfg.algorithm.flow_refresh_updates) != 2
                or int(cfg.env.train.total_num_envs) != 32
                or float(cfg.actor.optim.lr) != 3e-6
            ):
                raise ValueError("canonical-fixed570 expects M1 T32/R2/LR3e-6")
            apply_student_padding_profile(cfg.actor.model, padding_profile)
            cfg.rollout.model.rl_head_config = OmegaConf.create(
                OmegaConf.to_container(cfg.actor.model.rl_head_config, resolve=True)
            )
            apply_student_padding_profile(cfg.rollout.model, padding_profile)
            for expert in cfg.rollout.expert_models.values():
                # Actor teacher construction starts from the student model config.
                expert.opd_canonical_fixed_padding = False
        if resume_checkpoint is not None:
            cfg.runner.resume_dir = str(resume_checkpoint.resolve(strict=True))
            cfg.runner.resume_mode = "adaptive_round_boundary"
        if allocation_smoke:
            if (
                cfg.runner.rollout_allocation.mode != "adaptive"
                or not cfg.algorithm.sync_suite_loss_weights
            ):
                raise ValueError(
                    "allocation-smoke requires adaptive synchronized weights"
                )
            for key, value in ALLOCATION_SMOKE_OVERRIDES.items():
                cfg.runner.rollout_allocation[key] = value
        cfg.actor.model.model_path = str(checkpoint_paths["student"])
        for suite, path in checkpoint_paths.items():
            if suite != "student":
                cfg.rollout.expert_models[suite].model_path = str(path)
        cfg.runner.logger.log_path = str(args.output)
        cfg.runner.logger.experiment_name = "libero_production_smoke"
        cfg.runner.max_steps = args.updates
        cfg.runner.save_interval = save_interval
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
    student_input_contract(cfg.actor.model, padding_profile)
    student_input_contract(cfg.rollout.model, padding_profile)
    if resume_checkpoint is not None:
        args.resume_summary = validate_resume_checkpoint(resume_checkpoint, cfg)
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


def run_logged(
    command: list[str], env: dict[str, str], log: Path, *, timeout_seconds: float = 3600
) -> None:
    """Bound the child lifetime and abort on an unrecoverable background CUDA OOM.

    Ray can keep its driver alive after a collective thread exits. Watching the
    child's new log bytes prevents that failure from holding smoke-test GPUs.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with log.open("a") as stream, log.open("r") as reader:
        reader.seek(0, os.SEEK_END)
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline, tail = time.monotonic() + timeout_seconds, ""
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Smoke subprocess exceeded {timeout_seconds}s; see {log}"
                    )
                try:
                    code = child.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    code = None
                tail = tail + reader.read()
                if (
                    "torch.OutOfMemoryError:" in tail
                    or "torch.cuda.OutOfMemoryError:" in tail
                ):
                    raise RuntimeError(
                        f"CUDA OOM in smoke subprocess or worker thread; see {log}"
                    )
                tail = tail[-512:]
                if code is not None:
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    return
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
    parser.add_argument(
        "--config-name",
        choices=[CONFIG_NAME] + [f"libero_mopd_candidate_m{i}" for i in range(4)],
        default=CONFIG_NAME,
    )
    parser.add_argument(
        "--allocation-smoke",
        action="store_true",
        help="M2/M3 only: shorten controller coverage window for integration checks, not performance evidence",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Completed global_step_N directory; --updates is the total target, not additional updates",
    )
    parser.add_argument(
        "--student-padding-profile",
        choices=PADDING_PROFILES,
        default="native",
        help="Experimental canonical-fixed570 changes policy inputs and uses M1 B256/micro4; needs a new matched baseline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set actor/rollout/environment/allocation seeds together (0..2147483647); omitted preserves YAML seeds",
    )
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument(
        "--save-every-updates",
        type=int,
        help="Optional checkpoint interval at complete rollout rounds",
    )
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3600,
        help="Maximum driver runtime for this bounded smoke",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Default; write plan only")
    args = parser.parse_args()
    args.python = args.python.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new output directory for each dry-run or run")
    if args.cpus < 1 or args.timeout_seconds <= 0:
        raise ValueError("cpus and timeout-seconds must be positive")
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
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
    tree_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    )
    (args.output / "code_provenance.json").write_text(
        json.dumps(
            {
                "git_commit": revision,
                "working_tree_status": tree_status,
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "clean_worktree": not bool(tree_status.strip()),
            },
            indent=2,
        )
        + "\n"
    )
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
        "resume": getattr(args, "resume_summary", None),
        "save_interval": int(cfg.runner.save_interval),
        "rollout_rounds": args.updates // int(cfg.algorithm.flow_refresh_updates),
        "config_name": args.config_name,
        "seed_override": args.seed,
        "effective_seeds": {
            key: OmegaConf.select(cfg, key)
            for key in (
                "actor.seed",
                "rollout.seed",
                "env.train.seed",
                "runner.rollout_allocation.seed",
            )
        },
        "trajectories_per_round": int(cfg.env.train.total_num_envs),
        "rollout_steps": args.rollout_steps,
        "incomplete_episodes": args.rollout_steps < 520,
        "performance_evidence": False,
        "student_padding_profile": args.student_padding_profile,
        "student_input_contract": student_input_contract(
            cfg.actor.model, args.student_padding_profile
        ),
        "batch_recipe": {
            "global_batch_size": int(cfg.actor.global_batch_size),
            "micro_batch_size": int(cfg.actor.micro_batch_size),
            "microbatches_per_rank": int(cfg.actor.global_batch_size)
            // (4 * int(cfg.actor.micro_batch_size)),
        },
        "smoke_only": True,
        "allocation_smoke": args.allocation_smoke,
        "allocation_smoke_overrides": (
            dict(ALLOCATION_SMOKE_OVERRIDES) if args.allocation_smoke else {}
        ),
        "timeout_seconds": args.timeout_seconds,
        "differences_from_standalone": [
            "Production Ray/FSDP, all suites on each actor rank; see trajectories_per_round",
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
        run_logged(
            driver_command,
            env,
            args.output / "driver.log",
            timeout_seconds=args.timeout_seconds,
        )
        status.write_text("OK\n")
    except BaseException:
        status.write_text("FAILED\n")
        raise
    finally:
        stop_owned(owner)


if __name__ == "__main__":
    main()
