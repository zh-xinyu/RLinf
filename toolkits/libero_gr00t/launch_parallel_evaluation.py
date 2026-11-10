# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate fixed LIBERO checkpoints through training's environment/rollout path.

Default is a dry run. Dispatches one task per private single-GPU Ray cluster,
with several exact initial states in parallel and episode-private policy noise.
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from toolkits.libero_gr00t.evaluate_full_checkpoint import (
    baseline_identity,
    candidate_model_config,
    checkpoint_step,
    sha256,
)
from toolkits.libero_gr00t.launch_production import (
    REPO,
    choose_ports,
    read_runtime_environment,
    run_logged,
)
from toolkits.libero_gr00t.parallel_evaluation import (
    EXECUTION,
    HORIZONS,
    SUITES,
    audit_episodes,
    validate_selection,
    write_json,
)
from toolkits.robocasa_mopd.owned_ray import OWNER_KEY, check_gpus, stop_owned


def task_config(
    args: argparse.Namespace, task: int, output: Path, gpu: int = 0
) -> DictConfig:
    """Derive inference/normalization from the resolved training recipe."""
    model, reuse = candidate_model_config(args.config, args.student)
    training = OmegaConf.load(args.config)
    if list(training.env.train.task_suite_names) != list(SUITES):
        raise ValueError("Expected the frozen four-suite LIBERO task order")
    if dict(training.env.train.task_suite_horizons) != dict(
        zip(SUITES, HORIZONS, strict=True)
    ):
        raise ValueError("Training recipe does not use official suite horizons")
    if dict(training.env.train.task_suite_execution_lengths) != dict(
        zip(SUITES, EXECUTION, strict=True)
    ):
        raise ValueError("Training recipe does not use K=5/5/5/10")
    if args.weights and checkpoint_step(args.weights) % reuse:
        raise ValueError("Checkpoint is not at a complete rollout boundary")
    # Keep latent dimensions, text preparation and precision unchanged. Native
    # B1 is required for exact equivalence; active slots are skipped by adapter.
    with open_dict(model):
        model.opd_native_rollout_batch_size = 1
        model.opd_native_rollout_batch_head = False
        model.opd_skip_finished_rollout = False
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "evaluations/libero")
    ):
        cfg = compose(
            config_name="libero_four_suite_gr00t_mopd_eval",
            overrides=[f"hydra.searchpath=[file://{REPO}/examples/embodiment/config]"],
        )
    with open_dict(cfg):
        cfg.rollout.model = model
        cfg.rollout.seed = args.seed
        cfg.runner.ckpt_path = str(args.weights) if args.weights else None
        cfg.runner.logger.log_path = str(output)
        cfg.runner.logger.project_name = "FORMAL_libero_gr00t"
        cfg.runner.logger.experiment_name = "independent_evaluation"
        cfg.runner.logger.logger_backends = []
        env = cfg.env.eval
        env.task_suite_horizons = dict(zip(SUITES, HORIZONS, strict=True))
        env.task_suite_execution_lengths = dict(zip(SUITES, EXECUTION, strict=True))
        env.task_id_filter = [task]
        env.total_num_envs = args.envs_per_task
        env.rollout_epoch = len(args.trials) // args.envs_per_task
        env.auto_reset = False
        env.max_episode_steps = max(HORIZONS)
        env.max_steps_per_rollout_epoch = max(HORIZONS)
        env.seed = args.seed
        cfg.evaluation = {
            "task_id": task,
            "trials": args.trials,
            "seed": args.seed,
            "decisions": (HORIZONS[task // 10] + EXECUTION[task // 10] - 1)
            // EXECUTION[task // 10],
            "trace": args.trace,
            "output": str(output),
        }
        # Physical GPU is supplied to each isolated driver by its environment.
        cfg.cluster.component_placement = {
            "env,rollout": {"node_group": "libero_eval", "placement": str(gpu)}
        }
        cfg.cluster.node_groups = [
            {
                "label": "libero_eval",
                "node_ranks": "0",
                "env_configs": [
                    {
                        "node_ranks": "0",
                        "python_interpreter_path": str(args.python),
                    }
                ],
            }
        ]
    OmegaConf.resolve(cfg)
    return cfg


def driver(config_path: Path) -> None:
    """Launch the same worker classes and evaluation runner used by RLinf."""
    import torch.multiprocessing as mp

    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from toolkits.libero_gr00t.parallel_evaluation_workers import (
        ParallelEvalEnvWorker,
        ParallelEvalRolloutWorker,
    )

    mp.set_start_method("spawn", force=True)
    # Validation constructs Cluster too; choose its namespace before that call.
    Cluster.NAMESPACE = "libero_eval_" + uuid.uuid4().hex[:16]
    cfg = validate_cfg(OmegaConf.load(config_path))
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = ParallelEvalRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = ParallelEvalEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    runner.run()


def launch_task(args: argparse.Namespace, task: int, gpu: int) -> None:
    """Own all Ray processes for a task; never attach to or stop others' jobs."""
    output = args.output / f"task_{task:02d}"
    if not args.run:
        return
    if args.cancelled.is_set():
        raise RuntimeError("Evaluation dispatch cancelled")
    ports = choose_ports()
    owner = "opd-libero-parallel-eval-" + uuid.uuid4().hex
    root = Path(tempfile.mkdtemp(prefix="opdlpe-", dir="/tmp"))
    ray_dir = root / "ray"
    ray_dir.mkdir()
    env = {**os.environ, **read_runtime_environment(args.python)}
    env.pop(OWNER_KEY, None)
    env.pop("RAY_ADDRESS", None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        RLINF_NODE_RANK="0",
        RAY_TMPDIR=str(root),
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        RAY_USAGE_STATS_ENABLED="0",
        # Shared nodes can exceed Ray's default30-second registration window.
        # This changes startup tolerance only; episode/model settings are unchanged.
        RAY_raylet_start_wait_time_s="180",
        GPU_FREE_MB=str(20 * 1024),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        OPD_EVAL_MEMORY_GIB=str(args.memory_gib),
    )
    ray_cmd = [
        str(args.python),
        "-m",
        "ray.scripts.scripts",
        "start",
        "--head",
        "--node-ip-address=127.0.0.1",
        f"--num-cpus={args.cpus}",
        "--num-gpus=1",
        "--min-worker-port=0",
        "--max-worker-port=0",
        "--object-store-memory=1073741824",
        "--include-dashboard=true",
        "--dashboard-host=127.0.0.1",
        "--disable-usage-stats",
        f"--temp-dir={ray_dir}",
        *[f"--{name}={port}" for name, port in ports.items()],
    ]
    command = [
        str(args.python),
        "-m",
        "toolkits.libero_gr00t.launch_parallel_evaluation",
        "--driver-config",
        str(output / "resolved_config.yaml"),
    ]
    write_json(
        output / "launch.json",
        {
            "owner": owner,
            "gpu": gpu,
            "ray_command": ray_cmd,
            "driver_command": command,
            "ray_temp_dir": str(ray_dir),
        },
    )
    try:
        (output / "gpu_preflight.txt").write_text(check_gpus(1, env))
        (output / "status").write_text("RUNNING\n")
        run_logged(
            ray_cmd, {**env, OWNER_KEY: owner}, output / "ray.log", timeout_seconds=300
        )
        env["RAY_ADDRESS"] = (ray_dir / "ray_current_cluster").read_text().strip()
        if int(env["RAY_ADDRESS"].rsplit(":", 1)[1]) != ports["port"]:
            raise RuntimeError("Private Ray address differs from plan")
        run_logged(
            command,
            {**env, OWNER_KEY: owner},
            output / "driver.log",
            timeout_seconds=args.timeout_seconds,
        )
        rows = json.loads((output / "episodes.json").read_text())
        summary = audit_episodes(rows, [task], args.trials)
        write_json(output / "summary.json", summary)
        if not summary["complete"]:
            raise RuntimeError("Incomplete evaluation; scores withheld")
        (output / "status").write_text("OK\n")
    except BaseException:
        (output / "status").write_text("FAILED\n")
        raise
    finally:
        stop_owned(owner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--output", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--baseline", action="store_true")
    source.add_argument("--weights", type=Path)
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(40)))
    parser.add_argument("--trials", nargs="+", type=int, default=list(range(50)))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--envs-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.driver_config:
        driver(args.driver_config)
        return
    if not all((args.config, args.python, args.output)) or not (
        args.baseline or args.weights
    ):
        raise ValueError(
            "Require --config, --python, --output and --baseline or --weights"
        )
    validate_selection(args.tasks, args.trials, args.envs_per_task)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        raise ValueError("Specify distinct nonnegative physical GPU IDs")
    if (
        args.memory_gib < 8
        or args.memory_gib > 20
        or args.cpus < 2
        or args.timeout_seconds <= 0
    ):
        raise ValueError("Require memory in [8,20] GiB, cpus>=2 and a positive timeout")
    for name in ("config", "python", "student", "weights"):
        if getattr(args, name):
            # Runtime wrappers are scripts: retain their path rather than
            # dereferencing any symlink into the underlying interpreter.
            setattr(args, name, getattr(args, name).absolute())
            if not getattr(args, name).exists():
                raise FileNotFoundError(getattr(args, name))
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    model, _ = candidate_model_config(args.config, args.student)
    manifest = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ),
        "source_sha256": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("parallel_evaluation.py"),
                Path(__file__).with_name("parallel_evaluation_workers.py"),
                REPO / "rlinf/envs/libero/libero_env.py",
                REPO / "rlinf/config.py",
            )
        },
        "base_checkpoint": model.model_path,
        "base_checkpoint_identity": baseline_identity(Path(model.model_path)),
        "metadata_sha256": sha256(
            Path(model.model_path) / "experiment_cfg/metadata.json"
        ),
        "weights": str(args.weights) if args.weights else None,
        "weights_sha256": sha256(args.weights) if args.weights else None,
        "training_config": str(args.config),
        "training_config_sha256": sha256(args.config),
        "tasks": args.tasks,
        "trials": args.trials,
        "gpus": args.gpus,
        "envs_per_task": args.envs_per_task,
        "execution_lengths": dict(zip(SUITES, EXECUTION, strict=True)),
        "horizons": dict(zip(SUITES, HORIZONS, strict=True)),
        "rng_protocol": "episode-private torch seed = seed + local_task*100 + trial",
        "seed": args.seed,
        "mode": "run" if args.run else "dry_run",
    }
    # Hydra uses process-global state: compose serially before GPU threads.
    configs = {}
    for i, task in enumerate(args.tasks):
        configs[task] = task_config(
            args, task, args.output / f"task_{task:02d}", args.gpus[i % len(args.gpus)]
        )
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", manifest)
    for task, cfg in configs.items():
        output = args.output / f"task_{task:02d}"
        output.mkdir()
        OmegaConf.save(cfg, output / "resolved_config.yaml")

    args.cancelled = threading.Event()

    def interrupted(signum, _frame):
        args.cancelled.set()
        # Only this invocation's exact owner markers; never global Ray cleanup.
        for launch in args.output.glob("task_*/launch.json"):
            stop_owned(json.loads(launch.read_text())["owner"])
        raise KeyboardInterrupt(f"Received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def shard(gpu_tasks):
        gpu, tasks = gpu_tasks
        for task in tasks:
            if args.cancelled.is_set():
                return
            try:
                launch_task(args, task, gpu)
            except BaseException:
                args.cancelled.set()
                raise

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(
            pool.map(
                shard,
                [
                    (gpu, args.tasks[i :: len(args.gpus)])
                    for i, gpu in enumerate(args.gpus)
                ],
            )
        )
    if args.run:
        rows = [
            row
            for task in args.tasks
            for row in json.loads(
                (args.output / f"task_{task:02d}/episodes.json").read_text()
            )
        ]
        report = audit_episodes(rows, args.tasks, args.trials)
        write_json(args.output / "summary.json", report)
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            fields = [
                "suite",
                "task_id",
                "trial_id",
                "success_once",
                "simulation_steps",
                "settling_steps",
                "horizon",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(
                sorted(rows, key=lambda row: (row["task_id"], row["trial_id"]))
            )


if __name__ == "__main__":
    main()
