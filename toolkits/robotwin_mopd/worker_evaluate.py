# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate fixed Robotwin scenes through standard RLinf rollout/env workers.

One task, one resident environment and one physical GPU per driver. Different
jobs may evaluate different tasks concurrently; no actor/optimizer is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from toolkits.robotwin_mopd.author_pool import load_pool, validate_pool_protocol

REPO = Path(__file__).resolve().parents[2]


def compose_config(args, pool):
    """Retain production model transforms and declare only evaluation workers."""
    os.environ["EMBODIED_PATH"] = str(REPO / "examples/embodiment")
    os.environ["OPD_RUN_DIR"] = str(args.output)
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05", overrides=args.override)
    model = OmegaConf.to_container(cfg.actor.model, resolve=True)
    model["model_path"] = str(args.checkpoint.resolve())
    # Normalizers are supplied explicitly in the experiment plan, never inferred
    # from task similarity. These are the identical per-task training statistics.
    model["opd_task_norm_stats"] = dict(args.norm_stats)
    horizon = int(pool["protocol"]["task_horizons"][args.task])
    if horizon % int(model["num_action_chunks"]):
        raise ValueError("Native horizon must be divisible by executed chunk length")
    with open_dict(cfg):
        cfg.runner.task_type = "embodied_eval"
        cfg.runner.only_eval = True
        cfg.runner.ckpt_path = None
        cfg.runner.logger.experiment_name = args.output.name
        cfg.runner.logger.logger_backends = []
        cfg.cluster.component_placement = {
            "env,rollout": {"node_group": "robotwin_eval", "placement": str(args.gpu)}
        }
        cfg.cluster.node_groups = [
            {
                "label": "robotwin_eval",
                "node_ranks": "0",
                "env_configs": [
                    {"node_ranks": "0", "python_interpreter_path": sys.executable}
                ],
            }
        ]
        del cfg.env.train
        env = cfg.env.eval
        env.task_names = [args.task]
        env.task_horizons = pool["protocol"]["task_horizons"]
        env.total_num_envs = 1
        env.rollout_epoch = args.episodes
        env.max_episode_steps = horizon
        env.max_steps_per_rollout_epoch = horizon
        env.auto_reset = False
        env.ignore_terminations = False
        env.enable_offload = False
        env.is_eval = True
        env.instruction_type = "unseen"
        env.video_cfg.save_video = False
        env.fixed_evaluation_pool = str(args.pool.resolve())
        env.fixed_evaluation_tasks = pool["protocol"]["task_names"]
        env.fixed_evaluation_records = str(args.output / "episodes.jsonl")
        cfg.rollout.model = model
        cfg.rollout.enable_offload = False
        del cfg.rollout.expert_models
    OmegaConf.resolve(cfg)
    return cfg


def observation_digest(obs):
    """Hash physical policy inputs for paired execution-path diagnostics."""
    digest = hashlib.sha256()
    for key in ("main_images", "wrist_images", "states"):
        value = obs[key]
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps([obs["task_descriptions"], obs["task_groups"]]).encode())
    return digest.hexdigest()


def checked_metrics(metrics, records, requested):
    """Reject partial evaluations or disagreement with exact episode outcomes."""
    if len(records) != requested or [r["episode"] for r in records] != list(
        range(requested)
    ):
        raise RuntimeError("Missing, duplicated, or reordered evaluation episodes")
    successes = sum(row["success_once"] for row in records)
    if int(metrics.get("num_trajectories", -1)) != requested:
        raise RuntimeError("Worker trajectory count disagrees with fixed replay")
    if abs(float(metrics["success_once"]) - successes / requested) > 1e-6:
        raise RuntimeError("Worker success metric disagrees with raw episodes")
    if (
        abs(
            float(metrics["episode_len"])
            - sum(r["simulation_steps"] for r in records) / requested
        )
        > 1e-4
    ):
        raise RuntimeError("Worker episode length disagrees with exact control count")
    return successes


def run(args):
    """Launch the ordinary runner; specialize only scene replay and RNG seeding."""
    import numpy as np
    import torch
    import torch.multiprocessing as mp

    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Evaluation output must be empty: {args.output}")
    pool = load_pool(args.pool, task_names=args.tasks)
    if (
        args.task not in args.tasks
        or not 0 < args.episodes <= pool["protocol"]["episodes_per_task"]
    ):
        raise ValueError("Task or episode count is outside the declared pool")
    cfg = compose_config(args, pool)
    runtime = validate_pool_protocol(
        pool,
        task_names=args.tasks,
        task_config=OmegaConf.to_container(cfg.env.eval.task_config, resolve=True),
        task_horizons=dict(cfg.env.eval.task_horizons),
        robotwin_root=Path(os.environ["ROBOTWIN_PATH"]),
    )
    OmegaConf.save(cfg, args.output / "requested_config.yaml")
    started = time.time()
    mp.set_start_method("spawn", force=True)
    Cluster.NAMESPACE = (
        "robotwin_fixed_" + hashlib.sha256(str(args.output).encode()).hexdigest()[:16]
    )
    cfg = validate_cfg(cfg)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml")
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    trace_path = args.output / "actions.jsonl"
    trace_actions = args.trace_actions

    class FixedNoiseRolloutWorker(MultiStepRolloutWorker):
        def init_worker(self):
            torch.cuda.set_per_process_memory_fraction(
                getattr(args, "gpu_memory_fraction", 0.32)
            )
            self._fixed_query = 0
            return super().init_worker()

        def predict(self, env_obs, mode="train"):
            # EnvOutput deliberately transports only policy inputs. Evaluation
            # has an exact, fixed number of queries per episode, including frozen
            # terminal padding, so seed at these boundaries without changing the
            # training transport schema.
            episode, chunk = divmod(self._fixed_query, self.n_eval_chunk_steps)
            if mode != "eval" or episode >= args.episodes:
                raise ValueError("Fixed replay requires serial evaluation")
            entry = pool["tasks"][args.task][episode]
            if env_obs["task_groups"] != [args.task] or env_obs[
                "task_descriptions"
            ] != [entry["instruction"]]:
                raise RuntimeError("Worker query is out of sync with fixed replay")
            if chunk == 0:
                import random

                seed = int(entry["scene_seed"])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            input_digest = observation_digest(env_obs) if trace_actions else None
            actions, result = super().predict(env_obs, mode=mode)
            if trace_actions:
                row = {
                    "scene_seed": int(entry["scene_seed"]),
                    "chunk": chunk,
                    "episode": episode,
                    "observation_sha256": input_digest,
                    "actions": actions.detach().cpu().tolist(),
                }
                with trace_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            self._fixed_query += 1
            return actions, result

    rollout = FixedNoiseRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
    runner.init_workers()
    metrics = runner.evaluate()
    records = [
        json.loads(line)
        for line in (args.output / "episodes.jsonl").read_text().splitlines()
    ]
    successes = checked_metrics(metrics, records, args.episodes)
    source_files = [
        "toolkits/robotwin_mopd/worker_evaluate.py",
        "rlinf/envs/robotwin/fixed_eval_env.py",
        "rlinf/envs/robotwin/multitask_env.py",
        "rlinf/workers/env/env_worker.py",
        "rlinf/workers/rollout/hf/huggingface_worker.py",
    ]
    summary = {
        "task": args.task,
        "checkpoint": str(args.checkpoint.resolve()),
        "requested_episodes": args.episodes,
        "completed_episodes": len(records),
        "successes": successes,
        "success_rate": successes / args.episodes,
        "episodes": records,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "simulation_steps": sum(r["simulation_steps"] for r in records),
        "simulation_step_unit": "executed_control_targets",
        "evaluation_chain": "EmbodiedEvalRunner/EnvWorker/MultiStepRolloutWorker",
        "pool": str(args.pool.resolve()),
        "pool_content_sha256": pool["pool_content_sha256"],
        "pool_protocol": pool["protocol"],
        "verified_runtime": runtime,
        "started_unix": started,
        "finished_unix": time.time(),
        "gpu_memory_fraction": getattr(args, "gpu_memory_fraction", 0.32),
        "source_sha256": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runner.metric_logger.finish()
    print(f"WORKER_EVAL_COMPLETE {args.task}: {successes}/{args.episodes}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument(
        "--gpu", type=int, required=True, help="Physical GPU index in RLinf placement"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--norm-stats", action="append", required=True, metavar="TASK=PATH"
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--trace-actions", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.32)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction <= 0.5:
        parser.error("Evaluation requires 0 < gpu-memory-fraction <=0.5")
    pairs = [value.split("=", 1) for value in args.norm_stats]
    if any(len(pair) != 2 for pair in pairs) or len({p[0] for p in pairs}) != len(
        pairs
    ):
        parser.error("--norm-stats must contain unique TASK=PATH pairs")
    args.norm_stats = dict(pairs)
    if set(args.norm_stats) != set(args.tasks) or any(
        not Path(p).is_file() for p in args.norm_stats.values()
    ):
        parser.error("Supply an existing normalizer for every declared task")
    run(args)


if __name__ == "__main__":
    main()
