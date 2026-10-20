# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Calibrate routed teachers on the training protocol, separate from final tests.

Uses the ordinary GR00T model, RoboCasa wrapper, and action conversion. There is
no Ray cluster or training optimizer. GPU processes retain one teacher across
several tasks; only the simulator is rebuilt. Calibration seeds are held out
from the training task-occurrence sequence. Outputs include every episode and
actual environment control steps, including early success termination.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def compose_training(config_name: str) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        version_base="1.1", config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def catalog(cfg: Any) -> list[dict]:
    from rlinf.envs.robocasa365.utils import load_robocasa365_task_specs

    return [
        {
            "task_id": i,
            "task": s["task_name"],
            "suite": s["metadata_view"]["task_soup"],
            "horizon": int(s["horizon"]),
        }
        for i, s in enumerate(load_robocasa365_task_specs(cfg.env.train))
    ]


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    from omegaconf import open_dict

    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env
    from rlinf.models import get_model
    from rlinf.utils.utils import seed_everything

    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)
    cfg = compose_training(args.config)
    tasks = catalog(cfg)
    if args.task:
        tasks = [t for t in tasks if t["task"] == args.task]
    tasks = tasks[args.rank :: args.world_size]
    model, suite = None, None
    for task in tasks:
        if suite != task["suite"]:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            suite = task["suite"]
            model_cfg = copy.deepcopy(cfg.actor.model)
            with open_dict(model_cfg):
                model_cfg.model_path = cfg.rollout.expert_models[suite].model_path
                model_cfg.precision = cfg.rollout.model.precision
                model_cfg.action_head_precision = cfg.rollout.expert_models[
                    suite
                ].action_head_precision
            model = get_model(model_cfg).to("cuda").eval()
            model.requires_grad_(False)
        count = min(args.parallel_envs, args.episodes)
        if args.episodes % count:
            raise ValueError("episodes must divide by the parallel environment count.")
        env_cfg = copy.deepcopy(cfg.env.train)
        with open_dict(env_cfg):
            env_cfg.task_filter = [task["task"]]
            env_cfg.task_soup = [suite]
            env_cfg.task_sampling_strategy = "suite_quota"
            env_cfg.total_num_envs = count
        env = Robocasa365Env(env_cfg, count, 0, 1, None)
        records = []
        try:
            for start in range(0, args.episodes, count):
                rows = [
                    dict(
                        task,
                        task_id=0,
                        round=1,
                        epoch=start // count,
                        worker=args.rank,
                        slot=i,
                        occurrence=start + i,
                        reset_seed=(args.seed + task["task_id"] * 1000003 + start + i)
                        % (2**32),
                    )
                    for i in range(count)
                ]
                env.set_rollout_tasks(rows)
                obs, _ = env.reset()
                seed_everything(args.seed + task["task_id"] * 1009 + start)
                with torch.no_grad():
                    while not np.all(env._episode_terminated | env._episode_truncated):
                        actions, _ = model.predict_action_batch(
                            env_obs=obs, mode="train"
                        )
                        actions = prepare_actions(
                            actions,
                            env_type="robocasa365",
                            model_type="gr00t",
                            num_action_chunks=cfg.actor.model.num_action_chunks,
                            action_dim=cfg.actor.model.action_dim,
                            env_cfg=env_cfg,
                        )
                        observations, *_ = env.chunk_step(actions)
                        obs = observations[-1]
                records.extend(env.rollout_episode_records())
        finally:
            env.close()
        result = dict(
            task,
            episodes=len(records),
            successes=sum(r["success_once"] for r in records),
            simulation_steps=sum(r["simulation_steps"] for r in records),
            records=records,
        )
        (args.output / "tasks" / f"{task['task']}.json").write_text(
            json.dumps(result, indent=2)
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="robocasa365_mopd_adaptive_r2_gr00t_full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--parallel-envs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000000)
    parser.add_argument("--task", help="Single-task smoke; never a complete reference.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    from omegaconf import OmegaConf

    from rlinf.utils.rollout_allocation import training_signature

    if args.episodes < 1 or args.parallel_envs < 1:
        parser.error("Episode and parallel-env counts must be positive.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "tasks").mkdir()
    os.environ.setdefault("OPD_RUN_DIR", str(args.output))
    cfg = compose_training(args.config)
    tasks = catalog(cfg)
    if args.task and args.task not in {t["task"] for t in tasks}:
        parser.error("Unknown smoke task.")
    OmegaConf.save(cfg, args.output / "training_config.yaml")
    source_files = [
        "rlinf/envs/robocasa365/robocasa365_env.py",
        "rlinf/envs/robocasa/venv.py",
        "rlinf/envs/action_utils.py",
        "rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py",
        "evaluations/robocasa365/calibrate_teachers.py",
    ]
    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "profile": os.environ.get("OPD_ENV_PROFILE"),
        "sources": {
            f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in source_files
        },
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    gpus = [int(g) for g in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or any(g < 0 for g in gpus):
        parser.error("GPU list must contain unique nonnegative device indices.")

    def launch(pair):
        rank, gpu = pair
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            args.config,
            "--output",
            str(args.output),
            "--episodes",
            str(args.episodes),
            "--parallel-envs",
            str(args.parallel_envs),
            "--seed",
            str(args.seed),
            "--rank",
            str(rank),
            "--world-size",
            str(len(gpus)),
        ]
        if args.task:
            cmd += ["--task", args.task]
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL="egl",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        with (args.output / f"gpu-{gpu}.log").open("w") as stream:
            subprocess.run(
                cmd,
                env=env,
                cwd=REPO,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    start = time.time()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(launch, enumerate(gpus)))
    rows = [
        json.loads(p.read_text())
        for p in sorted((args.output / "tasks").glob("*.json"))
    ]
    expected = 1 if args.task else len(tasks)
    if len(rows) != expected or any(row["episodes"] != args.episodes for row in rows):
        raise RuntimeError("Calibration is incomplete.")
    reference = {
        "schema_version": 1,
        "status": "SMOKE" if args.task else "COMPLETE",
        "training_signature": training_signature(cfg),
        "teacher_checkpoints": {
            s: str(Path(e.model_path).resolve())
            for s, e in cfg.rollout.expert_models.items()
        },
        "tasks": [{k: v for k, v in row.items() if k != "records"} for row in rows],
        "seed": args.seed,
        "trajectories": sum(r["episodes"] for r in rows),
        "simulation_steps": sum(r["simulation_steps"] for r in rows),
        "wall_seconds": time.time() - start,
        "provenance": provenance,
    }
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2))
    print(
        json.dumps(
            {
                k: reference[k]
                for k in ("status", "trajectories", "simulation_steps", "wall_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
