# Copyright 2025 The RLinf Authors.
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

"""Standalone native RoboTwin evaluation with the exact MOPD policy transforms.

Example after sourcing the isolated runtime env.sh::

    CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/evaluate.py \
        --task adjust_bottle --checkpoint teacher --episodes 5 \
        --seed 200000 --output outputs/robotwin4/eval/adjust_teacher.json
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.robotwin.multitask_env import TASK_HORIZONS, RoboTwinMultiTaskEnv
from rlinf.models.embodiment.openpi import get_model


def write_summary(path: Path, metadata: dict, episodes: list[dict]) -> None:
    """Atomically publish partial results after every completed episode."""
    count = len(episodes)
    result = dict(
        metadata,
        completed_episodes=count,
        successes=sum(row["success_once"] for row in episodes),
        success_rate=sum(row["success_once"] for row in episodes) / count
        if count
        else None,
        simulation_steps=sum(row["simulation_steps"] for row in episodes),
        simulation_step_unit="executed_control_targets",
        episodes=episodes,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(TASK_HORIZONS))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", default="teacher", help="base, teacher, or checkpoint directory"
    )
    parser.add_argument("--config-name", default="robotwin4_mopd_pi05")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Native RoboTwin VLA evaluation requires a visible CUDA device"
        )
    repo = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    os.environ.setdefault("OPD_RUN_DIR", str(args.output.parent))
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    model_cfg = copy.deepcopy(cfg.actor.model)
    if args.checkpoint == "teacher":
        model_cfg.model_path = cfg.rollout.expert_models[args.task].model_path
    elif args.checkpoint != "base":
        model_cfg.model_path = str(Path(args.checkpoint).expanduser().resolve())
    metadata = {
        "task": args.task,
        "checkpoint": str(model_cfg.model_path),
        "requested_episodes": args.episodes,
        "initial_seed": args.seed,
        "config_name": args.config_name,
        "hydra_overrides": args.override,
        "chunk_length": int(model_cfg.num_action_chunks),
        "denoising_steps": int(model_cfg.num_steps),
        "task_horizon": int(cfg.env.eval.task_horizons[args.task]),
        "normalization": OmegaConf.to_container(
            model_cfg.opd_task_norm_stats, resolve=True
        ),
        "reset_filter": "unstable_scene_only; no planner or success filtering",
    }
    episodes = []
    write_summary(args.output, metadata, episodes)
    # Use the identical loader, task normalization, and native control path as
    # training, without constructing workers, distributed process groups or Ray.
    torch.manual_seed(args.seed)
    print(f"Loading policy: {model_cfg.model_path}", flush=True)
    model = get_model(model_cfg).to("cuda").eval()
    print("Policy loaded", flush=True)
    env_cfg = copy.deepcopy(cfg.env.eval)
    env_cfg.task_names = [args.task]
    env_cfg.total_num_envs = 1
    env_cfg.video_cfg.save_video = False
    env = RoboTwinMultiTaskEnv(env_cfg, 1, 0, 1, None)
    next_seed = args.seed
    try:
        for episode in range(args.episodes):
            started = time.monotonic()
            row = {
                "slot": 0,
                "task_id": 0,
                "task": args.task,
                "suite": args.task,
                "reset_seed": next_seed,
                "round": episode,
            }
            env.set_rollout_tasks([row])
            obs, _ = env.reset()
            print(f"Episode {episode}: environment reset", flush=True)
            actual_seed = env.rollout_episode_records()[0]["actual_reset_seed"]
            next_seed = actual_seed + 1  # Avoid repeating a retried scene.
            random.seed(actual_seed)
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)
            inference_seconds = 0.0
            simulation_seconds = 0.0
            chunks = 0
            with torch.inference_mode():
                while True:
                    before = time.monotonic()
                    actions, _ = model.predict_action_batch(
                        obs, mode="eval", compute_values=False
                    )
                    torch.cuda.synchronize()
                    inference_seconds += time.monotonic() - before
                    if not torch.isfinite(torch.as_tensor(actions)).all():
                        raise RuntimeError("Policy produced nonfinite actions")
                    before = time.monotonic()
                    observations, _, terminal, truncated, _ = env.chunk_step(actions)
                    simulation_seconds += time.monotonic() - before
                    obs = observations[-1]
                    chunks += 1
                    if bool((terminal | truncated).any()):
                        break
            result = env.rollout_episode_records()[0]
            result.update(
                episode=episode,
                wall_seconds=time.monotonic() - started,
                inference_seconds=inference_seconds,
                simulation_seconds=simulation_seconds,
                policy_chunks=chunks,
            )
            episodes.append(result)
            write_summary(args.output, metadata, episodes)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "success_rate": sum(r["success_once"] for r in episodes)
                / len(episodes),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
