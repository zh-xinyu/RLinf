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
import hashlib
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
        finished_unix=time.time() if count == metadata["requested_episodes"] else None,
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


def read_evaluation_pool(path: Path, task: str, episodes: int, cfg):
    """Validate a fixed pool against this evaluator's runtime before model loading."""
    from toolkits.robotwin_mopd import author_pool, feasible_pool

    header = json.loads(path.read_text())
    provider = (
        author_pool
        if header.get("protocol", {}).get("name") == author_pool.PROTOCOL_NAME
        else feasible_pool
    )
    pool = provider.load_pool(path)
    provider.validate_pool_protocol(
        pool,
        task_config=OmegaConf.to_container(cfg.env.eval.task_config, resolve=True),
        task_horizons=OmegaConf.to_container(cfg.env.eval.task_horizons, resolve=True),
        robotwin_root=Path(os.environ["ROBOTWIN_PATH"]),
    )
    if task not in pool["protocol"]["task_names"] or task not in pool["tasks"]:
        raise ValueError(f"Task {task} is not part of the evaluation pool")
    entries = pool["tasks"][task][:episodes]
    if len(entries) != episodes:
        raise ValueError(f"Requested {episodes} episodes but the pool has fewer scenes")
    return pool, entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(TASK_HORIZONS))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--evaluation-pool",
        type=Path,
        help="Optional validated expert-screened or author-cache unseen pool",
    )
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
    pool, pool_entries = None, None
    if args.evaluation_pool is not None:
        args.evaluation_pool = args.evaluation_pool.resolve()
        pool, pool_entries = read_evaluation_pool(
            args.evaluation_pool, args.task, args.episodes, cfg
        )
    model_cfg = copy.deepcopy(cfg.actor.model)
    if args.checkpoint == "teacher":
        model_cfg.model_path = cfg.rollout.expert_models[args.task].model_path
    elif args.checkpoint != "base":
        model_cfg.model_path = str(Path(args.checkpoint).expanduser().resolve())
    metadata = {
        "started_unix": time.time(),
        "task": args.task,
        "checkpoint": str(model_cfg.model_path),
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
        "requested_episodes": args.episodes,
        "initial_seed": args.seed,
        "config_name": args.config_name,
        "evaluator_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "env_adapter_source_sha256": hashlib.sha256(
            (repo / "rlinf/envs/robotwin/multitask_env.py").read_bytes()
        ).hexdigest(),
        "hydra_overrides": args.override,
        "chunk_length": int(model_cfg.num_action_chunks),
        "denoising_steps": int(model_cfg.num_steps),
        "task_horizon": int(cfg.env.eval.task_horizons[args.task]),
        "normalization": OmegaConf.to_container(
            model_cfg.opd_task_norm_stats, resolve=True
        ),
        "reset_filter": "unstable_scene_only; no planner or success filtering",
    }
    if pool is not None:
        from toolkits.robotwin_mopd.author_pool import NOISE_RULE, PROTOCOL_NAME

        is_author_pool = pool["protocol"]["name"] == PROTOCOL_NAME
        metadata["reset_filter"] = (
            "author-cache accepted scenes; no local online expert revalidation; no reseeding"
            if is_author_pool
            else "expert-screened fixed pool; no evaluation reseeding"
        )
        metadata["noise_rule"] = NOISE_RULE
        metadata["evaluation_protocol"] = pool["protocol"]["name"]
        if is_author_pool:
            from rlinf.envs.robotwin.runtime_profile import validate_robotwin4_runtime

            metadata["verified_runtime"] = validate_robotwin4_runtime()
            checkpoint = Path(model_cfg.model_path)
            metadata["checkpoint_files"] = {
                str(path.relative_to(checkpoint)): {
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.stat().st_size < 1024 * 1024
                    else None,
                }
                for path in sorted(checkpoint.rglob("*"))
                if path.is_file()
            }
            conversion = checkpoint / "conversion.json"
            if conversion.exists():
                metadata["checkpoint_conversion_receipt"] = json.loads(
                    conversion.read_text()
                )
            metadata["checkpoint_hash_note"] = (
                "Small files freshly hashed; large-weight digest, when present, "
                "comes from the preserved conversion receipt, not a fresh rehash."
            )
        metadata["initial_seed"] = pool_entries[0]["scene_seed"]
        metadata["evaluation_pool"] = {
            "path": str(args.evaluation_pool),
            "pool_content_sha256": pool["pool_content_sha256"],
            "file_sha256": hashlib.sha256(
                args.evaluation_pool.read_bytes()
            ).hexdigest(),
            "protocol": pool["protocol"],
            "selected_scene_seeds": [entry["scene_seed"] for entry in pool_entries],
            "selected_instructions": [entry["instruction"] for entry in pool_entries],
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
    if pool is not None:
        env_cfg.instruction_type = "unseen"
        for key, value in pool["protocol"]["screening_overrides"].items():
            OmegaConf.update(env_cfg.task_config, key, value, force_add=True)
    env = RoboTwinMultiTaskEnv(env_cfg, 1, 0, 1, None)
    next_seed = args.seed
    try:
        for episode in range(args.episodes):
            started = time.monotonic()
            fixed_episode = None if pool_entries is None else pool_entries[episode]
            if fixed_episode is not None:
                next_seed = fixed_episode["scene_seed"]
            row = {
                "slot": 0,
                "task_id": 0,
                "task": args.task,
                "suite": args.task,
                "reset_seed": next_seed,
                "round": episode,
            }
            env.set_rollout_tasks([row])
            obs, _ = env.reset(evaluation_episode=fixed_episode)
            print(f"Episode {episode}: environment reset", flush=True)
            actual_seed = env.rollout_episode_records()[0]["actual_reset_seed"]
            actual_scene_info = None
            if fixed_episode is not None:
                if pool["protocol"]["name"] == "author_cache_replay_unseen":
                    from toolkits.robotwin_mopd.author_pool import (
                        validate_replay_identity,
                        validate_scene_info,
                    )

                    validate_replay_identity(
                        fixed_episode, actual_seed, obs["task_descriptions"]
                    )
                    actual_scene_info = copy.deepcopy(env._slots[0].task.get_info())
                    metadata.setdefault("fixed_scene_checks", []).append(
                        {
                            "episode": episode,
                            "scene_seed": actual_seed,
                            "expected_episode_info": fixed_episode["episode_info"],
                            "actual_scene_info": actual_scene_info,
                            "matched": actual_scene_info
                            == fixed_episode["episode_info"],
                        }
                    )
                    # Persist a mismatch even though it must never become a policy result.
                    write_summary(args.output, metadata, episodes)
                    validate_scene_info(
                        fixed_episode["episode_info"], actual_scene_info
                    )
                if actual_seed != fixed_episode["scene_seed"]:
                    raise RuntimeError("Fixed pool seed changed during evaluation")
                if obs["task_descriptions"] != [fixed_episode["instruction"]]:
                    raise RuntimeError("Fixed pool prompt changed during reset")
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
                    if fixed_episode is not None and obs["task_descriptions"] != [
                        fixed_episode["instruction"]
                    ]:
                        raise RuntimeError("Fixed pool prompt changed during rollout")
                    chunks += 1
                    if bool((terminal | truncated).any()):
                        break
            result = env.rollout_episode_records()[0]
            if actual_scene_info is not None:
                result["actual_scene_info"] = actual_scene_info
            result.update(
                episode=episode,
                wall_seconds=time.monotonic() - started,
                inference_seconds=inference_seconds,
                simulation_seconds=simulation_seconds,
                policy_chunks=chunks,
            )
            if fixed_episode is not None:
                result.update(
                    pool_episode_index=episode,
                    evaluation_pool_sha256=pool["pool_content_sha256"],
                    instruction=fixed_episode["instruction"],
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
