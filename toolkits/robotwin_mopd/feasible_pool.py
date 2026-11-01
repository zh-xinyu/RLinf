# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Build fixed expert-screened RoboTwin scenes without retaining demonstrations.

Run only after training and other GPU users have released the selected device::

    CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/feasible_pool.py \
        --output outputs/robotwin4/evaluation_pool.json --episodes-per-task 5

Acceptance follows RoboTwin's scripted-expert rule, not learned-teacher success.
The candidate list is revalidated in the current Aloha-clean configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

TASKS = ("adjust_bottle", "handover_mic", "place_container_plate", "stack_bowls_two")
PROTOCOL_NAME = "robotwin_expert_screened_unseen"
SCREENING_OVERRIDES = {
    "is_test": True,
    "eval_mode": True,
    "render_freq": 0,
    "save_data": False,
    "eval_video_log": False,
}


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON using sorted keys and compact separators."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pool_content_sha256(pool: dict) -> str:
    """Hash all fields except this hash itself, including rejection metadata."""
    return canonical_sha256(
        {key: value for key, value in pool.items() if key != "pool_content_sha256"}
    )


def load_pool(path: str | Path, *, require_complete: bool = True) -> dict:
    """Read a complete, content-verified pool; partial files are diagnostic only."""
    pool = json.loads(Path(path).read_text())
    if pool.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation-pool schema")
    if pool.get("pool_content_sha256") != pool_content_sha256(pool):
        raise ValueError("Evaluation-pool content SHA256 mismatch")
    protocol = pool["protocol"]
    if protocol["name"] != PROTOCOL_NAME or protocol["instruction_type"] != "unseen":
        raise ValueError("Expected expert-screened unseen-instruction protocol")
    if require_complete and not pool.get("complete", False):
        raise ValueError("Evaluation pool is incomplete")
    for task in protocol["task_names"]:
        rows = pool["tasks"][task]
        if require_complete and len(rows) < protocol["episodes_per_task"]:
            raise ValueError(f"Insufficient accepted scenes for {task}")
        seeds = set()
        for row in rows:
            seed = row["scene_seed"]
            if type(seed) is not int or seed in seeds:
                raise ValueError(f"Invalid or duplicate scene seed for {task}")
            seeds.add(seed)
            if not isinstance(row["episode_info"], dict):
                raise ValueError(
                    "episode_info must be the expert's placeholder dictionary"
                )
            if (
                not isinstance(row["instruction"], str)
                or not row["instruction"].strip()
            ):
                raise ValueError("Every accepted scene needs a fixed instruction")
    return pool


def validate_pool_protocol(
    pool: dict,
    *,
    task_config: dict,
    task_horizons: dict,
    robotwin_root: str | Path | None = None,
) -> None:
    """Validate resolved evaluation config and optionally pinned simulator files."""
    protocol = pool["protocol"]
    if canonical_sha256(task_config) != protocol["task_config_sha256"]:
        raise ValueError("Evaluation task_config does not match the screened pool")
    for task in protocol["task_names"]:
        if int(task_horizons[task]) != protocol["task_horizons"][task]:
            raise ValueError(f"Evaluation horizon does not match pool for {task}")
    if robotwin_root is not None:
        root = Path(robotwin_root)
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != protocol["robotwin_revision"]:
            raise ValueError("RoboTwin revision differs from the screened pool")
        for name, expected in protocol["source_files_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Screened pool source-file mismatch: {name}")


def write_pool(path: Path, pool: dict) -> None:
    """Publish progress atomically with a hash, without any actions or images."""
    pool["pool_content_sha256"] = pool_content_sha256(pool)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(pool, indent=2) + "\n")
    temporary.replace(path)


def require_idle_visible_gpus() -> None:
    """Fail before simulator initialization if a selected GPU is occupied."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = [
        tuple(field.strip() for field in line.split(","))
        for line in output.splitlines()
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = (
        visible.split(",") if visible is not None else [row[0] for row in devices]
    )
    selected = [item.strip() for item in selected]
    rows = [row for row in devices if row[0] in selected or row[1] in selected]
    if not rows or len(rows) != len(selected):
        raise RuntimeError(
            "Expected CUDA_VISIBLE_DEVICES to select known GPU indices or UUIDs"
        )
    uuids = {row[1] for row in rows}
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    occupied = [
        line for line in processes.splitlines() if line.split(",")[0].strip() in uuids
    ]
    if occupied or any(int(row[2]) > 256 for row in rows):
        raise RuntimeError(
            f"Selected GPU is occupied; run the pool builder after it is idle: {rows}, {occupied}"
        )


def screen_candidate(
    *,
    task_name: str,
    seed: int,
    episode_index: int,
    setup_args: dict,
    task_factory: Callable,
    instruction_sampler: Callable,
) -> tuple[dict | None, dict]:
    """Run one scripted expert, retain only scene metadata, and always close it."""
    task = task_factory(task_name)
    started = time.monotonic()
    audit = {"scene_seed": seed}
    expert_info = None
    initial_info = None
    try:
        scene_args = copy.deepcopy(setup_args)
        scene_args.update(now_ep_num=episode_index, seed=seed)
        task.setup_demo(**scene_args)
        try:
            initial_info = copy.deepcopy(task.get_info())
        except Exception as error:
            audit["initial_info_error"] = f"{type(error).__name__}: {error}"
        result = task.play_once()
        if not (task.plan_success and task.check_success()):
            audit["reason"] = "expert_failed"
        else:
            expert_info = copy.deepcopy(result["info"])
    except Exception as error:
        audit["reason"] = (
            "unstable" if type(error).__name__ == "UnStableError" else "expert_error"
        )
        audit["error"] = f"{type(error).__name__}: {error}"
    finally:
        task.close_env(clear_cache=True)
        audit["screening_seconds"] = time.monotonic() - started
    if expert_info is None:
        return None, audit
    # A language/configuration bug must fail loudly, not silently select another
    # scene. Use the successful expert's placeholders, never recompute after motion.
    instruction = instruction_sampler(task_name, expert_info, seed)
    row = {"scene_seed": seed, "episode_info": expert_info, "instruction": instruction}
    audit["reason"] = "accepted"
    audit["initial_info_matches_expert"] = initial_info == expert_info
    audit["initial_info_missing_keys"] = sorted(
        set(expert_info) - set(initial_info or {})
    )
    return row, audit


def sample_unseen_instruction(task_name: str, episode_info: dict, seed: int) -> str:
    """Freeze a deterministic draw from the native unseen instruction generator."""
    from description.utils.generate_episode_instructions import (
        generate_episode_descriptions,
    )

    generated = generate_episode_descriptions(task_name, [episode_info], 100, seed)
    choices = generated[0]["unseen"] if generated else []
    if not choices:
        raise ValueError(f"No unseen instruction for {task_name}, seed={seed}")
    return str(random.Random(seed).choice(choices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--max-candidates-per-task", type=int, default=150)
    parser.add_argument("--config-name", default="robotwin4_mopd_pi05")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        args.episodes_per_task < 1
        or args.max_candidates_per_task < args.episodes_per_task
    ):
        parser.error("Candidate limit must be at least the positive episodes-per-task")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Task names must be unique")
    if args.output.exists():
        parser.error(
            "Output already exists; use a new attempt path, preserving partial evidence"
        )
    require_idle_visible_gpus()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from robotwin.envs.vector_env import class_decorator

    from rlinf.envs.robotwin.multitask_env import NativeTaskSlot

    repo = Path(__file__).resolve().parents[2]
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    os.environ.setdefault("OPD_RUN_DIR", str(args.output.parent))
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    task_config = OmegaConf.to_container(cfg.env.eval.task_config, resolve=True)
    horizons = {task: int(cfg.env.eval.task_horizons[task]) for task in args.tasks}
    args.seed_file = (
        args.seed_file or repo / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    )
    seed_bytes = args.seed_file.read_bytes()
    seed_data = json.loads(seed_bytes)
    source_names = [
        "envs/_base_task.py",
        "robotwin/envs/vector_env.py",
        "task_config/demo_clean.yml",
        "task_config/_camera_config.yml",
        "task_config/_embodiment_config.yml",
        "task_config/_eval_step_limit.yml",
        "description/utils/generate_episode_instructions.py",
        *[f"envs/{task}.py" for task in args.tasks],
        *[f"description/task_instruction/{task}.json" for task in args.tasks],
    ]
    protocol = {
        "name": PROTOCOL_NAME,
        "instruction_type": "unseen",
        "task_names": args.tasks,
        "episodes_per_task": args.episodes_per_task,
        "task_horizons": horizons,
        "task_config_sha256": canonical_sha256(task_config),
        "task_config": task_config,
        "robotwin_revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_names
        },
        "planner_backend": task_config.get("planner_backend", "curobo"),
        "seed_source": str(args.seed_file.resolve()),
        "seed_source_sha256": hashlib.sha256(seed_bytes).hexdigest(),
        "candidate_order": "original eval_seeds.json order, independently revalidated",
        "max_candidates_per_task": args.max_candidates_per_task,
        "acceptance_rule": "scripted expert play_once; plan_success && check_success",
        "screening_overrides": dict(SCREENING_OVERRIDES),
        "instruction_sampling": "native generator, max_descriptions=100, local RNG seeded by scene_seed; fixed unseen prompt",
        "demonstration_actions_saved": False,
    }
    pool = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "protocol": protocol,
        "tasks": {task: [] for task in args.tasks},
        "screening_audit": {task: [] for task in args.tasks},
    }
    write_pool(args.output, pool)
    for task_name in args.tasks:
        slot = NativeTaskSlot(
            task_name, horizons[task_name], task_config, instruction_type="unseen"
        )
        setup_args = copy.deepcopy(slot.args)
        setup_args.update(protocol["screening_overrides"])
        candidates = seed_data[task_name]["success_seeds"][
            : args.max_candidates_per_task
        ]
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"Duplicate candidate seeds for {task_name}")
        for seed in candidates:
            row, audit = screen_candidate(
                task_name=task_name,
                seed=int(seed),
                episode_index=len(pool["tasks"][task_name]),
                setup_args=setup_args,
                task_factory=class_decorator,
                instruction_sampler=sample_unseen_instruction,
            )
            pool["screening_audit"][task_name].append(audit)
            if row is not None:
                pool["tasks"][task_name].append(row)
            write_pool(args.output, pool)
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "accepted": len(pool["tasks"][task_name]),
                        **audit,
                    }
                ),
                flush=True,
            )
            if len(pool["tasks"][task_name]) >= args.episodes_per_task:
                break
        if len(pool["tasks"][task_name]) < args.episodes_per_task:
            raise RuntimeError(
                f"Insufficient accepted scenes for {task_name}; partial evidence retained at {args.output}"
            )
    pool["complete"] = True
    write_pool(args.output, pool)
    load_pool(args.output)
    print(f"FEASIBLE_POOL_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
