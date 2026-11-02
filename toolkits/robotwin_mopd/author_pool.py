# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Import fixed unseen scenes from the guarded author cache, without simulation.

This protocol replays author-accepted scenes; it does not claim that a local
scripted expert has revalidated them. Pool construction never loads a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
from pathlib import Path

from toolkits.robotwin_mopd.feasible_pool import (
    TASKS,
    canonical_sha256,
    pool_content_sha256,
    write_pool,
)

PROTOCOL_NAME = "author_cache_replay_unseen"
AUTHOR_REVISION = "fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
CACHE_SHA256 = "12ac49e3913308901113cc8ddd9f71f0987ce8c9ebd83f03c308c759bdbeb939"
NOISE_RULE = "torch.manual_seed(actual_scene_seed) once per episode; production predict_action_batch draws subsequent noise"


def file_sha256(path: Path) -> str:
    """Return a file's byte digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(root: Path) -> str:
    """Read the checkout revision without requiring a clean asset directory."""
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def author_cache(reference: Path) -> dict:
    """Check the pinned cache and all 54 released compatibility hashes."""
    if revision(reference) != AUTHOR_REVISION:
        raise ValueError("Author reference revision mismatch")
    path = reference / "pace/configs/expert_scenes.json"
    if file_sha256(path) != CACHE_SHA256:
        raise ValueError("Author cache SHA256 mismatch")
    cache = json.loads(path.read_text())
    hashes = cache["compatible_source_files_sha256"]
    if len(hashes) != 54:
        raise ValueError("Expected all 54 author source guards")
    for name, expected in hashes.items():
        if file_sha256(reference / name) != expected:
            raise ValueError(f"Author source guard mismatch: {name}")
    subprocess.run(
        [
            "git",
            "-C",
            str(reference),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "description",
        ],
        check=True,
        capture_output=True,
    )
    return cache


def source_hashes(root: Path) -> dict:
    """Fingerprint local task/control/camera/robot/language configuration files."""
    names = {
        "envs/_base_task.py",
        "robotwin/envs/vector_env.py",
        "task_config/demo_clean.yml",
        "task_config/_camera_config.yml",
        "task_config/_embodiment_config.yml",
        "task_config/_eval_step_limit.yml",
        "description/utils/generate_episode_instructions.py",
        *[f"envs/{task}.py" for task in TASKS],
        *[f"description/task_instruction/{task}.json" for task in TASKS],
    }
    names.update(str(p.relative_to(root)) for p in (root / "envs/robot").rglob("*.py"))
    names.update(str(p.relative_to(root)) for p in (root / "envs/camera").rglob("*.py"))
    return {name: file_sha256(root / name) for name in sorted(names)}


def author_instruction(reference: Path, task: str, info: dict, seed: int) -> str:
    """Freeze one unseen draw from the pinned author generator (100 candidates)."""
    path = reference / "description/utils/generate_episode_instructions.py"
    spec = importlib.util.spec_from_file_location("author_pool_language", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = random.getstate()
    try:
        random.seed(seed)
        descriptions = module.generate_episode_descriptions(task, [info], 100)
        choices = descriptions[0]["unseen"] if descriptions else []
        if not choices:
            raise ValueError(f"No unseen prompt for {task}, scene {seed}")
        return random.Random(seed).choice(choices)
    finally:
        random.setstate(state)


def load_pool(path: Path) -> dict:
    """Reject incomplete, tampered, duplicated, or ambiguously selected pools."""
    pool = json.loads(Path(path).read_text())
    if pool.get("schema_version") != 2:
        raise ValueError("Unsupported author-pool schema")
    if pool.get("pool_content_sha256") != pool_content_sha256(pool):
        raise ValueError("Author-pool content SHA256 mismatch")
    p = pool["protocol"]
    if p["name"] != PROTOCOL_NAME or p["instruction_type"] != "unseen":
        raise ValueError("Expected author_cache_replay_unseen protocol")
    if not pool.get("complete"):
        raise ValueError("Author pool incomplete")
    if (
        p.get("online_expert_revalidation") is not False
        or p.get("allow_reseed") is not False
    ):
        raise ValueError("Author pool must forbid reseeding and local-screening claims")
    if (
        p["author_revision"] != AUTHOR_REVISION
        or p["author_cache_sha256"] != CACHE_SHA256
    ):
        raise ValueError("Author provenance mismatch")
    if p["task_names"] != list(TASKS) or set(pool["tasks"]) != set(TASKS):
        raise ValueError("Expected the four predeclared tasks")
    n = p["episodes_per_task"]
    if type(n) is not int or n < 1 or p["selection_count"] != n:
        raise ValueError("Invalid per-task count")
    if type(p["selection_start"]) is not int or p["selection_start"] < 0:
        raise ValueError("Invalid selection start")
    if type(p["author_cohort"]) is not int or p["author_cohort"] not in (0, 1, 2):
        raise ValueError("Invalid author cohort")
    for task, rows in pool["tasks"].items():
        if len(rows) != n:
            raise ValueError(f"Wrong episode count for {task}")
        seeds = [row["scene_seed"] for row in rows]
        if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != n:
            raise ValueError(f"Invalid or duplicate scene seed for {task}")
        for row in rows:
            if not isinstance(row["episode_info"], dict):
                raise ValueError("Invalid episode_info")
            if (
                not isinstance(row["instruction"], str)
                or not row["instruction"].strip()
            ):
                raise ValueError("Missing fixed instruction")
    return pool


def runtime_identity(report: dict) -> dict:
    """Compare semantic/content identity while retaining paths as provenance."""
    keys = (
        "profile",
        "sapien_version",
        "sapien_distribution_version",
        "sapien_module_sha256",
        "robotwin_revision",
        "robotwin_source_sha256",
    )
    identity = {key: report.get(key) for key in keys}
    manifest = report.get("sapien_vendor_manifest", {})
    identity["sapien_wheel_sha256"] = manifest.get("wheel_sha256")
    identity["sapien_local_patches"] = manifest.get("local_patches", [])
    return identity


def validate_pool_protocol(
    pool: dict, *, task_config: dict, task_horizons: dict, robotwin_root: Path
) -> dict:
    """Check source/config/runtime/cache provenance before a model is loaded."""
    from rlinf.envs.robotwin.runtime_profile import validate_robotwin4_runtime

    p = pool["protocol"]
    if canonical_sha256(task_config) != p["task_config_sha256"]:
        raise ValueError("Evaluation task_config differs from author replay pool")
    if {task: int(task_horizons[task]) for task in TASKS} != p["task_horizons"]:
        raise ValueError("Evaluation horizons differ from author replay pool")
    if revision(robotwin_root) != p["robotwin_revision"]:
        raise ValueError("Local RoboTwin revision mismatch")
    if source_hashes(robotwin_root) != p["source_files_sha256"]:
        raise ValueError("Local runtime source-file mismatch")
    runtime = validate_robotwin4_runtime()
    if runtime_identity(runtime) != runtime_identity(p["runtime_profile"]):
        raise ValueError("Verified runtime profile differs from author replay pool")
    reference = Path(os.environ.get("PACE_REFERENCE_ROOT", p["author_reference_root"]))
    cache = author_cache(reference)
    start, count = p["selection_start"], p["selection_count"]
    for task in TASKS:
        selected = cache["tasks"][task][str(p["author_cohort"])][start : start + count]
        actual = [
            {k: row[k] for k in ("scene_seed", "episode_info")}
            for row in pool["tasks"][task]
        ]
        if actual != selected:
            raise ValueError(
                f"Pool scenes differ from predeclared author selection: {task}"
            )
        for row in pool["tasks"][task]:
            expected = author_instruction(
                reference, task, row["episode_info"], row["scene_seed"]
            )
            if row["instruction"] != expected:
                raise ValueError(f"Author instruction mismatch: {task}")
    return runtime


def validate_replay_identity(entry: dict, actual_seed: int, prompts: list[str]) -> None:
    """Reject providers that silently reseed or replace the fixed instruction."""
    if type(actual_seed) is not int or actual_seed != entry["scene_seed"]:
        raise RuntimeError("Author fixed-pool provider reseeded the scene")
    if prompts != [entry["instruction"]]:
        raise RuntimeError("Author fixed-pool provider changed the instruction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument(
        "--purpose", choices=("development", "formal_heldout"), required=True
    )
    parser.add_argument("--config-name", default="robotwin4_mopd_pi05")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 100:
        parser.error("Selection must lie within the 100-scene author cohort")
    if args.output.exists():
        parser.error("Output exists; pools are immutable")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from rlinf.envs.robotwin.runtime_profile import validate_robotwin4_runtime

    reference = args.reference_root.resolve()
    cache = author_cache(reference)
    runtime = validate_robotwin4_runtime()
    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    os.environ.setdefault("OPD_RUN_DIR", str(args.output.resolve().parent))
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    task_config = OmegaConf.to_container(cfg.env.eval.task_config, resolve=True)
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    protocol = {
        "name": PROTOCOL_NAME,
        "instruction_type": "unseen",
        "task_names": list(TASKS),
        "episodes_per_task": args.count,
        "selection_count": args.count,
        "selection_start": args.start,
        "author_cohort": args.cohort,
        "purpose": args.purpose,
        "author_revision": AUTHOR_REVISION,
        "author_cache_sha256": CACHE_SHA256,
        "author_reference_root": str(reference),
        "author_source_files_sha256": cache["compatible_source_files_sha256"],
        "author_language_source_sha256": file_sha256(
            reference / "description/utils/generate_episode_instructions.py"
        ),
        "online_expert_revalidation": False,
        "allow_reseed": False,
        "selection_rule": "ordered author cache slice; independent of learned-teacher performance",
        "instruction_sampling": "pinned author generator; max_descriptions=100; Python random seeded by scene_seed; random.Random(scene_seed).choice(unseen)",
        "noise_rule": NOISE_RULE,
        "screening_overrides": {},
        "task_config": task_config,
        "task_config_sha256": canonical_sha256(task_config),
        "task_horizons": {
            task: int(cfg.env.eval.task_horizons[task]) for task in TASKS
        },
        "robotwin_revision": revision(root),
        "source_files_sha256": source_hashes(root),
        "runtime_profile": runtime,
    }
    tasks = {}
    for task in TASKS:
        selected = cache["tasks"][task][str(args.cohort)][
            args.start : args.start + args.count
        ]
        tasks[task] = [
            dict(
                row,
                instruction=author_instruction(
                    reference, task, row["episode_info"], row["scene_seed"]
                ),
            )
            for row in selected
        ]
    pool = {"schema_version": 2, "complete": True, "protocol": protocol, "tasks": tasks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_pool(args.output, pool)
    pool = load_pool(args.output)
    validate_pool_protocol(
        pool,
        task_config=task_config,
        task_horizons=protocol["task_horizons"],
        robotwin_root=root,
    )
    print(f"AUTHOR_POOL_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
