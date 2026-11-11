# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU protocol checks and read-only training/checkpoint audits."""

from __future__ import annotations

import json
import math
import os
import re
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

from toolkits.robotwin_mopd.new4 import contracts as c


def compose_config(machine: dict, job: dict) -> DictConfig:
    """Resolve one recipe without importing Hydra in the bootstrap entrypoint."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    os.environ["OPD_RUN_DIR"] = str(Path(machine["output_root"]) / "preflight")
    with initialize_config_dir(
        version_base="1.1", config_dir=str(c.REPO / "examples/embodiment/config")
    ):
        cfg = compose(
            config_name="robotwin4_mopd_pi05", overrides=c.overrides(machine, job)
        )
    OmegaConf.resolve(cfg)
    return cfg


def validate_evaluation_pools(machine: dict, cfg) -> dict:
    """Verify held-out scenes, inference settings and disjoint train seed ranges."""
    from omegaconf import OmegaConf

    from toolkits.robotwin_mopd.author_pool import load_pool, validate_pool_protocol
    from toolkits.robotwin_mopd.new4.evaluation import pool_path

    report = {}
    for smoke, cohort, episodes, purpose in (
        (True, 0, 2, "development"),
        (False, 2, 100, "formal_heldout"),
    ):
        path = pool_path(machine, smoke=smoke)
        pool = load_pool(path, task_names=c.TASKS)
        protocol = pool["protocol"]
        if (
            protocol["author_cohort"] != cohort
            or protocol["episodes_per_task"] != episodes
            or protocol["selection_start"] != 0
            or protocol["purpose"] != purpose
        ):
            raise ValueError("Unexpected evaluation cohort/selection/purpose")
        validate_pool_protocol(
            pool,
            task_names=c.TASKS,
            task_config=OmegaConf.to_container(cfg.env.eval.task_config, resolve=True),
            task_horizons=dict(cfg.env.eval.task_horizons),
            robotwin_root=Path(os.environ["ROBOTWIN_PATH"]),
        )
        if not smoke:
            for task_id, task in enumerate(c.TASKS):
                for seed in (0, 1, 2):
                    first = c.seed_settings(seed)["env.train.seed"] + task_id * 1000003
                    # At most all1600 training slots assigned to this task, plus19 reset retries.
                    if any(
                        first <= row["scene_seed"] <= first + 50 * 32 + 19
                        for row in pool["tasks"][task]
                    ):
                        raise ValueError(
                            "Formal evaluation scenes overlap training seed range"
                        )
        report["dev2" if smoke else "formal100"] = pool["pool_content_sha256"]
    return report


def preflight(machine: dict) -> dict:
    """Check every declared job and the rebased calibration without running Ray."""
    from omegaconf import OmegaConf

    from rlinf.config import (
        validate_flow_refresh_cfg,
        validate_rollout_allocation_cfg,
        validate_suite_distillation_cfg,
    )
    from rlinf.utils.rollout_allocation import load_teacher_reference

    runtime = c.check_runtime(machine)
    tasks = [
        {"task": task, "suite": task, "horizon": c.RECIPE["task_horizons"][task]}
        for task in c.TASKS
    ]
    plans = []
    for job in c.jobs():
        cfg = compose_config(machine, job)
        validate_flow_refresh_cfg(cfg)
        validate_rollout_allocation_cfg(cfg)
        validate_suite_distillation_cfg(cfg)
        reference = Path(machine["output_root"]) / "prepared/teacher_reference.json"
        rates = load_teacher_reference(str(reference), cfg, tasks)
        reuse = c.RECIPE["methods"][job["method"]]["reuse"]
        assert list(cfg.env.train.task_names) == c.TASKS
        assert dict(cfg.env.train.task_horizons) == c.RECIPE["task_horizons"]
        assert cfg.actor.global_batch_size == 256 and cfg.env.train.total_num_envs == 32
        assert (
            cfg.runner.max_steps == 50 * reuse and cfg.runner.val_check_interval == -1
        )
        assert list(cfg.runner.save_steps) == [
            round_ * reuse for round_ in range(5, 51, 5)
        ]
        assert cfg.actor.optim.lr == 1e-5 and cfg.actor.optim.lr_warmup_steps == 4
        assert cfg.algorithm.sync_suite_loss_weights is False
        assert (
            cfg.algorithm.sync_suite_query_weights
            == c.RECIPE["methods"][job["method"]]["adaptive"]
        )
        assert (
            cfg.actor.cuda_cache_preallocate_gib
            == cfg.actor.cuda_cache_retention_gib
            == 0
        )
        assert cfg.runner.rolling_full_checkpoint is True
        assert list(rates.values()) == [0.9, 1.0, 0.9, 0.9]
        path = (
            Path(machine["output_root"])
            / "prepared/configs"
            / (c.job_id(job) + ".yaml")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, path)
        plans.append(
            {
                "job": c.job_id(job),
                "rounds": 50,
                "updates": 50 * reuse,
                "seeds": c.seed_settings(job["seed"]),
                "config": str(path),
            }
        )
    evaluation = validate_evaluation_pools(machine, cfg)
    record = {
        "evaluation_pools": evaluation,
        "status": "CPU_PREFLIGHT_PASS",
        "gpu_validation": "PENDING",
        "runtime": runtime,
        "plans": plans,
    }
    (Path(machine["output_root"]) / "prepared/preflight.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def checkpoint_audit(
    path: Path, expected_update: int, reuse: int, *, full: bool = False
) -> dict:
    """Check complete referenced byte ranges; do not load multi-GB policy tensors."""
    state = json.loads((path / "runner_state.json").read_text())
    allocation = json.loads((path / "rollout_allocation_state.json").read_text())
    weight = path / "actor/model_state_dict/full_weights.pt"
    if (
        state["optimizer_updates"] != expected_update
        or state["rollout_count"] != expected_update // reuse
        or state["env_steps"] != allocation["simulation_steps"]
        or allocation["rounds"] != expected_update // reuse
        or allocation["trajectories"] != allocation["rounds"] * 32
    ):
        raise ValueError(f"Checkpoint counters differ: {path}")
    if weight.stat().st_size < 16_000_000_000 or not zipfile.is_zipfile(weight):
        raise ValueError(f"Incomplete policy weight container: {weight}")
    result = {
        "weight_bytes": weight.stat().st_size,
        "round": state["rollout_count"],
        "updates": expected_update,
        "full_recovery": False,
    }
    if full:
        from torch.distributed.checkpoint import FileSystemReader, load_state_dict

        dcp = path / "actor/dcp_checkpoint"
        reader = FileSystemReader(str(dcp))
        metadata = reader.read_metadata()
        for key in ("model", "optimizers", "lr_schedulers", "rng"):
            if not any(
                name.startswith(f"fsdp_checkpoint.{key}.")
                for name in metadata.state_dict_metadata
            ):
                raise ValueError(f"Missing recovery state: {key}")
        extents = {}
        for entry in metadata.storage_data.values():
            relative = Path(entry.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid recovery shard path")
            extents[relative] = max(
                extents.get(relative, 0), entry.offset + entry.length
            )
        if not extents or any(
            not (dcp / name).is_file() or (dcp / name).stat().st_size < size
            for name, size in extents.items()
        ):
            raise ValueError("Missing/truncated recovery shards")
        schedule = {
            "fsdp_checkpoint": {
                "lr_schedulers": [{"last_epoch": None, "_step_count": None}]
            }
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_state_dict(schedule, reader, no_dist=True)
        if schedule["fsdp_checkpoint"]["lr_schedulers"][0] != {
            "last_epoch": expected_update,
            "_step_count": expected_update + 1,
        } or state.get("weights_only", True):
            raise ValueError("Recovery scheduler does not match completed updates")
        result["full_recovery"] = True
    return result


def audit_run(
    directory: Path, reuse: int, *, updates: int | None = None, start_update: int = 0
) -> dict:
    """Verify actual update, query, episode and recovery records after a run."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(directory / "resolved_config.yaml")
    total = cfg.runner.max_steps if updates is None else updates
    rows = [
        json.loads(line)
        for line in (directory / "optimizer_updates.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(line)
        for line in (directory / "rollout_episodes.jsonl").read_text().splitlines()
    ]
    if (directory / "status").read_text().strip() != "COMPLETE":
        raise ValueError("Training process did not finish successfully")
    if [row["update_step"] for row in rows] != list(range(start_update + 1, total + 1)):
        raise ValueError("Missing/duplicated optimizer updates")
    counts = Counter(row["round"] for row in episodes)
    if counts != dict.fromkeys(
        range(start_update // reuse + 1, total // reuse + 1), 32
    ):
        raise ValueError("Physical rollout rounds do not contain exactly32 episodes")
    quota = {}
    for row in rows:
        step, round_ = row["update_step"], row["rollout_count"]
        if round_ != (step - 1) // reuse + 1:
            raise ValueError("Rollout/update counter mismatch")
        for key in ("train/actor/velocity_mse", "train/actor/grad_norm"):
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite {key}")
        if not math.isclose(
            row["train/actor/lr"], 1e-5 * min((step - 1) / 4, 1), abs_tol=1e-11
        ):
            raise ValueError("Unexpected optimizer-update LR schedule")
        if f"allocation/quota/{c.TASKS[0]}" in row:
            quota[round_] = [row[f"allocation/quota/{task}"] for task in c.TASKS]
    for row in rows:
        q = quota[row["rollout_count"]]
        valid = [row[f"train/actor/valid_velocity_elements/{task}"] for task in c.TASKS]
        if sum(q) != 32 or any(
            not math.isclose(n / sum(valid), target / 32, abs_tol=1e-7)
            for n, target in zip(valid, q)
        ):
            raise ValueError("Actual query shares do not match physical quotas")
    for row in episodes:
        if (
            row["horizon"] != c.RECIPE["task_horizons"][row["task"]]
            or row["policy_update_step"] != (row["round"] - 1) * reuse
        ):
            raise ValueError("Episode horizon or generating-policy counter differs")
    root = directory / cfg.runner.logger.experiment_name / "checkpoints"
    final = root / f"global_step_{total}"
    checkpoint = checkpoint_audit(final, total, reuse, full=True)
    raw = (directory / "driver.log").read_text(errors="replace")
    match = re.search(r"/runs/([a-z0-9]+)", raw)
    return {
        "status": "PASS",
        "completed_round": total // reuse,
        "completed_update": total,
        "new_trajectories": len(episodes),
        "new_simulation_steps": sum(row["simulation_steps"] for row in episodes),
        "wandb_run_id": match.group(1) if match else None,
        "final_checkpoint": str(final),
        "checkpoint": checkpoint,
    }
