# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""No-learning specialist diagnostic using the real production train collector."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from rlinf.scheduler.cluster.config import ClusterConfig
from toolkits.libero_gr00t.teacher_reference import (
    HORIZONS,
    SUITES,
    read_suite,
    validate_contract,
    verify_capture,
)


def select_reference(folder: Path, cfg, tasks: list[int], trials: list[int]) -> dict:
    """Require content-bound native teacher episodes for every requested pair."""
    validate_contract(cfg)
    for values, limit, name in ((tasks, 10, "tasks"), (trials, 50, "trials")):
        if (
            not values
            or len(set(values)) != len(values)
            or any(
                type(value) is not int or value not in range(limit) for value in values
            )
        ):
            raise ValueError(f"Invalid unique {name} indices")
    manifest = verify_capture(folder)
    protocol = manifest["calibration_protocol"]
    if (
        protocol["noise_seed"] != "seed + local_task_id * 100 + trial"
        or protocol["environment_seed"] != manifest["seed"]
        or not set(trials).issubset(protocol["initial_state_indices"])
    ):
        raise ValueError("Reference does not support the requested reset/noise pairing")
    selected = []
    for suite in SUITES:
        if (
            Path(cfg.rollout.expert_models[suite].model_path).resolve()
            != Path(manifest["teacher_checkpoints"][suite]).resolve()
        ):
            raise ValueError("Teacher checkpoint differs from reference")
        selected.extend(
            row
            for row in read_suite(folder, manifest, suite)
            if row["task_id"] in tasks and row["trial"] in trials
        )
    if len(selected) != 4 * len(tasks) * len(trials):
        raise ValueError("Reference lacks exact selected episode coverage")
    return {
        "seed": manifest["seed"],
        "episodes": selected,
        "manifest": manifest,
        "capture": str(folder),
    }


def configure_probe(
    cfg,
    output: Path,
    python: Path,
    gpus: list[int],
    reference: dict,
    tasks: list[int],
    trials: list[int],
    memory_gib: float,
    capture_chunks: list[int],
    runtime_vars: dict,
):
    """Copy a training recipe; keep all changes confined to this diagnostic."""
    if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
        raise ValueError("Four distinct physical GPUs are required")
    if not 8 <= memory_gib <= 10:
        raise ValueError("memory-gib must be in [8, 10]")
    if len(set(capture_chunks)) != len(capture_chunks) or any(
        type(chunk) is not int or chunk < 1 or chunk >= 65 for chunk in capture_chunks
    ):
        raise ValueError("Capture chunks must be unique noninitial indices 1..64")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        cfg.runner.only_eval = False
        cfg.runner.val_check_interval = -1
        cfg.runner.resume_dir = None
        cfg.runner.ckpt_path = None
        cfg.runner.use_training_pipeline = False
        cfg.runner.overlap_env_bootstrap = False
        cfg.runner.per_worker_log_path = str(output / "worker_logs")
        cfg.runner.logger.log_path = str(output)
        cfg.runner.rollout_allocation.enabled = True
        cfg.env.train.total_num_envs = 4
        cfg.env.train.rollout_epoch = 1
        cfg.env.train.seed = reference["seed"]
        cfg.env.train.max_episode_steps = 520
        cfg.env.train.max_steps_per_rollout_epoch = 520
        cfg.env.train.video_cfg.save_video = False
        cfg.env.train.enable_offload = False
        cfg.rollout.pipeline_stage_num = 1
        cfg.rollout.enable_cuda_graph = False
        cfg.rollout.enable_torch_compile = False
        cfg.rollout.collect_prev_infos = False
        cfg.rollout.seed = reference["seed"]
        cfg.actor.global_batch_size = 16
        cfg.actor.micro_batch_size = 1
        cfg.cluster.component_placement = {
            "actor,env,rollout": {
                "node_group": "teacher_probe",
                "placement": ",".join(map(str, sorted(gpus))),
            }
        }
        cfg.cluster.node_groups = [
            {
                "label": "teacher_probe",
                "node_ranks": "0",
                "env_configs": [
                    {
                        "node_ranks": "0",
                        "python_interpreter_path": str(python),
                        "env_vars": [
                            {key: value} for key, value in runtime_vars.items()
                        ],
                    }
                ],
            }
        ]
        cfg.teacher_training_probe = {
            "tasks": tasks,
            "trials": trials,
            "seed": reference["seed"],
            "output": str(output),
            "memory_gib": memory_gib,
            "capture_chunks": capture_chunks,
        }
    ClusterConfig.from_dict_cfg(cfg.cluster)
    return cfg


def rank_config(cfg, rank: int):
    """Select one native teacher and suite per rank without changing core workers."""
    if type(rank) is not int or rank not in range(4):
        raise ValueError("Probe requires exactly four ranks")
    suite = SUITES[rank]
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(result):
        model = OmegaConf.merge(result.actor.model, result.rollout.expert_models[suite])
        model.opd_canonical_fixed_padding = False
        model.inference_autocast_dtype = "bf16"
        result.actor.model = model
        result.rollout.model = OmegaConf.create(
            OmegaConf.to_container(model, resolve=True)
        )
        result.env.train.task_suite_names = [suite]
        result.env.train.task_suite_horizons = {suite: HORIZONS[suite]}
        result.env.train.task_id_filter = None
        result.env.train.task_sampling_strategy = "suite_quota"
    return result


def make_plan(
    layouts: list[dict], task: int, trial: int, round_number: int, seed: int
) -> list[dict]:
    """Bind each slot to the environment's actual catalog and reference seeds."""
    if len(layouts) != 4 or {layout["worker"] for layout in layouts} != set(range(4)):
        raise ValueError("Expected four distinct environment ranks")
    plan = []
    for layout in sorted(layouts, key=lambda item: item["worker"]):
        rank = layout["worker"]
        if layout["envs"] != 1 or layout["epochs"] != 1:
            raise ValueError("Probe requires one training environment per rank")
        matches = [
            row
            for row in layout["tasks"]
            if row["task_id"] == task and row["suite"] == SUITES[rank]
        ]
        if len(matches) != 1:
            raise ValueError("Requested task is missing or ambiguous in native catalog")
        plan.append(
            {
                **matches[0],
                "worker": rank,
                "slot": 0,
                "epoch": 0,
                "round": round_number,
                "occurrence": trial,
                "reset_seed": seed,
                "policy_seed": seed + task * 100 + trial,
            }
        )
    return plan


def audit_episodes(
    records: list[dict], reference: dict, tasks: list[int], trials: list[int]
) -> dict:
    """Report SR only for exact complete native-horizon episode coverage."""
    expected = {
        (suite, task, trial) for suite in SUITES for task in tasks for trial in trials
    }
    seen = set()
    comparisons = []
    lookup = {
        (row["suite"], row["task_id"], row["trial"]): row
        for row in reference["episodes"]
    }
    for row in records:
        key = (row["suite"], row["task_id"], row["trial_id"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected training episode")
        seen.add(key)
        horizon = HORIZONS[row["suite"]]
        if (
            row["horizon"] != horizon
            or type(row["success_once"]) is not bool
            or type(row["simulation_steps"]) is not int
            or not 1 <= row["simulation_steps"] <= horizon
            or (not row["success_once"] and row["simulation_steps"] != horizon)
            or row["reset_seed"] != reference["seed"]
            or row["policy_seed"]
            != reference["seed"] + row["task_id"] * 100 + row["trial_id"]
        ):
            raise ValueError("Incomplete episode or mismatched horizon/seed")
        other = lookup[key]
        comparisons.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "trial_id": key[2],
                "training_success": row["success_once"],
                "reference_success": other["success_once"],
                "training_steps": row["simulation_steps"],
                "reference_steps": other["steps"],
            }
        )
    if seen != expected:
        raise ValueError("Missing training episodes; withhold success rates")
    return {
        "complete": True,
        "episodes": len(records),
        "optimizer_updates": 0,
        "training_successes": sum(row["success_once"] for row in records),
        "reference_successes": sum(row["reference_success"] for row in comparisons),
        "outcome_disagreements": sum(
            row["training_success"] != row["reference_success"] for row in comparisons
        ),
        "pairs": comparisons,
        "scope": "Actual env.train / HF.generate; one native specialist per rank. No student, actor, learning, or mixed-suite gradient-routing validation.",
    }


def run_driver(config_path: Path) -> None:
    """Launch real train workers, collect complete rounds, and audit all outcomes."""
    import torch.multiprocessing as mp

    from rlinf.scheduler import Channel, Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from toolkits.libero_gr00t.teacher_training_workers import (
        TeacherProbeEnvWorker,
        TeacherProbeRolloutWorker,
    )

    mp.set_start_method("spawn", force=True)
    cfg = OmegaConf.load(config_path)
    probe = cfg.teacher_training_probe
    output = Path(probe.output)
    reference = json.loads((output / "selected_reference.json").read_text())
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = TeacherProbeRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = TeacherProbeEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    rollout_init = rollout.init_worker()
    env_init = env.init_worker()
    rollout_init.wait()
    env_init.wait()
    layouts = env.describe_rollout_allocation().wait()
    before = env.get_physics_step_counts().wait()
    env_channel = Channel.create("TeacherProbeEnv")
    rollout_channel = Channel.create("TeacherProbeRollout")
    records, audits, captures = [], [], []
    round_number = 0
    for trial in probe.trials:
        for task in probe.tasks:
            round_number += 1
            plan = make_plan(layouts, task, trial, round_number, probe.seed)
            env.set_rollout_allocation(plan, 0).wait()
            rollout.set_probe_episode(task, trial).wait()
            env_handle = env.interact(
                input_channel=env_channel,
                rollout_channel=rollout_channel,
                reward_channel=None,
                actor_channel=None,
            )
            rollout_handle = rollout.generate(
                input_channel=rollout_channel, output_channel=env_channel
            )
            rollout_handle.wait()
            env_handle.wait()
            batch = [
                row
                for rank_rows in env.get_rollout_allocation_records().wait()
                for row in rank_rows
            ]
            if len(batch) != 4:
                raise ValueError(
                    "A completed training round must contain exactly four episodes"
                )
            records.extend(batch)
            with (output / "episodes.jsonl").open("a") as stream:
                for row in batch:
                    stream.write(json.dumps(row) + "\n")
            rollout_audits = rollout.get_probe_audit().wait()
            audits.append(
                {
                    "round": round_number,
                    "rollout": rollout_audits,
                    "env": env.get_probe_action_audit().wait(),
                }
            )
            by_suite = {row["suite"]: row for row in batch}
            for worker_audit in rollout_audits:
                episode = by_suite[worker_audit["suite"]]
                for capture in worker_audit["captures"]:
                    valid = min(
                        8,
                        max(
                            0, episode["simulation_steps"] - capture["chunk_index"] * 8
                        ),
                    )
                    captures.append(
                        {
                            **capture,
                            "episode_simulation_steps": episode["simulation_steps"],
                            "valid_controls": [index < valid for index in range(8)],
                        }
                    )
            (output / "capture_manifest.json").write_text(
                json.dumps(captures, indent=2) + "\n"
            )
    after = env.get_physics_step_counts().wait()
    deltas = [
        {key: last[key] - first[key] for key in last}
        for first, last in zip(before, after, strict=True)
    ]
    if (
        sum(row["policy_simulation_steps"] for row in deltas)
        != sum(row["simulation_steps"] for row in records)
        or sum(row["settling_simulation_steps"] for row in deltas) != len(records) * 10
        or sum(row["reset_episodes"] for row in deltas) != len(records)
    ):
        raise ValueError(
            "Actual train simulator calls do not match episode/settling counts"
        )
    report = audit_episodes(records, reference, list(probe.tasks), list(probe.trials))
    report.update(physics_deltas=deltas, worker_audits=audits)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run_driver(parser.parse_args().config)
