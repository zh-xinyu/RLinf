# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen routed teachers in the real training generate/interact path.

This diagnostic starts no actor and performs no updates or weight synchronization.
Use --student-checkpoint PATH for a single frozen policy evaluated through the
original HF mixed-batch predict method, including a specialist used as warm start.
Launch inside an owner-scoped four-GPU Ray instance, with CUDA_VISIBLE_DEVICES set
before Ray starts. For example (after sourcing the isolated runtime)::

    python -m toolkits.robotwin_mopd.teacher_train_rollout \
        --output outputs/robotwin4/diagnostics/teacher_train_t16 \
        --physical-gpus 4,5,6,7 --trajectories 16 --rounds 1

T16/T32 give four/eight mixed-task slots per rollout rank. The teacher uses the
training mode, ODE, action chunks, task normalizers, native task horizons, quota
allocator, and HF/Env communication used by training. CPU storage is an explicit
slower fallback; the default holds four frozen teachers on each rollout GPU.
"""

import argparse
import copy
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def select_rows(value: Any, indices: list[int], batch_size: int) -> Any:
    """Select a sub-batch while preserving optional observation fields."""
    if isinstance(value, dict):
        return {k: select_rows(v, indices, batch_size) for k, v in value.items()}
    if value is None:
        return None
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError("Expected a batch-first tensor/array")
        return value[indices]
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError("Expected one list item per observation")
        return [value[i] for i in indices]
    raise TypeError(f"Unsupported batch field {type(value)}")


def restore_rows(parts: list[Any], indices: list[list[int]], batch_size: int) -> Any:
    """Scatter teacher outputs back into the original mixed-task row order."""
    first = parts[0]
    if first is None:
        if any(part is not None for part in parts):
            raise ValueError("Inconsistent optional teacher output")
        return None
    if isinstance(first, dict):
        if any(set(part) != set(first) for part in parts):
            raise ValueError("Teacher result keys differ")
        return {
            key: restore_rows([part[key] for part in parts], indices, batch_size)
            for key in first
        }
    if isinstance(first, (torch.Tensor, np.ndarray)):
        shape = (batch_size, *first.shape[1:])
        output = (
            first.new_empty(shape)
            if isinstance(first, torch.Tensor)
            else np.empty(shape, dtype=first.dtype)
        )
        for part, rows in zip(parts, indices, strict=True):
            output[rows] = part
        return output
    raise TypeError(f"Unsupported teacher output {type(first)}")


def routed_predict(models: dict, env_obs: dict, mode: str = "train", activate=None):
    """Use each task's teacher exactly once and restore actions plus replay data."""
    labels = env_obs.get("task_groups")
    if labels is None or len(labels) != len(env_obs["states"]):
        raise ValueError("One task_groups label per observation is required")
    unknown = set(labels) - set(models)
    if unknown:
        raise ValueError(f"No teacher for task(s): {sorted(unknown)}")
    parts, rows = [], []
    with torch.no_grad():
        for name, model in models.items():
            selected = [i for i, label in enumerate(labels) if label == name]
            if not selected:
                continue
            if activate is not None:
                activate(name)
            actions, result = model.predict_action_batch(
                env_obs=select_rows(env_obs, selected, len(labels)), mode=mode
            )
            parts.append({"actions": actions, "result": result})
            rows.append(selected)
    merged = restore_rows(parts, rows, len(labels))
    return merged["actions"], merged["result"]


def diagnostic_placement(physical_gpus: list[int]) -> dict[str, str]:
    """Place ranks on physical IDs; RLinf discovery ignores the CUDA mask."""
    if len(physical_gpus) != 4 or sorted(set(physical_gpus)) != physical_gpus:
        raise ValueError("Use four distinct physical GPU IDs in ascending order")
    return {"actor,env,rollout": ",".join(str(index) for index in physical_gpus)}


def allowed_gpu_uuids(physical_gpus: list[int]) -> list[str]:
    """Resolve physical GPU indices without relying on CUDA logical numbering."""
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    mapping = {int(line.split(",")[0]): line.split(",")[1].strip() for line in lines}
    return [mapping[index] for index in physical_gpus]


def assert_gpu_allowed(allowed: list[str]) -> dict:
    """Fail before model/simulator loading if RLinf assigned a forbidden GPU."""
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    device_uuid = str(properties.uuid)
    normalized_uuid = device_uuid.removeprefix("GPU-").lower()
    normalized_allowed = {uuid.removeprefix("GPU-").lower() for uuid in allowed}
    if normalized_uuid not in normalized_allowed:
        raise RuntimeError(f"Refusing GPU {device_uuid}; allowed UUIDs={allowed}")
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": index,
        "gpu_uuid": device_uuid,
    }


def make_worker_classes():
    """Import distributed workers lazily so CPU routing tests start no Ray cluster."""
    from rlinf.models import get_model
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    class FrozenTeacherRolloutWorker(MultiStepRolloutWorker):
        def init_worker(self):
            start = time.perf_counter()
            self.diagnostic = assert_gpu_allowed(self.cfg.teacher_diagnostic.gpu_uuids)
            self.teachers = {}
            self._active_teacher = None
            self.predict_seconds = 0.0
            self.predict_calls = 0
            self.routed_samples = Counter()
            self.storage = self.cfg.teacher_diagnostic.teacher_storage
            self.student_checkpoint = self.cfg.teacher_diagnostic.get(
                "student_checkpoint"
            )
            self.action_min = float("inf")
            self.action_max = float("-inf")
            self.first_batch_prompts = None
            if self.student_checkpoint:
                model_cfg = copy.deepcopy(self.model_cfg)
                with open_dict(model_cfg):
                    model_cfg.model_path = self.student_checkpoint
                    model_cfg.precision = self.cfg.rollout.model.precision
                self.hf_model = get_model(model_cfg).eval().requires_grad_(False)
                self.setup_sample_params()
                self.diagnostic.update(
                    init_seconds=time.perf_counter() - start,
                    student_checkpoint=self.student_checkpoint,
                )
                self.log_info(f"Student diagnostic GPU mapping: {self.diagnostic}")
                return
            for name, expert in self.cfg.rollout.expert_models.items():
                model_cfg = copy.deepcopy(self.model_cfg)
                with open_dict(model_cfg):
                    model_cfg.model_path = expert.model_path
                    model_cfg.precision = self.cfg.rollout.model.precision
                    model_cfg.load_to_device = False
                model = get_model(model_cfg).eval().requires_grad_(False)
                parameter_bytes = sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                if not self.teachers and self.storage == "resident":
                    free, _ = torch.cuda.mem_get_info()
                    required = parameter_bytes * len(self.cfg.rollout.expert_models)
                    if required + 6 * 2**30 > free:
                        raise RuntimeError(
                            f"Four teachers need {required / 2**30:.1f} GiB plus "
                            f"6 GiB headroom, only {free / 2**30:.1f} GiB free; "
                            "use --teacher-storage cpu explicitly"
                        )
                if self.storage == "resident":
                    model.to(self.device)
                self.teachers[name] = model
                self.log_info(
                    f"Diagnostic teacher {name}: {parameter_bytes / 2**30:.2f} GiB"
                )
            self.hf_model = next(iter(self.teachers.values()))
            self.setup_sample_params()
            self.diagnostic.update(init_seconds=time.perf_counter() - start)
            self.log_info(f"Teacher diagnostic GPU mapping: {self.diagnostic}")

        def _activate(self, name):
            if self.storage != "cpu" or self._active_teacher == name:
                return
            if self._active_teacher is not None:
                self.teachers[self._active_teacher].to("cpu")
            self.teachers[name].to(self.device)
            self._active_teacher = name

        def predict(self, env_obs, mode="train"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if self.student_checkpoint:
                actions, result = super().predict(env_obs, mode=mode)
            else:
                actions, result = routed_predict(
                    self.teachers, env_obs, mode, self._activate
                )
            torch.cuda.synchronize()
            self.predict_seconds += time.perf_counter() - start
            self.predict_calls += 1
            self.routed_samples.update(env_obs["task_groups"])
            if not torch.isfinite(actions).all():
                raise FloatingPointError(
                    "Diagnostic policy produced non-finite actions"
                )
            self.action_min = min(self.action_min, float(actions.min()))
            self.action_max = max(self.action_max, float(actions.max()))
            if self.first_batch_prompts is None:
                self.first_batch_prompts = list(env_obs["task_descriptions"])
            result["expert_label_flag"] = False
            self._attach_teacher_ids(env_obs, result)
            return actions, result

        def diagnostic_metrics(self):
            return dict(
                self.diagnostic,
                rank=self._rank,
                predict_seconds=self.predict_seconds,
                predict_calls=self.predict_calls,
                routed_samples=dict(self.routed_samples),
                action_min=self.action_min,
                action_max=self.action_max,
                first_batch_prompts=self.first_batch_prompts,
                teacher_checkpoints={
                    name: str(spec.model_path)
                    for name, spec in self.cfg.rollout.expert_models.items()
                }
                if not self.student_checkpoint
                else {},
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )

    class DiagnosticEnvWorker(EnvWorker):
        def init_worker(self):
            self.diagnostic = assert_gpu_allowed(self.cfg.teacher_diagnostic.gpu_uuids)
            self.bootstrap_seconds = 0.0
            self.chunk_seconds = 0.0
            self.chunk_calls = 0
            start = time.perf_counter()
            super().init_worker()
            self.diagnostic["init_seconds"] = time.perf_counter() - start

        def _bootstrap_and_send_train(self, rollout_channel):
            start = time.perf_counter()
            result = super()._bootstrap_and_send_train(rollout_channel)
            self.bootstrap_seconds += time.perf_counter() - start
            return result

        def env_interact_step(self, chunk_actions, stage_id):
            start = time.perf_counter()
            result = super().env_interact_step(chunk_actions, stage_id)
            self.chunk_seconds += time.perf_counter() - start
            self.chunk_calls += 1
            return result

        def diagnostic_metrics(self):
            return dict(
                self.diagnostic,
                rank=self._rank,
                bootstrap_and_send_seconds=self.bootstrap_seconds,
                env_chunk_seconds=self.chunk_seconds,
                env_chunk_calls=self.chunk_calls,
            )

    return FrozenTeacherRolloutWorker, DiagnosticEnvWorker


def summarize_records(records: list[dict]) -> dict:
    """Use completed physical episode records, not padded training transitions."""
    tasks = sorted({row["task"] for row in records})
    return {
        task: {
            "episodes": sum(row["task"] == task for row in records),
            "successes": sum(
                bool(row["success_once"]) for row in records if row["task"] == task
            ),
            "success_rate": float(
                np.mean([row["success_once"] for row in records if row["task"] == task])
            ),
            "simulation_steps": sum(
                row["simulation_steps"] for row in records if row["task"] == task
            ),
        }
        for task in tasks
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical-gpus", default="4,5,6,7")
    parser.add_argument("--trajectories", type=int, choices=[16, 32], default=16)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--teacher-storage", choices=["resident", "cpu"], default="resident"
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        help="Optional single frozen policy; use the original HF mixed-batch predict path",
    )
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    physical = [int(x) for x in args.physical_gpus.split(",")]
    if len(physical) != 4 or len(set(physical)) != 4 or args.rounds < 1:
        parser.error("Exactly four distinct GPUs and positive rounds are required")
    # The owner must start Ray with the same mask. Do not change a live cluster.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.physical_gpus:
        parser.error(
            "Set CUDA_VISIBLE_DEVICES to --physical-gpus before starting Ray/driver"
        )
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    os.environ["OPD_RUN_DIR"] = str(args.output)
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    overrides = [
        *args.override,
        f"env.train.total_num_envs={args.trajectories}",
        "runner.only_eval=false",
        "runner.val_check_interval=-1",
        "runner.rollout_allocation.mode=uniform",
        f"runner.rollout_allocation.seed={args.seed}",
        "rollout.enable_offload=false",
        "rollout.pipeline_stage_num=1",
    ]
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05", overrides=overrides)
    with open_dict(cfg):
        cfg.cluster.component_placement = diagnostic_placement(physical)
        cfg.teacher_diagnostic = {
            "gpu_uuids": allowed_gpu_uuids(physical),
            "teacher_storage": args.teacher_storage,
            "student_checkpoint": str(args.student_checkpoint.expanduser().resolve())
            if args.student_checkpoint
            else None,
        }
    from rlinf.config import validate_cfg
    from rlinf.scheduler import Channel, Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.utils.rollout_allocation import RolloutAllocator

    cfg = validate_cfg(cfg)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml")
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    placement = HybridComponentPlacement(cfg, cluster)
    if any(placement.get_world_size(name) != 4 for name in ("actor", "env", "rollout")):
        raise RuntimeError("Diagnostic requires four logical ranks on the allowed GPUs")
    rollout_cls, env_cls = make_worker_classes()
    rollout = rollout_cls.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = env_cls.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    start = time.perf_counter()
    rollout_init = rollout.init_worker()
    env_init = env.init_worker()
    rollout_init.wait()
    env_init.wait()
    initialized = time.perf_counter()
    layouts = env.describe_rollout_allocation().wait()
    first = layouts[0]
    if any(
        {k: v for k, v in row.items() if k != "worker"}
        != {k: v for k, v in first.items() if k != "worker"}
        for row in layouts
    ):
        raise RuntimeError("Diagnostic worker layouts differ")
    allocator = RolloutAllocator(
        first["tasks"], 4, first["envs"], first["epochs"], cfg.runner.rollout_allocation
    )
    env_channel = Channel.create("TeacherDiagnosticEnv")
    rollout_channel = Channel.create("TeacherDiagnosticRollout")
    records, timings = [], []
    for round_index in range(args.rounds):
        env.set_rollout_allocation(allocator.plan(), policy_update_step=0).wait()
        round_start = time.perf_counter()
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
        rows = [
            row
            for worker_rows in env.get_rollout_allocation_records().wait()
            for row in worker_rows
        ]
        allocator.observe(rows)
        records.extend(rows)
        timings.append(
            {"round": round_index + 1, "seconds": time.perf_counter() - round_start}
        )
        summary = {
            "diagnostic": "frozen_student_actual_training_path"
            if args.student_checkpoint
            else "frozen_routed_teachers_actual_training_path",
            "student_checkpoint": cfg.teacher_diagnostic.student_checkpoint,
            "student_updates": 0,
            "initialization_seconds": initialized - start,
            "round_timings": timings,
            "tasks": summarize_records(records),
            "episodes": records,
            "rollout_workers": rollout.diagnostic_metrics().wait(),
            "env_workers": env.diagnostic_metrics().wait(),
        }
        (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "round": round_index + 1,
                    "tasks": summary["tasks"],
                    "timing": timings[-1],
                }
            ),
            flush=True,
        )
    # Ray lifecycle remains owned by the caller's scoped launcher.
    (args.output / "COMPLETE").write_text(
        "Frozen policy training-path diagnostic finished.\n"
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
