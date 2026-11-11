# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Immutable science recipe and relocatable resources for the new four tasks."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from toolkits.robotwin_mopd.formal.experiment import (
    digest,
)
from toolkits.robotwin_mopd.formal.experiment import (
    environment as runtime_environment,
)
from toolkits.robotwin_mopd.formal.experiment import (
    read_machine as read_runtime_machine,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECIPE = json.loads((HERE / "recipe.json").read_text())
TASKS = RECIPE["tasks"]


def job_id(job: dict) -> str:
    """Name a scientific arm and training repeat without conflating attempts."""
    return f"{job['method']}_s{job['seed']}"


def jobs() -> list[dict]:
    """Return the fixed eight collaborator jobs, never infer missing repeats."""
    result = [dict(row) for row in RECIPE["jobs"]]
    if len(result) != 8 or len({job_id(row) for row in result}) != 8:
        raise ValueError("The collaborator recipe must contain eight unique jobs")
    return result


def select_job(name: str) -> dict:
    """Resolve only an explicitly declared collaborator job."""
    for job in jobs():
        if job_id(job) == name:
            return job
    raise ValueError(f"Unknown collaborator job: {name}")


def read_machine(path: Path) -> dict:
    """Require explicit target paths and four-rank-compatible hardware batching."""
    machine = read_runtime_machine(path)
    if machine["profile"] != "blackwell-cu128-candidate":
        raise ValueError("This release requires the verified Blackwell profile")
    micro = machine.get("actor_micro_batch_size", 16)
    if type(micro) is not int or micro < 1 or 256 % (4 * micro):
        raise ValueError("microbatch must divide B256 over four actor ranks")
    for key, default in (("ray_port", 6451), ("dashboard_port", 8331)):
        value = machine.setdefault(key, default)
        if type(value) is not int or not 1024 <= value <= 65535:
            raise ValueError(f"Invalid {key}")
    if machine["ray_port"] == machine["dashboard_port"]:
        raise ValueError("Ray and dashboard ports must differ")
    for key, default, minimum in (
        ("num_cpus", 32, 4),
        ("minimum_gpu_free_mib", 60000, 0),
    ):
        value = machine.setdefault(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {key}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}", machine.get("node_label", "collab")):
        raise ValueError("node_label must be a short filename-safe label")
    cpus = machine.setdefault(
        "evaluation_cpus_per_task", max(2, machine["num_cpus"] // 4)
    )
    if type(cpus) is not int or cpus < 2 or cpus * 4 > machine["num_cpus"]:
        raise ValueError("Four evaluation tasks must fit the allocated CPU budget")
    fraction = machine.setdefault("evaluation_gpu_memory_fraction", 0.32)
    if isinstance(fraction, bool) or not 0 < float(fraction) <= 0.5:
        raise ValueError("Invalid evaluation_gpu_memory_fraction")
    minimum = machine.setdefault("evaluation_minimum_gpu_free_mib", 30000)
    if type(minimum) is not int or minimum < 0:
        raise ValueError("Invalid evaluation_minimum_gpu_free_mib")
    ports = [
        machine[key] + offset
        for key in ("ray_port", "dashboard_port")
        for offset in (0, 10, 11, 12, 13)
    ]
    workers = set(range(machine["ray_port"] + 20000, machine["ray_port"] + 20512))
    if (
        max(ports) > 65535
        or max(workers) > 65535
        or workers.intersection(ports)
        or len(set(ports)) != len(ports)
    ):
        raise ValueError("Training and evaluation Ray/dashboard ports must be distinct")
    return machine


def environment(machine: dict) -> dict:
    """Activate the pinned runtime, isolating logging and owned Ray ports."""
    env = runtime_environment(machine)
    env.update(
        OPD_CONTROL_RAY_PORT=str(machine["ray_port"]),
        OPD_CONTROL_DASHBOARD_PORT=str(machine["dashboard_port"]),
        RAY_ADDRESS=f"127.0.0.1:{machine['ray_port']}",
        GPU_FREE_MB=str(machine.get("minimum_gpu_free_mib", 60000)),
    )
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RESUME_FROM", "WANDB_NAME"):
        env.pop(key, None)
    env.pop("OPD_RAY_PROCESS_OWNER", None)
    env["WANDB_PROJECT"] = "FORMAL_robotwin_pi05_NEW"
    return env


def seed_settings(seed: int) -> dict[str, int]:
    """Use the same small-budget task-local seed partition as current runs."""
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("training seed must be 0, 1 or 2")
    return {
        "actor.seed": seed,
        "env.train.seed": 42 + 10000 * seed,
        "runner.rollout_allocation.seed": 42 + 10000 * seed,
        "+rollout.seed": 1000000000 + 10000 * seed,
    }


def run_name(machine: dict, job: dict) -> str:
    spec = RECIPE["methods"][job["method"]]
    allocation = "adaptive" if spec["adaptive"] else "uniform"
    return (
        f"new4_sft400_{allocation}_r{spec['reuse']}_t32_b256_rounds50"
        f"_seed{job['seed']}_{machine.get('node_label', 'collab')}"
    )


def mapping(values: dict) -> str:
    """Quote path values while retaining valid Hydra dictionary keys."""
    return (
        "{"
        + ",".join(f"{key}:{json.dumps(value)}" for key, value in values.items())
        + "}"
    )


def model_overrides(machine: dict) -> list[str]:
    """Replace old task routes and bind the verified per-task normalizers."""
    root = Path(machine["resource_root"])
    experts = (
        "{"
        + ",".join(
            task + ":{model_path:" + json.dumps(str(root / "teachers" / task)) + "}"
            for task in TASKS
        )
        + "}"
    )
    return [
        "~rollout.expert_models",
        "+rollout.expert_models=" + experts,
        "~actor.model.opd_task_norm_stats",
        "+actor.model.opd_task_norm_stats="
        + mapping(
            {
                task: str(root / "student/assets" / task / "norm_stats.json")
                for task in TASKS
            }
        ),
        "actor.model.model_path=" + json.dumps(str(root / "student")),
    ]


def overrides(machine: dict, job: dict) -> list[str]:
    """Vary only declared allocation, R and seed; B remains per-update B256."""
    spec = RECIPE["methods"][job["method"]]
    reuse, adaptive = spec["reuse"], spec["adaptive"]
    reference = Path(machine["output_root"]) / "prepared/teacher_reference.json"
    values = [
        *RECIPE["common_overrides"],
        *model_overrides(machine),
        "actor.global_batch_size=256",
        f"actor.micro_batch_size={machine.get('actor_micro_batch_size', 16)}",
        f"algorithm.flow_refresh_updates={reuse}",
        f"+algorithm.sync_suite_query_weights={str(adaptive).lower()}",
        f"runner.max_steps={50 * reuse}",
        "runner.save_interval=-1",
        "+runner.save_steps=" + json.dumps([r * reuse for r in RECIPE["save_rounds"]]),
        f"runner.rollout_allocation.mode={'adaptive' if adaptive else 'uniform'}",
        "runner.rollout_allocation.teacher_reference_file="
        + (json.dumps(str(reference)) if adaptive else "null"),
        *[f"{key}={value}" for key, value in seed_settings(job["seed"]).items()],
        "runner.logger.wandb_group="
        + (
            f"new4_sft400_reuse_r4_rounds50_seed{job['seed']}"
            if reuse == 4
            else f"new4_sft400_main_rounds50_seed{job['seed']}"
        ),
    ]
    for phase in ("train", "eval"):
        values += [
            f"env.{phase}.task_names=[" + ",".join(TASKS) + "]",
            f"~env.{phase}.task_horizons",
            f"+env.{phase}.task_horizons=" + mapping(RECIPE["task_horizons"]),
        ]
    return values


def train_command(
    machine: dict,
    job: dict,
    *,
    dry_run: bool = False,
    smoke: bool = False,
    resume_dir: Path | None = None,
) -> list[str]:
    name = run_name(machine, job)
    output = Path(machine["output_root"])
    values = overrides(machine, job)
    if smoke:
        name = (
            "new4_deployment_r2" if resume_dir is None else "new4_deployment_resume_r2"
        )
        values += [
            "runner.max_steps=" + ("4" if resume_dir is None else "6"),
            "runner.save_steps=[2,4]"
            if resume_dir is None
            else "runner.save_steps=[6]",
            "runner.logger.logger_backends=[tensorboard]",
        ]
        if resume_dir is not None:
            values += [
                "runner.resume_dir=" + json.dumps(str(resume_dir)),
                "+runner.resume_mode=adaptive_round_boundary",
            ]
    return [
        str(Path(machine["runtime_root"]) / "venv/bin/python"),
        "-m",
        "toolkits.robotwin_mopd.run",
        "--run-id",
        name,
        "--output-root",
        str(output / ("smoke" if smoke else "train")),
        "--num-gpus",
        "4",
        "--num-cpus",
        str(machine.get("num_cpus", 32)),
        *(["--dry-run"] if dry_run else []),
        *values,
    ]


def write_immutable(path: Path, record: dict) -> None:
    """Allow idempotent preparation, refusing to change an existing contract."""
    text = json.dumps(record, indent=2) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"Existing contract changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def check_runtime(machine: dict) -> dict:
    """Verify the installed payload; no old task evaluation scenes are rebound."""
    import ctypes

    import torch

    from rlinf.envs.robotwin.runtime_profile import validate_robotwin4_runtime
    from toolkits.robotwin_mopd.author_pool import runtime_identity
    from toolkits.robotwin_mopd.formal.blackwell import NCCL_SHA

    if torch.__version__.split("+")[0] != "2.7.1" or torch.version.cuda != "12.8":
        raise ValueError("Expected Torch2.7.1/cu128")
    actual = validate_robotwin4_runtime()
    expected = json.loads(
        (HERE.parent / "formal/blackwell_runtime_identity.json").read_text()
    )
    if runtime_identity(actual) != expected:
        raise ValueError(
            "Simulator content differs from the verified Blackwell profile"
        )
    library = Path(os.environ["OPD_ROBOTWIN_NCCL_LIBRARY"])
    version = ctypes.c_int()
    if (
        digest(library) != NCCL_SHA
        or ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(version))
        or version.value != 23102
    ):
        raise ValueError("Expected the pinned NCCL2.31.2 payload and loaded library")
    return {
        "runtime_identity": expected,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def prepare(machine: dict) -> None:
    """Hash transferred files, then relocate only protocol-equivalent paths."""
    manifest = json.loads((HERE / "resources.json").read_text())
    root = Path(machine["resource_root"])
    snapshots = {}
    for name, record in manifest["files"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError(f"Resource identity mismatch: {name}")
        stat = path.stat()
        snapshots[name] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": record["sha256"],
        }
    output = Path(machine["output_root"]) / "prepared"
    tokenizer = output / "openpi-data/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    source = root / "openpi-data/big_vision/paligemma_tokenizer.model"
    if tokenizer.exists():
        if tokenizer.is_symlink() or digest(tokenizer) != digest(source):
            raise ValueError("Prepared tokenizer differs")
    else:
        shutil.copyfile(source, tokenizer)
    reference = json.loads((HERE / "teacher_reference_source.json").read_text())
    reference["teacher_checkpoints"] = {
        task: str((root / "teachers" / task).resolve()) for task in TASKS
    }
    reference["training_signature"]["model"]["opd_task_norm_stats"] = {
        task: str(root / "student/assets" / task / "norm_stats.json") for task in TASKS
    }
    write_immutable(output / "teacher_reference.json", reference)
    write_immutable(
        output / "resource_verification.json",
        {
            "manifest_sha256": digest(HERE / "resources.json"),
            "files": snapshots,
        },
    )
    for name in ("formal100.json", "dev2.json"):
        write_immutable(
            output / "evaluation" / name, json.loads((HERE / name).read_text())
        )


def freeze(machine: dict) -> None:
    """Lock all eight paired runs to one science and hardware-batching recipe."""
    write_immutable(
        Path(machine["output_root"]) / "prepared/training_contract.json",
        {
            "recipe_sha256": digest(HERE / "recipe.json"),
            "resources_sha256": digest(HERE / "resources.json"),
            "machine": machine,
            "seed_settings": {str(seed): seed_settings(seed) for seed in (1, 2)},
            "evaluation": {
                "pool_content_sha256": json.loads(
                    (HERE / "formal100.json").read_text()
                )["pool_content_sha256"],
                "rounds": RECIPE["eval_rounds"],
                "episodes_per_task": 100,
            },
        },
    )
