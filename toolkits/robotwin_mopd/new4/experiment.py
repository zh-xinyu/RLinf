# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Portable, train-only collaborator queue: eight new4 SFT400 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from toolkits.robotwin_mopd.new4 import contracts as c
from toolkits.robotwin_mopd.new4.validate import audit_run, checkpoint_audit, preflight


def job_directory(machine: dict, job: dict) -> Path:
    """Keep each method/seed training run in its own directory."""
    return Path(machine["output_root"]) / "train" / c.run_name(machine, job)


def checkpoint_path(machine: dict, job: dict, round_: int) -> Path:
    """Map physical rounds to the runner's optimizer-update save layout."""
    name = c.run_name(machine, job)
    reuse = c.RECIPE["methods"][job["method"]]["reuse"]
    return (
        job_directory(machine, job)
        / name
        / "checkpoints"
        / f"global_step_{round_ * reuse}"
    )


def source_identity() -> dict:
    """Bind deployment evidence to runtime-affecting source, allowing doc edits."""
    suffixes = {".py", ".json", ".yaml", ".yml", ".sh", ".txt"}
    files = {
        path
        for root in (
            c.REPO / "rlinf",
            c.HERE.parent,
            c.REPO / "examples/embodiment/config",
        )
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes
    }
    files.update(
        {
            c.REPO / "toolkits/robocasa_mopd/owned_ray.py",
            c.REPO / "examples/embodiment/train_embodied_agent.py",
        }
    )
    return {str(path.relative_to(c.REPO)): c.digest(path) for path in sorted(files)}


def validate_deployment(machine: dict, env: dict) -> dict:
    """Require a real teacher round and full-model FSDP save/load/update cycle."""
    from toolkits.robotwin_mopd.new4.teacher_smoke import run as teacher_smoke

    tiny = Path(machine["output_root"]) / "smoke/tiny_fsdp"
    if tiny.exists():
        raise FileExistsError(tiny)
    subprocess.run(
        [
            str(Path(machine["runtime_root"]) / "venv/bin/python"),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            "toolkits.robotwin_mopd.formal.collective_smoke",
            "--output",
            str(tiny),
        ],
        env=env,
        cwd=c.REPO,
        check=True,
    )
    for rank in range(4):
        report = json.loads((tiny / f"rank{rank}.json").read_text())
        if (
            report["status"] != "PASS"
            or report.get("fresh_optimizer_restore") != "PASS"
        ):
            raise ValueError("Four-rank FSDP/fresh-optimizer recovery smoke failed")
    teacher_smoke(machine, env)
    job = {"method": "uniform_r2", "seed": 2}
    subprocess.run(
        c.train_command(machine, job, smoke=True), env=env, cwd=c.REPO, check=True
    )
    root = Path(machine["output_root"]) / "smoke"
    initial = audit_run(root / "new4_deployment_r2", 2, updates=4)
    checkpoint = Path(initial["final_checkpoint"])
    subprocess.run(
        c.train_command(machine, job, smoke=True, resume_dir=checkpoint),
        env=env,
        cwd=c.REPO,
        check=True,
    )
    resumed = root / "new4_deployment_resume_r2"
    continuation = audit_run(resumed, 2, updates=6, start_update=4)
    metadata = json.loads((resumed / "resume_metadata.json").read_text())
    if (
        metadata["mode"] != "adaptive_round_boundary"
        or not metadata["optimizer_moments_restored"]
        or len(metadata["actor_states"]) != 4
        or any(
            row["optimizer_updates"] != 4
            or row["scheduler_last_epoch"] != 4
            or row["optimizer_state_entries"] < 1
            for row in metadata["actor_states"]
        )
    ):
        raise ValueError("Full-model optimizer/scheduler recovery was not demonstrated")
    result = {
        "status": "TARGET_GPU_VALIDATION_PASS",
        "teacher_episodes_per_task": 8,
        "first_training": initial,
        "full_state_continuation": continuation,
        "source_identity": source_identity(),
        "microbatch": machine.get("actor_micro_batch_size", 16),
        "runtime": c.check_runtime(machine),
        "validated_unix": time.time(),
        "gpu_hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + env["CUDA_VISIBLE_DEVICES"],
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines(),
    }
    c.write_immutable(
        Path(machine["output_root"]) / "prepared/gpu_validation.json", result
    )
    return result


def require_deployment(machine: dict) -> None:
    proof = json.loads(
        (Path(machine["output_root"]) / "prepared/gpu_validation.json").read_text()
    )
    if (
        proof["status"] != "TARGET_GPU_VALIDATION_PASS"
        or proof["source_identity"] != source_identity()
        or proof["microbatch"] != machine.get("actor_micro_batch_size", 16)
        or proof["runtime"] != c.check_runtime(machine)
    ):
        raise ValueError(
            "Missing deployment validation, or source/runtime/microbatch changed"
        )
    receipt = json.loads(
        (
            Path(machine["output_root"]) / "prepared/resource_verification.json"
        ).read_text()
    )
    if receipt["manifest_sha256"] != c.digest(c.HERE / "resources.json"):
        raise ValueError("Verified resource manifest changed")
    for name, item in receipt["files"].items():
        path = Path(machine["resource_root"]) / name
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or path.stat().st_mtime_ns != item["mtime_ns"]
        ):
            raise ValueError(f"Verified resource changed; revalidate: {name}")


def execute_job(machine: dict, job: dict, env: dict) -> dict:
    """Skip only a fully audited complete run; never overwrite failed attempts."""
    directory = job_directory(machine, job)
    reuse = c.RECIPE["methods"][job["method"]]["reuse"]
    if directory.exists():
        return audit_run(directory, reuse)
    if shutil.disk_usage(Path(machine["output_root"])).free < 300 * 1024**3:
        raise RuntimeError(
            "Need300GiB free before a new50-round run; preserve existing checkpoints"
        )
    tags = f"formal,new4,sft400,{job['method']},seed{job['seed']}"
    subprocess.run(
        c.train_command(machine, job),
        env={**env, "WANDB_TAGS": tags},
        cwd=c.REPO,
        check=True,
    )
    return audit_run(directory, reuse)


def summarize(machine: dict) -> dict:
    """Keep all saves and raw run identities; defer every formal SR measurement."""
    rows, completed = [], []
    for job in c.jobs():
        directory = job_directory(machine, job)
        reuse = c.RECIPE["methods"][job["method"]]["reuse"]
        evidence = None
        try:
            evidence = audit_run(directory, reuse)
            completed.append(c.job_id(job))
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            pass
        for round_ in c.RECIPE["save_rounds"]:
            path = checkpoint_path(machine, job, round_)
            ready = False
            try:
                checkpoint_audit(path, round_ * reuse, reuse, full=round_ == 50)
                ready = True
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                pass
            rows.append(
                {
                    "job": c.job_id(job),
                    "method": job["method"],
                    "training_seed": job["seed"],
                    "R": reuse,
                    "rollout_round": round_,
                    "optimizer_update": round_ * reuse,
                    "checkpoint_directory": str(path),
                    "checkpoint_ready": ready,
                    "wandb_run_id": evidence.get("wandb_run_id", "")
                    if evidence
                    else "",
                    "training_state": "COMPLETE"
                    if evidence
                    else "PENDING_OR_INCOMPLETE",
                    "formal_evaluation_planned": round_ in c.RECIPE["eval_rounds"],
                    "episodes_per_task": 100,
                    **dict.fromkeys(c.TASKS, ""),
                    "task_macro_sr": "",
                    "evaluation_state": "PENDING",
                }
            )
    output = Path(machine["output_root"]) / "deliverables"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "checkpoint_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    record = {
        "completed_jobs": completed,
        "target_jobs": 8,
        "all_saved_policy_points": 80,
        "formal_policy_points": 40,
        "formal_episodes_deferred": 16000,
        "formal_scene_pool": "NOT_FROZEN; coordinate before evaluation",
        "updated_unix": time.time(),
    }
    (output / "training_summary.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    """Plan, validate or run the fixed train-only collaborator queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument(
        "action",
        choices=["plan", "prepare", "preflight", "validate", "run", "summarize"],
    )
    parser.add_argument("--job", help="Optional one of the eight declared job IDs")
    args = parser.parse_args()
    machine = c.read_machine(args.machine)
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "jobs": [
                        {
                            **job,
                            "id": c.job_id(job),
                            "command": c.train_command(machine, job),
                        }
                        for job in c.jobs()
                    ],
                    "evaluation": "DEFERRED",
                    "rollout_rounds": 400,
                    "optimizer_updates": 1100,
                },
                indent=2,
            )
        )
        return
    # Run in the dedicated runtime even when invoked using system python/Slurm.
    env = c.environment(machine)
    runtime_python = str(Path(machine["runtime_root"]) / "venv/bin/python")
    if (
        Path(os.sys.prefix).resolve()
        != (Path(machine["runtime_root"]) / "venv").resolve()
    ):
        subprocess.run(
            [
                runtime_python,
                "-m",
                "toolkits.robotwin_mopd.new4.experiment",
                *os.sys.argv[1:],
            ],
            env=env,
            cwd=c.REPO,
            check=True,
        )
        return
    if args.action == "summarize":
        print(json.dumps(summarize(machine), indent=2))
        return
    if args.action == "prepare":
        c.prepare(machine)
        return
    if args.action == "preflight":
        print(json.dumps(preflight(machine), indent=2))
        return
    if args.action == "validate":
        c.prepare(machine)
        preflight(machine)
        print(json.dumps(validate_deployment(machine, env), indent=2))
        return
    require_deployment(machine)
    c.freeze(machine)
    chosen = [c.select_job(args.job)] if args.job else c.jobs()
    for job in chosen:
        require_deployment(machine)
        print(json.dumps({"starting": c.job_id(job)}), flush=True)
        execute_job(machine, job, env)
        summarize(machine)
    print(json.dumps(summarize(machine), indent=2))


if __name__ == "__main__":
    main()
