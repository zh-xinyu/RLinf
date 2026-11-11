# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Portable train/evaluate queue: eight new4 SFT400 experiments."""

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


def validate_deployment(machine: dict, env: dict, machine_path: Path) -> dict:
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
    from toolkits.robotwin_mopd.new4.evaluation import evaluate_checkpoint

    evaluation = evaluate_checkpoint(
        machine,
        machine_path,
        job,
        0,
        Path(continuation["final_checkpoint"]),
        smoke=True,
    )
    result = {
        "evaluation_smoke": evaluation,
        "formal_pool_sha256": json.loads((c.HERE / "formal100.json").read_text())[
            "pool_content_sha256"
        ],
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
        or proof.get("evaluation_smoke", {}).get("state") != "COMPLETE"
        or proof.get("formal_pool_sha256")
        != json.loads((c.HERE / "formal100.json").read_text())["pool_content_sha256"]
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


def atomic_csv(path: Path, rows: list[dict]) -> None:
    """Publish a complete table atomically; never expose a half-written CSV."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def summarize(machine: dict) -> dict:
    """Rebuild checkpoint and final-round tables from validated raw evidence."""
    from toolkits.robotwin_mopd.new4 import evaluation

    rows, experiments, completed = [], [], []
    pool_hash = json.loads(evaluation.pool_path(machine).read_text())[
        "pool_content_sha256"
    ]
    for job in c.jobs():
        directory = job_directory(machine, job)
        reuse = c.RECIPE["methods"][job["method"]]["reuse"]
        evidence = None
        try:
            evidence = audit_run(directory, reuse)
            completed.append(c.job_id(job))
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            pass
        job_rows = []
        for round_ in c.RECIPE["save_rounds"]:
            path = checkpoint_path(machine, job, round_)
            ready = False
            try:
                checkpoint_audit(path, round_ * reuse, reuse, full=round_ == 50)
                ready = True
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                pass
            result = evaluation.checkpoint_results(machine, job, round_, path)
            row = {
                "job": c.job_id(job),
                "method": job["method"],
                "training_seed": job["seed"],
                "R": reuse,
                "rollout_round": round_,
                "optimizer_update": round_ * reuse,
                "checkpoint_directory": str(path),
                "checkpoint_ready": ready,
                "wandb_run_id": (evidence or {}).get("wandb_run_id", ""),
                "training_state": "COMPLETE" if evidence else "PENDING_OR_INCOMPLETE",
                "formal_evaluation_planned": True,
                "episodes_per_task": 100,
                **{
                    task: result["tasks"][task]["success_rate"]
                    if result["tasks"][task]
                    else ""
                    for task in c.TASKS
                },
                "task_macro_sr": result["macro_sr"]
                if result["macro_sr"] is not None
                else "",
                "evaluation_state": result["state"],
                "completed_tasks": result["completed_tasks"],
                "pool_content_sha256": pool_hash,
                "evaluation_directory": str(
                    evaluation.task_directory(machine, job, round_, c.TASKS[0]).parent
                ),
            }
            rows.append(row)
            job_rows.append(row)
        final = job_rows[-1]
        evaluated = sum(row["evaluation_state"] == "COMPLETE" for row in job_rows)
        experiments.append(
            {
                "job": c.job_id(job),
                "method": job["method"],
                "training_seed": job["seed"],
                "R": reuse,
                "reported_round": 50,
                "evaluated_checkpoints": evaluated,
                "expected_checkpoints": 10,
                "training_state": final["training_state"],
                **{task: final[task] for task in c.TASKS},
                "task_macro_sr": final["task_macro_sr"],
                "state": "COMPLETE"
                if evidence and evaluated == 10
                else "PENDING_OR_INCOMPLETE",
                "wandb_run_id": final["wandb_run_id"],
                "pool_content_sha256": pool_hash,
            }
        )
    output = Path(machine["output_root"]) / "deliverables"
    atomic_csv(output / "checkpoint_results.csv", rows)
    atomic_csv(output / "experiment_results.csv", experiments)
    record = {
        "completed_jobs": completed,
        "evaluated_jobs": [
            row["job"] for row in experiments if row["state"] == "COMPLETE"
        ],
        "target_jobs": 8,
        "all_saved_policy_points": 80,
        "formal_policy_points": 80,
        "evaluated_policy_points": sum(
            row["evaluation_state"] == "COMPLETE" for row in rows
        ),
        "formal_episodes": 32000,
        "formal_scene_pool_sha256": pool_hash,
        "evaluation_rounds": c.RECIPE["eval_rounds"],
        "updated_unix": time.time(),
    }
    # A checkpoint macro is published only after all four tasks are validated.
    temporary = output / "training_summary.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, output / "training_summary.json")
    return record


def evaluate_job(machine: dict, machine_path: Path, job: dict) -> None:
    """Finish all ten fixed checkpoints before advancing to another training job."""
    from toolkits.robotwin_mopd.new4.evaluation import evaluate_checkpoint

    reuse = c.RECIPE["methods"][job["method"]]["reuse"]
    audit_run(job_directory(machine, job), reuse)
    for round_ in c.RECIPE["eval_rounds"]:
        checkpoint = checkpoint_path(machine, job, round_)
        checkpoint_audit(checkpoint, round_ * reuse, reuse, full=round_ == 50)
        evaluate_checkpoint(
            machine,
            machine_path,
            job,
            round_,
            checkpoint,
            on_complete=lambda: summarize(machine),
        )


def run_queue(
    machine: dict,
    machine_path: Path,
    chosen: list[dict],
    env: dict,
    *,
    evaluation_only: bool = False,
) -> None:
    """Train, audit and evaluate each job in order; stop on any failed stage."""
    for job in chosen:
        require_deployment(machine)
        print(json.dumps({"starting": c.job_id(job)}), flush=True)
        if not evaluation_only:
            execute_job(machine, job, env)
        summarize(machine)
        evaluate_job(machine, machine_path, job)
        summarize(machine)


def main() -> None:
    """Plan, validate or run the fixed train/evaluate collaborator queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument(
        "action",
        choices=[
            "plan",
            "prepare",
            "preflight",
            "validate",
            "run",
            "evaluate",
            "summarize",
        ],
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
                    "evaluation": "AFTER_EACH_TRAINING",
                    "evaluation_rounds": c.RECIPE["eval_rounds"],
                    "episodes_per_task": 100,
                    "formal_policy_points": 80,
                    "formal_episodes": 32000,
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
        print(
            json.dumps(
                validate_deployment(machine, env, args.machine.resolve()), indent=2
            )
        )
        return
    require_deployment(machine)
    c.freeze(machine)
    chosen = [c.select_job(args.job)] if args.job else c.jobs()
    run_queue(
        machine,
        args.machine.resolve(),
        chosen,
        env,
        evaluation_only=args.action == "evaluate",
    )
    print(json.dumps(summarize(machine), indent=2))


if __name__ == "__main__":
    main()
