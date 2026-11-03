# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Portable, fail-closed RoboTwin formal experiment entrypoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECIPE = json.loads((HERE / "recipe.json").read_text())
TASKS = RECIPE["tasks"]


def digest(path: Path) -> str:
    """Hash large weights without materializing them in RAM."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_machine(path: Path) -> dict:
    """Require explicit machine-local paths and an isolated output directory."""
    cfg = json.loads(path.read_text())
    for key in ("runtime_root", "resource_root", "output_root", "pace_reference_root"):
        value = Path(cfg[key]).expanduser()
        if not value.is_absolute() or "absolute/path" in str(value):
            raise ValueError(f"Fill an absolute {key} in {path}")
        cfg[key] = str(value.resolve())
    if cfg["wandb_mode"] not in ("disabled", "offline", "online"):
        raise ValueError("Invalid wandb_mode")
    if cfg["profile"] not in ("action-cu124", "blackwell-cu128-candidate"):
        raise ValueError("Unknown runtime profile")
    return cfg


def environment(machine: dict) -> dict:
    """Activate runtime without inheriting its recorded repository location."""
    runtime = Path(machine["runtime_root"])
    result = subprocess.check_output(
        [
            "bash",
            "-c",
            'set -e; source "$1"; env -0',
            "activate",
            str(runtime / "env.sh"),
        ]
    )
    env = dict(
        item.split("=", 1) for item in result.decode().split("\0") if "=" in item
    )
    env.update(
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        PYTHONPATH=str(REPO) + ":" + env.get("PYTHONPATH", ""),
        ROBOTWIN_RUNTIME_ROOT=str(runtime),
        PACE_REFERENCE_ROOT=machine["pace_reference_root"],
        OPENPI_DATA_HOME=str(Path(machine["output_root"]) / "prepared/openpi-data"),
        WANDB_MODE=machine["wandb_mode"],
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        USE_TF="0",
        JAX_PLATFORMS="cpu",
    )
    if machine.get("vulkan_icd"):
        env["VK_ICD_FILENAMES"] = machine["vulkan_icd"]
        env["VK_DRIVER_FILES"] = machine["vulkan_icd"]
    if machine.get("wandb_entity"):
        env["WANDB_ENTITY"] = machine["wandb_entity"]
    else:
        env.pop("WANDB_ENTITY", None)
    return env


def run_id(method: str, seed: int) -> str:
    return f"robotwin_formal_sft2000_{method}_rounds20_seed{seed}"


def model_overrides(machine: dict) -> list[str]:
    root = Path(machine["resource_root"])
    overrides = []
    for task in TASKS:
        teacher = root / "teachers" / task
        overrides += [
            f"rollout.expert_models.{task}.model_path={teacher}",
            f"actor.model.opd_task_norm_stats.{task}={teacher}/assets/robotwin/norm_stats.json",
        ]
    return overrides


def train_command(
    machine: dict, method: str, seed: int, *, dry_run=False, smoke=False
) -> list[str]:
    """Use the same B/LR/T/warmup and vary only the declared method factors."""
    if seed != 0:
        raise ValueError(
            "First release authorizes seed0 only; design additional seeds after first round"
        )
    spec = RECIPE["methods"][method]
    reuse = spec["reuse"]
    output = Path(machine["output_root"])
    reference = output / "prepared/teacher_reference.json"
    name = run_id(method, seed) + ("_smoke" if smoke else "")
    overrides = [
        *RECIPE["common_overrides"],
        *model_overrides(machine),
        f"algorithm.flow_refresh_updates={reuse}",
        f"actor.model.model_path={machine['resource_root']}/student",
        f"runner.max_steps={20 * reuse}",
        f"runner.save_interval={5 * reuse}",
        f"runner.rollout_allocation.mode={'adaptive' if spec['adaptive'] else 'uniform'}",
        f"runner.rollout_allocation.teacher_reference_file={reference if spec['adaptive'] else 'null'}",
        f"algorithm.sync_suite_loss_weights={str(spec['adaptive']).lower()}",
        "actor.seed=0",
        "env.train.seed=42",
        "runner.rollout_allocation.seed=42",
        "+rollout.seed=1000000000",
        "runner.logger.wandb_group=robotwin_formal_sft2000_100ep",
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
        *(["--dry-run"] if dry_run else []),
        *(["--smoke"] if smoke else []),
        *overrides,
    ]


def eval_jobs(machine: dict, target: str, seed: int) -> list[dict]:
    """All four saved checkpoints and shared references use 100 episodes/task."""
    output = Path(machine["output_root"])
    if target in ("student", "teachers"):
        checkpoints = [
            (
                0,
                "teacher"
                if target == "teachers"
                else str(Path(machine["resource_root"]) / "student"),
            )
        ]
        name = target
    else:
        name = run_id(target, seed)
        reuse = RECIPE["methods"][target]["reuse"]
        checkpoints = [
            (
                r,
                str(
                    output / "train" / name / "checkpoints" / f"global_step_{r * reuse}"
                ),
            )
            for r in RECIPE["save_rounds"]
        ]
    jobs = []
    for round_, checkpoint in checkpoints:
        for task in TASKS:
            destination = output / "eval" / name / f"round{round_}_{task}.json"
            command = [
                str(Path(machine["runtime_root"]) / "venv/bin/python"),
                "-u",
                "-m",
                "toolkits.robotwin_mopd.evaluate",
                "--task",
                task,
                "--checkpoint",
                checkpoint,
                "--episodes",
                "100",
                "--evaluation-pool",
                str(HERE / "formal100.json"),
                "--output",
                str(destination),
            ]
            for value in model_overrides(machine):
                command += ["--override", value]
            jobs.append(
                {
                    "target": target,
                    "round": round_,
                    "task": task,
                    "checkpoint": checkpoint,
                    "output": str(destination),
                    "command": command,
                }
            )
    return jobs


def verify_resources(machine: dict) -> dict:
    """Freshly verify every transferred model/tokenizer before binding calibration."""
    manifest = json.loads((HERE / "resources.json").read_text())
    root = Path(machine["resource_root"])
    for name, record in manifest["files"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Resource identity mismatch: {name}")
    return {
        "manifest_sha256": digest(HERE / "resources.json"),
        "file_count": len(manifest["files"]),
    }


def prepare(machine: dict) -> None:
    """Relocate only verified paths, keeping source calibration counts unchanged."""
    verified = verify_resources(machine)
    output = Path(machine["output_root"]) / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    # OpenPI resolves symlinks and then requires the tokenizer to stay below its
    # cache root. Materialize this small file even when model resources are links
    # or mounted read-only; keep the large checkpoints in their original storage.
    tokenizer = output / "openpi-data/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    source = (
        Path(machine["resource_root"])
        / "openpi-data/big_vision/paligemma_tokenizer.model"
    )
    if tokenizer.exists():
        if tokenizer.is_symlink() or digest(tokenizer) != digest(source):
            raise ValueError("Existing prepared tokenizer differs; refuse overwrite")
    else:
        shutil.copyfile(source, tokenizer)
    reference = json.loads((HERE / "teacher_reference_source.json").read_text())
    for task in TASKS:
        teacher = Path(machine["resource_root"]) / "teachers" / task
        reference["teacher_checkpoints"][task] = str(teacher.resolve())
        reference["training_signature"]["model"]["opd_task_norm_stats"][task] = str(
            teacher / "assets/robotwin/norm_stats.json"
        )
    target = output / "teacher_reference.json"
    text = json.dumps(reference, indent=2) + "\n"
    if target.exists() and target.read_text() != text:
        raise ValueError("Prepared reference differs; refuse overwrite")
    target.write_text(text)
    (output / "resource_verification.json").write_text(
        json.dumps(verified, indent=2) + "\n"
    )


def execute_evaluations(
    machine: dict, target: str, seed: int, env: dict, *, smoke: bool = False
) -> None:
    """Run one checkpoint at a time, four tasks in parallel on allocated GPUs."""
    from toolkits.robocasa_mopd.owned_ray import check_gpus

    check_gpus(4, env)
    devices = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 4:
        raise ValueError("Exactly four explicitly allocated GPUs are required")
    jobs = eval_jobs(machine, target, seed)
    if smoke:
        for job in jobs:
            command = job["command"]
            command[command.index("--episodes") + 1] = "2"
            command[command.index("--evaluation-pool") + 1] = str(HERE / "dev20.json")
            job["output"] = str(
                Path(machine["output_root"]) / "smoke/teachers" / f"{job['task']}.json"
            )
            command[command.index("--output") + 1] = job["output"]
    # Check the entire request before launching; never silently overwrite partial results.
    for job in jobs:
        if Path(job["output"]).exists():
            raise FileExistsError(job["output"])
        if job["checkpoint"] != "teacher" and not Path(job["checkpoint"]).is_dir():
            raise FileNotFoundError(job["checkpoint"])
    for offset in range(0, len(jobs), 4):
        children = []
        try:
            for device, job in zip(devices, jobs[offset : offset + 4], strict=True):
                path = Path(job["output"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix(".log").open("x") as log:
                    child = subprocess.Popen(
                        job["command"],
                        cwd=REPO,
                        env={**env, "CUDA_VISIBLE_DEVICES": device},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                children.append(child)
            codes = [child.wait() for child in children]
            if any(codes):
                raise RuntimeError(f"Evaluation failed: {codes}; preserve logs/results")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()


def summarize(machine: dict) -> None:
    """Validate episode identity and emit a table with missing values left blank."""
    pool = json.loads((HERE / "formal100.json").read_text())
    rows = []
    for target in ("teachers", "student", "m0", "m1", "m2"):
        jobs = eval_jobs(machine, target, 0)
        for offset in range(0, len(jobs), 4):
            group = jobs[offset : offset + 4]
            row = {
                "experiment": target,
                "training_seed": "" if target in ("teachers", "student") else 0,
                "rollout_round": group[0]["round"],
                "episodes_per_task": 100,
                "status": "PENDING",
            }
            valid = 0
            for job in group:
                path = Path(job["output"])
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                if (
                    result.get("chunk_length") != 25
                    or result.get("denoising_steps") != 10
                    or result.get("task_horizon")
                    != RECIPE["task_horizons"][job["task"]]
                ):
                    raise ValueError(f"Wrong fixed-horizon inference protocol: {path}")
                expected_seeds = [
                    entry["scene_seed"] for entry in pool["tasks"][job["task"]]
                ]
                if (
                    result.get("evaluation_pool", {}).get("pool_content_sha256")
                    != pool["pool_content_sha256"]
                    or result.get("requested_episodes") != 100
                    or result.get("task") != job["task"]
                ):
                    raise ValueError(f"Wrong evaluation identity: {path}")
                if result.get("completed_episodes") != 100:
                    row["status"] = "INCOMPLETE"
                    continue
                if (
                    result["evaluation_pool"]["selected_scene_seeds"] != expected_seeds
                    or len(result["episodes"]) != 100
                ):
                    raise ValueError(f"Wrong evaluation scene count: {path}")
                for episode, expected in zip(
                    result["episodes"], pool["tasks"][job["task"]]
                ):
                    if (
                        episode.get("actual_reset_seed") != expected["scene_seed"]
                        or episode.get("instruction") != expected["instruction"]
                    ):
                        raise ValueError(f"Wrong episode seed/instruction: {path}")
                expected_checkpoint = (
                    Path(machine["resource_root"]) / "teachers" / job["task"]
                    if target == "teachers"
                    else Path(job["checkpoint"])
                )
                if (
                    Path(result["checkpoint"]).resolve()
                    != expected_checkpoint.resolve()
                ):
                    raise ValueError(f"Wrong checkpoint: {path}")
                successes = sum(ep["success_once"] for ep in result["episodes"])
                if any(
                    type(ep["success_once"]) is not bool for ep in result["episodes"]
                ):
                    raise ValueError(f"Invalid success flags: {path}")
                if successes != result["successes"]:
                    raise ValueError(f"Inconsistent success count: {path}")
                row[job["task"]] = successes / 100
                valid += 1
            if valid == 4:
                row["status"] = "COMPLETE"
                row["task_macro_sr"] = sum(row[t] for t in TASKS) / 4
            rows.append(row)
    output = Path(machine["output_root"]) / "deliverables"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "checkpoint_results.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "experiment",
                "training_seed",
                "rollout_round",
                "episodes_per_task",
                *TASKS,
                "task_macro_sr",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "prepare",
            "preflight",
            "train",
            "smoke",
            "teacher-smoke",
            "collective-smoke",
            "eval",
            "summarize",
        ),
    )
    parser.add_argument(
        "target", nargs="?", choices=("m0", "m1", "m2", "student", "teachers")
    )
    args = parser.parse_args()
    machine = read_machine(args.machine)
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "training": {
                        m: train_command(machine, m, 0) for m in RECIPE["methods"]
                    },
                    "evaluation": [
                        j
                        for t in ("student", "teachers", "m0", "m1", "m2")
                        for j in eval_jobs(machine, t, 0)
                    ],
                },
                indent=2,
            )
        )
        return
    if args.action == "prepare":
        prepare(machine)
        return
    if args.action == "summarize":
        summarize(machine)
        return
    env = environment(machine)
    runtime_python = str(Path(machine["runtime_root"]) / "venv/bin/python")
    if (
        args.action in ("eval", "teacher-smoke")
        and Path(sys.prefix) != Path(machine["runtime_root"]) / "venv"
    ):
        subprocess.run(
            [
                runtime_python,
                "-m",
                "toolkits.robotwin_mopd.formal.experiment",
                *sys.argv[1:],
            ],
            env=env,
            cwd=REPO,
            check=True,
        )
        return
    if args.action == "preflight":
        for method in RECIPE["methods"]:
            subprocess.run(
                train_command(machine, method, 0, dry_run=True),
                env=env,
                cwd=REPO,
                check=True,
            )
        subprocess.run(
            [
                str(Path(machine["runtime_root"]) / "venv/bin/python"),
                "-m",
                "toolkits.robotwin_mopd.formal.validate",
                "--machine",
                str(args.machine.resolve()),
            ],
            env=env,
            cwd=REPO,
            check=True,
        )
    elif args.action in ("train", "smoke"):
        if args.target not in RECIPE["methods"]:
            parser.error("train/smoke requires m0, m1 or m2")
        if not (
            Path(machine["output_root"]) / "prepared/resource_verification.json"
        ).exists():
            raise ValueError("Run prepare first")
        subprocess.run(
            train_command(machine, args.target, 0, smoke=args.action == "smoke"),
            env=env,
            cwd=REPO,
            check=True,
        )
    elif args.action == "collective-smoke":
        output = Path(machine["output_root"]) / "smoke/tiny_fsdp"
        if output.exists():
            raise FileExistsError(output)
        if len(env.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 4:
            parser.error("collective-smoke requires four allocated visible GPUs")
        subprocess.run(
            [
                runtime_python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                "-m",
                "toolkits.robotwin_mopd.formal.collective_smoke",
                "--output",
                str(output),
            ],
            cwd=REPO,
            env=env,
            check=True,
        )
    elif args.action == "teacher-smoke":
        execute_evaluations(machine, "teachers", 0, env, smoke=True)
    elif args.action == "eval":
        if args.target is None:
            parser.error("eval requires a target")
        execute_evaluations(machine, args.target, 0, env)


if __name__ == "__main__":
    main()
