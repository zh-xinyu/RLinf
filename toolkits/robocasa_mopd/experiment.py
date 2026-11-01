# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Reproduce the RoboCasa matrix using one explicit machine JSON, without shell exports."""

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INITIALIZATION = "Joint-SFT-Full"
ROUNDS = (25, 50, 75, 100, 125, 150)
ARMS = {
    "m0": ("robocasa-M0-uniform-r1-joint-sft-full-seed0-formal", 1),
    "m1": ("robocasa-M1-uniform-r2-joint-sft-full-seed0-formal", 2),
    "m2": ("robocasa-M2-adaptive-sync-r2-joint-sft-full-seed0-formal", 2),
    "m3": ("robocasa-M3-adaptive-sync-r1-joint-sft-full-seed0-formal", 1),
}


def load_machine(path: Path) -> dict:
    """Validate local settings without executing a shell configuration."""
    value = json.loads(path.read_text())
    required = {
        "hostname",
        "runtime_root",
        "resource_root",
        "student_checkpoint",
        "output_root",
        "wandb_entity",
        "wandb_mode",
    }
    if set(value) != required or any(
        not isinstance(v, str) or (not v and k != "wandb_entity")
        for k, v in value.items()
    ):
        raise ValueError(
            f"Machine JSON requires exactly these string keys: {sorted(required)}; "
            "only wandb_entity may be empty"
        )
    if value["hostname"] != socket.gethostname().split(".")[0]:
        raise ValueError(
            "Machine hostname does not match; select the correct machine JSON"
        )
    if value["wandb_mode"] not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled")
    if value["wandb_entity"].startswith("REPLACE_"):
        raise ValueError("Choose your W&B username/team slug, or leave it empty")
    if value["wandb_mode"] == "online" and not value["wandb_entity"].strip():
        raise ValueError("Online logging requires your W&B username or team slug")
    for key in ("runtime_root", "resource_root", "student_checkpoint", "output_root"):
        path = Path(value[key]).expanduser()
        if not path.is_absolute() or str(path).startswith("/absolute/path"):
            raise ValueError(f"Fill in a real absolute path for {key}")
        value[key] = str(path.resolve())
    runtime = Path(value["runtime_root"])
    value["bootstrap_root"] = str(runtime.with_name(runtime.name + "-bootstrap"))
    value["teacher_root"] = str(
        Path(value["resource_root"]) / "teachers/gr00t_n1-5/foundation_model_learning"
    )
    value["teacher_reference"] = str(
        Path(value["resource_root"]) / "teacher_reference.json"
    )
    return value


def environment(machine: dict) -> dict[str, str]:
    """Derive runtime and resource variables explicitly, independent of active venvs."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    for key in (
        "PYTHONHOME",
        "CONDA_PREFIX",
        "RLINF_PYTHON_EXECUTABLE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "GPU_FREE_MB",
        "WANDB_ENTITY",
        "WANDB_DISABLED",
    ):
        env.pop(key, None)
    runtime = Path(machine["runtime_root"])
    env.update(
        PATH=str(runtime / "venv/bin") + os.pathsep + os.environ.get("PATH", ""),
        VIRTUAL_ENV=str(runtime / "venv"),
        PYTHONPATH=str(REPO),
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        PYTHONNOUSERSITE="1",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        RLINF_NODE_RANK="0",
        OPD_ENV_PROFILE="gr00t-n15-robocasa365-20261019",
        OPD_RUNTIME_ROOT=str(runtime),
        OPD_STUDENT_CHECKPOINT=machine["student_checkpoint"],
        OPD_MODEL_ROOT=machine["teacher_root"],
        OPD_TEACHER_REFERENCE=machine["teacher_reference"],
        OPD_RUN_DIR=str(REPO / "outputs/train"),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        HF_HOME=str(runtime / "cache/huggingface"),
        HF_MODULES_CACHE=str(runtime / "cache/huggingface/modules"),
        WANDB_MODE=machine["wandb_mode"],
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    if machine["wandb_entity"] and machine["wandb_mode"] != "disabled":
        env["WANDB_ENTITY"] = machine["wandb_entity"]
    return env


def bind_outputs(machine: dict) -> None:
    """Create a new outputs link or validate an existing directory; never replace data."""
    target = Path(machine["output_root"])
    link = REPO / "outputs"
    if link.is_symlink() or link.exists():
        if link.resolve() != target or not target.is_dir():
            raise ValueError(
                f"Existing {link} does not resolve to output_root={target}; no data moved"
            )
        return
    target.mkdir(parents=True, exist_ok=True)
    if link != target:
        link.symlink_to(target, target_is_directory=True)


def weights_path(arm: str, round_number: int) -> Path:
    """Map physical rounds to the existing runner's nested checkpoint directory."""
    name, repeats = ARMS[arm]
    return (
        REPO
        / "outputs/train"
        / name
        / name
        / "checkpoints"
        / f"global_step_{round_number * repeats}"
        / "actor/model_state_dict/full_weights.pt"
    )


def eval_id(arm: str, round_number: int = 0) -> str:
    """Return stable evaluation identifiers used by the collector and email template."""
    return (
        "robocasa-Joint-SFT-Full-round0"
        if arm == "joint-sft-full"
        else f"{ARMS[arm][0]}-round{round_number}-eval50"
    )


def commands_for(args, machine: dict) -> list[list[str]]:
    """Build explicit subprocess commands without executing downloads or GPU work."""
    python = str(Path(machine["runtime_root"]) / "venv/bin/python")
    launch = [python, "-m", "toolkits.robocasa_mopd.run"]
    resources = [python, "-m", "toolkits.robocasa_mopd.resources"]
    if args.command == "verify":
        return [
            [
                python,
                str(REPO / "requirements/opd_profile.py"),
                "verify",
                "--runtime",
                machine["runtime_root"],
                *(["--gpu"] if args.gpu else []),
            ]
        ]
    if args.command == "login":
        if machine["wandb_mode"] != "online":
            return []
        return [[str(Path(machine["runtime_root"]) / "venv/bin/wandb"), "login"]]
    if args.command == "download-teachers":
        return [
            resources
            + [
                "download-teachers",
                "--destination",
                str(Path(machine["resource_root"]) / "teachers"),
            ]
        ]
    if args.command == "assets":
        return [
            [python, "-m", "robocasa.scripts.setup_macros"],
            [python, "-m", "robocasa.scripts.download_kitchen_assets", "--type", "all"],
        ]
    if args.command == "prepare":
        return [resources + ["prepare", "--output", machine["teacher_reference"]]]
    if args.command == "preflight":
        return [
            launch + ["train", arm, "--run-id", f"check-{arm}", "--dry-run"]
            for arm in ARMS
        ]
    if args.command == "smoke":
        return [
            launch + ["train", "m2", "--run-id", "setup-m2-smoke", "--smoke"],
            launch + ["eval", "--run-id", "setup-joint-sft-full-eval-smoke", "--smoke"],
        ]
    if args.command == "train":
        return [launch + ["train", args.arm, "--run-id", ARMS[args.arm][0]]]
    if args.command == "eval":
        if args.arm == "joint-sft-full":
            if args.round is not None:
                raise ValueError(
                    "Joint-SFT-Full is evaluated once at round0; omit --round"
                )
            return [launch + ["eval", "--run-id", eval_id("joint-sft-full")]]
        return [
            launch
            + [
                "eval",
                "--run-id",
                eval_id(args.arm, r),
                "--weights",
                str(weights_path(args.arm, r)),
            ]
            for r in ([args.round] if args.round else ROUNDS)
        ]
    if args.command == "summarize":
        return [
            [
                python,
                "-m",
                "toolkits.robocasa_mopd.results",
                "--output",
                str(REPO / "outputs/deliverables/checkpoint_results.csv"),
            ]
        ]
    raise ValueError(args.command)


def execute(command: list[str], env: dict) -> None:
    """Run one step and stop immediately on failure."""
    print("+ " + json.dumps(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def install(machine: dict, dry_run: bool) -> None:
    """Bootstrap uv once, then delegate to the existing versioned installer."""
    runtime = Path(machine["runtime_root"])
    bootstrap = Path(machine["bootstrap_root"])
    if runtime.exists():
        raise ValueError(
            "Runtime already exists; use verify, or choose a new path for installation"
        )
    commands = []
    if not bootstrap.exists():
        commands.append([sys.executable, "-m", "venv", str(bootstrap)])
    commands += [
        [str(bootstrap / "bin/python"), "-m", "pip", "install", "uv==0.12.0"],
        [
            str(bootstrap / "bin/python"),
            str(REPO / "requirements/opd_profile.py"),
            "install",
            "--runtime",
            str(runtime),
        ],
    ]
    if dry_run:
        print(json.dumps(commands, indent=2))
        return
    env = environment(machine)
    env["PATH"] = str(bootstrap / "bin") + os.pathsep + env["PATH"]
    env.pop("VIRTUAL_ENV", None)
    for command in commands:
        execute(command, env)


def main() -> None:
    """Select one machine and execute one documented reproduction stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; no file writes, downloads, or GPU use",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "paths",
        "install",
        "login",
        "download-teachers",
        "assets",
        "prepare",
        "preflight",
        "smoke",
        "summarize",
    ):
        sub.add_parser(name)
    sub.add_parser("verify").add_argument("--gpu", action="store_true")
    sub.add_parser("train").add_argument("arm", choices=list(ARMS))
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("arm", choices=["joint-sft-full", *ARMS])
    evaluate.add_argument("--round", type=int, choices=ROUNDS)
    args = parser.parse_args()
    machine = load_machine(args.machine)
    if args.command == "paths":
        print(json.dumps(machine, indent=2))
        return
    if args.command == "install":
        install(machine, args.dry_run)
        return
    commands = commands_for(args, machine)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    python = Path(machine["runtime_root"]) / "venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"Install the fixed runtime first: {python}")
    if args.command in {"prepare", "preflight", "smoke", "train", "eval", "summarize"}:
        bind_outputs(machine)
    env = environment(machine)
    for command in commands:
        execute(command, env)


if __name__ == "__main__":
    main()
