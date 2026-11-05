# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Capture native specialist calibration and export a complete adaptive reference.

Prepare on CPU, run each suite on one explicitly selected free GPU, then finalize.
Historical screening JSONL is deliberately not accepted without this capture.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.utils.rollout_allocation import load_teacher_reference, training_signature

REPO = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (240, 320, 320, 520), strict=True))


def digest(path: Path) -> str:
    """Hash content without reading checkpoint weights into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity() -> dict[str, str]:
    """Capture the local policy, preprocessing, and sampler implementation."""
    files = list((REPO / "rlinf/models/embodiment/gr00t").rglob("*.py"))
    files += [
        REPO / "toolkits/libero_gr00t/probe.py",
        Path(__file__),
        REPO / "examples/embodiment/config/model/gr00t.yaml",
    ]
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(files)}


def checkpoint_identity(path: Path) -> dict[str, str]:
    """Require metadata/config and hash all inference weight shards."""
    files = [path / "config.json", path / "experiment_cfg/metadata.json"]
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No safetensors shards in {path}")
    files += shards
    index = path / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
    return {str(p.relative_to(path)): digest(p) for p in files}


def validate_contract(cfg) -> None:
    """Reject settings that the existing native specialist probe cannot reproduce."""
    env = cfg.env.train
    required_env = {
        "env_type": "libero",
        "task_sampling_strategy": "suite_quota",
        "auto_reset": False,
        "ignore_terminations": False,
        "freeze_finished_episodes": True,
        "reset_settle_steps": 10,
        "task_id_filter": None,
    }
    for key, value in required_env.items():
        if env.get(key) != value:
            raise ValueError(f"Unsupported calibration environment {key}")
    if list(env.task_suite_names) != list(SUITES):
        raise ValueError("Calibration requires the four canonical suites")
    if OmegaConf.to_container(env.task_suite_horizons) != HORIZONS:
        raise ValueError("Calibration requires native suite horizons")
    for key in ("camera_heights", "camera_widths"):
        if env.init_params.get(key) != 256:
            raise ValueError("Calibration requires 256px images")
    if set(cfg.rollout.expert_models) != set(SUITES):
        raise ValueError("Exactly one routed specialist per suite is required")
    merged = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    common = {
        "model_type": "gr00t",
        "precision": "bf16",
        "action_dim": 7,
        "num_action_chunks": 8,
        "denoising_steps": 4,
    }
    teacher_contract = {
        "precision": "bf16",
        "action_head_precision": "bf16",
        "obs_converter_type": "libero",
        "embodiment_tag": "libero_franka",
        "embodiment_projector_id": 31,
        "opd_fixed_text_padding": True,
        "disable_image_tiling": True,
        "data_config_class": "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoFrankaDataConfig",
    }
    for key, value in common.items():
        if merged.get(key) != value:
            raise ValueError(f"Unsupported calibration model {key}")
    head = merged.rl_head_config
    if (head.noise_method, head.noise_level, head.disable_dropout) != (
        "flow_ode",
        0.3,
        True,
    ):
        raise ValueError("Calibration requires dropout-free ODE4/noise0.3")
    student = Path(cfg.actor.model.model_path).resolve()
    for expert in cfg.rollout.expert_models.values():
        if Path(expert.model_path).resolve() == student:
            raise ValueError("Student and teacher must be different checkpoint paths")
        for key, value in teacher_contract.items():
            if expert.get(key, merged.get(key)) != value:
                raise ValueError(f"Unsupported calibration teacher {key}")


def prepare(config: Path, python: Path, output: Path, episodes: int, seed: int) -> dict:
    """Capture identities before any specialist run; never import a prior screen."""
    if output.exists():
        raise FileExistsError(output)
    if type(episodes) is not int or not 20 <= episodes <= 50:
        raise ValueError("Require 20..50 unique official initial states per task")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)
    validate_contract(cfg)
    paths = {
        s: str(Path(cfg.rollout.expert_models[s].model_path).resolve(strict=True))
        for s in SUITES
    }
    identities = {s: checkpoint_identity(Path(p)) for s, p in paths.items()}
    manifest = {
        "schema_version": 1,
        "status": "PREPARED",
        "training_signature": training_signature(cfg),
        "teacher_checkpoints": paths,
        "teacher_file_sha256": identities,
        "source_sha256": source_identity(),
        "python": str(python.resolve(strict=True)),
        "student": str(Path(cfg.actor.model.model_path).resolve(strict=True)),
        "episodes_per_task": episodes,
        "seed": seed,
        "calibration_protocol": {
            "sampler": "native_teacher_ode",
            "initial_state_indices": list(range(episodes)),
            "noise_seed": "seed + local_task_id * 100 + trial",
            "environment_seed": seed,
            "denoise_steps": 4,
            "execute": 8,
            "settling_steps": 10,
            "first_success_stops_episode": True,
            "formal_evaluation": False,
        },
    }
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "training_config.yaml", resolve=True)
    manifest["training_config_sha256"] = digest(output / "training_config.yaml")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_capture(folder: Path, suite: str | None = None) -> dict:
    """Reject changed sources, recipe or teacher weights since preparation."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PREPARED":
        raise ValueError("Not a prepared calibration capture")
    if source_identity() != manifest["source_sha256"]:
        raise ValueError("Calibration implementation changed after prepare")
    if digest(folder / "training_config.yaml") != manifest["training_config_sha256"]:
        raise ValueError("Calibration configuration changed after prepare")
    for name in SUITES if suite is None else [suite]:
        if (
            checkpoint_identity(Path(manifest["teacher_checkpoints"][name]))
            != manifest["teacher_file_sha256"][name]
        ):
            raise ValueError(f"Teacher content changed for {name}")
    return manifest


def probe_command(manifest: dict, suite: str, output: Path) -> list[str]:
    """Build the existing, native teacher-only probe invocation."""
    return [
        manifest["python"],
        "-u",
        "toolkits/libero_gr00t/probe.py",
        "--mode",
        "eval",
        "--student",
        manifest["student"],
        "--teacher",
        manifest["teacher_checkpoints"][suite],
        "--suite",
        suite,
        "--output",
        str(output.resolve()),
        "--eval-roles",
        "teacher",
        "--tasks",
        *map(str, range(10)),
        "--trials",
        str(manifest["episodes_per_task"]),
        "--seed",
        str(manifest["seed"]),
        "--execute",
        "8",
        "--denoise-steps",
        "4",
        "--noise-level",
        "0.3",
        "--student-head-precision",
        "fp32",
        "--student-text-padding",
        "native",
        "--memory-gib",
        "24",
    ]


def run_suite(folder: Path, suite: str, timeout: int) -> None:
    """Run one complete suite on one selected GPU, with bounded owned cleanup."""
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    manifest = verify_capture(folder, suite)
    import torch

    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly expose exactly one free CUDA device")
    if torch.cuda.mem_get_info()[0] < 30 * 2**30:
        raise RuntimeError("Calibration requires at least 30GiB free before launch")
    output = folder / f"{suite}.jsonl"
    state = folder / f"{suite}.status.json"
    if output.exists() or state.exists():
        raise FileExistsError("Do not overwrite a calibration suite attempt")
    command = probe_command(manifest, suite, output)
    info = {
        "status": "RUNNING",
        "started_at": time.time(),
        "command": command,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    state.write_text(json.dumps(info, indent=2) + "\n")
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(name, None)
    child = None
    try:
        with (folder / f"{suite}.log").open("w") as stream:
            child = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = child.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Calibration probe returned {code}")
        verify_capture(folder, suite)
        info.update(
            status="COMPLETE", ended_at=time.time(), episode_file_sha256=digest(output)
        )
    except BaseException as error:
        info.update(status="FAILED", error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.write_text(json.dumps(info, indent=2) + "\n")


def read_suite(folder: Path, manifest: dict, suite: str) -> list[dict]:
    """Require exact unique task/trial coverage and terminal episode evidence."""
    path = folder / f"{suite}.jsonl"
    state = json.loads((folder / f"{suite}.status.json").read_text())
    if state.get("status") != "COMPLETE" or state.get("episode_file_sha256") != digest(
        path
    ):
        raise ValueError("Calibration suite did not complete or log changed")
    start, end = state.get("started_at"), state.get("ended_at")
    if (
        not all(type(t) in (float, int) and math.isfinite(t) for t in (start, end))
        or end < start
    ):
        raise ValueError("Missing or invalid calibration wall-clock interval")
    if state["command"] != probe_command(manifest, suite, path):
        raise ValueError("Calibration command does not match capture")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if (
        not records
        or records[0].get("kind") != "protocol"
        or records[-1].get("kind") != "complete"
    ):
        raise ValueError("Missing calibration protocol/completion records")
    protocol = records[0]
    for key, value in {
        "mode": "eval",
        "suite": suite,
        "teacher": manifest["teacher_checkpoints"][suite],
        "eval_roles": ["teacher"],
        "tasks": list(range(10)),
        "trials": manifest["episodes_per_task"],
        "seed": manifest["seed"],
        "execute": 8,
        "denoise_steps": 4,
        "noise_level": 0.3,
        "world_size": 1,
    }.items():
        if protocol.get(key) != value:
            raise ValueError(f"Calibration protocol mismatch: {key}")
    rows = records[1:-1]
    pairs = set()
    for row in rows:
        if (
            row.get("kind") != "episode"
            or row.get("role") != "teacher"
            or row.get("suite") != suite
        ):
            raise ValueError("Unexpected calibration record/teacher routing")
        pair = (row.get("task_id"), row.get("trial"))
        if any(type(n) is not int for n in pair) or pair in pairs:
            raise ValueError("Invalid or duplicate calibration task/trial")
        pairs.add(pair)
        if (
            type(row.get("success_once")) is not bool
            or row.get("complete_episode") is not True
        ):
            raise ValueError("Incomplete or invalid calibration outcome")
        if row.get("horizon") != HORIZONS[suite] or row.get("settling_steps") != 10:
            raise ValueError("Calibration horizon/settling mismatch")
        steps = row.get("steps")
        if type(steps) is not int or not 1 <= steps <= HORIZONS[suite]:
            raise ValueError("Invalid calibration simulator count")
        if not row["success_once"] and steps != HORIZONS[suite]:
            raise ValueError("Truncated failed calibration episode")
    expected = {(t, i) for t in range(10) for i in range(manifest["episodes_per_task"])}
    if pairs != expected or len(rows) != len(expected):
        raise ValueError("Incomplete calibration task/trial coverage")
    return rows


def finalize(folder: Path) -> dict:
    """Export only complete content-bound calibration; reject old screening logs."""
    output = folder / "teacher_reference.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_capture(folder)
    if not 20 <= manifest["episodes_per_task"] <= 50:
        raise ValueError("Require at least20 calibration episodes per task")
    tasks, policy_steps = [], 0
    for suite in SUITES:
        rows = read_suite(folder, manifest, suite)
        policy_steps += sum(row["steps"] for row in rows)
        for task_id in range(10):
            selected = [r for r in rows if r["task_id"] == task_id]
            tasks.append(
                {
                    "task": f"{suite}/{task_id}",
                    "suite": suite,
                    "horizon": HORIZONS[suite],
                    "episodes": len(selected),
                    "successes": sum(r["success_once"] for r in selected),
                    "simulation_steps": sum(r["steps"] for r in selected),
                    "settling_steps": sum(r["settling_steps"] for r in selected),
                }
            )
    states = [json.loads((folder / f"{s}.status.json").read_text()) for s in SUITES]
    wall_seconds = max(s["ended_at"] for s in states) - min(
        s["started_at"] for s in states
    )
    result = {
        "wall_seconds": wall_seconds,
        "schema_version": 1,
        "status": "COMPLETE",
        "training_signature": manifest["training_signature"],
        "teacher_checkpoints": manifest["teacher_checkpoints"],
        "tasks": tasks,
        "provenance": manifest,
        "capture_manifest_sha256": digest(folder / "manifest.json"),
        "episode_files_sha256": {s: digest(folder / f"{s}.jsonl") for s in SUITES},
        "calibration_budget": {
            "trajectories": len(tasks) * manifest["episodes_per_task"],
            "simulation_steps": policy_steps,
            "settling_steps": len(tasks) * manifest["episodes_per_task"] * 10,
        },
    }
    temporary = output.with_suffix(".pending")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    try:
        load_teacher_reference(
            str(temporary), OmegaConf.load(folder / "training_config.yaml"), tasks
        )
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    def interrupt_owned_run(signum, frame):
        raise KeyboardInterrupt("Calibration interrupted")

    signal.signal(signal.SIGTERM, interrupt_owned_run)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--python", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--episodes-per-task", type=int, default=20)
    prep.add_argument("--seed", type=int, default=1234)
    run = sub.add_parser("run-suite")
    run.add_argument("--capture", type=Path, required=True)
    run.add_argument("--suite", choices=SUITES, required=True)
    run.add_argument("--timeout-seconds", type=int, default=14400)
    end = sub.add_parser("finalize")
    end.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            args.config, args.python, args.output, args.episodes_per_task, args.seed
        )
    elif args.action == "run-suite":
        run_suite(args.capture.resolve(), args.suite, args.timeout_seconds)
    else:
        finalize(args.capture.resolve())


if __name__ == "__main__":
    main()
