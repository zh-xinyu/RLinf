# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Launch the frozen RoboCasa experiment matrix on a dedicated eight-GPU node.

Source the fixed runtime and machine-local paths first. Each invocation owns
its Ray processes, refuses an existing cluster, and never invokes ray stop.
"""

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from evaluations.robocasa365.formal_eval import manifest_tasks, sha256
from rlinf.config import validate_suite_distillation_cfg
from rlinf.utils.rollout_allocation import load_teacher_reference
from toolkits.robocasa_mopd.owned_ray import (
    OWNER_KEY,
    check_gpus,
    require_free_ray,
    stop_owned,
)
from toolkits.robocasa_mopd.resources import INPUTS, checkpoint_paths, verify_files

REPO = Path(__file__).resolve().parents[2]


def training_config(experiment: str, run_id: str, smoke: bool = False):
    """Resolve one matrix arm without inheriting pilot resume or environment overrides."""
    config_name = f"robocasa365_mopd_formal_{experiment.lower()}"
    output = REPO / "outputs" / ("smoke" if smoke else "train") / run_id
    overrides = [
        f"runner.logger.log_path={json.dumps(str(output))}",
        f"runner.logger.experiment_name={run_id}",
        f"hydra.run.dir={json.dumps(str(output))}",
    ]
    if smoke or os.environ.get("WANDB_MODE") == "disabled":
        overrides.append("runner.logger.logger_backends=[tensorboard]")
    if smoke:
        r = 1 if experiment.lower() in {"m0", "m3"} else 2
        overrides += [
            f"runner.max_steps={2 * r}",
            f"runner.save_interval={2 * r}",
        ]
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    validate_suite_distillation_cfg(cfg)
    command = [
        sys.executable,
        "-u",
        str(REPO / "examples/embodiment/train_embodied_agent.py"),
        "--config-name",
        config_name,
        *overrides,
    ]
    return cfg, output, command


def check_reference(cfg) -> None:
    """Accept only the shared, content-verified calibration for adaptive arms."""
    if cfg.runner.rollout_allocation.mode != "adaptive":
        return
    path = cfg.runner.rollout_allocation.teacher_reference_file
    reference = json.loads(Path(path).read_text())
    if reference.get("checkpoint_manifest_sha256") != sha256(
        INPUTS / "checkpoints.json"
    ):
        raise ValueError(
            "Run resources.py prepare to verify inputs and bind the reference"
        )
    frozen = json.loads((INPUTS / "teacher_reference.portable.json").read_text())
    for key, value in frozen.items():
        if key != "teacher_checkpoints" and reference.get(key) != value:
            raise ValueError(f"Shared teacher reference changed: {key}")
    load_teacher_reference(path, cfg, manifest_tasks())


def run_command(command: list[str], log: Path, env: dict) -> None:
    """Run a driver in its own process group and terminate only that group on interruption."""
    print("Running:", json.dumps(command), flush=True)
    with log.open("a") as stream:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main() -> None:
    """Run training or independent evaluation with recorded inputs and isolated Ray."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    train = sub.add_parser("train")
    train.add_argument("experiment", choices=["m0", "m1", "m2", "m3"])
    train.add_argument(
        "--smoke", action="store_true", help="Two full physical rounds, no W&B"
    )
    evaluate = sub.add_parser("eval")
    evaluate.add_argument(
        "--weights", type=Path, help="Omit only for the shared Joint-SFT-Full baseline"
    )
    evaluate.add_argument(
        "--intermediate", action="store_true", help="20 episodes/task; otherwise 50"
    )
    evaluate.add_argument(
        "--smoke", action="store_true", help="OpenDrawer, one episode"
    )
    for command in (train, evaluate):
        command.add_argument("--run-id", required=True)
        command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", args.run_id):
        parser.error("Use a safe, unique run-id of at most 101 characters")
    if args.mode == "eval" and args.smoke and args.intermediate:
        parser.error("--smoke and --intermediate are mutually exclusive")
    for key in ("OPD_RUNTIME_ROOT", "OPD_STUDENT_CHECKPOINT", "OPD_MODEL_ROOT"):
        if not os.environ.get(key):
            parser.error(f"Set {key} after sourcing your runtime")
    runtime = Path(os.environ["OPD_RUNTIME_ROOT"]).resolve()
    if Path(sys.prefix).resolve() != (runtime / "venv").resolve():
        parser.error("Use the selected runtime's Python")
    env = dict(os.environ)
    env.update(
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        PYTHONPATH=str(REPO),
        PYTHONNOUSERSITE="1",
        MUJOCO_GL="egl",
        RLINF_NODE_RANK="0",
        RAY_ADDRESS="127.0.0.1:6379",
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    if env["CUDA_VISIBLE_DEVICES"] != "0,1,2,3,4,5,6,7":
        parser.error("The frozen single-node recipe requires physical GPUs 0 through 7")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("RAY_DASHBOARD_STARTUP_TIMEOUT_S", "300")
    env.setdefault("RAY_DASHBOARD_SUBPROCESS_MODULE_WAIT_READY_TIMEOUT", "180")
    env.pop(OWNER_KEY, None)
    os.environ.update(env)
    os.environ.setdefault("OPD_RUN_DIR", str(REPO / "outputs/train"))
    env["OPD_RUN_DIR"] = os.environ["OPD_RUN_DIR"]
    cfg = None
    if args.mode == "train":
        cfg, output, command = training_config(args.experiment, args.run_id, args.smoke)
        check_reference(cfg)
        commands = [command]
    else:
        output = REPO / "outputs" / ("smoke" if args.smoke else "eval") / args.run_id
        entry = str(REPO / "evaluations/robocasa365/formal_eval.py")
        plan = [
            sys.executable,
            entry,
            "plan",
            "--run-id",
            args.run_id,
            "--base-checkpoint",
            env["OPD_STUDENT_CHECKPOINT"],
            "--gpus",
            "0,1,2,3,4,5,6,7",
        ]
        if args.weights:
            plan += ["--weights", str(args.weights.resolve(strict=True))]
        if args.intermediate:
            plan += ["--intermediate"]
        if args.smoke:
            plan += ["--smoke-task", "OpenDrawer", "--smoke-episodes", "1"]
        commands = [
            plan,
            [sys.executable, entry, "run", "--plan", str(output / "plan.json")],
        ]
    if args.dry_run:
        print(json.dumps({"commands": commands, "output": str(output)}, indent=2))
        if cfg is not None:
            print(OmegaConf.to_yaml(cfg))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    # Prevent two launchers under the same Unix account from racing for Ray ports.
    with open(f"/tmp/opd-single-job-{os.getuid()}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_free_ray()
        check_gpus(8, env)
        subprocess.run(
            ["git", "diff", "--exit-code", "--quiet", "HEAD"], cwd=REPO, check=True
        )
        manifest = json.loads((INPUTS / "checkpoints.json").read_text())
        paths = checkpoint_paths(
            Path(env["OPD_STUDENT_CHECKPOINT"]), Path(env["OPD_MODEL_ROOT"])
        )
        for name, path in paths.items():
            verify_files(path, manifest["checkpoints"][name])
        launch = REPO / "outputs/launches" / args.run_id
        launch.mkdir(parents=True, exist_ok=False)
        if cfg is not None:
            output.mkdir(parents=True)
            OmegaConf.save(cfg, output / "resolved_config.yaml")
        metadata = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "checkpoint_manifest_sha256": sha256(INPUTS / "checkpoints.json"),
            "hostname": socket.gethostname(),
            "commands": commands,
            "profile": env.get("OPD_ENV_PROFILE"),
            "runtime": str(runtime),
            "output": str(output),
        }
        (launch / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
        owner = f"opd-formal-{uuid.uuid4().hex}"
        ray_env = {**env, OWNER_KEY: owner}
        ray_command = [
            str(runtime / "venv/bin/ray"),
            "start",
            "--head",
            "--port=6379",
            "--num-cpus=32",
            "--num-gpus=8",
            "--include-dashboard=true",
            "--dashboard-host=127.0.0.1",
            "--dashboard-port=8265",
            f"--temp-dir={tempfile.mkdtemp(prefix='opd-ray-', dir='/tmp')}",
        ]
        (launch / "status").write_text("RUNNING\n")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        try:
            run_command(ray_command, launch / "ray.log", ray_env)
            for command in commands:
                run_command(command, launch / "driver.log", env)
            (launch / "status").write_text("OK\n")
        except BaseException:
            (launch / "status").write_text("FAILED\n")
            raise
        finally:
            stop_owned(owner)
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
