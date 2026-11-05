# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Plan or run a bounded independent LIBERO evaluation on a private Ray cluster.

Default: four representative tasks, all 50 official states (200 episodes).
--all-tasks evaluates all 2,000 task/state pairs. Dry run is the default.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.scheduler.cluster.config import ClusterConfig
from toolkits.libero_gr00t.evaluate_full_checkpoint import baseline_identity, sha256
from toolkits.libero_gr00t.launch_production import (
    REPO,
    choose_ports,
    read_runtime_environment,
    run_logged,
)
from toolkits.libero_gr00t.padding_profile import (
    PADDING_PROFILES,
    apply_student_padding_profile,
    student_input_contract,
)
from toolkits.libero_gr00t.summarize_evaluation import EPISODE, SUITES
from toolkits.robocasa_mopd.owned_ray import OWNER_KEY, check_gpus, stop_owned


def build_config(args):
    """Compose a four-rank, 16-slot recipe with a conservative completion budget."""
    gpus = sorted(args.gpus)
    if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
        raise ValueError("Specify four distinct nonnegative physical GPU IDs")
    tasks = list(range(40)) if args.all_tasks else [0, 10, 20, 30]
    # Retain 50 official states/task. Per-rank pools differ by at most one item.
    budget = math.ceil(math.ceil(len(tasks) * 50 / 4) / 4) * 520
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "evaluations/libero")
    ):
        cfg = compose(
            config_name="libero_four_suite_gr00t_mopd_eval",
            overrides=[f"hydra.searchpath=[file://{REPO}/examples/embodiment/config]"],
        )
    with open_dict(cfg):
        cfg.rollout.model.model_path = str(args.student)
        apply_student_padding_profile(
            cfg.rollout.model, getattr(args, "student_padding_profile", "native")
        )
        cfg.runner.ckpt_path = str(args.weights) if args.weights else None
        cfg.runner.logger.log_path = str(args.output)
        cfg.runner.logger.experiment_name = "libero_independent_evaluation"
        cfg.env.eval.task_id_filter = tasks
        cfg.env.eval.max_steps_per_rollout_epoch = budget
        cfg.cluster.component_placement = {
            "env,rollout": {
                "node_group": "libero_eval",
                "placement": ",".join(map(str, gpus)),
            }
        }
        cfg.cluster.node_groups = [
            {
                "label": "libero_eval",
                "node_ranks": "0",
                "env_configs": [
                    {
                        "node_ranks": "0",
                        "python_interpreter_path": str(args.python),
                    }
                ],
            }
        ]
    OmegaConf.resolve(cfg)
    ClusterConfig.from_dict_cfg(cfg.cluster)
    return cfg


def validate_checkpoint(args, cfg) -> dict:
    """Bind an existing strict CPU/full-model validation to this evaluation.

    A successful standalone reload protocol avoids constructing an extra model
    beside the four Ray workers. This launcher verifies artifact identity and
    inference settings, not merely a user-provided success boolean.
    """
    profile = getattr(args, "student_padding_profile", "native")
    input_contract = student_input_contract(cfg.rollout.model, profile)
    metadata = args.student / "experiment_cfg/metadata.json"
    if args.weights is None:
        if args.validation_protocol:
            raise ValueError("Baseline must not supply a trained-weight protocol")
        return {
            "baseline": True,
            "student_padding_profile": profile,
            "student_input_contract": input_contract,
            "metadata_sha256": sha256(metadata),
            "base_checkpoint_identity": baseline_identity(args.student),
        }
    if args.validation_protocol is None:
        raise ValueError("Trained weights require --validation-protocol")
    protocol = json.loads(args.validation_protocol.read_text())
    if protocol.get("strict") is not True or protocol.get("state_tensor_count", 0) < 1:
        raise ValueError("Checkpoint protocol lacks a strict full-state validation")
    if protocol.get("base_comparison", {}).get("unchanged_backbone_tensors", 0) < 1:
        raise ValueError("Checkpoint protocol lacks frozen-backbone validation")
    if protocol.get("weights_sha256") != sha256(args.weights):
        raise ValueError("Checkpoint differs from the validated weights hash")
    if protocol.get("metadata_sha256") != sha256(metadata):
        raise ValueError("Base metadata differs from the validated metadata hash")
    if Path(protocol["base_checkpoint"]).resolve() != args.student:
        raise ValueError("Use the same base checkpoint path as the validation protocol")
    validated = OmegaConf.create(protocol["model_config"])
    student_input_contract(validated, profile)
    # Historical native protocols predate this field; missing canonical flag
    # means False. Canonical profiles always require explicit captured identity.
    if protocol.get("student_padding_profile", "native") != profile:
        raise ValueError("Validated student padding profile differs")
    if "student_input_contract" in protocol or profile != "native":
        if protocol.get("student_input_contract") != input_contract:
            raise ValueError("Validated student input contract differs")
    for key in (
        "model_type",
        "precision",
        "action_head_precision",
        "inference_autocast_dtype",
        "action_dim",
        "num_action_chunks",
        "denoising_steps",
        "obs_converter_type",
        "embodiment_tag",
        "embodiment_projector_id",
        "data_config_class",
        "opd_fixed_text_padding",
        "disable_image_tiling",
        "skip_backbone_logits",
        "rl_head_config.noise_method",
        "rl_head_config.noise_level",
        "rl_head_config.disable_dropout",
        "rl_head_config.padding_value",
    ):
        if OmegaConf.select(validated, key) != OmegaConf.select(cfg.rollout.model, key):
            raise ValueError(f"Validated model contract differs at {key}")
    return {
        "baseline": False,
        "student_padding_profile": profile,
        "student_input_contract": input_contract,
        "weights_sha256": protocol["weights_sha256"],
        "metadata_sha256": protocol["metadata_sha256"],
        "validation_protocol": str(args.validation_protocol),
        "validation_protocol_sha256": sha256(args.validation_protocol),
    }


def audit_selected(logs: list[Path], tasks: list[int]) -> tuple[dict, list[dict]]:
    """Require exactly the selected tasks times all 50 states before reporting SR."""
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(type(task) is not int or task not in range(40) for task in tasks)
    ):
        raise ValueError("tasks must be nonempty unique integer IDs in [0, 39]")
    expected = {(task, trial) for task in tasks for trial in range(50)}
    records, sources, duplicates = {}, [], 0
    for log in logs:
        data = log.read_bytes()
        sources.append({"path": str(log), "sha256": hashlib.sha256(data).hexdigest()})
        for match in EPISODE.finditer(data.decode(errors="replace")):
            key, success = (int(match[1]), int(match[2])), match[3] == "True"
            if key not in expected:
                raise ValueError(f"Unexpected task/trial pair: {key}")
            if key in records:
                if records[key] != success:
                    raise ValueError(f"Conflicting task/trial result: {key}")
                duplicates += 1
            records[key] = success
    missing = sorted(expected - records.keys())
    rows = [
        {
            "suite": SUITES[task // 10],
            "task_id": task,
            "trial_id": trial,
            "success": int(success),
        }
        for (task, trial), success in sorted(records.items())
    ]
    report = {
        "complete": not missing,
        "episodes": len(records),
        "expected_episodes": len(expected),
        "task_ids": tasks,
        "missing_pairs": missing,
        "identical_log_duplicates": duplicates,
        "source_logs": sources,
        "suite_success_rates": None,
        "macro_success_rate": None,
    }
    if not missing:
        rates = {
            suite: sum(row["success"] for row in rows if row["suite"] == suite)
            / sum(row["suite"] == suite for row in rows)
            for suite in SUITES
            if any(row["suite"] == suite for row in rows)
        }
        report.update(
            suite_success_rates=rates,
            macro_success_rate=sum(rates.values()) / len(rates),
        )
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path)
    source.add_argument("--baseline", action="store_true")
    parser.add_argument("--validation-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs=4, type=int, required=True)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument(
        "--student-padding-profile",
        choices=PADDING_PROFILES,
        default="native",
        help="Explicit changed-policy profile; canonical-fixed570 requires a matched baseline",
    )
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("python", "student", "weights", "validation_protocol"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    if args.cpus < 1 or args.timeout_seconds <= 0:
        raise ValueError("cpus and timeout-seconds must be positive")
    for path in (
        args.python.parent.parent / "env.sh",
        args.student / "config.json",
        args.student / "experiment_cfg/metadata.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    runtime_vars = read_runtime_environment(args.python)
    cfg = build_config(args)
    validation = validate_checkpoint(args, cfg)
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml")
    owner, ports = "opd-libero-eval-" + uuid.uuid4().hex, choose_ports()
    ray_root = Path(tempfile.mkdtemp(prefix="opdle-", dir="/tmp"))
    ray_dir = ray_root / "ray"
    ray_dir.mkdir()
    env = {**os.environ, **runtime_vars}
    env.pop(OWNER_KEY, None)
    env.update(
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        RLINF_NODE_RANK="0",
        RAY_ADDRESS=f"127.0.0.1:{ports['port']}",
        RAY_TMPDIR=str(ray_root),
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        CUDA_VISIBLE_DEVICES=",".join(map(str, sorted(args.gpus))),
        GPU_FREE_MB=str(20 * 1024),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        RAY_USAGE_STATS_ENABLED="0",
    )
    ray_command = [
        str(args.python),
        "-m",
        "ray.scripts.scripts",
        "start",
        "--head",
        "--node-ip-address=127.0.0.1",
        f"--num-cpus={args.cpus}",
        "--num-gpus=4",
        "--object-store-memory=2147483648",
        "--include-dashboard=true",
        "--dashboard-host=127.0.0.1",
        "--disable-usage-stats",
        f"--temp-dir={ray_dir}",
        *[f"--{name}={port}" for name, port in ports.items()],
    ]
    driver_command = [
        str(args.python),
        str(REPO / "evaluations/eval_embodied_agent.py"),
        "--config-path",
        str(args.output),
        "--config-name",
        "resolved_config",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ]
    manifest = {
        "mode": "run" if args.run else "dry_run",
        "owner": owner,
        "physical_gpus": sorted(args.gpus),
        "ray_temp_dir": str(ray_dir),
        "ray_address": env["RAY_ADDRESS"],
        "ray_command": ray_command,
        "driver_command": driver_command,
        "timeout_seconds": args.timeout_seconds,
        "student": str(args.student),
        "weights": str(args.weights) if args.weights else None,
        "task_ids": list(cfg.env.eval.task_id_filter),
        "trials_per_task": 50,
        "expected_episodes": len(cfg.env.eval.task_id_filter) * 50,
        "scheduling_steps": int(cfg.env.eval.max_steps_per_rollout_epoch),
        "rng_protocol": "rank-local rollout seed 1234+rank; shared official states",
        "gpu_free_preflight": "20 GiB/card; availability check, not reservation",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ),
        **validation,
    }
    manifest_path, status = args.output / "launch.json", args.output / "status"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.run:
        status.write_text("DRY_RUN\n")
        return
    try:
        (args.output / "gpu_preflight.txt").write_text(check_gpus(4, env))
    except BaseException:
        status.write_text("PREFLIGHT_FAILED\n")
        raise
    ray_env = {**env, OWNER_KEY: owner}
    ray_env.pop("RAY_ADDRESS", None)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    status.write_text("RUNNING\n")
    try:
        run_logged(ray_command, ray_env, args.output / "ray.log", timeout_seconds=120)
        private_address = (ray_dir / "ray_current_cluster").read_text().strip()
        if int(private_address.rsplit(":", 1)[1]) != ports["port"]:
            raise RuntimeError("Private Ray address differs from launch plan")
        env["RAY_ADDRESS"] = private_address
        manifest["ray_address"] = private_address
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        run_logged(
            driver_command,
            env,
            args.output / "driver.log",
            timeout_seconds=args.timeout_seconds,
        )
        logs = sorted(args.output.rglob("EnvGroup/rank_*.log"))
        if len(logs) != 4:
            raise RuntimeError(
                f"Expected four environment-worker logs, found {len(logs)}"
            )
        report, rows = audit_selected(logs, manifest["task_ids"])
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["suite", "task_id", "trial_id", "success"]
            )
            writer.writeheader()
            writer.writerows(rows)
        if not report["complete"]:
            raise RuntimeError("Incomplete task/trial coverage; success rates withheld")
        status.write_text("OK\n")
    except BaseException:
        status.write_text("FAILED\n")
        raise
    finally:
        stop_owned(owner)


if __name__ == "__main__":
    main()
