# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the frozen GR00T N1.5 RoboCasa365 target-task evaluation through RLinf.

This program requires an isolated, already-started single-node Ray cluster. It
does not start, stop, or modify Ray on a shared machine. The ``plan`` command is
CPU-only; ``run`` dispatches one RLinf evaluation driver per task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
CONFIG_NAME = "robocasa365_mopd_gr00t_formal"
MANIFEST = EVAL_DIR / "target50_manifest.json"
PROTOCOL = "rlinf-robocasa365-target50-native-v1"
EXPECTED_COMMITS = {
    "gr00t": "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96",
    "robocasa": "921c9a5736a8d0ea5589657898aadcfa55a6a195",
}
EXPECTED_PACKAGES = {
    "ray": "2.56.1",
    "torch": "2.6.0",
    "transformers": "4.51.3",
    "mujoco": "3.3.1",
    "robosuite": "1.5.2",
    "numpy": "2.2.5",
    "hydra-core": "1.4.0.dev6",
    "omegaconf": "2.4.0.dev13",
}
SUITE_COUNTS = {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16}
EPISODES_PER_TASK = 50
INTERMEDIATE_EPISODES_PER_TASK = 20
ENVS_PER_TASK = 5
SEED = 7
CHUNK = 16
SOURCE_FILES = (
    "evaluations/robocasa365/formal_eval.py",
    "evaluations/robocasa365/robocasa365_mopd_gr00t_formal.yaml",
    "evaluations/robocasa365/target50_manifest.json",
    "examples/embodiment/config/env/robocasa365.yaml",
    "examples/embodiment/config/model/gr00t.yaml",
    "rlinf/config.py",
    "rlinf/runners/embodied_eval_runner.py",
    "rlinf/workers/env/env_worker.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "rlinf/envs/action_utils.py",
    "rlinf/envs/robocasa/venv.py",
    "rlinf/envs/robocasa365/robocasa365_env.py",
    "rlinf/envs/robocasa365/utils.py",
    "rlinf/utils/metric_utils.py",
    "rlinf/utils/placement.py",
    "rlinf/models/embodiment/gr00t/simulation_io.py",
    "rlinf/models/embodiment/gr00t/embodiment_tags.py",
    "rlinf/models/embodiment/gr00t/gr00t_n1d5/__init__.py",
    "rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py",
    "rlinf/models/embodiment/gr00t/gr00t_n1d5/modality_config.py",
)


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Identify all in-repository files that determine the evaluation result."""
    return {name: sha256(REPO / name) for name in SOURCE_FILES}


def base_model_hashes(base: Path) -> dict[str, str]:
    """Identify every loaded HF model shard and its index."""
    files = [*base.glob("*.safetensors"), *base.glob("*.safetensors.index.json")]
    if not any(path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError(f"No safetensors model shards in {base}")
    return {path.name: sha256(path) for path in sorted(files)}


def manifest_tasks() -> list[dict[str, str | int]]:
    """Load the exact 50 tasks and horizons selected for this protocol."""
    suites = json.loads(MANIFEST.read_text())
    if set(suites) != set(SUITE_COUNTS):
        raise ValueError("The frozen suite set changed")
    tasks = []
    for suite, count in SUITE_COUNTS.items():
        entries = suites[suite]
        if len(entries) != count:
            raise ValueError(f"Expected {count} tasks in {suite}")
        for name, horizon in entries.items():
            if not isinstance(horizon, int) or horizon < 1:
                raise ValueError(f"Invalid horizon for {name}: {horizon}")
            tasks.append({"suite": suite, "task": name, "horizon": horizon})
    if len({task["task"] for task in tasks}) != 50:
        raise ValueError("The frozen task list contains duplicates")
    return tasks


def validate_registry(tasks: list[dict[str, str | int]]) -> None:
    """Reject changed RoboCasa task membership or task horizons."""
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    for suite in SUITE_COUNTS:
        expected = {
            str(task["task"]): int(task["horizon"])
            for task in tasks
            if task["suite"] == suite
        }
        if set(TASK_SET_REGISTRY[suite]) != set(expected):
            raise ValueError(f"RoboCasa registry task drift in {suite}")
        for name, horizon in expected.items():
            if int(get_task_horizon(name)) != horizon:
                raise ValueError(f"RoboCasa registry horizon drift for {name}")


def git_output(path: Path, *args: str) -> str:
    """Read Git metadata from a source checkout."""
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def validate_runtime() -> dict[str, str]:
    """Require the standard local official checkouts, never a cold-start fork."""
    runtime = Path(os.environ["OPD_RUNTIME_ROOT"]).resolve(strict=True)
    result = {"runtime_root": str(runtime)}
    for package, commit in EXPECTED_COMMITS.items():
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Cannot import {package} from the OPD runtime")
        actual = Path(spec.origin).resolve().parents[1]
        expected = (runtime / "repos" / package).resolve(strict=True)
        if actual != expected:
            raise RuntimeError(f"{package} imports from {actual}, expected {expected}")
        if git_output(actual, "rev-parse", "HEAD") != commit:
            raise RuntimeError(f"{package} is not at frozen commit {commit}")
        result[f"{package}_source"] = str(actual)
        result[f"{package}_commit"] = commit
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if versions != EXPECTED_PACKAGES:
        raise RuntimeError(f"Frozen package versions changed: {versions}")
    result["package_versions"] = versions
    return result


def check_checkpoint(base: Path, weights: Path | None) -> None:
    """Check model metadata and the optional RLinf full-weight override."""
    for name in ("config.json", "experiment_cfg/metadata.json"):
        if not (base / name).is_file():
            raise FileNotFoundError(base / name)
    if weights is not None and not weights.is_file():
        raise FileNotFoundError(weights)


def parse_gpus(value: str) -> list[int]:
    """Parse distinct physical GPU IDs."""
    ids = [int(part) for part in value.split(",")]
    if not ids or any(gpu < 0 for gpu in ids) or len(ids) != len(set(ids)):
        raise ValueError("Use distinct nonnegative physical GPU IDs")
    return ids


def rounded_budget(horizon: int) -> int:
    """Round only the rollout tensor length, not the episode physics horizon."""
    return math.ceil(horizon / CHUNK) * CHUNK


def compose_task_config(plan: dict, task: dict, gpu: int):
    """Resolve a single-task config under the frozen task-native protocol."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    task_dir = Path(plan["output"]) / "tasks" / task["task"]
    os.environ.update(
        EMBODIED_PATH=str(REPO / "examples" / "embodiment"),
        OPD_RUN_DIR=str(task_dir),
        MOPD_PLACEMENT=str(gpu),
    )
    with initialize_config_dir(version_base="1.1", config_dir=str(EVAL_DIR)):
        cfg = compose(config_name=CONFIG_NAME)
    cfg.rollout.model.model_path = plan["base_checkpoint"]
    cfg.runner.ckpt_path = plan["weights"]
    cfg.runner.logger.experiment_name = task["task"]
    cfg.env.eval.task_soup = [task["suite"]]
    cfg.env.eval.task_filter = [task["task"]]
    cfg.env.eval.total_num_envs = plan["envs_per_task"]
    cfg.env.eval.rollout_epoch = plan["episodes_per_task"] // plan["envs_per_task"]
    cfg.env.eval.seed = plan["seed"]
    cfg.env.eval.max_episode_steps = rounded_budget(task["horizon"])
    cfg.env.eval.max_steps_per_rollout_epoch = rounded_budget(task["horizon"])
    cfg.env.eval.video_cfg.save_video = plan["save_video"]
    cfg.cluster.component_placement["env,rollout"] = str(gpu)
    OmegaConf.resolve(cfg)
    if cfg.env.eval.episode_horizon_source != "task_horizon" or cfg.env.eval.auto_reset:
        raise ValueError(
            "Formal evaluation must stop at native task horizons without auto-reset"
        )
    if (
        not cfg.env.eval.ignore_terminations
        or not cfg.env.eval.use_fixed_reset_state_ids
    ):
        raise ValueError("Formal evaluation success/reset semantics changed")
    return cfg


def checked_metrics(metrics: dict, episodes: int, horizon: int | None = None) -> dict:
    """Require complete binary outcomes and the native episode length."""
    plain = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in metrics.items()
    }
    if plain.get("num_trajectories") != episodes:
        raise ValueError(f"Expected {episodes} trajectories, got {plain}")
    if horizon is not None and not math.isclose(
        float(plain.get("episode_len", -1)), horizon, abs_tol=1e-6
    ):
        raise ValueError(f"Expected native episode length {horizon}, got {plain}")
    for key in ("success_once", "success_at_end"):
        value = float(plain[key])
        count = round(value * episodes)
        if not 0 <= value <= 1 or abs(value - count / episodes) > 1e-6:
            raise ValueError(f"Invalid binary metric {key}={value}")
        plain[f"{key}_count"] = count
    return plain


def validate_plan(plan: dict) -> None:
    """Refuse a plan with protocol, source, or episode-budget drift."""
    if plan["protocol"] != PROTOCOL or plan["source_sha256"] != source_hashes():
        raise RuntimeError("Evaluation protocol source changed since planning")
    if plan["manifest_sha256"] != sha256(MANIFEST):
        raise RuntimeError("Frozen task manifest changed since planning")
    if plan["seed"] != SEED:
        raise RuntimeError("Formal evaluation seed changed")
    expected = manifest_tasks()
    if plan["mode"] in {"formal", "intermediate"}:
        budget = (
            EPISODES_PER_TASK
            if plan["mode"] == "formal"
            else INTERMEDIATE_EPISODES_PER_TASK
        )
        if plan["tasks"] != expected or plan["episodes_per_task"] != budget:
            raise RuntimeError("Formal task/episode budget changed")
        if plan["envs_per_task"] != ENVS_PER_TASK:
            raise RuntimeError("Formal env count changed")
        if (
            Path(plan["output"]).resolve().parent
            != (REPO / "outputs" / "eval").resolve()
        ):
            raise RuntimeError("Formal output must be under outputs/eval")
    elif plan["mode"] == "smoke":
        if len(plan["tasks"]) != 1 or plan["tasks"][0] not in expected:
            raise RuntimeError("Smoke task is not from the frozen manifest")
        if plan["envs_per_task"] != 1:
            raise RuntimeError("Smoke env count changed")
        if (
            Path(plan["output"]).resolve().parent
            != (REPO / "outputs" / "smoke").resolve()
        ):
            raise RuntimeError("Smoke output must be under outputs/smoke")
    else:
        raise RuntimeError("Unknown evaluation mode")
    if (
        plan["envs_per_task"] < 1
        or plan["episodes_per_task"] < 1
        or plan["episodes_per_task"] % plan["envs_per_task"]
    ):
        raise RuntimeError("Episode count must be divisible by env count")
    if (
        not plan["gpus"]
        or any(gpu < 0 for gpu in plan["gpus"])
        or len(set(plan["gpus"])) != len(plan["gpus"])
    ):
        raise RuntimeError("Invalid GPU assignment")


def make_plan(args: argparse.Namespace) -> Path:
    """Write an immutable run plan and its code/runtime provenance."""
    tasks = manifest_tasks()
    runtime = validate_runtime()
    validate_registry(tasks)
    base = args.base_checkpoint.resolve(strict=True)
    weights = args.weights.resolve(strict=True) if args.weights else None
    check_checkpoint(base, weights)
    smoke = args.smoke_task is not None
    intermediate = getattr(args, "intermediate", False)
    if smoke and intermediate:
        raise ValueError("Smoke and intermediate evaluation are mutually exclusive")
    if smoke:
        tasks = [task for task in tasks if task["task"] == args.smoke_task]
        if len(tasks) != 1:
            raise ValueError(f"Unknown smoke task: {args.smoke_task}")
    root = REPO / "outputs" / ("smoke" if smoke else "eval")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("run-id must be a single safe directory name")
    if smoke and args.smoke_episodes < 1:
        raise ValueError("Smoke episodes must be positive")
    output = root / args.run_id
    plan = {
        "protocol": PROTOCOL,
        "mode": "smoke" if smoke else ("intermediate" if intermediate else "formal"),
        "output": str(output),
        "hostname": socket.gethostname(),
        "rlinf_git_commit": git_output(REPO, "rev-parse", "HEAD"),
        "source_sha256": source_hashes(),
        "manifest_sha256": sha256(MANIFEST),
        "runtime": runtime,
        "base_checkpoint": str(base),
        "base_config_sha256": sha256(base / "config.json"),
        "base_metadata_sha256": sha256(base / "experiment_cfg" / "metadata.json"),
        "base_model_sha256": base_model_hashes(base),
        "weights": str(weights) if weights else None,
        "weights_sha256": sha256(weights) if weights else None,
        "tasks": tasks,
        "episodes_per_task": (
            args.smoke_episodes
            if smoke
            else (INTERMEDIATE_EPISODES_PER_TASK if intermediate else EPISODES_PER_TASK)
        ),
        "envs_per_task": 1 if smoke else ENVS_PER_TASK,
        "seed": SEED,
        "gpus": parse_gpus(args.gpus),
        "save_video": args.save_video,
        "metric": "RLinf independent eval/success_once",
        "horizon_rule": "task-native physics horizon; tensor budget rounded to 16",
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    (output / "status").write_text("PLANNED\n")
    print(f"PLAN {path}: {len(tasks)} tasks x {plan['episodes_per_task']} episodes")
    return path


def evaluate_task(plan: dict, task_name: str, gpu: int) -> None:
    """Run one exact task through the ordinary RLinf evaluation components."""
    import torch.multiprocessing as mp
    from omegaconf import OmegaConf

    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    validate_plan(plan)
    task = next(entry for entry in plan["tasks"] if entry["task"] == task_name)
    output = Path(plan["output"]) / "tasks" / task_name
    mp.set_start_method("spawn", force=True)
    Cluster.NAMESPACE = (
        "opdformal_" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    )
    cfg = compose_task_config(plan, task, gpu)
    OmegaConf.save(cfg, output / "requested_config.yaml")
    cfg = validate_cfg(cfg)
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)

    class FormalEvalRolloutWorker(MultiStepRolloutWorker):
        """Seed flow-matching noise after loading the selected model weights."""

        def init_worker(self):
            result = super().init_worker()
            from rlinf.utils.utils import seed_everything

            seed_everything(int(plan["seed"]) + self._rank)
            return result

    rollout = FormalEvalRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )

    class CapturedRunner(EmbodiedEvalRunner):
        def evaluate(self):
            metrics = super().evaluate()
            checked = checked_metrics(
                metrics, plan["episodes_per_task"], task["horizon"]
            )
            (output / "metrics.json").write_text(
                json.dumps(
                    {
                        "task": task_name,
                        "suite": task["suite"],
                        "gpu": gpu,
                        "metrics": checked,
                    },
                    indent=2,
                )
                + "\n"
            )
            return metrics

    runner = CapturedRunner(cfg=cfg, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


def launch_task(plan: dict, plan_path: Path, task: dict, gpu: int) -> None:
    """Run one task in an isolated driver and retain status/logs on failure."""
    output = Path(plan["output"]) / "tasks" / task["task"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "status").write_text("RUNNING\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "task",
        "--plan",
        str(plan_path),
        "--task",
        task["task"],
        "--gpu",
        str(gpu),
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "driver.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        (output / "status").write_text(f"EXIT_{result.returncode}\n")
        raise RuntimeError(f"{task['task']} failed; see {output / 'driver.log'}")
    if not (output / "metrics.json").is_file():
        (output / "status").write_text("MISSING_METRICS\n")
        raise RuntimeError(f"{task['task']} produced no metrics")
    (output / "status").write_text("OK\n")
    print(f"COMPLETE {task['task']} on GPU {gpu}", flush=True)


def summarize(plan: dict) -> dict:
    """Publish suite/overall scores only after every episode is accounted for."""
    validate_plan(plan)
    output = Path(plan["output"])
    rows = []
    for task in plan["tasks"]:
        task_dir = output / "tasks" / task["task"]
        if (task_dir / "status").read_text().strip() != "OK":
            raise RuntimeError(f"Task incomplete: {task['task']}")
        result = json.loads((task_dir / "metrics.json").read_text())
        if result["task"] != task["task"] or result["suite"] != task["suite"]:
            raise RuntimeError(f"Task identity mismatch: {task['task']}")
        metrics = checked_metrics(
            result["metrics"], plan["episodes_per_task"], task["horizon"]
        )
        rows.append(
            {
                "suite": task["suite"],
                "task": task["task"],
                "horizon": task["horizon"],
                "episodes": plan["episodes_per_task"],
                "success_once_count": metrics["success_once_count"],
                "success_once": metrics["success_once"],
                "success_at_end_count": metrics["success_at_end_count"],
                "success_at_end": metrics["success_at_end"],
            }
        )
    with (output / "per_task_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    suites = {}
    for suite in SUITE_COUNTS:
        selected = [row for row in rows if row["suite"] == suite]
        if not selected:
            continue
        suites[suite] = {
            "tasks": len(selected),
            "episodes": sum(row["episodes"] for row in selected),
            "successes": sum(row["success_once_count"] for row in selected),
            "success_once": sum(row["success_once_count"] for row in selected)
            / sum(row["episodes"] for row in selected),
        }
    summary = {
        "protocol": PROTOCOL,
        "mode": plan["mode"],
        "tasks": len(rows),
        "episodes": sum(row["episodes"] for row in rows),
        "successes": sum(row["success_once_count"] for row in rows),
        "success_once": sum(row["success_once_count"] for row in rows)
        / sum(row["episodes"] for row in rows),
        "suites": suites,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(plan: dict, plan_path: Path) -> None:
    """Dispatch one task first, then distribute remaining tasks across GPUs."""
    validate_plan(plan)
    if not os.environ.get("RAY_ADDRESS"):
        raise RuntimeError(
            "Start an isolated Ray cluster and set RAY_ADDRESS before run"
        )
    if os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError(
            "Set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 before ray start"
        )
    if validate_runtime() != plan["runtime"]:
        raise RuntimeError("OPD runtime changed since planning")
    base = Path(plan["base_checkpoint"])
    if sha256(base / "config.json") != plan["base_config_sha256"]:
        raise RuntimeError("Base config changed since planning")
    if (
        sha256(base / "experiment_cfg" / "metadata.json")
        != plan["base_metadata_sha256"]
    ):
        raise RuntimeError("Base normalization metadata changed since planning")
    if base_model_hashes(base) != plan["base_model_sha256"]:
        raise RuntimeError("Base model weights changed since planning")
    if plan["weights"] and sha256(Path(plan["weights"])) != plan["weights_sha256"]:
        raise RuntimeError("Evaluation weights changed since planning")
    output = Path(plan["output"])
    (output / "status").write_text("RUNNING\n")
    try:
        tasks = plan["tasks"]
        gpus = plan["gpus"]
        launch_task(plan, plan_path, tasks[0], gpus[0])
        remaining = tasks[1:]
        if remaining:
            shards = [remaining[i :: len(gpus)] for i in range(len(gpus))]

            def run_shard(gpu: int, shard: list[dict]) -> None:
                for task in shard:
                    launch_task(plan, plan_path, task, gpu)

            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = [
                    pool.submit(run_shard, gpu, shard)
                    for gpu, shard in zip(gpus, shards, strict=True)
                ]
                for future in futures:
                    future.result()
        print(json.dumps(summarize(plan), indent=2), flush=True)
        (output / "status").write_text(
            {
                "formal": "OK\n",
                "intermediate": "OK_INTERMEDIATE\n",
                "smoke": "OK_SMOKE\n",
            }[plan["mode"]]
        )
    except BaseException:
        (output / "status").write_text("FAILED\n")
        raise


def main() -> None:
    """Parse the explicit plan, run, task, and summary subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("plan")
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--base-checkpoint", type=Path, required=True)
    prep.add_argument("--weights", type=Path)
    prep.add_argument("--gpus", default="0")
    prep.add_argument("--save-video", action="store_true")
    prep.add_argument(
        "--intermediate",
        action="store_true",
        help="Evaluate all 50 tasks with 20 episodes each; final defaults to 50",
    )
    prep.add_argument("--smoke-task")
    prep.add_argument("--smoke-episodes", type=int, default=1)
    for name in ("run", "task", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        if name == "task":
            command.add_argument("--task", required=True)
            command.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args)
        return
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.mode == "run":
        run(plan, args.plan.resolve())
    elif args.mode == "task":
        if (
            args.task not in {task["task"] for task in plan["tasks"]}
            or args.gpu not in plan["gpus"]
        ):
            raise ValueError("Task/GPU not in the frozen run plan")
        evaluate_task(plan, args.task, args.gpu)
    else:
        print(json.dumps(summarize(plan), indent=2))


if __name__ == "__main__":
    main()
