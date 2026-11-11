# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export and bind teacher calibration by content without collecting new physics."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from rlinf.utils.rollout_allocation import load_teacher_reference, training_signature
from toolkits.libero_gr00t import launch_production as production
from toolkits.libero_gr00t.run_uniform import machine_config
from toolkits.libero_gr00t.teacher_reference import checkpoint_identity, digest

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = (240, 240, 320, 480)


def tasks() -> list[dict]:
    """Describe all forty fixed tasks without importing/starting a simulator."""
    return [
        {"task": f"{suite}/{task}", "suite": suite, "horizon": HORIZONS[index]}
        for index, suite in enumerate(SUITES)
        for task in range(10)
    ]


def publish(path: Path, result: dict) -> None:
    """Write a new artifact atomically; identical repeated binds are idempotent."""
    text = json.dumps(result, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if path.read_text() != text:
                raise FileExistsError(f"Existing reference differs: {path}")
            return
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(text)
        temporary.replace(path)


def validate_budget(reference: dict) -> None:
    """Require the calibration counters consumed by the training runner."""
    wall = reference.get("wall_seconds")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Calibration needs its measured nonnegative wall_seconds")
    if any(
        type(row.get("simulation_steps")) is not int
        or not row["episodes"]
        <= row["simulation_steps"]
        <= row["episodes"] * row["horizon"]
        for row in reference["tasks"]
    ):
        raise ValueError("Calibration needs valid measured simulation_steps per task")


def export_bundle(
    config: Path, reference: Path, output: Path, minimum: int = 50
) -> dict:
    """Package validated SR counts and teacher hashes, omitting personal paths."""
    cfg = OmegaConf.load(config)
    load_teacher_reference(str(reference), cfg, tasks())
    source = json.loads(reference.read_text())
    validate_budget(source)
    if minimum < 1 or any(row["episodes"] < minimum for row in source["tasks"]):
        raise ValueError(
            f"Every teacher task needs at least {minimum} calibrated episodes"
        )
    result = {
        "schema_version": 1,
        "minimum_episodes_per_task": minimum,
        "source_reference_sha256": digest(reference),
        "teacher_files_sha256": {
            suite: checkpoint_identity(
                Path(cfg.rollout.expert_models[suite].model_path)
            )
            for suite in SUITES
        },
        "reference": {
            key: source[key]
            for key in (
                "schema_version",
                "status",
                "training_signature",
                "tasks",
                "wall_seconds",
                "calibration_budget",
            )
            if key in source
        },
        "calibration_provenance": {
            key: source.get("provenance", {})[key]
            for key in (
                "capture_manifest_sha256",
                "episode_files_sha256",
                "source_sha256",
            )
            if key in source.get("provenance", {})
        },
    }
    publish(output, result)
    return result


def bind_config(machine: Path, repeats: int, output: Path):
    """Compose the actual Adaptive protocol before relocating calibration."""
    settings = machine_config(machine)
    paths = {
        "student": Path(settings["student"]),
        "libero_spatial": Path(settings["spatial_teacher"]),
        **{
            suite: Path(settings["models"]) / folder
            for suite, folder in (
                ("libero_object", "object"),
                ("libero_goal", "goal"),
                ("libero_10", "long"),
            )
        },
    }
    args = SimpleNamespace(
        python=Path(settings["python"]),
        output=output.parent,
        gpus=settings["gpus"],
        seed=0,
        config_name=f"libero_mopd_adaptive_release_r{repeats}",
        updates=100 * repeats,
        save_every_updates=10 * repeats,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        native_microbatch_size=settings["microbatch_size"],
        skip_finished_rollout=True,
        actor_cache_retention_gib=settings["actor_cache_retention_gib"],
        actor_cache_preallocate_gib=settings["actor_cache_preallocate_gib"],
    )
    # Only the file destination is unresolved during binding; it is not read by
    # config composition. Restore the caller's process-local environment later.
    previous = os.environ.get("OPD_LIBERO_TEACHER_REFERENCE")
    os.environ["OPD_LIBERO_TEACHER_REFERENCE"] = str(output.absolute())
    try:
        return production.build_config(args, paths)
    finally:
        if previous is None:
            os.environ.pop("OPD_LIBERO_TEACHER_REFERENCE", None)
        else:
            os.environ["OPD_LIBERO_TEACHER_REFERENCE"] = previous


def bind_bundle(bundle_path: Path, cfg, output: Path) -> dict:
    """Change only checkpoint locations after exact content/protocol verification."""
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("schema_version") != 1:
        raise ValueError("Unknown teacher reference bundle schema")
    reference = dict(bundle["reference"])
    validate_budget(reference)
    if reference["training_signature"] != training_signature(cfg):
        raise ValueError(
            "Calibration protocol differs; relocation cannot alter inference semantics"
        )
    minimum = bundle["minimum_episodes_per_task"]
    if (
        type(minimum) is not int
        or minimum < 1
        or any(row["episodes"] < minimum for row in reference["tasks"])
    ):
        raise ValueError(
            "Teacher episode coverage differs from the declared bundle minimum"
        )
    paths = {}
    for suite in SUITES:
        path = Path(cfg.rollout.expert_models[suite].model_path).resolve(strict=True)
        if checkpoint_identity(path) != bundle["teacher_files_sha256"][suite]:
            raise ValueError(f"Teacher checkpoint content differs: {suite}")
        paths[suite] = str(path)
    reference["teacher_checkpoints"] = paths
    reference["provenance"] = {
        "kind": "Content-verified relocation; no new environment interaction",
        "bundle_sha256": digest(bundle_path),
        "source_reference_sha256": bundle["source_reference_sha256"],
        "teacher_files_sha256": bundle["teacher_files_sha256"],
        "calibration_provenance": bundle.get("calibration_provenance", {}),
    }
    # Use the real shared loader on a private temporary file before publication.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "reference.json"
        candidate.write_text(json.dumps(reference))
        load_teacher_reference(str(candidate), cfg, tasks())
    publish(output, reference)
    return reference


def main() -> None:
    """Export after owner calibration; bind locally through the shared runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--reference", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--minimum-episodes", type=int, default=50)
    bind = commands.add_parser("bind")
    bind.add_argument("--bundle", type=Path, required=True)
    bind.add_argument("--machine", type=Path, required=True)
    bind.add_argument("--r", type=int, choices=(1, 2, 4), required=True)
    bind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_bundle(args.config, args.reference, args.output, args.minimum_episodes)
    else:
        bind_bundle(
            args.bundle, bind_config(args.machine, args.r, args.output), args.output
        )


if __name__ == "__main__":
    main()
