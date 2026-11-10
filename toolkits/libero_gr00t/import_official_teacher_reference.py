# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bind completed official-prefix calibration to the matching mixed-K training recipe."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.utils.execution_protocol import suite_execution_lengths
from rlinf.utils.rollout_allocation import load_teacher_reference, training_signature
from toolkits.libero_gr00t.teacher_reference import checkpoint_identity, digest

CASES = {
    "libero_spatial": "spatial_K5",
    "libero_object": "object_K5",
    "libero_goal": "goal_K5",
    "libero_10": "long_K10",
}


def import_reference(config: Path, capture: Path, output: Path) -> dict:
    """Verify raw counts/protocol/content; preserve the original calibration provenance."""
    if output.exists():
        raise FileExistsError(output)
    cfg = OmegaConf.load(config)
    lengths = suite_execution_lengths(
        cfg.env.train, int(cfg.actor.model.num_action_chunks)
    )
    if lengths != dict(zip(CASES, [5, 5, 5, 10], strict=True)):
        raise ValueError("Reference import requires the frozen official prefixes")
    state = json.loads((capture / "queue_status.json").read_text())
    manifest = json.loads((capture / "manifest.json").read_text())
    if state["phase"] != "complete" or manifest["initial_states"] != list(range(20)):
        raise ValueError("Require complete20-state/task teacher calibration")
    tasks, paths, hashes = [], {}, {}
    starts, ends = [], []
    for suite, case in CASES.items():
        expert = cfg.rollout.expert_models[suite]
        if (
            expert.get("opd_rotary_precision") != "bf16"
            or expert.action_head_precision != "bf16"
        ):
            raise ValueError("Teacher precision must match the BF16 calibration")
        if (
            not expert.opd_fixed_text_padding
            or expert.embodiment_tag != "libero_franka"
        ):
            raise ValueError("Teacher observation contract differs from calibration")
        path = Path(expert.model_path).resolve(strict=True)
        if str(path) != manifest["cases"][case]["checkpoint"]:
            raise ValueError("Teacher path differs from calibration")
        if checkpoint_identity(path) != manifest["teacher_files_sha256"][suite]:
            raise ValueError("Teacher weights/metadata differ from calibration")
        paths[suite] = str(path)
        folder = capture / case
        protocol = json.loads((folder / "protocol.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        starts.append(float(protocol["start_time"]))
        ends.append(float(summary["finished_at"]))
        rows = [
            json.loads(line)
            for line in (folder / "episodes.jsonl").read_text().splitlines()
        ]
        horizon = int(cfg.env.train.task_suite_horizons[suite])
        if protocol["execute"] != lengths[suite] or protocol["horizon"] != horizon:
            raise ValueError("Calibration execution/horizon mismatch")
        if summary["status"] != "COMPLETE" or summary["episode_file_sha256"] != digest(
            folder / "episodes.jsonl"
        ):
            raise ValueError("Calibration rows changed or never completed")
        if len(rows) != 200 or {(r["task_id"], r["trial"]) for r in rows} != {
            (t, i) for t in range(10) for i in range(20)
        }:
            raise ValueError("Calibration task/state coverage is incomplete")
        for row in rows:
            if (
                row["suite"] != suite
                or row["execute"] != lengths[suite]
                or row["horizon"] != horizon
                or row["complete_episode"] is not True
                or type(row["success_once"]) is not bool
                or not 1 <= row["steps"] <= horizon
                or (not row["success_once"] and row["steps"] != horizon)
            ):
                raise ValueError("Calibration contains invalid or truncated episodes")
        for task_id in range(10):
            selected = [r for r in rows if r["task_id"] == task_id]
            tasks.append(
                {
                    "task": f"{suite}/{task_id}",
                    "suite": suite,
                    "horizon": horizon,
                    "episodes": 20,
                    "successes": sum(r["success_once"] for r in selected),
                    "simulation_steps": sum(r["steps"] for r in selected),
                    "settling_steps": 200,
                }
            )
        hashes[case] = digest(folder / "episodes.jsonl")
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "wall_seconds": max(ends) - min(starts),
        "tasks": tasks,
        "training_signature": training_signature(cfg),
        "teacher_checkpoints": paths,
        "calibration_budget": {
            "trajectories": 800,
            "simulation_steps": sum(t["simulation_steps"] for t in tasks),
            "settling_steps": 8000,
        },
        "provenance": {
            "kind": "Import completed independent official-prefix calibration; no new physics",
            "capture": str(capture.resolve()),
            "capture_manifest_sha256": digest(capture / "manifest.json"),
            "episode_files_sha256": hashes,
            "note": "Full16 latent is unchanged by returned storage width10; physics executes the calibrated5/5/5/10 prefixes. Real mixed-train-path acceptance remains separate.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pending")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    load_teacher_reference(str(temporary), cfg, tasks)
    temporary.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_reference(args.config, args.capture, args.output)


if __name__ == "__main__":
    main()
