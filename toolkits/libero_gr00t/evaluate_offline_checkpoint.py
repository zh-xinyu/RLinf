# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate an offline student with the recorded suite-specific execution protocol."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from toolkits.libero_gr00t import probe
from toolkits.libero_gr00t.collect_teacher_trajectories import PROTOCOLS
from toolkits.libero_gr00t.offline_observations import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, nargs="+", default=[40, 41])
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError(cli.output)
    if torch.cuda.device_count() != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Expose exactly one evaluation GPU")
    cli.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        16 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    args = SimpleNamespace(
        student=cli.checkpoint,
        student_head_precision="fp32",
        student_text_padding="native",
        execute=10,
        denoise_steps=4,
        noise_level=0.3,
        seed=1234,
    )
    model = probe.load_model(cli.checkpoint, args).cuda().eval().requires_grad_(False)
    from rlinf.models.embodiment.gr00t.gr00t_n1d5.memory_utils import (
        enable_feature_only_backbone,
    )

    enable_feature_only_backbone(model)
    egl = probe.renderer_index()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    from libero.libero import benchmark

    protocol = {
        "checkpoint": str(cli.checkpoint),
        "trials": cli.trials,
        "tasks": list(range(10)),
        "seed": 1234,
        "suites": PROTOCOLS,
        "student_padding": "native",
        "head_precision": "fp32",
        "metadata_sha256": file_hash(cli.checkpoint / "experiment_cfg/metadata.json"),
        "scope": f"{40 * len(cli.trials)}-episode pilot, reset states disjoint from teacher collection0..29",
    }
    (cli.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for name, (execute, horizon) in PROTOCOLS.items():
        args.suite, args.execute = name, execute
        model.output_action_chunks = execute
        model.action_head.action_chunk = execute
        probe.HORIZONS[name] = horizon
        suite = benchmark.get_benchmark_dict()[name]()
        for trial in cli.trials:
            for task in range(10):
                seed = args.seed + task * 100 + trial
                torch.manual_seed(seed)
                np.random.seed(seed)
                pool, row = probe.collect(
                    model, suite, task, trial, args, max_chunks=None
                )
                del pool
                if not row["complete_episode"]:
                    raise RuntimeError("Incomplete evaluation episode")
                row.update(suite=name, execute=execute)
                rows.append(row)
                with (cli.output / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps(
                        {
                            "episodes": len(rows),
                            "successes": sum(r["success_once"] for r in rows),
                        }
                    ),
                    flush=True,
                )
    expected = {(s, t, r) for s in PROTOCOLS for t in range(10) for r in cli.trials}
    if {(r["suite"], r["task_id"], r["trial"]) for r in rows} != expected or len(
        rows
    ) != len(expected):
        raise ValueError("Evaluation coverage mismatch")
    summary = {
        "complete": True,
        "episodes": len(rows),
        "successes": sum(r["success_once"] for r in rows),
        "success_rate": sum(r["success_once"] for r in rows) / len(rows),
        "suite_success_rates": {
            s: np.mean([r["success_once"] for r in rows if r["suite"] == s]).item()
            for s in PROTOCOLS
        },
        "simulation_steps": sum(r["steps"] for r in rows),
    }
    (cli.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
