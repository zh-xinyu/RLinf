# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a saved pilot action head in a new process against its pinned base."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from toolkits.libero_gr00t.pilot import evaluate, load_head
from toolkits.libero_gr00t.probe import SUITES, load_model, renderer_index


def main() -> None:
    """Load the saved artifact and replay its independent evaluation protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--student", type=Path, help="Relocated base checkpoint")
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--tasks", nargs="+", type=int)
    parser.add_argument("--trials", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError(cli.output)
    if torch.cuda.device_count() != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Select exactly one CUDA_VISIBLE_DEVICES explicitly")
    manifest = json.loads((cli.head / "manifest.json").read_text())
    cfg = manifest["recipe"]
    args = SimpleNamespace(**cfg)
    args.student = cli.student or Path(manifest["base_checkpoint"])
    args.suite = cli.suite
    args.eval_tasks = cli.tasks if cli.tasks is not None else cfg["eval_tasks"]
    args.eval_initial_states = (
        cli.trials if cli.trials is not None else cfg["eval_initial_states"]
    )
    if any(task not in range(10) for task in args.eval_tasks):
        raise ValueError("LIBERO suite task indices must be in [0, 9]")
    if any(trial < 0 for trial in args.eval_initial_states):
        raise ValueError("Initial-state indices must be nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(
        min(
            args.memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory,
            1,
        )
    )
    egl = renderer_index()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    student = load_model(args.student, args).cuda().eval().requires_grad_(False)
    load_head(student, cli.head)
    digest = hashlib.sha256()
    with (cli.head / "action_head.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    cli.output.parent.mkdir(parents=True, exist_ok=True)

    def write(row):
        row = {"suite": cli.suite, **row}
        with cli.output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    write(
        {
            "kind": "reload_protocol",
            "head": str(cli.head.resolve()),
            "head_sha256": digest.hexdigest(),
            "base_checkpoint": str(args.student.resolve()),
            "base_repo": manifest["base_repo"],
            "base_revision": manifest["base_revision"],
            "tasks": args.eval_tasks,
            "trials": args.eval_initial_states,
            "precision": args.student_head_precision,
            "text_padding": args.student_text_padding,
            "egl_index": egl,
        }
    )
    evaluate(
        student,
        benchmark.get_benchmark_dict()[cli.suite](),
        args,
        manifest["rollout_round"],
        manifest["optimizer_step"],
        write,
    )


if __name__ == "__main__":
    main()
