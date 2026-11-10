# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Record the former standalone path for exact paired worker-evaluation checks."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from toolkits.libero_gr00t.evaluate_full_checkpoint import (
    candidate_model_config,
    prepare_evaluation_weights,
)
from toolkits.libero_gr00t.parallel_evaluation import (
    EXECUTION,
    HORIZONS,
    SUITES,
    action_digest,
    validate_selection,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--tasks", nargs="+", type=int, default=[0])
    parser.add_argument("--trials", nargs="+", type=int, default=[40, 41])
    args = parser.parse_args()
    validate_selection(args.tasks, args.trials, 1)
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 20 * 2**30:
        raise ValueError("Select one GPU with >=20GiB free")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        16 * 2**30 / torch.cuda.mem_get_info()[1]
    )
    from libero.libero import benchmark

    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
    from toolkits.libero_gr00t import probe

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(probe.renderer_index())
    config, _ = candidate_model_config(args.config)
    config.opd_native_rollout_batch_size = 1
    config.opd_skip_finished_rollout = False
    model = get_model(config).eval().cpu().requires_grad_(False)
    reload = prepare_evaluation_weights(model, args.weights, Path(config.model_path))
    model._probe_expected_state_dim = 11
    model.cuda()
    probe.HORIZONS = dict(zip(SUITES, HORIZONS, strict=True))
    args.output.mkdir(parents=True)
    write_json(args.output / "reload.json", reload)
    rows, trace = [], []
    original = model.predict_action_batch
    current = None

    def record(obs, **kwargs):
        actions, payload = original(obs, **kwargs)
        trace.append(
            {
                "task_id": current[0],
                "trial_id": current[1],
                "actions_sha256": action_digest(actions),
            }
        )
        return actions, payload

    model.predict_action_batch = record
    for task_id in args.tasks:
        suite = benchmark.get_benchmark_dict()[SUITES[task_id // 10]]()
        for trial in args.trials:
            current = task_id, trial
            torch.manual_seed(1234 + (task_id % 10) * 100 + trial)
            episode_args = SimpleNamespace(
                seed=1234, suite=SUITES[task_id // 10], execute=EXECUTION[task_id // 10]
            )
            _, result = probe.collect(
                model, suite, task_id % 10, trial, episode_args, max_chunks=None
            )
            rows.append(
                {
                    "task_id": task_id,
                    "trial_id": trial,
                    "suite": SUITES[task_id // 10],
                    "horizon": result["horizon"],
                    "success_once": result["success_once"],
                    "simulation_steps": result["steps"],
                }
            )
            write_json(args.output / "episodes.json", rows)
            write_json(args.output / "action_trace.json", trace)


if __name__ == "__main__":
    main()
