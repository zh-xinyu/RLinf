# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real-model live-output/RNG parity and latency for latched terminal slots."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
from toolkits.libero_gr00t.benchmark_rollout_batching import cpu_tree, raw_batch
from toolkits.libero_gr00t.evaluate_full_checkpoint import candidate_model_config
from toolkits.libero_gr00t.offline_observations import (
    SUITES,
    ObservationPool,
    file_hash,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        raise ValueError("Use a fresh output and positive repeats")
    if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 24 * 2**30:
        raise ValueError("Select one GPU with at least24GiB free")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        16 * 2**30 / torch.cuda.mem_get_info()[1]
    )
    cfg, _ = candidate_model_config(args.config)
    cfg.opd_native_rollout_batch_size = 1
    cfg.opd_skip_finished_rollout = True
    model = get_model(cfg).eval().requires_grad_(False).cuda()
    pool = ObservationPool(args.pool)
    groups = [pool.sample(suite, 4, 617 + i) for i, suite in enumerate(SUITES)]
    obs = raw_batch([groups[s][i] for i in range(4) for s in range(4)])
    args.output.mkdir(parents=True)
    receipt = {
        "complete": False,
        "config_sha256": file_hash(args.config),
        "pool_manifest_sha256": file_hash(args.pool / "manifest.json"),
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
        "scope": "Real policy on synthetic latched masks; no physics or success-rate claim",
    }
    for live in [16, 8, 4, 0]:
        mask = torch.arange(16) < live
        reference = None
        for skip in [False, True]:
            model.opd_skip_finished_rollout = skip
            durations = []
            for repeat in range(args.repeats):
                torch.manual_seed(782 + repeat)
                model.predict_action_batch(
                    dict(obs, opd_rollout_active_mask=torch.ones(16, dtype=torch.bool))
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                action, payload = model.predict_action_batch(
                    dict(obs, opd_rollout_active_mask=mask)
                )
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
            current = {
                "action": cpu_tree(action),
                "inputs": cpu_tree(payload["forward_inputs"]),
                "cpu_rng": torch.get_rng_state(),
                "gpu_rng": torch.cuda.get_rng_state(),
            }
            if not skip:
                reference = current
                baseline_seconds = statistics.median(durations)
            else:
                exact_action = torch.equal(
                    current["action"][mask], reference["action"][mask]
                )
                exact_inputs = all(
                    torch.equal(value[mask], reference["inputs"][key][mask])
                    for key, value in current["inputs"].items()
                )
                exact_rng = torch.equal(
                    current["cpu_rng"], reference["cpu_rng"]
                ) and torch.equal(current["gpu_rng"], reference["gpu_rng"])
                case = {
                    "active_rows": live,
                    "total_rows": 16,
                    "baseline_seconds": baseline_seconds,
                    "optimized_seconds": statistics.median(durations),
                    "speedup": baseline_seconds / statistics.median(durations),
                    "live_actions_bitwise_equal": exact_action,
                    "live_payload_bitwise_equal": exact_inputs,
                    "rng_bitwise_equal": exact_rng,
                    "passed": exact_action and exact_inputs and exact_rng,
                    "actual_model_forwards": model.opd_last_active_rollout_rows,
                }
                receipt["cases"].append(case)
                (args.output / "results.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                print(json.dumps(case), flush=True)
    receipt["complete"] = True
    receipt["passed"] = all(case["passed"] for case in receipt["cases"])
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
