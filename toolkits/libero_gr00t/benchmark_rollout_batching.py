# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-checkpoint native batching parity and latency, without physics."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    restore_opd_text_padding,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d5.native_rollout import (
    native_rollout_plan,
    pack_native_rollout_inputs,
    slice_rollout_result,
)
from toolkits.libero_gr00t.evaluate_full_checkpoint import candidate_model_config
from toolkits.libero_gr00t.offline_observations import (
    SUITES,
    ObservationPool,
    file_hash,
)


def cpu_tree(value):
    """Snapshot inference payloads before another case reuses device memory."""
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


def raw_batch(rows: list[dict]) -> dict:
    """Concatenate already oriented, unnormalized decision observations."""
    return {
        key: sum((row[key] for row in rows), [])
        if key == "task_descriptions"
        else torch.cat([row[key] for row in rows])
        for key in rows[0]
    }


def errors(value: torch.Tensor, reference: torch.Tensor) -> dict:
    """Report absolute and relative differences without hiding BF16 roundoff."""
    difference = value.float() - reference.float()
    return {
        "maximum_absolute": float(difference.abs().max()),
        "mean_absolute": float(difference.abs().mean()),
        "relative_l2": float(
            difference.norm() / reference.float().norm().clamp_min(1e-8)
        ),
    }


@torch.no_grad()
def fixed_state_velocities(model, payload: dict, limit: int) -> torch.Tensor:
    """Compare vectors at the same baseline flow states, not displaced trajectories."""
    count = payload["chains"].shape[0]
    rows = [
        restore_opd_text_padding(slice_rollout_result(payload, i)) for i in range(count)
    ]
    result = [None] * count
    for indices in native_rollout_plan(rows, limit):
        inputs = pack_native_rollout_inputs(rows, indices)
        from transformers.feature_extraction_utils import BatchFeature

        normalized = {
            key: inputs[key]
            for key in [
                "state",
                "state_mask",
                "eagle_input_ids",
                "eagle_attention_mask",
                "embodiment_id",
            ]
        }
        for key in ["eagle_pixel_values", "eagle_image_sizes"]:
            normalized[key] = inputs[key].reshape(-1, *inputs[key].shape[2:])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_inputs, action_inputs = model.prepare_input(normalized)
            backbone = model.backbone(backbone_inputs)
            if (
                getattr(model, "opd_native_rollout_batch_head", False)
                or len(indices) == 1
            ):
                vectors = model.action_head.get_velocity_from_backbone(
                    backbone,
                    action_inputs,
                    inputs["chains"][:, 1].cuda(),
                    torch.ones(len(indices), device="cuda", dtype=torch.long),
                )
            else:
                vectors = torch.cat(
                    [
                        model.action_head.get_velocity_from_backbone(
                            BatchFeature(
                                data=slice_rollout_result(dict(backbone), row)
                            ),
                            BatchFeature(
                                data=slice_rollout_result(dict(action_inputs), row)
                            ),
                            inputs["chains"][row : row + 1, 1].cuda(),
                            torch.ones(1, device="cuda", dtype=torch.long),
                        )
                        for row in range(len(indices))
                    ]
                )
            vectors = (
                vectors[:, : model.output_action_chunks, : model.valid_action_dim]
                .float()
                .cpu()
            )
        for local, original in enumerate(indices):
            result[original] = vectors[local : local + 1]
    return torch.cat(result)


@torch.no_grad()
def benchmark_case(
    model, rows: list[dict], sizes: list[int], repeats: int, seed: int
) -> dict:
    """Measure one mixed-task pattern with identical per-environment random draws."""
    observation = raw_batch(rows)
    results, baseline = [], None
    for size in sizes:
        model.opd_native_rollout_batch_size = size
        model.predict_action_batch(dict(observation), mode="train")
        torch.cuda.synchronize()
        times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            actions, payload = model.predict_action_batch(
                dict(observation), mode="train"
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        current = {
            "actions": cpu_tree(actions),
            "payload": cpu_tree(payload),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        entry = {
            "maximum_batch_size": size,
            "seconds": times,
            "median_seconds": statistics.median(times),
            "actual_batches": [1] * len(rows)
            if size == 1
            else model.opd_last_native_rollout_batches,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        if baseline is None:
            baseline = current
            baseline["vectors"] = fixed_state_velocities(
                model, current["payload"]["forward_inputs"], 1
            )
            entry["static_gate_passed"] = True
        else:
            before, after = (
                baseline["payload"]["forward_inputs"],
                current["payload"]["forward_inputs"],
            )
            vectors = fixed_state_velocities(model, before, size)
            exact_inputs = all(
                torch.equal(after[key], value)
                for key, value in before.items()
                if key != "chains"
            )
            chains = errors(after["chains"], before["chains"])
            velocities = errors(vectors, baseline["vectors"])
            physical = errors(current["actions"][..., :6], baseline["actions"][..., :6])
            gripper_flips = int(
                (current["actions"][..., 6] != baseline["actions"][..., 6]).sum()
            )
            same_initial = torch.equal(after["chains"][:, 0], before["chains"][:, 0])
            same_rng = torch.equal(
                current["cpu_rng"], baseline["cpu_rng"]
            ) and torch.equal(current["cuda_rng"], baseline["cuda_rng"])
            entry.update(
                exact_inputs=exact_inputs,
                identical_initial_noise=same_initial,
                identical_rng_state=same_rng,
                chains=chains,
                fixed_state_velocity=velocities,
                physical_actions=physical,
                gripper_flips=gripper_flips,
                static_gate_passed=bool(
                    exact_inputs
                    and same_initial
                    and same_rng
                    and gripper_flips == 0
                    and physical["maximum_absolute"] <= 0.005
                    and velocities["relative_l2"] <= 0.01
                ),
            )
        entry["speedup_over_b1"] = (
            results[0]["median_seconds"] / entry["median_seconds"] if results else 1.0
        )
        results.append(entry)
    return {
        "observations": len(rows),
        "task_descriptions": observation["task_descriptions"],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument(
        "--batch-head",
        action="store_true",
        help="Experimental full head batching; default keeps B1 head kernels",
    )
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.sizes[0] != 1
        or min(args.sizes) < 1
        or args.repeats < 1
    ):
        raise ValueError(
            "Use a new output, positive sizes/repeats and B1 as the first reference"
        )
    if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 24 * 2**30:
        raise ValueError("Explicitly select one GPU with at least24GiB free")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        16 * 2**30 / torch.cuda.mem_get_info()[1]
    )
    args.output.mkdir(parents=True)
    cfg, _ = candidate_model_config(args.config)
    cfg.opd_native_rollout_batch_size = 1
    cfg.opd_native_rollout_batch_head = args.batch_head
    model = get_model(cfg).eval().requires_grad_(False).cuda()
    pool = ObservationPool(args.pool)
    sampled = [pool.sample(suite, 4, args.seed + i) for i, suite in enumerate(SUITES)]
    patterns = {
        "mixed_four_suite_16": [sampled[s][i] for i in range(4) for s in range(4)],
        "repeated_task_16": [sampled[s][0] for _ in range(4) for s in range(4)],
    }
    receipt = {
        "complete": False,
        "config": str(args.config.resolve()),
        "config_sha256": file_hash(args.config),
        "pool_manifest_sha256": file_hash(args.pool / "manifest.json"),
        "model_config": OmegaConf.to_container(cfg, resolve=True),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "scope": "Static parity/latency only; no closed-loop success-rate claim",
        "gate": {
            "physical_action_maximum_absolute": 0.005,
            "velocity_relative_l2": 0.01,
            "gripper_flips": 0,
            "inputs_noise_rng": "exact",
        },
        "cases": {},
    }
    for name, rows in patterns.items():
        receipt["cases"][name] = benchmark_case(
            model, rows, args.sizes, args.repeats, args.seed
        )
        (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps({"case": name, "results": receipt["cases"][name]["results"]}),
            flush=True,
        )
    receipt["complete"] = True
    receipt["static_gate_passed"] = all(
        row["static_gate_passed"]
        for case in receipt["cases"].values()
        for row in case["results"]
    )
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
