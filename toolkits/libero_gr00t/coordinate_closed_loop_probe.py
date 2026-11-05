# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded paired frozen-teacher closed-loop coordinate-transport diagnostic.

Tasks 0 and 1, initial-state index 0, four suites, native/continuous modes:
16 complete episodes total. No training or student weights. Same episode seed
and per-chunk teacher RNG draws in both modes; each continuous chunk starts from
exactly the native teacher's sampled z. Returned native payloads are retained
only for collect()'s schema checks and MUST NOT be used as training transitions.

Explicit --run-gpu and one free visible CUDA device are required. The JSONL
contains a protocol row, paired episode rows, and an aggregate diagnostic. This
small test detects gross policy damage, not statistically established SR gaps.
"""

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from toolkits.libero_gr00t.coordinate_sampler_audit import decode_action, integrate
from toolkits.libero_gr00t.native_batch_smoke import load_policy
from toolkits.libero_gr00t.probe import SUITES, collect, renderer_index


class FrozenCoordinateOracle:
    """Return native actions or a frozen continuous-coordinate teacher oracle."""

    _probe_expected_state_dim = 8

    def __init__(self, teacher, scale, shift, low, high, mode):
        self.teacher = teacher
        self.scale, self.shift = scale.cuda(), shift.cuda()
        self.low, self.high = low, high
        self.mode = mode
        self.delta_squares, self.max_abs, self.gripper_flips, self.values = (
            0.0,
            0.0,
            0,
            0,
        )

    @torch.no_grad()
    def predict_action_batch(self, env_obs, mode="eval", **kwargs):
        from transformers.feature_extraction_utils import BatchFeature

        actions, payload = self.teacher.predict_action_batch(
            env_obs, mode=mode, **kwargs
        )
        if self.mode == "native":
            return actions, payload
        normalized = dict(payload["forward_inputs"])
        z = normalized["chains"][:, 0].cuda()
        for key in ("eagle_pixel_values", "eagle_image_sizes"):
            normalized[key] = normalized[key].reshape(-1, *normalized[key].shape[2:])
        # Preserve the native sampler's RNG stream for the next physical chunk.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            backbone_input, action_input = self.teacher.prepare_input(normalized)
            cached = self.teacher.backbone(backbone_input)

            def velocity(x, k):
                result = self.teacher.action_head.get_velocity_from_backbone(
                    backbone_output=BatchFeature(data=dict(cached)),
                    action_input=action_input,
                    x_t=x,
                    idx=torch.full((1,), k, dtype=torch.long, device="cuda"),
                    denoise_steps=4,
                )
                if result.shape != (1, 16, 32):
                    raise ValueError(f"Expected full16x32velocity, got {result.shape}")
                return result

            endpoint, _, _ = integrate(
                z, velocity, self.scale, self.shift, "continuous", 4
            )
        transformed = decode_action(endpoint, self.low, self.high, execute=8)
        delta = transformed - actions.cpu().float()
        self.delta_squares += float(delta.square().sum())
        self.max_abs = max(self.max_abs, float(delta.abs().max()))
        self.gripper_flips += int((transformed[..., 6] != actions.cpu()[..., 6]).sum())
        self.values += delta.numel()
        return transformed, payload

    def diagnostics(self):
        return {
            "counterfactual_same_observation_action_rmse": (
                self.delta_squares / self.values
            )
            ** 0.5
            if self.values
            else None,
            "counterfactual_action_max_abs": self.max_abs if self.values else None,
            "counterfactual_gripper_flips": self.gripper_flips if self.values else None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--student", type=Path, required=True, help="Metadata only")
    parser.add_argument(
        "--teachers", type=Path, nargs=4, required=True, help="Spatial Object Goal Long"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=24)
    args = parser.parse_args()
    if not args.run_gpu:
        print(
            "No GPU work performed; explicitly select one free GPU and add --run-gpu."
        )
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly expose exactly one free CUDA device")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.execute, args.denoise_steps = 8, 4
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(
        min(
            0.95,
            args.memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory,
        )
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(renderer_index())
    from libero.libero import benchmark

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write(row):
        with args.output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    write(
        {
            "kind": "protocol",
            "tasks": [0, 1],
            "initial_state": 0,
            "modes": ["native", "continuous"],
            "episodes": 16,
            "seed": args.seed,
            "episode_rng_seed": "seed + task_id * 100 + initial_state",
            "teacher_paths": [str(p) for p in args.teachers],
            "student_metadata": str(args.student),
            "denoise_steps": 4,
            "execute": 8,
            "training": False,
            "native_payload_training_use": False,
            "scope": "small paired frozen-teacher diagnostic, not a formal SR estimate",
        }
    )
    rows = []
    started = time.perf_counter()
    for suite_name, path in zip(SUITES, args.teachers, strict=True):
        teacher = load_policy(path, args, native=False).cuda()
        from gr00t.data.schema import DatasetMetadata

        from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
            _libero_action_bounds,
        )

        raw_metadata = json.loads(
            (args.student / "experiment_cfg/metadata.json").read_text()
        )
        if len(raw_metadata) != 1:
            raise ValueError("Expected one student embodiment")
        metadata = DatasetMetadata.model_validate(next(iter(raw_metadata.values())))
        _, low, high = _libero_action_bounds(metadata)
        scale, shift = teacher.opd_action_endpoint_affine_to(metadata)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        args.suite = suite_name
        for task in (0, 1):
            for mode in ("native", "continuous"):
                episode_seed = args.seed + task * 100
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                oracle = FrozenCoordinateOracle(teacher, scale, shift, low, high, mode)
                pool, result = collect(oracle, suite, task, 0, args, max_chunks=None)
                row = {
                    "kind": "episode",
                    "suite": suite_name,
                    "mode": mode,
                    "episode_seed": episode_seed,
                    **result,
                    **oracle.diagnostics(),
                }
                rows.append(row)
                write(row)
                del pool, oracle
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    write(
        {
            "kind": "summary",
            "episodes": len(rows),
            "elapsed_s": time.perf_counter() - started,
            "success_counts": {
                mode: sum(
                    bool(row["success_once"]) for row in rows if row["mode"] == mode
                )
                for mode in ("native", "continuous")
            },
            "episodes_per_mode": 8,
            "completed": True,
        }
    )


if __name__ == "__main__":
    main()
