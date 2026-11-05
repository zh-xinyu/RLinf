# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen four-step teacher sampler audit; no training or simulator interaction.

For F_t(x)=A_t*x+t*b, A_t=1+t*(s-1), native Euler x+=x+h*v:
  F_(t+h)(x+h*v)-F_t(x) = h*[A_(t+h)*v+(s-1)*x+b].
The continuous pushforward instead uses A_t*v and has local state defect
h**2*(s-1)*v. This audit compares both frozen oracle samplers with the same z,
including all 16x32 flow coordinates. Only physical first-8x7 actions execute.

Explicit --run-gpu and exactly one visible free CUDA device are required.
Teachers are loaded sequentially; the student contributes metadata only.
Both a FP32 Euler reference and the official low-precision native sampler are
reported, so solver-coordinate error is not conflated with rounding effects.
"""

import argparse
import copy
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from toolkits.libero_gr00t.native_batch_smoke import load_policy
from toolkits.libero_gr00t.probe import SUITES, observation_batch


def errors(delta):
    value = delta.detach().float().cpu()
    return {
        "rmse": float(value.square().mean().sqrt()),
        "max_abs": float(value.abs().max()),
    }


def physical_errors(actual, expected):
    delta = (actual - expected).float().cpu()
    dims = tuple(range(delta.ndim - 1))
    return {
        **errors(delta),
        "per_dim_rmse": delta.square().mean(dim=dims).sqrt().tolist(),
        "per_dim_max_abs": delta.abs().amax(dim=dims).tolist(),
        "per_dim_signed_mean": delta.mean(dim=dims).tolist(),
        "gripper_disagreement_fraction": float(
            (actual[..., 6] != expected[..., 6]).float().mean()
        ),
    }


def decode_action(x, low, high, execute):
    """Actual min-max metadata, followed by LIBERO's RLDS gripper convention."""
    physical = ((x[..., :execute, :7].cpu().float() + 1) / 2) * (high - low) + low
    physical[..., 6] = torch.sign(1 - 2 * physical[..., 6])
    return physical


def integrate(z, velocity, scale, shift, kind, steps):
    """FP32 Euler oracle; teacher network precision remains unchanged."""
    h = 1.0 / steps
    x = z.float().clone()
    states, velocities = [], []
    for k in range(steps):
        t = k / steps
        a = 1 + t * (scale - 1)
        source = (x - t * shift) / a
        v = velocity(source, k).float()
        states.append(source.detach().clone())
        velocities.append(v.detach().clone())
        if kind == "continuous":
            transported = a * v + (scale - 1) * source + shift
        elif kind == "discrete":
            transported = (a + h * (scale - 1)) * v + (scale - 1) * source + shift
        elif kind == "native":
            transported = v
        else:
            raise ValueError(kind)
        x = x + h * transported
    return x, states, velocities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument(
        "--student",
        type=Path,
        required=True,
        help="Metadata only; no student weights loaded",
    )
    parser.add_argument(
        "--teachers", type=Path, nargs=4, required=True, help="Spatial Object Goal Long"
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="Noise samples per saved suite observation; max32",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=24)
    parser.add_argument("--execute", type=int, default=8)
    parser.add_argument("--denoise-steps", type=int, default=4)
    args = parser.parse_args()
    if not args.run_gpu:
        print(
            "No GPU work performed; pass --run-gpu with one free CUDA_VISIBLE_DEVICES."
        )
        return
    if args.denoise_steps != 4 or args.execute != 8 or not 1 <= args.seeds <= 32:
        raise ValueError(
            "Bounded audit requires four steps, eight executed actions and 1..32 seeds"
        )
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly select exactly one free CUDA device")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(
        min(
            1.0,
            args.memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory,
        )
    )
    report = {
        "scope": "frozen teacher action audit, four saved task0 observations; no success-rate evidence",
        "training_changed": False,
        "seed": args.seed,
        "seeds_per_suite": args.seeds,
        "step_size": 0.25,
        "student_metadata": str(args.student),
        "suites": [],
    }
    for suite_index, (suite, path) in enumerate(
        zip(SUITES, args.teachers, strict=True)
    ):
        teacher = load_policy(path, args, native=False).cuda()
        from gr00t.data.schema import DatasetMetadata
        from transformers.feature_extraction_utils import BatchFeature

        from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
            _libero_action_bounds,
        )

        metadata = json.loads(
            (args.student / "experiment_cfg/metadata.json").read_text()
        )
        if len(metadata) != 1:
            raise ValueError("Expected one student embodiment")
        student_metadata = DatasetMetadata.model_validate(next(iter(metadata.values())))
        _, target_low, target_high = _libero_action_bounds(student_metadata)
        _, source_low, source_high = _libero_action_bounds(teacher.metadata)
        scale, shift = teacher.opd_action_endpoint_affine_to(student_metadata)
        scale, shift = scale.cuda(), shift.cuda()
        with np.load(
            args.observations / f"{suite}_obs.npz", allow_pickle=False
        ) as saved:
            raw = observation_batch(dict(saved), str(saved["task_description"].item()))
        branches = {
            key: []
            for key in (
                "official",
                "native_fp32",
                "continuous",
                "discrete",
                "identity_continuous",
                "identity_discrete",
            )
        }
        local_errors, local_identity_residuals = [], []
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            # Get independent full teacher preprocessing once, then cache VLM output.
            torch.manual_seed(args.seed)
            _, payload = teacher.predict_action_batch(copy.deepcopy(raw), mode="eval")
            normalized = dict(payload["forward_inputs"])
            for key in ("eagle_pixel_values", "eagle_image_sizes"):
                normalized[key] = normalized[key].reshape(
                    -1, *normalized[key].shape[2:]
                )
            backbone_input, action_input = teacher.prepare_input(normalized)
            cached_backbone = teacher.backbone(backbone_input)

            def velocity(
                x, k, policy=teacher, backbone=cached_backbone, inputs=action_input
            ):
                # process_backbone_output mutates its BatchFeature dictionary;
                # a fresh shallow container prevents applying VL attention twice.
                fresh_backbone = BatchFeature(data=dict(backbone))
                result = policy.action_head.get_velocity_from_backbone(
                    backbone_output=fresh_backbone,
                    action_input=inputs,
                    x_t=x,
                    idx=torch.full((1,), k, dtype=torch.long, device="cuda"),
                    denoise_steps=4,
                )
                if result.shape != (1, 16, 32):
                    raise ValueError(
                        f"Audit requires complete16x32velocity, got {result.shape}"
                    )
                return result

            for seed_index in range(args.seeds):
                torch.manual_seed(args.seed + suite_index * 1000 + seed_index)
                native_actions, native_payload = teacher.predict_action_batch(
                    copy.deepcopy(raw), mode="eval"
                )
                native_chain = native_payload["forward_inputs"]["chains"]
                z = native_chain[:, 0].cuda()
                official = native_chain[:, -1].cuda().float()
                # Verify metadata-only physical decoder against real full pipeline.
                torch.testing.assert_close(
                    decode_action(official, source_low, source_high, 8),
                    native_actions.cpu().float(),
                    rtol=1e-5,
                    atol=1e-6,
                )
                reference, states, velocities = integrate(
                    z,
                    velocity,
                    torch.ones_like(scale),
                    torch.zeros_like(shift),
                    "native",
                    4,
                )
                branches["official"].append((official * scale + shift).cpu())
                branches["native_fp32"].append((reference * scale + shift).cpu())
                for kind in ("continuous", "discrete"):
                    actual, _, _ = integrate(z, velocity, scale, shift, kind, 4)
                    branches[kind].append(actual.cpu())
                    identity, _, _ = integrate(
                        z,
                        velocity,
                        torch.ones_like(scale),
                        torch.zeros_like(shift),
                        kind,
                        4,
                    )
                    # Identity control compared in source coordinates.
                    torch.testing.assert_close(identity, reference, rtol=0, atol=1e-6)
                    branches[f"identity_{kind}"].append(
                        (identity * scale + shift).cpu()
                    )
                for k, (x, v) in enumerate(zip(states, velocities, strict=True)):
                    t, h = k / 4, 0.25
                    a = 1 + t * (scale - 1)
                    mapped_next = (a + h * (scale - 1)) * (x + h * v) + (t + h) * shift
                    continuous_next = (
                        a * x + t * shift + h * (a * v + (scale - 1) * x + shift)
                    )
                    defect = mapped_next - continuous_next
                    expected_defect = h * h * (scale - 1) * v
                    torch.testing.assert_close(
                        defect, expected_defect, rtol=1e-4, atol=2e-6
                    )
                    local_errors.append(defect.cpu())
                    local_identity_residuals.append((defect - expected_defect).cpu())
            tensors = {key: torch.cat(items) for key, items in branches.items()}
            physical = {
                key: decode_action(value, target_low, target_high, 8)
                for key, value in tensors.items()
            }
            metrics = {}
            for kind in (
                "continuous",
                "discrete",
                "identity_continuous",
                "identity_discrete",
            ):
                metrics[kind] = {
                    "normalized_full16x32_vs_fp32_reference": errors(
                        tensors[kind] - tensors["native_fp32"]
                    ),
                    "normalized_executed8x7_vs_fp32_reference": errors(
                        (tensors[kind] - tensors["native_fp32"])[:, :8, :7]
                    ),
                    "physical_vs_fp32_reference": physical_errors(
                        physical[kind], physical["native_fp32"]
                    ),
                    "physical_vs_official_sampler": physical_errors(
                        physical[kind], physical["official"]
                    ),
                }
            report["suites"].append(
                {
                    "suite": suite,
                    "teacher": str(path),
                    "endpoint_scale": scale[:7].cpu().tolist(),
                    "endpoint_shift": shift[:7].cpu().tolist(),
                    "branches": metrics,
                    "native_fp32_vs_official_physical": physical_errors(
                        physical["native_fp32"], physical["official"]
                    ),
                    "continuous_one_step_defect": errors(torch.cat(local_errors)),
                    "discrete_identity_algebra_residual": errors(
                        torch.cat(local_identity_residuals)
                    ),
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["suites"][-1]), flush=True)
        del (
            teacher,
            cached_backbone,
            action_input,
            backbone_input,
            normalized,
            payload,
            native_payload,
            native_chain,
            velocity,
        )
        gc.collect()
        torch.cuda.empty_cache()
    report["completed"] = True
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
