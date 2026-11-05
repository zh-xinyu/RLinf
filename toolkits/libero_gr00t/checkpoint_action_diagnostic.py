# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compare fixed-observation velocity and action agreement across checkpoints.

Run once for the base student and once per saved full checkpoint, using the same
fixtures and seeds. This is not an SR test: teacher actions on these observations
need not succeed, and paired-noise action distance is not distributional distance.
No simulator or optimizer is created. GPU work requires --run-gpu explicitly.
"""

import argparse
import copy
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from toolkits.libero_gr00t.evaluate_full_checkpoint import (
    SUITES,
    candidate_model_config,
    checkpoint_step,
    load_full_weights,
    sha256,
)


def read_fixtures(observations: Path | None, manifest: Path | None) -> list[dict]:
    """Resolve explicit fixture provenance; never infer an unknown initial state."""
    if (observations is None) == (manifest is None):
        raise ValueError("Provide exactly one observations directory or manifest")
    if manifest is not None:
        rows = json.loads(manifest.read_text())
        if not isinstance(rows, list):
            raise ValueError("Fixture manifest must be a JSON list")
        base = manifest.parent
    else:
        rows = [{"suite": suite, "path": f"{suite}_obs.npz"} for suite in SUITES]
        base = observations
    if not 1 <= len(rows) <= 40:
        raise ValueError("Require 1..40 fixed observations")
    result, seen = [], set()
    for row in rows:
        if row.get("suite") not in SUITES:
            raise ValueError("Unknown suite in fixture manifest")
        path = (base / row["path"]).resolve(strict=True)
        if path in seen:
            raise ValueError("Duplicate fixture path")
        seen.add(path)
        for field, limit in (("task_id", 10), ("initial_state_id", 50)):
            value = row.get(field)
            if value is not None and (
                type(value) is not int or value not in range(limit)
            ):
                raise ValueError(f"Invalid {field}")
        result.append({**row, "path": str(path), "sha256": sha256(path)})
    return result


def checkpoint_identity(path: Path) -> dict:
    """Hash actual weights and metadata, not just a mutable checkpoint directory."""
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights in {path}")
    files += [path / "config.json", path / "experiment_cfg/metadata.json"]
    index = path / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
    return {
        "path": str(path.resolve()),
        "files_sha256": {str(f.relative_to(path)): sha256(f) for f in files},
    }


def seed_all(seed: int) -> None:
    """Replay all preprocessing and policy-noise RNG streams for each fixture."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compare_sample(
    student_velocities: torch.Tensor,
    teacher_velocities: torch.Tensor,
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    kl_weights: torch.Tensor,
) -> dict:
    """Report each time separately and avoid averaging gripper into motion error."""
    if student_velocities.shape != (4, 8, 7) or teacher_velocities.shape != (4, 8, 7):
        raise ValueError("Require four flow times and the executed 8x7 prefix")
    if student_actions.shape != (8, 7) or teacher_actions.shape != (8, 7):
        raise ValueError("Require physical actions shaped 8x7")
    if kl_weights.shape != (4,):
        raise ValueError("Require four M3 time weights")
    tensors = (
        student_velocities,
        teacher_velocities,
        student_actions,
        teacher_actions,
        kl_weights,
    )
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("Nonfinite diagnostic values")
    error = (student_velocities.float() - teacher_velocities.float()).square()
    action_delta = student_actions.float() - teacher_actions.float()
    per_time = error.mean(dim=(1, 2))
    return {
        "velocity_mse_per_time": per_time.tolist(),
        "velocity_mse_per_time_and_dim": error.mean(dim=1).tolist(),
        "velocity_mse_uniform_time_mean": float(per_time.mean()),
        "m3_weighted_velocity_mse": float((per_time * kl_weights.float()).mean()),
        "physical_action_rmse_first6": float(
            action_delta[:, :6].square().mean().sqrt()
        ),
        "physical_action_rmse_per_dim": action_delta.square()
        .mean(dim=0)
        .sqrt()
        .tolist(),
        "physical_action_max_abs_first6": float(action_delta[:, :6].abs().max()),
        "gripper_disagreement_fraction": float(
            (student_actions[:, 6] != teacher_actions[:, 6]).float().mean()
        ),
    }


def raw_fixture(row: dict) -> dict:
    """Read saved raw simulator observation without starting physics."""
    from toolkits.libero_gr00t.probe import observation_batch

    with np.load(row["path"], allow_pickle=False) as saved:
        return observation_batch(dict(saved), str(saved["task_description"].item()))


def velocity(policy, inputs, state, step):
    """Use the exact NFT path and executed action prefix used during training."""
    from rlinf.models.embodiment.base_policy import ForwardType

    return policy(
        forward_type=ForwardType.NFT,
        forward_inputs=inputs,
        nft_inputs={
            "x_t": state.cuda(),
            "timesteps": torch.tensor([step], device="cuda"),
        },
        compute_values=False,
    )["v_theta"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="Omit for update0 base student")
    parser.add_argument("--student", type=Path)
    parser.add_argument("--teachers", type=Path, nargs=4, required=True)
    fixtures = parser.add_mutually_exclusive_group(required=True)
    fixtures.add_argument("--observations", type=Path)
    fixtures.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--minimum-free-gib", type=float, default=20)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_gpu:
        print("No GPU work performed; explicitly select one device and pass --run-gpu")
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.seeds <= 8 or not 0 <= args.seed < 2**31:
        raise ValueError("Require 1..8 seeds and a nonnegative seed below 2**31")
    if not 8 <= args.memory_gib <= args.minimum_free_gib:
        raise ValueError("Require 8 <= memory-gib <= minimum-free-gib")
    rows = read_fixtures(args.observations, args.fixture_manifest)
    cfg, reuse = candidate_model_config(args.config, args.student)
    step = checkpoint_step(args.weights) if args.weights else 0
    if step % reuse:
        raise ValueError("Checkpoint must be at a complete rollout boundary")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly select exactly one CUDA_VISIBLE_DEVICES")
    free, total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(
            f"Need {args.minimum_free_gib} GiB free, found {free / 2**30:.2f}"
        )
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(min(args.memory_gib * 2**30 / total, 1))

    from rlinf.algorithms.flow_opd import (
        map_target_flow_state_to_source,
        transport_source_velocity_to_target,
    )
    from rlinf.algorithms.losses import flow_sde_kl_weights
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
    from toolkits.libero_gr00t.native_batch_smoke import load_policy

    report = {
        "complete": False,
        "scope": "Fixed raw observations and paired noise; no task-success evidence",
        "optimizer_update": step,
        "rollout_round": step // reuse,
        "seed": args.seed,
        "seeds_per_fixture": args.seeds,
        "fixtures": rows,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "model_config": OmegaConf.to_container(cfg, resolve=True),
        "student_identity": checkpoint_identity(Path(cfg.model_path)),
        "weights": str(args.weights.resolve()) if args.weights else None,
        "weights_sha256": sha256(args.weights) if args.weights else None,
        "teacher_identities": [checkpoint_identity(p) for p in args.teachers],
        "physical_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_cap_gib": args.memory_gib,
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    student = get_model(cfg).cpu().eval().requires_grad_(False)
    if args.weights:
        report["full_weight_validation"] = load_full_weights(student, args.weights)
    target_metadata = student.metadata
    student.cuda()
    staged = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for fixture_index, row in enumerate(rows):
            for noise_index in range(args.seeds):
                seed = args.seed + fixture_index * 1000 + noise_index
                seed_all(seed)
                actions, payload = student.predict_action_batch(
                    copy.deepcopy(raw_fixture(row)), mode="eval"
                )
                inputs = payload["forward_inputs"]
                chains = inputs["chains"]
                if chains.shape != (1, 5, 16, 32):
                    raise ValueError(f"Unexpected full chain shape: {chains.shape}")
                vs = [
                    velocity(student, inputs, chains[:, k], k)[0].cpu()
                    for k in range(4)
                ]
                staged.append(
                    {
                        "fixture_index": fixture_index,
                        "noise_index": noise_index,
                        "seed": seed,
                        "chains": chains.detach().cpu(),
                        "velocities": torch.stack(vs),
                        "actions": actions[0].detach().cpu(),
                    }
                )
                del payload, inputs, chains, actions, vs
    del student
    gc.collect()
    torch.cuda.empty_cache()
    args.execute, args.denoise_steps = 8, 4
    weights = flow_sde_kl_weights(4, float(cfg.rl_head_config.noise_level)).cpu()
    report["kl_weights"] = weights.tolist()
    for suite, teacher_path in zip(SUITES, args.teachers, strict=True):
        local = [
            item for item in staged if rows[item["fixture_index"]]["suite"] == suite
        ]
        if not local:
            continue
        teacher = load_policy(teacher_path, args, native=False).cuda()
        scale, shift = teacher.opd_action_endpoint_affine_to(target_metadata)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for item in local:
                row = rows[item["fixture_index"]]
                seed_all(item["seed"])
                actions, payload = teacher.predict_action_batch(
                    copy.deepcopy(raw_fixture(row)), mode="eval"
                )
                inputs = payload["forward_inputs"]
                teacher_noise = inputs["chains"][:, 0].cpu()
                if not torch.equal(teacher_noise, item["chains"][:, 0]):
                    raise ValueError(
                        "Native samplers did not receive exactly paired noise"
                    )
                targets = []
                for k in range(4):
                    idx = torch.tensor([k])
                    source = map_target_flow_state_to_source(
                        item["chains"][:, k], idx, 4, scale, shift
                    )
                    vt = velocity(teacher, inputs, source, k).cpu()
                    target = transport_source_velocity_to_target(
                        source[:, :8], vt, idx, 4, scale, shift
                    )
                    targets.append(target[0])
                metrics = compare_sample(
                    item["velocities"],
                    torch.stack(targets),
                    item["actions"],
                    actions[0].cpu(),
                    weights,
                )
                report["samples"].append(
                    {
                        "suite": suite,
                        "fixture_index": item["fixture_index"],
                        "noise_index": item["noise_index"],
                        "seed": item["seed"],
                        "paired_noise_exact": True,
                        **metrics,
                    }
                )
                del payload, inputs, actions, teacher_noise, targets, source, vt, target
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if len(report["samples"]) != len(rows) * args.seeds:
        raise RuntimeError("Incomplete diagnostic sample coverage")
    report["complete"] = True
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "samples": len(report["samples"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
