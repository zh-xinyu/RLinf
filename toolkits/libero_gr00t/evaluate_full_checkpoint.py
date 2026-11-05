# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a base policy or strictly reloaded production weights on LIBERO.

This uses one explicitly selected GPU and no Ray. It validates the full-weight
artifact and real policy inference, not the distributed evaluation scheduler.
"""

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from toolkits.libero_gr00t.padding_profile import (
    PADDING_PROFILES,
    student_input_contract,
)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def load_full_weights(model: torch.nn.Module, path: Path) -> dict:
    """Validate the complete CPU state before copying into an unsharded model.

    Reject partial heads, nested training checkpoints, wrong shapes/dtypes and
    nonfinite values and frozen-backbone drift. Never silently coerce BF16 head
    weights into an FP32 model. Other state (for example a version counter) is
    compared and reported, but is not classified as frozen VLM parameters.
    """
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = model.state_dict()
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("Expected a flat full-model tensor state dict")
    if set(state) != set(expected):
        raise ValueError(
            "Full-weight key mismatch: "
            f"missing={sorted(set(expected) - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - set(expected))[:5]}"
        )
    count = 0
    comparison = {
        "changed_action_head_tensors": 0,
        "unchanged_action_head_tensors": 0,
        "unchanged_backbone_tensors": 0,
        "other_changed_state_keys": [],
        "other_unchanged_state_tensors": 0,
    }
    for key, target in expected.items():
        tensor = state[key]
        if tensor.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tensor.shape} != {target.shape}"
            )
        if tensor.dtype != target.dtype:
            raise ValueError(
                f"Dtype mismatch for {key}: {tensor.dtype} != {target.dtype}"
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Nonfinite checkpoint tensor: {key}")
        identical = torch.equal(tensor, target.cpu())
        if key.startswith("backbone."):
            if not identical:
                raise ValueError(
                    f"Frozen backbone drift relative to base checkpoint: {key}"
                )
            comparison["unchanged_backbone_tensors"] += 1
        elif key.startswith("action_head."):
            field = (
                "unchanged_action_head_tensors"
                if identical
                else "changed_action_head_tensors"
            )
            comparison[field] += 1
        elif identical:
            comparison["other_unchanged_state_tensors"] += 1
        else:
            comparison["other_changed_state_keys"].append(key)
        count += tensor.numel()
    model.load_state_dict(state, strict=True)
    return {
        "state_tensor_count": len(state),
        "state_element_count": count,
        "strict": True,
        "base_comparison": comparison,
    }


def candidate_model_config(
    config_path: Path,
    student: Path | None = None,
    *,
    student_padding_profile: str = "native",
):
    """Extract the exact supported actor input contract from a resolved run config."""
    cfg = OmegaConf.load(config_path)
    model = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    required = {
        "model_type": "gr00t",
        "precision": "bf16",
        "action_head_precision": "fp32",
        "inference_autocast_dtype": "bf16",
        "obs_converter_type": "libero_grouped",
        "embodiment_projector_id": 31,
        "disable_image_tiling": True,
        "num_action_chunks": 8,
        "denoising_steps": 4,
    }
    for key, value in required.items():
        if model.get(key) != value:
            raise ValueError(f"Unsupported reload contract {key}: expected {value!r}")
    student_input_contract(model, student_padding_profile)
    rollout_model = OmegaConf.select(cfg, "rollout.model")
    if rollout_model is not None:
        # Actor and rollout can differ in loading controls, but not in the
        # declared student policy input contract.
        effective_rollout = OmegaConf.merge(model, rollout_model)
        student_input_contract(effective_rollout, student_padding_profile)
    if model.rl_head_config.noise_method != "flow_ode":
        raise ValueError("Reload smoke requires the candidate ODE inference policy")
    if not model.rl_head_config.get("disable_dropout", False):
        raise ValueError("Reload smoke requires disabled action-head dropout")
    if student is not None:
        model.model_path = str(student.resolve(strict=True))
    return model, int(cfg.algorithm.flow_refresh_updates)


def sha256(path: Path) -> str:
    """Hash a checkpoint or metadata file without materializing its bytes in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_identity(student: Path) -> dict:
    """Hash the actual base config, metadata, index and all local weight shards."""
    shards = sorted(student.glob("*.safetensors"))
    if not shards:
        raise ValueError("Baseline checkpoint has no safetensors weights")
    files = [student / "config.json", student / "experiment_cfg/metadata.json", *shards]
    index = student / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text()).get("weight_map", {})
        if not mapping:
            raise ValueError("Baseline shard index has no weight map")
        for name in set(mapping.values()):
            shard = (student / name).resolve()
            if shard.parent != student.resolve() or shard not in [
                p.resolve() for p in shards
            ]:
                raise ValueError(
                    f"Baseline shard index references unavailable weights: {name}"
                )
        files.append(index)
    records = [
        {
            "path": str(path.relative_to(student)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(files)
    ]
    return {
        "files": records,
        "content_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def checkpoint_step(path: Path) -> int:
    """Read the production checkpoint update ID from its directory name."""
    for parent in path.parents:
        if match := re.fullmatch(r"global_step_(\d+)", parent.name):
            return int(match[1])
    raise ValueError("Weights must live below a global_step_<N> checkpoint directory")


def prepare_evaluation_weights(
    model: torch.nn.Module, weights: Path | None, base: Path
) -> dict:
    """Keep baseline weights intact or strictly load a complete trained policy."""
    if weights is None:
        return {
            "baseline": True,
            "strict": False,
            "weights": None,
            "weights_sha256": None,
            "base_checkpoint_identity": baseline_identity(base),
        }
    loaded = load_full_weights(model, weights)
    return {
        "baseline": False,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        **loaded,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Require an explicit base-policy or trained-weight evaluation source."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path)
    source.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--config", type=Path, required=True, help="Training resolved_config.yaml"
    )
    parser.add_argument(
        "--student", type=Path, help="Relocated original base checkpoint"
    )
    parser.add_argument(
        "--student-padding-profile",
        choices=PADDING_PROFILES,
        default="native",
        help="Must match the training config; canonical-fixed570 needs its own baseline",
    )
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--tasks", nargs="+", type=int, default=[0])
    parser.add_argument("--trials", nargs="+", type=int, default=[1])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--minimum-free-gib", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new evaluation output directory")
    for values, limit, name in ((args.tasks, 10, "tasks"), (args.trials, 50, "trials")):
        if len(set(values)) != len(values) or any(
            value not in range(limit) for value in values
        ):
            raise ValueError(f"{name} must contain unique indices in [0, {limit - 1}]")
    if len(set(args.suites)) != len(args.suites):
        raise ValueError("Duplicate evaluation suites")
    if not 8 <= args.memory_gib <= args.minimum_free_gib:
        raise ValueError("Require 8 <= memory-gib <= minimum-free-gib")
    model_cfg, reuse = candidate_model_config(
        args.config, args.student, student_padding_profile=args.student_padding_profile
    )
    step = 0 if args.baseline else checkpoint_step(args.weights)
    if step % reuse:
        raise ValueError("Checkpoint is not at a complete rollout-round boundary")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly select exactly one CUDA_VISIBLE_DEVICES")
    free, total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(
            f"Need {args.minimum_free_gib} GiB free; found {free / 2**30:.2f}"
        )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(min(args.memory_gib * 2**30 / total, 1))

    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
    from toolkits.libero_gr00t.pilot import evaluate
    from toolkits.libero_gr00t.probe import renderer_index

    egl = renderer_index()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    model = get_model(model_cfg).cpu().eval().requires_grad_(False)
    loaded = prepare_evaluation_weights(model, args.weights, Path(model_cfg.model_path))
    model._probe_expected_state_dim = 11
    model.cuda()
    args.output.mkdir(parents=True)
    protocol = {
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "base_checkpoint": model_cfg.model_path,
        "metadata_sha256": sha256(
            Path(model_cfg.model_path) / "experiment_cfg/metadata.json"
        ),
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
        "student_padding_profile": args.student_padding_profile,
        "student_input_contract": student_input_contract(
            model_cfg, args.student_padding_profile
        ),
        "suites": args.suites,
        "tasks": args.tasks,
        "trials": args.trials,
        "seed": args.seed,
        "rng_protocol": "per-task/trial reseeding via pilot.evaluate",
        "optimizer_update": step,
        "rollout_round": step // reuse,
        "physical_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "egl_index": egl,
        "initial_free_gib": free / 2**30,
        "memory_cap_gib": args.memory_gib,
        "distributed_evaluator_verified": False,
        **loaded,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    for suite in args.suites:

        def write(row):
            row = {"suite": suite, **row}
            with (args.output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        eval_args = SimpleNamespace(
            suite=suite,
            seed=args.seed,
            execute=8,
            eval_tasks=args.tasks,
            eval_initial_states=args.trials,
        )
        success, episodes = evaluate(
            model,
            benchmark.get_benchmark_dict()[suite](),
            eval_args,
            step // reuse,
            step,
            write,
        )
        results.append({"suite": suite, "successes": success, "episodes": episodes})
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "suites": results,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "formal_performance_evidence": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
