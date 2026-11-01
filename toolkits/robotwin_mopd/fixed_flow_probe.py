# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compare checkpoints on fixed reference observations and all ten flow states.

Example (source the isolated runtime; choose a free GPU explicitly)::

    CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/fixed_flow_probe.py \
      --task adjust_bottle --checkpoint /path/to/joint/global_step_20 \
      --output /path/to/new_fixed_probe.json

The reference defaults to the place_container_plate SFT specialist. This is a
short unfiltered-seen mechanism diagnostic, not an episode success evaluation.
The companion .reference.pt stores CPU observations, noise, and reference chains.
Use --cache-input with that file to compare later checkpoints without resetting
or simulating again. No optimizer or training worker is constructed.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.losses import flow_sde_kl_weights
from rlinf.envs.robotwin.multitask_env import TASK_HORIZONS, RoboTwinMultiTaskEnv
from rlinf.models.embodiment.openpi import get_model
from toolkits.robotwin_mopd.flow_probe import (
    aggregate_statistics,
    partition_statistics,
    prepare_context,
    probe_noise_seed,
    sample_complete_flow,
)


def tensor_hash(tensor: torch.Tensor) -> str:
    """Hash exact bytes together with shape and dtype, including BF16 tensors."""
    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps([list(tensor.shape), str(tensor.dtype)]).encode()
    return hashlib.sha256(
        header + tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def input_hash(value) -> str:
    """Hash the complete observation tree without omitting prompts or task IDs."""

    def manifest(item):
        if torch.is_tensor(item):
            return {"tensor_sha256": tensor_hash(item)}
        if isinstance(item, dict):
            return {key: manifest(val) for key, val in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [manifest(val) for val in item]
        return item

    return hashlib.sha256(
        json.dumps(manifest(value), sort_keys=True).encode()
    ).hexdigest()


def cpu_snapshot(value):
    """Detach and clone tensors so simulation cannot mutate cached observations."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(val) for key, val in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(val) for val in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(val) for val in value)
    return copy.deepcopy(value)


def validate_layout(model) -> None:
    """Keep the diagnostic tied to the current RoboTwin flow and action contract."""
    layout = tuple(
        int(getattr(model.config, name))
        for name in (
            "action_horizon",
            "action_dim",
            "action_chunk",
            "action_env_dim",
            "num_steps",
        )
    )
    if layout != (50, 32, 25, 14, 10):
        raise ValueError(
            f"Expected 50x32 latent, 25x14 execution, ten ODE steps: {layout}"
        )


def amp_context(amp: str):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" else nullcontext()
    )


@torch.inference_mode()
def make_reference_sample(model, obs, *, scene_seed: int, index: int):
    """Cache one original-rollout reference flow without an outer AMP context."""
    context = prepare_context(model, obs)
    device = context["state"].device
    noise_seed = probe_noise_seed(scene_seed, index)
    noise = torch.randn(
        (1, 50, 32),
        device=device,
        dtype=model.action_in_proj.weight.dtype,
        generator=torch.Generator(device=device).manual_seed(noise_seed),
    )
    output = sample_complete_flow(model, context, noise)
    if output["chains"].shape != (1, 11, 50, 32):
        raise ValueError("Reference must retain all eleven ODE states")
    if not torch.equal(output["chains"][:, 0], noise):
        raise ValueError("Reference flow changed the supplied initial noise")
    sample = {
        "probe_index": index,
        "observation": cpu_snapshot(obs),
        "normalized_state": cpu_snapshot(context["state"]),
        "noise_seed": noise_seed,
        "noise": cpu_snapshot(noise),
        "x_t": cpu_snapshot(output["chains"][:, :10]),
        "ode_grid": model._get_timesteps(10, device).detach().cpu(),
    }
    sample["observation_sha256"] = input_hash(sample["observation"])
    sample["noise_sha256"] = tensor_hash(sample["noise"])
    sample["x_t_sha256"] = tensor_hash(sample["x_t"])
    actions = model.output_transform(
        {"actions": output["actions"], "state": context["observation"].state},
        task_ids=context["task_ids"],
    )["actions"]
    return sample, actions


@torch.inference_mode()
def collect_reference(model, env, *, requested_seed: int, probes: int):
    """Collect initial reference-visited observations, never execute the last chunk."""
    obs, _ = env.reset()
    actual_seed = int(env.rollout_episode_records()[0]["actual_reset_seed"])
    prompt = obs["task_descriptions"][0]
    samples, executed = [], 0
    flags = {"terminated": False, "truncated": False}
    for index in range(probes):
        if obs["task_descriptions"] != [prompt]:
            raise ValueError("Prompt changed within reference episode")
        sample, actions = make_reference_sample(
            model, obs, scene_seed=actual_seed, index=index
        )
        sample["executed_control_targets_before_observation"] = executed
        samples.append(sample)
        if index + 1 == probes:
            break
        if not torch.isfinite(torch.as_tensor(actions)).all():
            raise ValueError("Nonfinite reference actions")
        observations, _, terminated, truncated, _ = env.chunk_step(actions)
        executed = int(env.rollout_episode_records()[0]["simulation_steps"])
        flags = {
            "terminated": bool(terminated.any()),
            "truncated": bool(truncated.any()),
        }
        if any(flags.values()):
            break
        obs = observations[-1]
    return samples, {
        "requested_scene_seed": requested_seed,
        "actual_reset_seed": actual_seed,
        "instruction": prompt,
        "executed_control_targets": executed,
        "native_terminal_flags": flags,
    }


@torch.inference_mode()
def score_fixed_inputs(model, samples: list[dict], *, amp: str) -> list[torch.Tensor]:
    """Evaluate every model at the identical cached observations, x_t, and times."""
    outputs = []
    for sample in samples:
        if input_hash(sample["observation"]) != sample["observation_sha256"]:
            raise ValueError("Cached observation or prompt changed")
        if tensor_hash(sample["x_t"]) != sample["x_t_sha256"]:
            raise ValueError("Cached reference x_t changed")
        if tensor_hash(sample["noise"]) != sample["noise_sha256"]:
            raise ValueError("Cached noise changed")
        with amp_context(amp):
            context = prepare_context(model, cpu_snapshot(sample["observation"]))
            device = context["state"].device
            if not torch.equal(context["state"].cpu(), sample["normalized_state"]):
                raise ValueError("Model normalized state differs from reference")
            grid = model._get_timesteps(10, device)
            if not torch.equal(grid.cpu(), sample["ode_grid"]):
                raise ValueError("Model ODE grid differs from reference")
            velocities = []
            for index in range(10):
                # Clone even after .to(): CPU fake models must not alias the cache.
                x_t = sample["x_t"][:, index].to(device).clone()
                velocity, _ = model.get_velocity(
                    context["state"],
                    x_t,
                    grid[index].expand(x_t.shape[0]),
                    context["prefix_masks"],
                    context["cache"],
                )
                velocities.append(velocity.detach().cpu())
            outputs.append(torch.stack(velocities, dim=1))
    return outputs


def summarize_errors(
    candidate, teacher, *, weights: torch.Tensor, grid: torch.Tensor
) -> dict:
    """Use uniform ten-bin averaging, retaining M3's unnormalized time weights."""
    if len(candidate) != len(teacher) or not candidate:
        raise ValueError("Candidate and teacher must share nonempty observation cache")
    if len(weights) != 10:
        raise ValueError("Expected ten M3 weights")
    per_bin = []
    for index in range(10):
        statistics = aggregate_statistics(
            [
                partition_statistics(left[:, index], right[:, index])
                for left, right in zip(candidate, teacher, strict=True)
            ]
        )
        per_bin.append(
            {
                "flow_index": index,
                "time": float(grid[index]),
                "m3_weight": float(weights[index]),
                "statistics": statistics,
            }
        )
    aggregate = aggregate_statistics([row["statistics"] for row in per_bin])
    summary = {}
    for block, stats in aggregate.items():
        weighted_sum = sum(
            float(weights[k]) * per_bin[k]["statistics"][block]["squared_error_sum"]
            for k in range(10)
        )
        summary[block] = {
            "elements": stats["elements"],
            "unweighted_mse": stats["mse"],
            "m3_weighted_mse": weighted_sum / stats["elements"],
        }
    return {"per_time_bin": per_bin, "aggregate": summary}


def parameter_dtypes(model) -> dict:
    """Record actual loader precision rather than infer it from YAML null values."""
    counts = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype, 0) + parameter.numel()
    return counts


def release_model(model):
    model.to("cpu")
    # Caller deletes its own reference; no optimizer exists.
    gc.collect()
    torch.cuda.empty_cache()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASK_HORIZONS), required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat for later checkpoints; base is supported",
    )
    parser.add_argument(
        "--reference-checkpoint", help="Default: place_container_plate SFT specialist"
    )
    parser.add_argument("--scene-seed", type=int)
    parser.add_argument("--probes", type=int, choices=[1, 2], default=1)
    parser.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    parser.add_argument(
        "--cache-input",
        type=Path,
        help="Reuse a previous .reference.pt without simulation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-name", default="robotwin4_mopd_pi05")
    args = parser.parse_args()
    if args.scene_seed is not None and args.scene_seed < 0:
        parser.error("scene-seed must be nonnegative")
    from toolkits.robotwin_mopd.feasible_pool import require_idle_visible_gpus

    require_idle_visible_gpus()
    if not torch.cuda.is_available():
        raise RuntimeError("Use an explicitly free CUDA GPU")
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    cache_path = output.with_suffix(".reference.pt")
    if output.exists() or (not args.cache_input and cache_path.exists()):
        raise FileExistsError("Use fresh output paths; diagnostic evidence is retained")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    os.environ.setdefault("OPD_RUN_DIR", str(output.parent))
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name=args.config_name)
    reference_path = (
        args.reference_checkpoint
        or cfg.rollout.expert_models.place_container_plate.model_path
    )

    def model_config(path):
        model_cfg = copy.deepcopy(cfg.actor.model)
        if path != "base":
            model_cfg.model_path = str(Path(path).expanduser().resolve())
        return model_cfg

    reference_cfg = model_config(reference_path)
    teacher_cfg = model_config(cfg.rollout.expert_models[args.task].model_path)
    normalization = OmegaConf.to_container(
        cfg.actor.model.opd_task_norm_stats, resolve=True
    )
    report = {
        "schema_version": 1,
        "started_unix": time.time(),
        "finished_unix": None,
        "task": args.task,
        "protocol": "unfiltered_seen_fixed_reference_flow",
        "episode_outcome_evaluated": False,
        "truncated_diagnostic": True,
        "reference_checkpoint": str(reference_cfg.model_path),
        "teacher_checkpoint": str(teacher_cfg.model_path),
        "comparison_amp": args.amp,
        "reference_generation_amp": "none",
        "comparison_path": "cached-prefix get_velocity, full50x32; no nft output projection",
        "normalization": normalization,
        "models": [],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Fixed-reference diagnostic; not current-policy training MSE or episode success rate.",
    }
    write_json(output, report)
    print(f"Loading fixed reference: {reference_cfg.model_path}", flush=True)
    reference = get_model(reference_cfg).to("cuda").eval()
    reference_dtypes = parameter_dtypes(reference)
    validate_layout(reference)
    if args.cache_input:
        cache_path = args.cache_input.resolve()
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("task", args.task),
            ("reference_checkpoint", str(reference_cfg.model_path)),
            ("normalization", normalization),
        ):
            if payload[key] != expected:
                raise ValueError(f"Reference cache {key} mismatch")
        samples, provenance = payload["samples"], payload["provenance"]
        if (
            args.scene_seed is not None
            and args.scene_seed != provenance["requested_scene_seed"]
        ):
            raise ValueError("Requested scene seed differs from saved cache")
    else:
        seed = (
            args.scene_seed
            if args.scene_seed is not None
            else (200001 if args.task == "adjust_bottle" else 200000)
        )
        env_cfg = copy.deepcopy(cfg.env.eval)
        env_cfg.task_names = [args.task]
        env_cfg.total_num_envs = 1
        env_cfg.video_cfg.save_video = False
        env_cfg.instruction_type = "seen"
        env = RoboTwinMultiTaskEnv(env_cfg, 1, 0, 1, None)
        env.set_rollout_tasks(
            [
                {
                    "slot": 0,
                    "task_id": 0,
                    "task": args.task,
                    "suite": args.task,
                    "reset_seed": seed,
                    "round": 0,
                }
            ]
        )
        try:
            samples, provenance = collect_reference(
                reference, env, requested_seed=seed, probes=args.probes
            )
        finally:
            env.close()
        payload = {
            "schema_version": 1,
            "task": args.task,
            "reference_checkpoint": str(reference_cfg.model_path),
            "normalization": normalization,
            "provenance": provenance,
            "samples": samples,
        }
        torch.save(payload, cache_path)
    report["completed_observation_probes"] = len(samples)
    report["completed_velocity_comparisons_per_model"] = len(samples) * 10
    report.update(provenance)
    report["reference_cache"] = str(cache_path)
    report["reference_cache_sha256"] = hashlib.sha256(
        cache_path.read_bytes()
    ).hexdigest()
    report["observations"] = [
        {
            key: value
            for key, value in sample.items()
            if key
            not in {"observation", "normalized_state", "noise", "x_t", "ode_grid"}
        }
        | {
            "latent_dtype": str(sample["x_t"].dtype),
            "state_dtype": str(sample["normalized_state"].dtype),
            "ode_grid": sample["ode_grid"].tolist(),
        }
        for sample in samples
    ]
    write_json(output, report)
    reference_values = score_fixed_inputs(reference, samples, amp=args.amp)
    release_model(reference)
    del reference
    print(f"Loading task teacher: {teacher_cfg.model_path}", flush=True)
    teacher = get_model(teacher_cfg).to("cuda").eval()
    report["teacher_parameter_elements_by_dtype"] = parameter_dtypes(teacher)
    validate_layout(teacher)
    teacher_values = score_fixed_inputs(teacher, samples, amp=args.amp)
    report["teacher_velocity_dtype"] = str(teacher_values[0].dtype)
    release_model(teacher)
    del teacher
    weights = flow_sde_kl_weights(10, float(cfg.actor.model.openpi.noise_level))
    report["m3_noise_level"] = float(cfg.actor.model.openpi.noise_level)

    def record(path, values, dtypes):
        report["models"].append(
            {
                "checkpoint": path,
                "parameter_elements_by_dtype": dtypes,
                "velocity_dtype": str(values[0].dtype),
                **summarize_errors(
                    values, teacher_values, weights=weights, grid=samples[0]["ode_grid"]
                ),
            }
        )
        write_json(output, report)

    record(str(reference_cfg.model_path), reference_values, reference_dtypes)
    for path in args.checkpoint:
        candidate_cfg = model_config(path)
        print(f"Loading comparison: {candidate_cfg.model_path}", flush=True)
        candidate = get_model(candidate_cfg).to("cuda").eval()
        validate_layout(candidate)
        values = score_fixed_inputs(candidate, samples, amp=args.amp)
        record(str(candidate_cfg.model_path), values, parameter_dtypes(candidate))
        release_model(candidate)
        del candidate
    report["finished_unix"] = time.time()
    write_json(output, report)
    print(f"FIXED_FLOW_PROBE_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
