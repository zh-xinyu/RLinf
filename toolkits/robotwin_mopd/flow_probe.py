# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Short, standalone full-latent flow diagnostics on student-visited observations.

No loss, optimizer, worker, or queue is modified. This is NOT a success-rate
assessment: by default only four decision observations are inspected, executing
at most three student chunks to reach them. Source the isolated runtime env.sh,
then run only on an explicitly free GPU::

    CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/flow_probe.py \
        --task adjust_bottle --checkpoint /path/to/global_step_40 \
        --evaluation-pool /path/to/evaluation_pool.json --probes 4 \
        --flow-indices 0 4 9 --output /path/to/flow_probe_adjust.json

Use --checkpoint base for a separate base-policy reference. Velocity comparison
uses the SAME student-path latent for both models, including all 50 x 32 values.
Endpoint comparison uses independent complete ODE trajectories with identical
initial noise. Errors are in normalized latent coordinates, not physical units.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.robotwin.multitask_env import TASK_HORIZONS, RoboTwinMultiTaskEnv
from rlinf.models.embodiment.openpi import get_model
from toolkits.robotwin_mopd.evaluate import read_evaluation_pool


def probe_noise_seed(scene_seed: int, probe_index: int) -> int:
    """A separate deterministic draw per physical observation, shared by models."""
    return (int(scene_seed) + 104729 * int(probe_index)) % (2**63 - 1)


def finish_statistics(totals: dict) -> dict:
    count = totals["elements"]
    teacher_norm = totals["teacher_squared_sum"]
    student_norm = totals["student_squared_sum"]
    denominator = math.sqrt(teacher_norm * student_norm)
    return dict(
        totals,
        mse=totals["squared_error_sum"] / count,
        student_rms=math.sqrt(student_norm / count),
        teacher_rms=math.sqrt(teacher_norm / count),
        cosine=totals["dot_sum"] / denominator if denominator > 0 else None,
    )


def partition_statistics(
    student: torch.Tensor, teacher: torch.Tensor, *, chunk: int = 25, physical: int = 14
) -> dict:
    """Report disjoint quadrants and useful marginals without projecting first."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("Expected matching [batch, full_horizon, latent_dim] tensors")
    horizon, dimensions = student.shape[1:]
    if not 0 < chunk < horizon or not 0 < physical < dimensions:
        raise ValueError("Both horizon and dimension partitions must be nonempty")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Nonfinite flow output")
    time_slices = {
        f"first{chunk}": slice(0, chunk),
        f"last{horizon - chunk}": slice(chunk, horizon),
        f"all{horizon}": slice(None),
    }
    dimension_slices = {
        f"physical{physical}": slice(0, physical),
        f"padded{dimensions - physical}": slice(physical, dimensions),
        f"all{dimensions}": slice(None),
    }
    result = {}
    for time_name, time_slice in time_slices.items():
        for dimension_name, dimension_slice in dimension_slices.items():
            left = student[:, time_slice, dimension_slice].float()
            right = teacher[:, time_slice, dimension_slice].float()
            totals = {
                "elements": left.numel(),
                "squared_error_sum": (left - right)
                .square()
                .sum(dtype=torch.float64)
                .item(),
                "student_squared_sum": left.square().sum(dtype=torch.float64).item(),
                "teacher_squared_sum": right.square().sum(dtype=torch.float64).item(),
                "dot_sum": (left * right).sum(dtype=torch.float64).item(),
            }
            result[f"{time_name}/{dimension_name}"] = finish_statistics(totals)
    return result


def aggregate_statistics(samples: list[dict]) -> dict:
    if not samples:
        return {}
    result = {}
    for block in samples[0]:
        totals = {
            key: sum(sample[block][key] for sample in samples)
            for key in (
                "elements",
                "squared_error_sum",
                "student_squared_sum",
                "teacher_squared_sum",
                "dot_sum",
            )
        }
        result[block] = finish_statistics(totals)
    return result


def prepare_context(model, env_obs) -> dict:
    """Follow the same observation transforms and prefix cache as eval inference."""
    from openpi.models.model import Observation

    inputs = model.obs_processor(env_obs)
    processed = model.precision_processor(
        model.input_transform(inputs, transpose=False)
    )
    observation = Observation.from_dict(processed)
    images, image_masks, tokens, token_masks, state = model._preprocess_observation(
        observation, train=False
    )
    prefix, prefix_masks, cache = model._build_prefix_cache(
        images, image_masks, tokens, token_masks
    )
    return {
        "observation": observation,
        "state": state,
        "prefix": prefix,
        "prefix_masks": prefix_masks,
        "cache": cache,
        "task_ids": inputs.get("opd_task_ids"),
    }


def sample_complete_flow(model, context: dict, noise: torch.Tensor) -> dict:
    return model._sample_actions_with_prefix_cache(
        context["state"],
        context["prefix"],
        context["prefix_masks"],
        context["cache"],
        noise=noise.clone(),
        mode="eval",
        compute_values=False,
    )


@torch.inference_mode()
def probe_observation(
    student,
    teacher,
    env_obs: dict,
    *,
    scene_seed: int,
    probe_index: int,
    flow_indices: list[int],
) -> tuple[dict, torch.Tensor]:
    student_context = prepare_context(student, env_obs)
    teacher_context = prepare_context(teacher, env_obs)
    if not torch.equal(student_context["state"], teacher_context["state"]):
        raise ValueError("Student/teacher normalized state coordinates differ")
    device = student_context["state"].device
    steps = int(student.config.num_steps)
    grid = student._get_timesteps(steps, device)
    teacher_grid = teacher._get_timesteps(int(teacher.config.num_steps), device)
    if not torch.equal(grid, teacher_grid):
        raise ValueError("Student/teacher ODE grids differ")
    if any(index < 0 or index >= steps for index in flow_indices):
        raise ValueError(
            "Flow indices must select pre-update points of the true ODE grid"
        )
    if student.action_in_proj.weight.dtype != teacher.action_in_proj.weight.dtype:
        raise ValueError("Student/teacher latent input dtypes differ")
    seed = probe_noise_seed(scene_seed, probe_index)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (1, int(student.config.action_horizon), int(student.config.action_dim)),
        dtype=student.action_in_proj.weight.dtype,
        device=device,
        generator=generator,
    )
    student_flow = sample_complete_flow(student, student_context, noise)
    teacher_flow = sample_complete_flow(teacher, teacher_context, noise)
    if not torch.equal(student_flow["chains"][:, 0], teacher_flow["chains"][:, 0]):
        raise RuntimeError("The two models did not receive identical initial noise")
    if student_flow["chains"].shape[1] != steps + 1:
        raise ValueError("Unexpected complete ODE chain layout")
    samples = []
    for index in flow_indices:
        x_t = student_flow["chains"][:, index]
        timestep = grid[index].expand(x_t.shape[0])
        student_velocity, _ = student.get_velocity(
            student_context["state"],
            x_t,
            timestep,
            student_context["prefix_masks"],
            student_context["cache"],
        )
        teacher_velocity, _ = teacher.get_velocity(
            teacher_context["state"],
            x_t,
            timestep,
            teacher_context["prefix_masks"],
            teacher_context["cache"],
        )
        samples.append(
            {
                "flow_index": index,
                "time": float(grid[index]),
                "velocity_on_student_path": partition_statistics(
                    student_velocity, teacher_velocity
                ),
            }
        )
    result = {
        "probe_index": probe_index,
        "noise_seed": seed,
        "initial_noise_sha256": hashlib.sha256(
            noise.float().cpu().numpy().tobytes()
        ).hexdigest(),
        "ode_grid": grid.cpu().tolist(),
        "velocity_samples": samples,
        "same_noise_endpoint": partition_statistics(
            student_flow["actions"], teacher_flow["actions"]
        ),
    }
    # Exactly the normal eval output transform: execute only the first 25 physical
    # actions; the comparison above has retained the full 50 x 32 latent output.
    actions = student.output_transform(
        {
            "actions": student_flow["actions"],
            "state": student_context["observation"].state,
        },
        task_ids=student_context["task_ids"],
    )["actions"]
    return result, actions


def save_report(
    path: Path,
    metadata: dict,
    probes: list[dict],
    *,
    executed: int,
    terminal: dict,
    finished: bool,
) -> None:
    velocity_samples = [
        sample["velocity_on_student_path"]
        for probe in probes
        for sample in probe["velocity_samples"]
    ]
    by_index = {}
    indices = sorted(
        {
            sample["flow_index"]
            for probe in probes
            for sample in probe["velocity_samples"]
        }
    )
    for index in indices:
        selected = [
            sample
            for probe in probes
            for sample in probe["velocity_samples"]
            if sample["flow_index"] == index
        ]
        by_index[str(index)] = {
            "time": selected[0]["time"],
            "observation_samples": len(selected),
            "statistics": aggregate_statistics(
                [sample["velocity_on_student_path"] for sample in selected]
            ),
        }
    report = dict(
        metadata,
        completed_observation_probes=len(probes),
        completed_velocity_comparisons=len(velocity_samples),
        completed_endpoint_comparisons=len(probes),
        executed_control_targets=executed,
        native_terminal_flags=terminal,
        finished_unix=time.time() if finished else None,
        probes=probes,
        aggregate_velocity_on_student_path=aggregate_statistics(velocity_samples),
        aggregate_velocity_by_flow_index=by_index,
        aggregate_same_noise_endpoint=aggregate_statistics(
            [probe["same_noise_endpoint"] for probe in probes]
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def collect_probe_observations(
    student,
    teacher,
    env,
    scene: dict,
    *,
    max_probes: int,
    flow_indices: list[int],
    on_progress=None,
):
    """Inspect at most max_probes decision states, never execute the final chunk."""
    probes, executed = [], 0
    terminal = {"terminated": False, "truncated": False}
    obs, _ = env.reset(evaluation_episode=scene)
    if env.rollout_episode_records()[0]["actual_reset_seed"] != scene["scene_seed"]:
        raise RuntimeError("Fixed evaluation scene changed")
    for index in range(max_probes):
        if obs["task_descriptions"] != [scene["instruction"]]:
            raise RuntimeError("Fixed prompt changed")
        probe, actions = probe_observation(
            student,
            teacher,
            obs,
            scene_seed=scene["scene_seed"],
            probe_index=index,
            flow_indices=flow_indices,
        )
        probe["executed_control_targets_before_observation"] = executed
        probes.append(probe)
        if on_progress is not None:
            on_progress(probes, executed, terminal)
        print(
            f"Probe {index + 1}/{max_probes}: {len(flow_indices)} velocity comparisons + endpoint",
            flush=True,
        )
        if index + 1 == max_probes:
            break
        observations, _, terminations, truncations, _ = env.chunk_step(actions)
        executed = env.rollout_episode_records()[0]["simulation_steps"]
        terminal = {
            "terminated": bool(terminations.any()),
            "truncated": bool(truncations.any()),
        }
        if any(terminal.values()):
            break
        obs = observations[-1]
    return probes, executed, terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(TASK_HORIZONS))
    parser.add_argument(
        "--checkpoint", required=True, help="Student checkpoint directory or base"
    )
    parser.add_argument("--evaluation-pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--probes",
        type=int,
        default=4,
        help="Maximum decision observations, not episodes",
    )
    parser.add_argument("--flow-indices", nargs="+", type=int, default=[0, 4, 9])
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--config-name", default="robotwin4_mopd_pi05")
    args = parser.parse_args()
    if (
        args.probes < 1
        or args.episode_index < 0
        or len(set(args.flow_indices)) != len(args.flow_indices)
    ):
        parser.error(
            "Probes must be positive, episode-index nonnegative, flow indices unique"
        )
    if args.output.exists():
        raise FileExistsError(
            "Use a new output path; previous diagnostic evidence is retained"
        )
    from toolkits.robotwin_mopd.feasible_pool import require_idle_visible_gpus

    require_idle_visible_gpus()
    os.environ.setdefault("USE_TF", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic only on a free CUDA GPU")
    repo = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    os.environ.setdefault("OPD_RUN_DIR", str(args.output.parent))
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name=args.config_name)
    pool, entries = read_evaluation_pool(
        args.evaluation_pool, args.task, args.episode_index + 1, cfg
    )
    scene = entries[args.episode_index]
    student_cfg = copy.deepcopy(cfg.actor.model)
    if args.checkpoint != "base":
        student_cfg.model_path = str(Path(args.checkpoint).expanduser().resolve())
    teacher_cfg = copy.deepcopy(cfg.actor.model)
    teacher_cfg.model_path = cfg.rollout.expert_models[args.task].model_path
    metadata = {
        "started_unix": time.time(),
        "task": args.task,
        "student_checkpoint": str(student_cfg.model_path),
        "teacher_checkpoint": str(teacher_cfg.model_path),
        "evaluation_pool": str(args.evaluation_pool.resolve()),
        "pool_content_sha256": pool["pool_content_sha256"],
        "protocol": pool["protocol"],
        "scene_seed": scene["scene_seed"],
        "instruction": scene["instruction"],
        "episode_index": args.episode_index,
        "requested_observation_probes": args.probes,
        "flow_indices": args.flow_indices,
        "truncated_diagnostic": True,
        "episode_outcome_evaluated": False,
        "sampling_scope": "Initial student-visited observations of one fixed unseen scene, selected ODE times only; not an estimator of the training-MSE distribution.",
        "coordinate_system": "normalized full flow latent; velocity evaluated on the same student-path x_t",
        "causal_interpretation": "Descriptive diagnostic only; does not establish why success rate changes or stagnates.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization": OmegaConf.to_container(
            student_cfg.opd_task_norm_stats, resolve=True
        ),
    }
    probes, executed = [], 0
    terminal = {"terminated": False, "truncated": False}
    save_report(
        args.output,
        metadata,
        probes,
        executed=executed,
        terminal=terminal,
        finished=False,
    )
    print(
        f"Loading student={student_cfg.model_path} and teacher={teacher_cfg.model_path}",
        flush=True,
    )
    student = get_model(student_cfg).to("cuda").eval()
    teacher = get_model(teacher_cfg).to("cuda").eval()
    for model in (student, teacher):
        layout = (
            int(model.config.action_horizon),
            int(model.config.action_dim),
            int(model.config.action_chunk),
            int(model.config.action_env_dim),
        )
        if layout != (50, 32, 25, 14) or int(model.config.num_steps) != 10:
            raise ValueError(
                f"Expected RoboTwin 50x32 latent, 25x14 execution, 10 ODE steps; got {layout}"
            )
    env_cfg = copy.deepcopy(cfg.env.eval)
    env_cfg.task_names = [args.task]
    env_cfg.total_num_envs = 1
    env_cfg.video_cfg.save_video = False
    env_cfg.instruction_type = "unseen"
    for key, value in pool["protocol"]["screening_overrides"].items():
        OmegaConf.update(env_cfg.task_config, key, value, force_add=True)
    env = RoboTwinMultiTaskEnv(env_cfg, 1, 0, 1, None)
    env.set_rollout_tasks(
        [
            {
                "slot": 0,
                "task_id": 0,
                "task": args.task,
                "suite": args.task,
                "reset_seed": scene["scene_seed"],
                "round": 0,
            }
        ]
    )
    try:
        probes, executed, terminal = collect_probe_observations(
            student,
            teacher,
            env,
            scene,
            max_probes=args.probes,
            flow_indices=args.flow_indices,
            on_progress=lambda observations, count, flags: save_report(
                args.output,
                metadata,
                observations,
                executed=count,
                terminal=flags,
                finished=False,
            ),
        )
        save_report(
            args.output,
            metadata,
            probes,
            executed=executed,
            terminal=terminal,
            finished=True,
        )
    finally:
        env.close()
    print(f"FLOW_PROBE_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
