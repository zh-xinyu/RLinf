# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Probe sampled M4 OPD variance and gradient geometry without updating weights.

The probe uses fixed RoboCasa reset observations and never calls ``env.step``. It first
measures how a single Flow-SDE transition changes the emitted action at several noise
levels. It then checks the sampled reverse-KL reward against its analytic expectation and,
unless disabled, compares real FSDP action-head gradients from M4 with the exact local-KL
gradient (M3-Hybrid) at the same student-visited states. No optimizer is constructed.
"""

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from rlinf.algorithms.losses import flow_sde_transition_coeffs
from toolkits.mopd.probe_gradient_geometry import (
    gradient_geometry,
    load_trainable_student,
    tensor_list_dot,
)
from toolkits.mopd.probe_precision_init import compose_mopd_config
from toolkits.mopd.probe_supervision_matrix import (
    SUITE_TASKS,
    collect_fixed_observations,
    inference_model_config,
    load_model,
    rms,
    seeded_action,
)

GR00T_RANDOM = (
    "rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model.random.randint"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=(0.1, 0.25, 0.5),
        help="Flow-SDE noise levels used for action drift and reward statistics.",
    )
    parser.add_argument(
        "--mc-draws",
        type=int,
        default=4096,
        help="Cheap Gaussian draws used for conditional reward statistics.",
    )
    parser.add_argument(
        "--gradient-draws",
        type=int,
        default=4,
        help="Real backward samples per suite for estimating M4 gradient noise.",
    )
    parser.add_argument(
        "--backend",
        choices=("fsdp", "autocast"),
        default="fsdp",
        help="Use exact single-rank FSDP compute (default) or approximate autocast.",
    )
    parser.add_argument("--master-port", type=int, default=29573)
    parser.add_argument(
        "--skip-gradients",
        action="store_true",
        help="Skip model backward passes and only run reward/action diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (normally within outputs/smoke/<run>/).",
    )
    return parser.parse_args()


def reverse_kl_reward(
    delta_over_sigma: torch.Tensor, epsilon: torch.Tensor
) -> torch.Tensor:
    """Return ``log p_teacher - log p_student`` for shared-covariance Gaussians.

    ``delta_over_sigma`` is ``(mu_student - mu_teacher) / sigma`` and ``epsilon``
    is the standard-normal noise used to sample from the student transition. Leading
    Monte Carlo dimensions may be added to ``epsilon``; normal broadcasting applies.
    """
    delta = delta_over_sigma.float()
    eps = epsilon.float()
    return -eps * delta - 0.5 * delta.square()


def summarize_reverse_kl(
    delta_over_sigma: torch.Tensor,
    *,
    num_draws: int,
    seed: int,
) -> dict[str, float]:
    """Summarize conditional M4 reward bias and variance for one transition."""
    if num_draws <= 1:
        raise ValueError("num_draws must be greater than one.")
    delta = delta_over_sigma.detach().float().cpu()
    generator = torch.Generator().manual_seed(seed)
    epsilon = torch.randn((num_draws, *delta.shape), generator=generator)
    rewards = reverse_kl_reward(delta.unsqueeze(0), epsilon)
    draw_means = rewards.flatten(start_dim=1).mean(dim=1)
    analytic_kl = 0.5 * delta.square().mean()
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=True)
    return {
        "analytic_reverse_kl": float(analytic_kl),
        "sampled_reward_mean": float(reward_mean),
        "mean_error": float(reward_mean + analytic_kl),
        "element_reward_std": float(reward_std),
        "element_abs_mean_over_std": float(
            reward_mean.abs() / reward_std.clamp_min(1e-30)
        ),
        "batch_mean_reward_std": float(draw_means.std(unbiased=True)),
        "batch_mean_abs_mean_over_std": float(
            draw_means.mean().abs() / draw_means.std(unbiased=True).clamp_min(1e-30)
        ),
    }


def _mean_tensor_lists(samples: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Average corresponding tensors in a non-empty list of gradient lists."""
    if not samples:
        raise ValueError("At least one gradient sample is required.")
    width = len(samples[0])
    assert all(len(sample) == width for sample in samples)
    return [
        sum(
            (sample[idx].float() for sample in samples),
            torch.zeros_like(samples[0][idx]).float(),
        )
        / len(samples)
        for idx in range(width)
    ]


def summarize_stochastic_gradients(
    samples: list[list[torch.Tensor]],
    analytic: list[torch.Tensor],
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Compare sampled M4 gradients with their analytic local-KL target."""
    mean_gradient = _mean_tensor_lists(samples)
    analytic_norm = max(tensor_list_dot(analytic, analytic), 0.0) ** 0.5
    mean_norm = max(tensor_list_dot(mean_gradient, mean_gradient), 0.0) ** 0.5
    mean_cosine = tensor_list_dot(mean_gradient, analytic) / max(
        mean_norm * analytic_norm, 1e-30
    )

    sample_norms = []
    sample_cosines = []
    noise_squared = 0.0
    for sample in samples:
        sample_norm = max(tensor_list_dot(sample, sample), 0.0) ** 0.5
        sample_norms.append(sample_norm)
        sample_cosines.append(
            tensor_list_dot(sample, analytic) / max(sample_norm * analytic_norm, 1e-30)
        )
        residual = [
            sample_tensor.float() - mean_tensor
            for sample_tensor, mean_tensor in zip(sample, mean_gradient, strict=True)
        ]
        noise_squared += max(tensor_list_dot(residual, residual), 0.0)
    noise_norm = (noise_squared / len(samples)) ** 0.5
    mean_noise_norm = noise_norm / len(samples) ** 0.5

    return (
        {
            "analytic_grad_norm": analytic_norm,
            "sample_grad_norms": sample_norms,
            "sample_cosines_to_analytic": sample_cosines,
            "mean_grad_norm": mean_norm,
            "mean_cosine_to_analytic": mean_cosine,
            "single_draw_gradient_noise_rms": noise_norm,
            "single_draw_gradient_snr": mean_norm / max(noise_norm, 1e-30),
            "mean_estimate_gradient_noise_rms": mean_noise_norm,
            "mean_estimate_gradient_snr": mean_norm / max(mean_noise_norm, 1e-30),
        },
        mean_gradient,
    )


def compare_gradients(
    measured: list[torch.Tensor], analytic: list[torch.Tensor]
) -> dict[str, float]:
    """Compare two deterministic gradient lists without inventing noise metrics."""
    measured_norm = max(tensor_list_dot(measured, measured), 0.0) ** 0.5
    analytic_norm = max(tensor_list_dot(analytic, analytic), 0.0) ** 0.5
    return {
        "analytic_grad_norm": analytic_norm,
        "measured_grad_norm": measured_norm,
        "cosine_to_analytic": tensor_list_dot(measured, analytic)
        / max(measured_norm * analytic_norm, 1e-30),
    }


def _map_tensors(value: Any, transform) -> Any:
    """Recursively transform tensors in standard nested containers."""
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    return value


def _fixed_flow_inputs(
    model: torch.nn.Module,
    obs: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Collect deterministic ODE states while preserving the model's M4 config."""
    rl_config = model.action_head.rl_config
    old_method = rl_config.noise_method
    old_joint = rl_config.joint_logprob
    try:
        rl_config.noise_method = "flow_ode"
        rl_config.joint_logprob = True
        _, result = seeded_action(model, obs, seed, mode="train")
    finally:
        rl_config.noise_method = old_method
        rl_config.joint_logprob = old_joint
    return result["forward_inputs"]


def _action_drift_sweep(
    model: torch.nn.Module,
    obs: dict[str, Any],
    *,
    seed: int,
    noise_levels: list[float],
) -> dict[str, Any]:
    """Compare forced one-transition SDE actions with deterministic evaluation."""
    eval_action, _ = seeded_action(model, obs, seed, mode="eval")
    num_steps = int(model.action_head.num_inference_timesteps)
    rl_config = model.action_head.rl_config
    old_method = rl_config.noise_method
    old_joint = rl_config.joint_logprob
    old_noise = float(rl_config.noise_level)
    result = {"eval_action_rms": rms(eval_action), "noise_levels": {}}
    try:
        rl_config.noise_method = "flow_sde"
        rl_config.joint_logprob = False
        for noise_level in noise_levels:
            rl_config.noise_level = noise_level
            steps = []
            for step in range(num_steps):
                with patch(GR00T_RANDOM, return_value=step):
                    action, rollout = seeded_action(model, obs, seed, mode="train")
                selected = rollout["forward_inputs"]["denoise_inds"][:, 0]
                assert torch.all(selected == step)
                drift = rms(action - eval_action)
                steps.append(
                    {
                        "step": step,
                        "action_rms": rms(action),
                        "action_drift_rms": drift,
                        "drift_over_eval": drift / max(rms(eval_action), 1e-12),
                    }
                )
            result["noise_levels"][str(noise_level)] = {
                "steps": steps,
                "mean_action_drift_rms": float(
                    np.mean([item["action_drift_rms"] for item in steps])
                ),
                "mean_drift_over_eval": float(
                    np.mean([item["drift_over_eval"] for item in steps])
                ),
            }
    finally:
        rl_config.noise_method = old_method
        rl_config.joint_logprob = old_joint
        rl_config.noise_level = old_noise
    return result


def _cache_flow_states(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    obs: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    """Cache matched student/teacher velocities on deterministic student states."""
    from rlinf.models.embodiment.base_policy import ForwardType

    forward_inputs = _fixed_flow_inputs(student, obs, seed)
    chains = forward_inputs["chains"]
    num_steps = int(student.action_head.num_inference_timesteps)
    steps = []
    for step in range(num_steps):
        x_t = chains[:, step]
        timesteps = torch.full(
            (x_t.shape[0],), step, dtype=torch.long, device=x_t.device
        )
        velocities = []
        for model in (student, teacher):
            with torch.no_grad():
                output = model(
                    forward_type=ForwardType.NFT,
                    forward_inputs=forward_inputs,
                    nft_inputs={"x_t": x_t, "timesteps": timesteps},
                    compute_values=False,
                )
            velocities.append(output["v_theta"].detach().float().cpu())
        steps.append(
            {
                "x_t": x_t.detach().float().cpu(),
                "student_velocity": velocities[0],
                "teacher_velocity": velocities[1],
            }
        )
    cpu_inputs = _map_tensors(
        forward_inputs, lambda tensor: tensor.detach().cpu().clone()
    )
    return cpu_inputs, steps


def _transition_scalars(
    num_steps: int, noise_level: float, step: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return GR00T transition scale, signed velocity coefficient, and std."""
    w0, w1, c, sigma = flow_sde_transition_coeffs(num_steps, noise_level)
    return (
        (w0 + w1)[step].float().to(device),
        (-c[step]).float().to(device),
        sigma[step].float().to(device),
    )


def _suite_gradient(
    *,
    student: torch.nn.Module,
    forward_inputs: dict[str, Any],
    steps: list[dict[str, torch.Tensor]],
    noise_level: float,
    backend: str,
    method: str,
    seed: int,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Backpropagate one analytic or sampled uniform-timestep suite objective."""
    from rlinf.models.embodiment.base_policy import ForwardType

    if method not in {"analytic", "sampled"}:
        raise ValueError(f"Unknown gradient method: {method}")
    trainable = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    student.zero_grad(set_to_none=True)
    num_steps = len(steps)
    losses = []
    rewards = []

    for step, cached in enumerate(steps):
        x_t = cached["x_t"].to(next(student.parameters()).device)
        teacher_velocity = cached["teacher_velocity"].to(x_t.device)
        old_velocity = cached["student_velocity"].to(x_t.device)
        timesteps = torch.full(
            (x_t.shape[0],), step, dtype=torch.long, device=x_t.device
        )
        compute_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if backend == "autocast"
            else nullcontext()
        )
        with compute_context:
            output = student(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_inputs={"x_t": x_t, "timesteps": timesteps},
                compute_values=False,
            )
        velocity = output["v_theta"].float()
        scale, c_k, sigma = _transition_scalars(
            num_steps, noise_level, step, x_t.device
        )
        x_action = x_t[:, : velocity.shape[1], : velocity.shape[2]].float()
        current_mu = scale * x_action + c_k * velocity
        teacher_mu = scale * x_action + c_k * teacher_velocity

        if method == "analytic":
            loss = 0.5 * ((current_mu - teacher_mu) / sigma).square().mean()
        else:
            generator = torch.Generator().manual_seed(seed + step * 1009)
            epsilon = torch.randn(old_velocity.shape, generator=generator).to(
                x_t.device
            )
            old_mu = scale * x_action + c_k * old_velocity
            x_next = old_mu + sigma * epsilon
            reward = -0.5 * ((x_next - teacher_mu) / sigma).square()
            reward += 0.5 * ((x_next - old_mu) / sigma).square()
            logprob = (
                -torch.log(sigma)
                - 0.5 * math.log(2 * math.pi)
                - 0.5 * ((x_next - current_mu) / sigma).square()
            )
            loss = (-logprob * reward.detach()).mean()
            rewards.append(float(reward.detach().mean().cpu()))
        losses.append(float(loss.detach().cpu()))
        (loss / num_steps).backward()
        del output, velocity, loss

    missing = [idx for idx, parameter in enumerate(trainable) if parameter.grad is None]
    assert not missing, f"Trainable parameters without gradients: {missing[:3]}"
    gradients = [parameter.grad.detach().cpu().clone() for parameter in trainable]
    student.zero_grad(set_to_none=True)
    metrics = {"mean_loss": float(np.mean(losses))}
    if rewards:
        metrics["mean_reward"] = float(np.mean(rewards))
    return gradients, metrics


def _combined_gradient(gradients: dict[str, list[torch.Tensor]]) -> list[torch.Tensor]:
    """Return the equal-suite mean gradient."""
    return _mean_tensor_lists(list(gradients.values()))


def _gradient_probe(
    *,
    cfg,
    device: torch.device,
    cached: dict[str, dict[str, Any]],
    noise_level: float,
    draws: int,
    backend: str,
    master_port: int,
    seed: int,
) -> dict[str, Any]:
    """Run real action-head backwards and compare M4 with analytic M3-Hybrid."""
    if draws <= 0:
        raise ValueError("gradient_draws must be positive.")
    student = load_trainable_student(
        cfg, device, backend=backend, master_port=master_port
    )
    print(
        f"gradient student loaded: FP32 action-head masters, backend={backend}",
        flush=True,
    )
    analytic_gradients = {}
    mean_m4_gradients = {}
    suites = {}
    try:
        for suite, suite_cache in cached.items():
            forward_inputs = _map_tensors(
                suite_cache["forward_inputs"], lambda tensor: tensor.to(device)
            )
            analytic, analytic_metrics = _suite_gradient(
                student=student,
                forward_inputs=forward_inputs,
                steps=suite_cache["steps"],
                noise_level=noise_level,
                backend=backend,
                method="analytic",
                seed=seed,
            )
            samples = []
            sample_metrics = []
            for draw in range(draws):
                gradients, metrics = _suite_gradient(
                    student=student,
                    forward_inputs=forward_inputs,
                    steps=suite_cache["steps"],
                    noise_level=noise_level,
                    backend=backend,
                    method="sampled",
                    seed=seed + draw * 100_003,
                )
                samples.append(gradients)
                sample_metrics.append(metrics)
            summary, mean_gradient = summarize_stochastic_gradients(samples, analytic)
            summary["analytic"] = analytic_metrics
            summary["samples"] = sample_metrics
            suites[suite] = summary
            analytic_gradients[suite] = analytic
            mean_m4_gradients[suite] = mean_gradient
            print(
                f"suite={suite} M4mean/M3 cosine="
                f"{summary['mean_cosine_to_analytic']:.6f} "
                f"single_draw_snr={summary['single_draw_gradient_snr']:.4f} "
                f"mean_snr={summary['mean_estimate_gradient_snr']:.4f}",
                flush=True,
            )
            del forward_inputs
            torch.cuda.empty_cache()

        m4_geometry = gradient_geometry(mean_m4_gradients)
        m3_geometry = gradient_geometry(analytic_gradients)
        combined_summary = compare_gradients(
            _combined_gradient(mean_m4_gradients),
            _combined_gradient(analytic_gradients),
        )
        return {
            "suites": suites,
            "m4_mean_gradient_geometry": m4_geometry,
            "analytic_gradient_geometry": m3_geometry,
            "combined_m4_vs_analytic": combined_summary,
        }
    finally:
        del student
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    """Run the fixed-observation, no-update M4 probe."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The M4 probe requires a CUDA device.")
    if any(noise_level <= 0 for noise_level in args.noise_levels):
        raise ValueError("Every noise level must be positive.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = compose_mopd_config("robocasa365_mopd_m4_gr00t")
    observations = collect_fixed_observations(cfg, args.num_envs)

    rollout_cfg = inference_model_config(
        cfg.actor.model, str(cfg.actor.model.model_path)
    )
    rollout_student = load_model(rollout_cfg, device)
    print("M4 rollout student BF16 compute copy loaded", flush=True)

    action_drift = {}
    cached = {}
    reward_statistics = {}
    num_steps = int(rollout_student.action_head.num_inference_timesteps)
    for suite_idx, (suite, obs) in enumerate(observations.items()):
        suite_seed = args.seed + suite_idx * 10_000
        action_drift[suite] = _action_drift_sweep(
            rollout_student,
            obs,
            seed=suite_seed,
            noise_levels=list(args.noise_levels),
        )
        teacher_path = str(cfg.rollout.expert_models[suite].model_path)
        teacher_cfg = inference_model_config(cfg.actor.model, teacher_path)
        teacher = load_model(teacher_cfg, device)
        forward_inputs, steps = _cache_flow_states(
            student=rollout_student,
            teacher=teacher,
            obs=obs,
            seed=suite_seed,
        )
        cached[suite] = {"forward_inputs": forward_inputs, "steps": steps}
        reward_statistics[suite] = {}
        for noise_level in args.noise_levels:
            step_stats = []
            for step, step_cache in enumerate(steps):
                _, c_k, sigma = _transition_scalars(
                    num_steps, noise_level, step, torch.device("cpu")
                )
                delta = (
                    c_k
                    * (step_cache["student_velocity"] - step_cache["teacher_velocity"])
                    / sigma
                )
                metrics = summarize_reverse_kl(
                    delta,
                    num_draws=args.mc_draws,
                    seed=suite_seed + step * 1009 + int(noise_level * 1_000_000),
                )
                metrics["step"] = step
                step_stats.append(metrics)
            reward_statistics[suite][str(noise_level)] = step_stats
        print(
            f"cached suite={suite}: action drift + {num_steps} matched flow states",
            flush=True,
        )
        del teacher
        torch.cuda.empty_cache()

    del rollout_student
    torch.cuda.empty_cache()

    gradient_results = None
    if not args.skip_gradients:
        configured_noise = float(cfg.actor.model.rl_head_config.noise_level)
        gradient_results = _gradient_probe(
            cfg=cfg,
            device=device,
            cached=cached,
            noise_level=configured_noise,
            draws=args.gradient_draws,
            backend=args.backend,
            master_port=args.master_port,
            seed=args.seed + 1_000_000,
        )

    output = {
        "student_checkpoint": str(cfg.actor.model.model_path),
        "config_name": "robocasa365_mopd_m4_gr00t",
        "compute_backend": args.backend if not args.skip_gradients else None,
        "compute_dtype": "bfloat16",
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "environment_step_performed": False,
        "seed": args.seed,
        "num_observations_per_suite": args.num_envs,
        "mc_draws": args.mc_draws,
        "gradient_draws": 0 if args.skip_gradients else args.gradient_draws,
        "suites": {
            suite: {
                "tasks": SUITE_TASKS[suite],
                "action_drift": action_drift[suite],
                "reward_statistics": reward_statistics[suite],
            }
            for suite in observations
        },
        "gradients": gradient_results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)
    print("M4_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
