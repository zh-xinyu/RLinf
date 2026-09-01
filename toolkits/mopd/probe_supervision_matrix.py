# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure routed teacher/student disagreement by task suite and flow timestep.

This is a fixed-observation diagnostic, not an environment rollout. It resets a small,
deterministic observation batch for two tasks in each suite, evaluates the undistilled
student and the matching routed teacher, and then closes the environments before model
scoring. Models use BF16 inference copies, matching FSDP's configured compute dtype; the
separate precision-init probe validates the student's FP32 optimizer masters.
"""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

REPO = Path(__file__).resolve().parents[2]
SUITE_TASKS = {
    "atomic_seen": ["TurnOnMicrowave", "CloseToasterOvenDoor"],
    "composite_seen": ["ScrubCuttingBoard", "RinseSinkBasin"],
    "composite_unseen": ["CuttingToolSelection", "CategorizeCondiments"],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (normally under ignored outputs/mopd/).",
    )
    return parser.parse_args()


def compose_mopd_config() -> DictConfig:
    """Compose the M1 config used for deterministic student flow states."""
    os.environ.setdefault("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    config_dir = REPO / "examples/embodiment/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="robocasa365_mopd_gr00t")
    OmegaConf.resolve(cfg)
    return cfg


def collect_fixed_observations(
    cfg: DictConfig, num_envs: int
) -> dict[str, dict[str, Any]]:
    """Reset and immediately close one deterministic env batch per task suite."""
    if num_envs <= 0 or num_envs % 2:
        raise ValueError(
            "--num-envs must be a positive multiple of two tasks per suite."
        )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env

    observations = {}
    for suite, tasks in SUITE_TASKS.items():
        env_cfg = copy.deepcopy(cfg.env.eval)
        with open_dict(env_cfg):
            env_cfg.total_num_envs = num_envs
            env_cfg.num_envs = num_envs
            env_cfg.group_size = 1
            env_cfg.task_soup = [suite]
            env_cfg.task_filter = tasks
            env_cfg.task_sampling_strategy = "ordered"
            env_cfg.use_fixed_reset_state_ids = True

        env = Robocasa365Env(
            env_cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        try:
            obs, _ = env.reset()
        finally:
            env.close()
        assert set(obs["task_groups"]) == {suite}, (
            f"Expected only {suite!r} observations, got {set(obs['task_groups'])}."
        )
        observations[suite] = obs
        print(
            f"observations ready: suite={suite} tasks={tasks} n={num_envs}",
            flush=True,
        )
    return observations


def inference_model_config(base: DictConfig, model_path: str) -> DictConfig:
    """Build a BF16 compute copy matching the actor's FSDP forward precision."""
    cfg = copy.deepcopy(base)
    with open_dict(cfg):
        cfg.model_path = model_path
        cfg.precision = "bf16"
        cfg.action_head_precision = "bf16"
        cfg.load_to_device = True
    return cfg


def load_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Load one frozen model on the requested device."""
    from rlinf.models import get_model

    model = get_model(cfg).to(device)
    model.eval()
    model.requires_grad_(False)
    action_head_dtypes = {
        parameter.dtype for parameter in model.action_head.parameters()
    }
    assert action_head_dtypes == {torch.bfloat16}, (
        f"The supervision probe must match BF16 FSDP compute, got {action_head_dtypes}."
    )
    return model


def seeded_action(
    model: torch.nn.Module,
    obs: dict[str, Any],
    seed: int,
    *,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one action prediction with a reproducible initial flow-noise draw."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        action, result = model.predict_action_batch(copy.deepcopy(obs), mode=mode)
    return action.float().numpy(), result


def rms(value: np.ndarray) -> float:
    """Compute a float64 root mean square."""
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2)))


def velocity_metrics(
    student_velocity: torch.Tensor, teacher_velocity: torch.Tensor
) -> dict:
    """Summarize one suite/timestep velocity comparison."""
    student = student_velocity.float().flatten(1)
    teacher = teacher_velocity.float().flatten(1)
    delta = student - teacher
    cosine = torch.nn.functional.cosine_similarity(student, teacher, dim=1).mean()
    return {
        "mse": float(delta.square().mean().cpu()),
        "cosine": float(cosine.cpu()),
        "student_rms": float(student.square().mean().sqrt().cpu()),
        "teacher_rms": float(teacher.square().mean().sqrt().cpu()),
    }


def diagnose_suite(
    *,
    suite: str,
    obs: dict[str, Any],
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    seed: int,
) -> dict[str, Any]:
    """Measure action and velocity disagreement for one routed suite."""
    from rlinf.models.embodiment.base_policy import ForwardType

    student_action, _ = seeded_action(student, obs, seed, mode="eval")
    teacher_action, _ = seeded_action(teacher, obs, seed, mode="eval")
    student_action_2, _ = seeded_action(student, obs, seed + 1, mode="eval")

    action_gap = rms(student_action - teacher_action)
    noise_gap = rms(student_action - student_action_2)
    action = {
        "student_rms": rms(student_action),
        "teacher_rms": rms(teacher_action),
        "student_teacher_rms": action_gap,
        "student_noise_rms": noise_gap,
        "gap_over_teacher": action_gap / max(rms(teacher_action), 1e-12),
        "gap_over_noise": action_gap / max(noise_gap, 1e-12),
    }

    _, student_result = seeded_action(student, obs, seed, mode="train")
    forward_inputs = student_result["forward_inputs"]
    chains = forward_inputs["chains"]
    num_steps = int(student.action_head.num_inference_timesteps)
    assert chains.shape[1] == num_steps + 1

    steps = []
    for step in range(num_steps):
        x_t = chains[:, step]
        timesteps = torch.full((x_t.shape[0],), step, dtype=torch.long)
        velocities = []
        for model in (student, teacher):
            with torch.no_grad():
                output = model(
                    forward_type=ForwardType.NFT,
                    forward_inputs=forward_inputs,
                    nft_inputs={"x_t": x_t, "timesteps": timesteps},
                    compute_values=False,
                )
            velocities.append(output["v_theta"])
        metrics = velocity_metrics(*velocities)
        metrics["step"] = step
        steps.append(metrics)

    mse = [step["mse"] for step in steps]
    summary = {
        "uniform_mse": float(np.mean(mse)),
        "late_over_early_mse": mse[-1] / max(mse[0], 1e-12),
    }
    result = {
        "suite": suite,
        "tasks": SUITE_TASKS[suite],
        "num_observations": len(obs["task_groups"]),
        "action": action,
        "velocity_steps": steps,
        "velocity_summary": summary,
    }

    print(f"\n=== {suite} ===", flush=True)
    print(
        "action: "
        f"student={action['student_rms']:.4f} teacher={action['teacher_rms']:.4f} "
        f"gap={action_gap:.4f} noise={noise_gap:.4f} "
        f"gap/noise={action['gap_over_noise']:.2f}",
        flush=True,
    )
    for metrics in steps:
        print(
            f"step {metrics['step']}: mse={metrics['mse']:.6f} "
            f"cos={metrics['cosine']:.6f} "
            f"|student|={metrics['student_rms']:.4f} "
            f"|teacher|={metrics['teacher_rms']:.4f}",
            flush=True,
        )
    print(
        f"uniform mse={summary['uniform_mse']:.6f} "
        f"late/early={summary['late_over_early_mse']:.2f}",
        flush=True,
    )
    return result


def main() -> None:
    """Run the three-suite fixed-observation diagnostic."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The supervision matrix probe requires a CUDA device.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = compose_mopd_config()
    observations = collect_fixed_observations(cfg, args.num_envs)

    student_cfg = inference_model_config(
        cfg.actor.model, str(cfg.actor.model.model_path)
    )
    student = load_model(student_cfg, device)
    print("student BF16 compute copy loaded", flush=True)

    results = []
    for suite, obs in observations.items():
        teacher_path = str(cfg.rollout.expert_models[suite].model_path)
        teacher_cfg = inference_model_config(cfg.actor.model, teacher_path)
        teacher = load_model(teacher_cfg, device)
        print(f"routed teacher loaded: suite={suite}", flush=True)
        results.append(
            diagnose_suite(
                suite=suite,
                obs=obs,
                student=student,
                teacher=teacher,
                seed=args.seed + len(results) * 1000,
            )
        )
        del teacher
        torch.cuda.empty_cache()

    total_uniform_mse = sum(
        result["velocity_summary"]["uniform_mse"] for result in results
    )
    print("\n=== equal-count suite loss share ===", flush=True)
    for result in results:
        share = result["velocity_summary"]["uniform_mse"] / max(
            total_uniform_mse, 1e-12
        )
        result["velocity_summary"]["equal_count_loss_share"] = share
        print(f"{result['suite']}: {share:.1%}", flush=True)

    output = {
        "student_checkpoint": str(cfg.actor.model.model_path),
        "compute_dtype": "bfloat16",
        "seed": args.seed,
        "suites": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)
    print("SUPERVISION_MATRIX_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
