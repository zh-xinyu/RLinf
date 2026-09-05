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
"""Measure routed M1 gradient magnitudes and interference without updating weights."""

import argparse
import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from toolkits.mopd.probe_supervision_matrix import (
    SUITE_TASKS,
    collect_fixed_observations,
    compose_mopd_config,
    inference_model_config,
    load_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        choices=("fsdp", "autocast"),
        default="fsdp",
        help="Use exact single-rank FSDP compute (default) or an approximate autocast path.",
    )
    parser.add_argument("--master-port", type=int, default=29571)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (normally within outputs/smoke/<run>/).",
    )
    return parser.parse_args()


def tensor_list_dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    """Compute a float64 CPU dot product between two parameter-gradient lists."""
    assert len(left) == len(right)
    total = 0.0
    for left_tensor, right_tensor in zip(left, right, strict=True):
        assert left_tensor.shape == right_tensor.shape
        total += float(
            (left_tensor.float() * right_tensor.float()).sum(dtype=torch.float64)
        )
    return total


def gradient_geometry(gradients: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    """Compute per-suite norms, pairwise cosines, and aggregate cancellation."""
    names = list(gradients)
    squared_norms = {
        name: max(tensor_list_dot(gradients[name], gradients[name]), 0.0)
        for name in names
    }
    norms = {name: value**0.5 for name, value in squared_norms.items()}

    pairwise = {}
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            dot = tensor_list_dot(gradients[left], gradients[right])
            cosine = dot / max(norms[left] * norms[right], 1e-30)
            pairwise[f"{left}__{right}"] = {"dot": dot, "cosine": cosine}

    combined_squared = sum(squared_norms.values())
    for pair in pairwise.values():
        combined_squared += 2.0 * pair["dot"]
    combined_norm = max(combined_squared, 0.0) ** 0.5
    norm_sum = sum(norms.values())
    return {
        "suite_grad_norms": norms,
        "pairwise": pairwise,
        "sum_gradient_norm": combined_norm,
        "sum_individual_norms": norm_sum,
        "cancellation_ratio": combined_norm / max(norm_sum, 1e-30),
    }


def load_trainable_student(
    cfg, device: torch.device, *, backend: str, master_port: int
) -> torch.nn.Module:
    """Load FP32 action-head masters and optionally wrap them with FSDP."""
    from rlinf.models import get_model

    student = get_model(copy.deepcopy(cfg.actor.model)).to(device)
    student.eval()
    trainable = [
        (name, parameter)
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all(name.startswith("action_head.") for name, _ in trainable)
    assert all(parameter.dtype == torch.float32 for _, parameter in trainable)
    if backend == "fsdp":
        import torch.distributed as dist

        from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
        from rlinf.hybrid_engines.fsdp.utils import create_device_mesh
        from rlinf.utils.logging import get_logger

        if dist.is_initialized():
            raise RuntimeError("The standalone FSDP probe needs a fresh process group.")
        os.environ["LOCAL_RANK"] = str(device.index or 0)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{master_port}",
            rank=0,
            world_size=1,
        )
        mesh = create_device_mesh(1)
        strategy = FSDPStrategyBase.create(
            cfg.actor, world_size=1, dp_group=None, logger=get_logger()
        )
        student = strategy.wrap_model(student, mesh)
    return student


def student_flow_inputs(
    student: torch.nn.Module,
    obs: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Collect deterministic M1 flow states from the BF16 rollout copy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        _, result = student.predict_action_batch(copy.deepcopy(obs), mode="train")
    return result["forward_inputs"]


def suite_gradient(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    forward_inputs: dict[str, Any],
    backend: str,
) -> tuple[list[torch.Tensor], list[float]]:
    """Backpropagate the expected uniform-timestep M1 loss for one suite."""
    from rlinf.models.embodiment.base_policy import ForwardType

    trainable = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    student.zero_grad(set_to_none=True)
    chains = forward_inputs["chains"]
    num_steps = int(student.action_head.num_inference_timesteps)
    step_losses = []

    for step in range(num_steps):
        x_t = chains[:, step]
        timesteps = torch.full((x_t.shape[0],), step, dtype=torch.long)
        nft_inputs = {"x_t": x_t, "timesteps": timesteps}
        with torch.no_grad():
            teacher_output = teacher(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_inputs=nft_inputs,
                compute_values=False,
            )
        compute_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if backend == "autocast"
            else nullcontext()
        )
        with compute_context:
            student_output = student(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_inputs=nft_inputs,
                compute_values=False,
            )
        loss = (
            (student_output["v_theta"].float() - teacher_output["v_theta"].float())
            .square()
            .mean()
        )
        step_losses.append(float(loss.detach().cpu()))
        (loss / num_steps).backward()
        del teacher_output, student_output, loss

    missing = [idx for idx, parameter in enumerate(trainable) if parameter.grad is None]
    assert not missing, f"Trainable parameters without gradients: {missing[:3]}"
    gradients = [parameter.grad.detach().cpu().clone() for parameter in trainable]
    student.zero_grad(set_to_none=True)
    return gradients, step_losses


def main() -> None:
    """Run the no-update routed-gradient geometry probe."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The gradient geometry probe requires a CUDA device.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = compose_mopd_config()
    observations = collect_fixed_observations(cfg, args.num_envs)

    rollout_cfg = inference_model_config(
        cfg.actor.model, str(cfg.actor.model.model_path)
    )
    rollout_student = load_model(rollout_cfg, device)
    flow_inputs = {
        suite: student_flow_inputs(rollout_student, obs, args.seed + suite_idx * 1000)
        for suite_idx, (suite, obs) in enumerate(observations.items())
    }
    del rollout_student
    torch.cuda.empty_cache()

    student = load_trainable_student(
        cfg,
        device,
        backend=args.backend,
        master_port=args.master_port,
    )
    print(
        f"student loaded: FP32 action-head masters, backend={args.backend}", flush=True
    )

    gradients = {}
    suite_metrics = {}
    for suite in observations:
        teacher_path = str(cfg.rollout.expert_models[suite].model_path)
        teacher_cfg = inference_model_config(cfg.actor.model, teacher_path)
        teacher = load_model(teacher_cfg, device)
        suite_gradients, step_losses = suite_gradient(
            student=student,
            teacher=teacher,
            forward_inputs=flow_inputs[suite],
            backend=args.backend,
        )
        gradients[suite] = suite_gradients
        grad_norm = tensor_list_dot(suite_gradients, suite_gradients) ** 0.5
        suite_metrics[suite] = {
            "tasks": SUITE_TASKS[suite],
            "step_losses": step_losses,
            "uniform_loss": sum(step_losses) / len(step_losses),
            "grad_norm": grad_norm,
        }
        print(
            f"suite={suite} uniform_loss={suite_metrics[suite]['uniform_loss']:.6f} "
            f"grad_norm={grad_norm:.6f}",
            flush=True,
        )
        del teacher
        torch.cuda.empty_cache()

    geometry = gradient_geometry(gradients)
    print("\n=== pairwise gradient cosine ===", flush=True)
    for pair, metrics in geometry["pairwise"].items():
        print(f"{pair}: {metrics['cosine']:.6f}", flush=True)
    print(
        f"cancellation ratio: {geometry['cancellation_ratio']:.6f} "
        "(||sum g|| / sum ||g||)",
        flush=True,
    )

    output = {
        "student_checkpoint": str(cfg.actor.model.model_path),
        "compute_backend": args.backend,
        "compute_dtype": "bfloat16",
        "optimizer_step_performed": False,
        "seed": args.seed,
        "suites": suite_metrics,
        "geometry": geometry,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {args.output}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print("GRADIENT_GEOMETRY_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
