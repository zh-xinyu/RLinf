# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small four-suite MOPD pilot with independent evaluation and saved action heads.

This runner isolates model/learning feasibility before the Ray/FSDP experiment.
Microbatch one avoids padding different prompts together. Equal accumulation
counts and rank gradient means implement an equal-suite global batch exactly.
"""

import argparse
import hashlib
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from rlinf.algorithms.flow_opd import (
    map_target_flow_state_to_source,
    transport_source_velocity_to_target,
    validate_cross_metadata_flow_objective,
)
from rlinf.algorithms.losses import compute_analytic_local_kl_loss, flow_sde_kl_weights
from rlinf.models.embodiment.base_policy import ForwardType
from toolkits.libero_gr00t.probe import (
    SUITES,
    audit_teacher_inputs,
    collect,
    load_model,
    parameter_fingerprint,
    renderer_index,
    tensors_to,
)


def validate_recipe(cfg: dict) -> None:
    """Reject silently different batches and overlapping evaluation initial states."""
    if cfg["world_size"] != 4 or cfg["global_batch_size"] % 4:
        raise ValueError("Require four equal suite ranks and B divisible by four")
    if cfg["micro_batch_size"] != 1:
        raise ValueError("Native-prompt pilot currently requires microbatch one")
    if set(cfg["train_initial_states"]) & set(cfg["eval_initial_states"]):
        raise ValueError("Training/evaluation initial-state indices must be disjoint")
    for key in ["rounds", "reuse", "global_batch_size", "eval_interval_rounds"]:
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["reuse_mode"] != "refresh":
        raise ValueError("Pilot supports refresh only; cached mode is not implemented")


def average_parameter_gradients(parameters, world_size: int, backend: str):
    """All-reduce the global mean once, after local microbatch accumulation."""
    for _, parameter in parameters:
        if backend == "nccl":
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(world_size)
        else:
            gradient = (
                torch.zeros_like(parameter, device="cpu")
                if parameter.grad is None
                else parameter.grad.cpu()
            )
            dist.all_reduce(gradient)
            parameter.grad = (gradient / world_size).to(parameter.device)


def save_head(student, args, step: int, round_index: int) -> Path:
    """Save a portable head artifact referencing its unchanged base checkpoint."""
    destination = args.output.parent / f"checkpoint_update_{step:05d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".partial")
    temporary.mkdir()
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in student.state_dict().items()
        if name.startswith("action_head.")
    }
    save_file(tensors, str(temporary / "action_head.safetensors"))
    manifest = {
        "base_checkpoint": str(args.student),
        "optimizer_step": step,
        "rollout_round": round_index,
        "student_text_padding": args.student_text_padding,
        "student_head_precision": args.student_head_precision,
        "optimizer_state_saved": False,
        "resume_supported": False,
        "keys": sorted(tensors),
        "base_repo": args.student_repo,
        "base_revision": args.student_revision,
        "recipe": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.rename(destination)
    return destination


def load_head(student, path: Path):
    """Load an evaluated head only when every action-head key is present."""
    tensors = load_file(str(path / "action_head.safetensors"), device="cpu")
    expected = {
        name for name in student.state_dict() if name.startswith("action_head.")
    }
    if set(tensors) != expected:
        raise ValueError("Saved head key set differs from current model")
    student.load_state_dict(tensors, strict=False)


def evaluate(student, suite, args, round_index: int, step: int, write):
    """Use held-out initial states without perturbing the training RNG stream."""
    cpu_rng, cuda_rng, numpy_rng = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        np.random.get_state(),
    )
    python_rng = random.getstate()
    started = time.perf_counter()
    successes = episodes = simulation_steps = settling_steps = 0
    try:
        for task in args.eval_tasks:
            for trial in args.eval_initial_states:
                torch.manual_seed(args.seed + task * 100 + trial)
                pool, row = collect(student, suite, task, trial, args, max_chunks=None)
                del pool
                if not row["complete_episode"]:
                    raise RuntimeError("Evaluation episode was truncated")
                successes += int(row["success_once"])
                episodes += 1
                simulation_steps += row["steps"]
                settling_steps += row["settling_steps"]
                write(
                    {
                        "kind": "evaluation_episode",
                        "round": round_index,
                        "optimizer_step": step,
                        **row,
                    }
                )
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    write(
        {
            "kind": "evaluation",
            "round": round_index,
            "optimizer_step": step,
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "simulation_steps": simulation_steps,
            "settling_steps": settling_steps,
            "elapsed_s": time.perf_counter() - started,
        }
    )
    return successes, episodes


def endpoint_audit(student, teacher):
    """Check actual checkpoint normalizers against decoded physical commands."""
    from gr00t.data.transform.state_action import StateActionTransform

    transform = next(
        t
        for t in student._modality_transform.transforms
        if isinstance(t, StateActionTransform)
        and any(k.startswith("action.") for k in t.apply_to)
    )
    scale, shift = teacher.opd_action_endpoint_affine_to(
        student.metadata,
        target_action_transform=transform,
        target_action_keys=student._modality_config["action"].modality_keys,
    )
    source = torch.linspace(-0.8, 0.8, 16 * 32).reshape(1, 16, 32)
    target = source * scale + shift
    a = teacher._get_unnormalized_action(source)
    b = student._get_unnormalized_action(target)
    a = torch.cat(
        [
            torch.as_tensor(a[k])
            for k in teacher._modality_config["action"].modality_keys
        ],
        dim=-1,
    )
    b = torch.cat(
        [
            torch.as_tensor(b[k])
            for k in student._modality_config["action"].modality_keys
        ],
        dim=-1,
    )
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    return scale, shift, float((a - b).abs().max())


def train(student, teacher, suite, args, write, rank, world_size):
    validate_cross_metadata_flow_objective(
        "analytic_local_kl", "nft", m3_mode="student_coordinate_sde"
    )
    for name, p in student.named_parameters():
        p.requires_grad_(name.startswith("action_head."))
    teacher.requires_grad_(False)
    parameters = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [p for _, p in parameters],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
    )
    scale, shift, error = endpoint_audit(student, teacher)
    write(
        {
            "kind": "endpoint_audit",
            "max_physical_action_error": error,
            "scale": scale.tolist(),
            "shift": shift.tolist(),
        }
    )
    weights = flow_sde_kl_weights(args.denoise_steps, args.noise_level)
    local_batch = args.global_batch_size // world_size
    step = physical_steps = settling_steps = trajectories = 0
    evaluate(student, suite, args, 0, 0, write)
    dist.barrier()
    device = "cuda" if args.backend == "nccl" else "cpu"
    for round_index in range(1, args.rounds + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed + rank * 100000 + round_index)
        task = (round_index - 1) % 10
        trial = args.train_initial_states[
            ((round_index - 1) // 10) % len(args.train_initial_states)
        ]
        pool, rollout = collect(
            student, suite, task, trial, args, max_chunks=args.rollout_chunks or None
        )
        physical_steps += rollout["steps"]
        settling_steps += rollout["settling_steps"]
        trajectories += 1
        write(
            {
                "kind": "rollout",
                "round": round_index,
                "optimizer_step": step,
                "local_trajectories": trajectories,
                "local_simulation_steps": physical_steps,
                "local_settling_steps": settling_steps,
                **rollout,
            }
        )
        # Audit/cache each teacher's preprocessing independently of student padding.
        for item in pool:
            item["teacher_inputs"] = audit_teacher_inputs(
                item["inputs"], item["raw"], teacher
            )
        for reuse_index in range(args.reuse):
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            learning_rate = args.lr * min((step + 1) / max(args.warmup_updates, 1), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            indices = torch.randint(len(pool), (local_batch,)).tolist()
            staged = []
            noise_hash = hashlib.sha256()
            refresh_started = time.perf_counter()
            # Stage fresh student states first; no teacher is resident during refresh.
            for index in indices:
                inputs = tensors_to(pool[index]["inputs"], "cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    chains = student(
                        forward_type=ForwardType.FLOW_REFRESH, forward_inputs=inputs
                    )["chains"]
                k = torch.randint(args.denoise_steps, (1,), device="cuda")
                x = chains[torch.arange(1, device="cuda"), k].detach().cpu()
                noise_hash.update(chains[:, 0].float().cpu().numpy().tobytes())
                staged.append({"index": index, "x": x, "k": k.cpu()})
                del chains, inputs, k
            refresh_s = time.perf_counter() - refresh_started
            teacher_started = time.perf_counter()
            torch.cuda.empty_cache()
            teacher.cuda()
            for item in staged:
                source_x = map_target_flow_state_to_source(
                    item["x"].cuda(), item["k"], args.denoise_steps, scale, shift
                )
                inputs = tensors_to(pool[item["index"]]["teacher_inputs"], "cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    v = teacher(
                        forward_type=ForwardType.NFT,
                        forward_inputs=inputs,
                        nft_inputs={"x_t": source_x, "timesteps": item["k"]},
                    )["v_theta"]
                    item["target"] = transport_source_velocity_to_target(
                        source_x[:, : v.shape[1]],
                        v,
                        item["k"],
                        args.denoise_steps,
                        scale,
                        shift,
                    ).cpu()
                del inputs, source_x, v
            teacher.cpu()
            torch.cuda.empty_cache()
            teacher_s = time.perf_counter() - teacher_started
            backward_started = time.perf_counter()
            diagnostics = []
            for item in staged:
                inputs = tensors_to(pool[item["index"]]["inputs"], "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = student(
                        forward_type=ForwardType.NFT,
                        forward_inputs=inputs,
                        nft_inputs={"x_t": item["x"], "timesteps": item["k"]},
                    )["v_theta"]
                    loss, metrics = compute_analytic_local_kl_loss(
                        v, item["target"].cuda(), weights[item["k"]].cuda()
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite local loss")
                (loss / local_batch).backward()
                diagnostics.append(metrics)
                del inputs, v, loss
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for _, p in parameters
            ):
                raise FloatingPointError("Nonfinite local gradients")
            backward_s = time.perf_counter() - backward_started
            sync_started = time.perf_counter()
            average_parameter_gradients(parameters, world_size, args.backend)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in parameters], 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            fingerprint = parameter_fingerprint(parameters)
            fingerprints = [None] * world_size
            dist.all_gather_object(fingerprints, fingerprint)
            if len(set(fingerprints)) != 1:
                raise RuntimeError("Suite student replicas diverged")
            local_mse = float(np.mean([d["actor/velocity_mse"] for d in diagnostics]))
            local_loss = float(np.mean([d["actor/analytic_kl"] for d in diagnostics]))
            mean = torch.tensor(
                [local_mse, local_loss], dtype=torch.float64, device=device
            )
            dist.all_reduce(mean)
            mean /= world_size
            torch.cuda.synchronize()
            write(
                {
                    "kind": "update",
                    "optimizer_step": step,
                    "round": round_index,
                    "reuse_index": reuse_index,
                    "local_velocity_mse": local_mse,
                    "local_analytic_kl": local_loss,
                    "joint_velocity_mse": float(mean[0]),
                    "joint_analytic_kl": float(mean[1]),
                    "lr": learning_rate,
                    "grad_norm": float(norm),
                    "global_batch_size": args.global_batch_size,
                    "local_unique_observations": len(set(indices)),
                    "local_pool_observations": len(pool),
                    "fresh_noise_sha256": noise_hash.hexdigest(),
                    "parameter_fingerprint": fingerprint,
                    "local_simulation_steps": physical_steps,
                    "local_trajectories": trajectories,
                    "refresh_s": refresh_s,
                    "teacher_s": teacher_s,
                    "backward_s": backward_s,
                    "synchronize_optimizer_s": time.perf_counter() - sync_started,
                    "elapsed_s": time.perf_counter() - started,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            del staged
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        del pool
        optimizer.zero_grad(set_to_none=True)
        if round_index % args.eval_interval_rounds == 0 or round_index == args.rounds:
            if rank == 0:
                destination = save_head(student, args, step, round_index)
                write(
                    {
                        "kind": "checkpoint",
                        "round": round_index,
                        "optimizer_step": step,
                        "path": str(destination),
                    }
                )
                # Check that the actual artifact can be loaded before reporting it usable.
                before = parameter_fingerprint(parameters)
                load_head(student, destination)
                if parameter_fingerprint(parameters) != before:
                    raise RuntimeError("Saved head failed reload equality")
            dist.barrier()
            evaluate(student, suite, args, round_index, step, write)
            dist.barrier()
    write(
        {
            "kind": "complete",
            "optimizer_steps": step,
            "rollout_rounds": args.rounds,
            "local_simulation_steps": physical_steps,
            "local_settling_steps": settling_steps,
            "local_trajectories": trajectories,
            "formal_training": False,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(cli.config), resolve=True)
    validate_recipe(cfg)
    args = SimpleNamespace(
        **cfg, **vars(cli), mode="mopd", tasks=list(range(10)), trials=1
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.world_size:
        raise ValueError("Launcher world size differs from recipe")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(
        args.memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    dist.init_process_group(
        args.backend,
        timeout=timedelta(minutes=20),
        **({"device_id": torch.device("cuda:0")} if args.backend == "nccl" else {}),
    )
    egl = renderer_index()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    suite = benchmark.get_benchmark_dict()[args.suite]()

    def write(row):
        row = {"rank": rank, "suite": args.suite, **row}
        with args.output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    write(
        {
            "kind": "protocol",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "egl_index": egl,
        }
    )
    try:
        student = load_model(args.student, args).cuda().eval()
        teacher = load_model(args.teacher, args).cpu().eval()
        train(student, teacher, suite, args, write, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
