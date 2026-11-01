# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Isolated four-task demonstration flow matching, without rollout or teachers.

Launch with torchrun on an explicitly selected GPU subset and a separate TCP
port. This entrypoint alone binds a demonstration forward to the loaded model;
the shared MOPD forward, model source, and checkpoint parameter names are intact.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader

from toolkits.robotwin_mopd.joint_sft_data import (
    TASKS,
    JointDemoDataset,
    RankBatchIndices,
    collate_demo_rows,
)


def make_flow_targets(actions, noise, timesteps):
    """Construct OpenPI's straight probability path and conditional velocity."""
    if actions.shape != noise.shape or timesteps.shape != actions.shape[:1]:
        raise ValueError("Flow target shapes must be [B,H,D], [B,H,D], [B]")
    t = timesteps[:, None, None]
    return t * noise + (1 - t) * actions, noise - actions


def demo_forward(
    self, *, observation, actions, noise=None, timesteps=None, augment_images=True
):
    """Evaluate full-horizon conditional flow matching through the joint model."""
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
    if timesteps is None:
        timesteps = self.sample_time(actions.shape[0], actions.device)
    x_t, target = make_flow_targets(actions, noise, timesteps)
    images, masks, tokens, token_masks, state = self._preprocess_observation(
        observation, train=augment_images
    )
    predicted, _ = self.get_joint_velocity(
        images, masks, tokens, token_masks, state, x_t, timesteps
    )
    if predicted.shape != target.shape:
        raise ValueError(
            f"Full 50x32 FM shape mismatch: {predicted.shape}, {target.shape}"
        )
    squared = (predicted.float() - target.float()).square()
    return {
        "loss_by_sample": squared.mean(dim=(1, 2)),
        "physical_loss_by_sample": squared[..., :14].mean(dim=(1, 2)),
        "padded_loss_by_sample": squared[..., 14:].mean(dim=(1, 2)),
    }


def prepare_batch(model, raw):
    """Apply the same task-specific delta/quantile/token/image transforms as OPD."""
    from openpi.models.model import Observation

    processed = model.input_transform(raw, transpose=False)
    actions = processed.pop("actions").to(
        device=torch.cuda.current_device(), dtype=torch.float32
    )
    processed = model.precision_processor(processed)
    if actions.shape[1:] != (50, 32):
        raise ValueError(f"Expected normalized actions [B,50,32], got {actions.shape}")
    return Observation.from_dict(processed), actions


def cast_trainable_fp32(model):
    """Preserve parameter identity, ties, frozen tensors and buffers."""
    with torch.no_grad():
        for parameter in model.parameters():
            if (
                parameter.requires_grad
                and parameter.is_floating_point()
                and parameter.dtype != torch.float32
            ):
                parameter.data = parameter.data.to(torch.float32)
    return model


def compose_config(args):
    """Reuse the tested model/FSDP configuration, with explicit offline overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    os.environ["OPD_RUN_DIR"] = str(args.output)
    with initialize_config_dir(
        config_dir=str(repo / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05")
    with open_dict(cfg):
        cfg.actor.global_batch_size = args.batch_size
        cfg.actor.micro_batch_size = args.microbatch
        cfg.actor.model.model_path = args.model_path or cfg.actor.model.model_path
        cfg.actor.model.openpi.nft_full_horizon = True
        cfg.actor.model.openpi.nft_joint_forward = True
        cfg.actor.enable_offload = False
        cfg.actor.fsdp_config.gradient_checkpointing = True
        cfg.actor.fsdp_config.enable_gradient_accumulation = False
        cfg.actor.fsdp_config.limit_all_gathers = True
        cfg.actor.fsdp_config.save_full_model_weights = True
        cfg.actor.optim.lr = args.lr
        cfg.actor.optim.value_lr = args.lr
        cfg.actor.optim.lr_warmup_steps = args.warmup
        cfg.actor.optim.total_training_steps = args.steps
        cfg.actor.optim.weight_decay = 1e-10
        cfg.actor.optim.lr_scheduler = "constant"
    OmegaConf.resolve(cfg)
    return cfg


def make_manager(cfg, world_size, rank):
    """Reuse RLinf's exact FSDP optimizer, clipping and checkpoint implementation."""
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
    from rlinf.models import get_model

    class JointSFTManager(FSDPModelManager):
        def model_provider_func(self):
            model = cast_trainable_fp32(get_model(self._cfg.model))
            model.forward = MethodType(demo_forward, model)
            return model

    manager = JointSFTManager(cfg.actor, world_size, rank)
    manager.setup_model_and_optimizer()
    return manager


@torch.no_grad()
def verify_official_forward(model, raw, output):
    """Compare identical observation/noise/time/augmentation against OpenPI SFT."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    observation, actions = prepare_batch(model, raw)
    noise = model.sample_noise(actions.shape, actions.device)
    timesteps = model.sample_time(len(actions), actions.device)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        candidate = demo_forward(
            model,
            observation=observation,
            actions=actions,
            noise=noise,
            timesteps=timesteps,
        )["loss_by_sample"]
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        reference = PI0Pytorch.forward(
            model, observation, actions, noise=noise, time=timesteps
        ).mean(dim=(1, 2))
    max_absolute = (candidate - reference).abs().max().item()
    max_relative = (
        ((candidate - reference).abs() / reference.abs().clamp_min(1e-8)).max().item()
    )
    report = {
        "candidate": candidate.tolist(),
        "official": reference.tolist(),
        "max_absolute_loss_difference": max_absolute,
        "max_relative_loss_difference": max_relative,
        "same_noise_time_and_augmentation_rng": True,
        "passed": max_relative < 0.01,
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(f"Demo forward differs from official SFT: {report}")
    return report


def source_identity():
    """Record source hashes even when this isolated development is uncommitted."""
    repo = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__),
        Path(__file__).with_name("joint_sft_data.py"),
        repo / "rlinf/models/embodiment/openpi/openpi_action_model.py",
        repo / "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
    ]
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }


def suite_totals(losses, task_ids):
    """Accumulate sums/counts before distributed reduction, not rank means."""
    totals = torch.zeros(
        (4, 4), dtype=torch.float64, device=losses["loss_by_sample"].device
    )
    for task in range(4):
        mask = task_ids.to(totals.device) == task
        totals[task, 0] = mask.sum()
        for col, key in enumerate(
            ("loss_by_sample", "physical_loss_by_sample", "padded_loss_by_sample"),
            start=1,
        ):
            totals[task, col] = losses[key].detach()[mask].double().sum()
    return totals


def totals_metrics(totals):
    """Return task metrics and the correctly weighted global loss."""
    metrics = {"loss": (totals[:, 1].sum() / totals[:, 0].sum().clamp_min(1)).item()}
    for index, task in enumerate(TASKS):
        count = totals[index, 0].item()
        metrics[f"{task}/samples"] = int(count)
        for col, key in enumerate(("loss", "physical_loss", "padded_loss"), start=1):
            metrics[f"{task}/{key}"] = (totals[index, col] / max(1, count)).item()
    return metrics


def memory_metrics(per_rank):
    """Summarize cumulative allocator peaks; units are decimal GB, not GiB."""
    return {
        "peak_allocated_gb": max(values[0] for values in per_rank),
        "peak_reserved_gb": max(values[1] for values in per_rank),
        "peak_memory_by_rank": [
            {"rank": rank, "allocated_gb": values[0], "reserved_gb": values[1]}
            for rank, values in enumerate(per_rank)
        ],
    }


def gather_memory_metrics(world_size):
    """Collect both peaks once per update; no allocator flush or device reset."""
    local = torch.tensor(
        [
            torch.cuda.max_memory_allocated() / 1e9,
            torch.cuda.max_memory_reserved() / 1e9,
        ],
        dtype=torch.float64,
        device="cuda",
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    return memory_metrics([values.tolist() for values in gathered])


def wandb_payload(namespace, step, metrics):
    """Keep a shared update axis and omit nested JSON-only memory detail."""
    return {
        "optimizer_step": step,
        **{
            f"{namespace}/{key}": value
            for key, value in metrics.items()
            if key != "step" and isinstance(value, (float, int))
        },
    }


def initialize_wandb(args, rank):
    """Explicit opt-in, rank-zero-only cloud logging; JSONL always remains local."""
    if rank != 0 or not args.wandb_project:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or args.output.name,
        dir=str(args.output),
        config=json.loads((args.output / "launch.json").read_text()),
        settings=wandb.Settings(init_timeout=60),
    )
    run.define_metric("optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("validation/*", step_metric="optimizer_step")
    (args.output / "wandb_run.json").write_text(
        json.dumps(
            {
                "id": run.id,
                "url": run.url,
                "project": args.wandb_project,
                "group": args.wandb_group,
                "name": run.name,
            },
            indent=2,
        )
        + "\n"
    )
    return run


@torch.no_grad()
def validate(manager, dataset, world_size, rank, seed, batches=4):
    """Fixed held-out scenes/noise/time; no image augmentation or policy rollout."""
    model = manager.model
    model.eval()
    totals = torch.zeros((4, 4), device="cuda", dtype=torch.float64)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed + rank)
        for index in range(batches):
            rows = [
                dataset[(index * world_size + rank) * 4 + task] for task in range(4)
            ]
            for row in rows:
                raw = collate_demo_rows([row])
                observation, actions = prepare_batch(model, raw)
                with manager.amp_context:
                    losses = model(
                        observation=observation, actions=actions, augment_images=False
                    )
                totals += suite_totals(losses, raw["opd_task_ids"])
    torch.distributed.all_reduce(totals)
    model.train()
    return totals_metrics(totals)


def build_parser():
    """Expose reproducible offline settings without changing MOPD config files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--holdout-episodes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--wandb-project", help="Explicit opt-in project; default is local JSONL only."
    )
    parser.add_argument("--wandb-group", default="robotwin_joint_sft_initialization")
    parser.add_argument(
        "--wandb-name", help="Defaults to the output directory name; use a pilot name."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="One GPU, no training; compare the full model against official SFT.",
    )
    parser.add_argument(
        "--inspect-data",
        action="store_true",
        help="CPU-only schema and action alignment audit.",
    )
    return parser


def main():
    """Train a shared four-task policy on official demonstrations only."""
    args = build_parser().parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.batch_size % (args.microbatch * world_size) or args.batch_size % 4:
        raise ValueError("Batch size must divide microbatch*world_size and four tasks")
    dataset = JointDemoDataset(
        args.data,
        seed=args.seed,
        samples=args.steps * args.batch_size,
        holdout_episodes=args.holdout_episodes,
    )
    if args.inspect_data:
        print(json.dumps(dataset.protocol, indent=2))
        return
    cfg = compose_config(args)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        from omegaconf import OmegaConf

        (args.output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
        (args.output / "launch.json").write_text(
            json.dumps(
                {
                    "arguments": {
                        key: str(val) if isinstance(val, Path) else val
                        for key, val in vars(args).items()
                    },
                    "world_size": world_size,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "data": dataset.protocol,
                    "initialization": "joint demonstration conditional flow matching from shared base model",
                    **source_identity(),
                },
                indent=2,
            )
            + "\n"
        )
    if args.verify_only:
        if world_size != 1:
            raise ValueError("--verify-only is a single-GPU check")
        from rlinf.models import get_model

        model = cast_trainable_fp32(get_model(cfg.actor.model)).cuda().train()
        task_reports = {}
        for task_id, task in enumerate(TASKS):
            raw = collate_demo_rows([dataset[task_id]])
            task_reports[task] = verify_official_forward(
                model, raw, args.output / f"official_forward_equivalence_{task}.json"
            )
        report = {
            "tasks": task_reports,
            "passed": all(row["passed"] for row in task_reports.values()),
        }
        (args.output / "official_forward_equivalence.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report), flush=True)
        return
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.distributed.barrier()
    wandb_run = initialize_wandb(args, rank)
    manager = make_manager(cfg, world_size, rank)
    manager.model.train()
    sampler = RankBatchIndices(args.batch_size, world_size, rank, args.steps)
    loader_kwargs = (
        {
            "multiprocessing_context": "spawn",
            "prefetch_factor": 2,
            "persistent_workers": True,
        }
        if args.workers
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=args.microbatch,
        sampler=sampler,
        collate_fn=collate_demo_rows,
        num_workers=args.workers,
        **loader_kwargs,
    )
    iterator = iter(loader)
    validation = (
        JointDemoDataset(
            args.data,
            seed=9173,
            split="validation",
            holdout_episodes=args.holdout_episodes,
        )
        if args.holdout_episodes
        else None
    )
    accumulation = args.batch_size // world_size // args.microbatch
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if validation is not None:
        metrics = validate(manager, validation, world_size, rank, 9173)
        if rank == 0:
            with (args.output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": 0, **metrics}) + "\n")
            if wandb_run is not None:
                wandb_run.log(wandb_payload("validation", 0, metrics))
    for step in range(1, args.steps + 1):
        step_start = time.monotonic()
        manager.optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros((4, 4), dtype=torch.float64, device="cuda")
        for _ in range(accumulation):
            raw = next(iterator)
            observation, actions = prepare_batch(manager.model, raw)
            # Synchronize each microbatch to keep FP32 gradient storage sharded.
            with (
                manager.before_micro_batch(manager.model, is_last_micro_batch=True),
                manager.amp_context,
            ):
                losses = manager.model(observation=observation, actions=actions)
                loss = losses["loss_by_sample"].mean() / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite FM loss at step {step}, rank {rank}"
                )
            manager.grad_scaler.scale(loss).backward()
            totals += suite_totals(losses, raw["opd_task_ids"])
        grad_norm, lrs = manager.optimizer_step()
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        manager.lr_scheduler.step()
        torch.distributed.all_reduce(totals)
        memory = gather_memory_metrics(world_size)
        elapsed = time.monotonic() - step_start
        if rank == 0:
            row = {
                "step": step,
                "examples": step * args.batch_size,
                "seconds": elapsed,
                "elapsed_seconds": time.monotonic() - start,
                "lr": lrs[0],
                "grad_norm": float(grad_norm),
                **memory,
                **totals_metrics(totals),
            }
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if wandb_run is not None:
                wandb_run.log(wandb_payload("train", step, row))
        if validation is not None and (
            step % args.validation_interval == 0 or step == args.steps
        ):
            metrics = validate(manager, validation, world_size, rank, 9173)
            if rank == 0:
                with (args.output / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, **metrics}) + "\n")
                if wandb_run is not None:
                    wandb_run.log(wandb_payload("validation", step, metrics))
        if step % args.save_interval == 0 or step == args.steps:
            checkpoint = args.output / "checkpoints" / f"global_step_{step}" / "actor"
            manager.save_checkpoint(str(checkpoint), step=step)
            if rank == 0:
                (checkpoint.parent / "metadata.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "method": "joint_demo_flow_matching",
                            "checkpoint_kind": "full_weights_and_optimizer",
                            "resume_supported_by_entrypoint": False,
                            "data": dataset.protocol,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    torch.distributed.barrier()
    if rank == 0:
        if wandb_run is not None:
            wandb_run.finish()
        (args.output / "COMPLETE").write_text(
            json.dumps(
                {"updates": args.steps, "wall_seconds": time.monotonic() - start}
            )
            + "\n"
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
