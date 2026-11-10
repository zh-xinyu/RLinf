# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native GR00T flow-matching SFT on teacher-generated demonstrations.

No teacher model, student ODE sampling, M3 weighting, or environment is used by
the optimizer. The original student's input/output metadata is kept unchanged.
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

from toolkits.libero_gr00t.offline_distillation import (
    bucket_rows,
    concatenate,
    fingerprint,
    preprocess,
    save_checkpoint,
)
from toolkits.libero_gr00t.offline_observations import SUITES, file_hash
from toolkits.libero_gr00t.probe import load_model
from toolkits.libero_gr00t.teacher_sft_data import TeacherActionPool, preprocess_sft


def restore_native_time_sampler(head) -> None:
    """Keep the native Beta law in FP32 after loading an RL checkpoint in BF16.

    Distributions are not module buffers, so action_head.float() does not cast
    the Beta distribution created under the checkpoint loader's default dtype.
    Torch's Dirichlet sampler does not implement BF16.
    """
    head.beta_dist = torch.distributions.Beta(
        torch.tensor(head.config.noise_beta_alpha, dtype=torch.float32),
        torch.tensor(head.config.noise_beta_beta, dtype=torch.float32),
    )


def native_sft_loss(student, inputs: dict) -> torch.Tensor:
    """Use the pinned GR00T upstream SFT forward, bypassing its RL-only override."""
    from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

    normalized = dict(inputs)
    for key in ("eagle_pixel_values", "eagle_image_sizes"):
        normalized[key] = normalized[key].reshape(-1, *normalized[key].shape[2:])
    backbone_inputs, action_inputs = student.prepare_input(normalized)
    with torch.no_grad():
        backbone_output = student.backbone(backbone_inputs)
    # Explicit native parent call: RL subclass.forward expects PPO chains.
    output = FlowmatchingActionHead.forward(
        student.action_head, backbone_output, action_inputs
    )
    return output["loss"]


def audit_example(student, example: tuple) -> dict:
    """Check identical inference observations and physical-action round trips."""
    raw, physical, valid = example
    inputs = preprocess_sft(student, raw, physical, valid)
    reference = preprocess(student, raw)
    for key, expected in reference.items():
        if key != "opd_raw_states":
            torch.testing.assert_close(inputs[key], expected, rtol=0, atol=0)
    action_dict = student._modality_transform.unapply(
        {"action": inputs["action"].float()}
    )
    physical_roundtrip = student.action_convert_fn(action_dict, chunk_size=16)
    # BF16 labels incur quantization; gripper sign must still match exactly.
    np.testing.assert_allclose(
        physical_roundtrip[..., :6], physical[None, ..., :6], atol=0.008, rtol=0.01
    )
    np.testing.assert_array_equal(physical_roundtrip[..., -1], physical[None, ..., -1])
    mask = inputs["action_mask"]
    if tuple(inputs["action"].shape) != (1, 16, 32):
        raise ValueError("Native action labels have an unexpected shape")
    if int(mask.sum()) != int(valid.sum()) * 7 or mask[..., 7:].any():
        raise ValueError("Action dimension or terminal masking is incorrect")
    return {
        "native_text_length": int(inputs["eagle_attention_mask"].sum()),
        "valid_timesteps": int(valid.sum()),
        "state_shape": list(inputs["state"].shape),
        "action_shape": list(inputs["action"].shape),
        "observation_parity": True,
        "physical_action_roundtrip": True,
    }


def train(cfg: dict, updates: int, resume: Path | None) -> None:
    """Accumulate suite-balanced gradients and average across learner ranks."""
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if cfg["global_batch_size"] % (world * len(SUITES)):
        raise ValueError("Global batch must divide evenly across ranks and suites")
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    torch.cuda.set_per_process_memory_fraction(
        cfg["allocator_gib"]
        * 2**30
        / torch.cuda.get_device_properties(local).total_memory
    )
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    run = None
    try:
        for seed_fn in (random.seed, np.random.seed, torch.manual_seed):
            seed_fn(cfg["seed"])
        args = SimpleNamespace(
            student=Path(cfg["student"]),
            execute=16,
            denoise_steps=4,
            noise_level=0.3,
            student_head_precision="fp32",
            student_text_padding="native",
        )
        student = load_model(args.student, args).cuda().eval()
        from rlinf.models.embodiment.gr00t.gr00t_n1d5.memory_utils import (
            enable_feature_only_backbone,
        )

        enable_feature_only_backbone(student)
        restore_native_time_sampler(student.action_head)
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(name.startswith("action_head."))
        student.action_head.train()
        parameters = [p for p in student.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("Trainable master parameters must be FP32")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg["learning_rate"],
            betas=tuple(cfg["adam_betas"]),
            eps=1e-8,
            weight_decay=cfg["weight_decay"],
            foreach=False,
        )
        start = 0
        if resume:
            from safetensors.torch import load_file

            if not (resume / "COMPLETE").exists():
                raise ValueError("Incomplete resume checkpoint")
            receipt = json.loads((resume / "manifest.json").read_text())
            recipe_hash = hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest()
            if receipt["recipe_sha256"] != recipe_hash:
                raise ValueError("Resume recipe differs")
            if receipt["head_sha256"] != file_hash(resume / "action_head.safetensors"):
                raise ValueError("Resume head content changed")
            if receipt["observation_pool_manifest_sha256"] != file_hash(
                Path(cfg["pool"]) / "manifest.json"
            ):
                raise ValueError("Resume dataset differs")
            tensors = load_file(str(resume / "action_head.safetensors"))
            expected = {k for k in student.state_dict() if k.startswith("action_head.")}
            if set(tensors) != expected:
                raise ValueError("Resume requires the complete action head")
            student.load_state_dict(tensors, strict=False)
            optimizer.load_state_dict(
                torch.load(
                    resume / "optimizer.pt", map_location="cpu", weights_only=True
                )
            )
            start = int(receipt["optimizer_updates"])
        if not start < updates <= cfg["maximum_updates"]:
            raise ValueError("Invalid training update range")
        output = Path(cfg["output"])
        output.mkdir(parents=True, exist_ok=True)
        pool = TeacherActionPool(Path(cfg["pool"]))
        checks = {
            suite: audit_example(student, pool.sample(suite, 1, 1234 + i)[0])
            for i, suite in enumerate(SUITES)
        }
        identities = [None] * world
        dist.all_gather_object(identities, fingerprint(student))
        if len(set(identities)) != 1:
            raise ValueError("Rank initialization differs")
        preflight = {
            "passed": True,
            "rank": rank,
            "world_size": world,
            "resume_update": start,
            "examples": checks,
            "training_objective": "Upstream GR00T conditional flow matching",
            "teacher_forwards": 0,
            "student_flow_refreshes": 0,
            "training_simulation_steps": 0,
            "metadata_sha256": file_hash(args.student / "experiment_cfg/metadata.json"),
            "head_fingerprint": identities[0],
        }
        (output / f"preflight_rank{rank}.json").write_text(
            json.dumps(preflight, indent=2) + "\n"
        )
        if rank == 0:
            import wandb

            run = wandb.init(
                project=cfg["wandb_project"],
                entity=cfg["wandb_entity"],
                name=cfg["run_name"],
                id=cfg["wandb_id"],
                resume="must" if resume else "never",
                group="teacher_rollout_sft_initialization",
                job_type="offline_training",
                config=cfg,
                dir=str(output),
            )
            run.define_metric("optimizer_update")
            run.define_metric("offline_sft/*", step_metric="optimizer_update")
        size = cfg["global_batch_size"] // world // len(SUITES)
        for step in range(start + 1, updates + 1):
            begun = time.perf_counter()
            # Per-update deterministic streams also make a saved-boundary resume exact.
            seed = cfg["seed"] + rank * 1000003 + step * 7919
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            optimizer.zero_grad(set_to_none=True)
            lr = cfg["learning_rate"] * min(step / max(cfg["warmup_updates"], 1), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            stats = torch.zeros((4, 2), device="cuda", dtype=torch.float64)
            for si, suite in enumerate(SUITES):
                examples = pool.sample(suite, size, seed + si * 104729)
                rows = [preprocess_sft(student, *example) for example in examples]
                # Match the native valid-entry objective independently of how
                # prompt lengths or microbatch sizes partition this suite.
                suite_entries = torch.tensor(
                    sum(int(row["action_mask"].sum()) for row in rows),
                    device="cuda",
                    dtype=torch.float64,
                )
                dist.all_reduce(suite_entries)
                for group in bucket_rows(rows, cfg["microbatch"]):
                    entries = sum(int(row["action_mask"].sum()) for row in group)
                    inputs = concatenate(group)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, cache_enabled=False
                    ):
                        loss = native_sft_loss(student, inputs)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite native SFT loss")
                    (loss * (world * entries / (suite_entries * 4))).backward()
                    stats[si, 0] += loss.detach().double() * entries
                    stats[si, 1] += entries
                    del inputs, loss
                del rows, examples
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite SFT gradients")
            optimizer.step()
            dist.all_reduce(stats)
            if rank == 0:
                row = {
                    "optimizer_update": step,
                    "offline_sft/flow_matching_loss": float(
                        (stats[:, 0] / stats[:, 1]).mean()
                    ),
                    "offline_sft/grad_norm": float(norm),
                    "offline_sft/lr": lr,
                    "offline_sft/update_seconds": time.perf_counter() - begun,
                    "offline_sft/cumulative_samples": step * cfg["global_batch_size"],
                    "offline_sft/training_simulation_steps": 0,
                    "offline_sft/teacher_forwards": 0,
                    "offline_sft/peak_allocated_gib": torch.cuda.max_memory_allocated()
                    / 2**30,
                    "offline_sft/peak_reserved_gib": torch.cuda.max_memory_reserved()
                    / 2**30,
                }
                row.update(
                    {
                        f"offline_sft/loss/{s}": float(stats[i, 0] / stats[i, 1])
                        for i, s in enumerate(SUITES)
                    }
                )
                with (output / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                run.log(row)
            if step in cfg["save_updates"] or step == updates:
                identities = [None] * world
                dist.all_gather_object(identities, fingerprint(student))
                if len(set(identities)) != 1:
                    raise RuntimeError("Distributed parameters diverged")
                save_checkpoint(student, optimizer, cfg, step, rank)
        if rank == 0:
            (output / f"completed_through_{updates}.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "optimizer_updates": updates,
                        "teacher_forwards": 0,
                        "student_flow_refreshes": 0,
                        "training_rollouts": 0,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        if run:
            run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--updates", required=True, type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.updates, args.resume)


if __name__ == "__main__":
    main()
