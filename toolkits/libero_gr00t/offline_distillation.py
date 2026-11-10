# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Routed M3 flow distillation on a fixed teacher-observation pool, without physics."""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from rlinf.algorithms.flow_opd import (
    map_target_flow_state_to_source,
    transport_source_velocity_to_target,
    validate_cross_metadata_flow_objective,
)
from rlinf.algorithms.losses import compute_analytic_local_kl_loss, flow_sde_kl_weights
from rlinf.models.embodiment.base_policy import ForwardType
from toolkits.libero_gr00t.offline_observations import (
    SUITES,
    ObservationPool,
    file_hash,
)
from toolkits.libero_gr00t.probe import audit_teacher_inputs, load_model


def bucket_rows(rows: list[dict], microbatch: int) -> list[list[dict]]:
    """Batch equal-length native prompts without inserting policy-visible padding."""
    if microbatch < 1 or not rows:
        raise ValueError("Expected positive microbatch and nonempty rows")
    groups = defaultdict(list)
    for row in rows:
        length = int(row["eagle_attention_mask"].sum())
        if row["eagle_input_ids"].shape[-1] != length:
            raise ValueError("Native input still contains text padding")
        groups[length].append(row)
    return [
        group[i : i + microbatch]
        for group in groups.values()
        for i in range(0, len(group), microbatch)
    ]


def preprocess(model, raw: dict) -> dict:
    """Match production observation conversion while avoiding an unused flow sample."""
    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        restore_opd_text_padding,
    )

    raw = dict(raw)
    raw["states"] = raw["states"].to(torch.bfloat16).float()
    converted = model.obs_convert_fn(raw)
    converted = {
        k: v if isinstance(v, np.ndarray) else np.asarray(v)
        for k, v in converted.items()
    }
    normalized = model.apply_transforms(converted)
    for key, value in normalized.items():
        if value.dtype == torch.float32:
            normalized[key] = value.to(torch.bfloat16)
    normalized = restore_opd_text_padding(normalized)
    for key in ("eagle_pixel_values", "eagle_image_sizes"):
        normalized[key] = normalized[key].reshape(
            1, model.image_nums, *normalized[key].shape[1:]
        )
    normalized["opd_raw_states"] = raw["states"].clone()
    return normalized


def concatenate(rows: list[dict]) -> dict:
    """Move a homogeneous native-text microbatch to the current CUDA device."""
    return {key: torch.cat([row[key] for row in rows], dim=0).cuda() for key in rows[0]}


def fingerprint(model) -> str:
    """Bind selected entries from each trainable tensor for distributed agreement."""
    checksum = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.detach().reshape(-1)
            checksum.update(name.encode())
            checksum.update(
                torch.cat([value[:8], value[-8:]]).float().cpu().numpy().tobytes()
            )
    return checksum.hexdigest()


def cpu_tree(value):
    """Copy optimizer state to CPU for portable, device-independent resume."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(student, optimizer, cfg: dict, step: int, rank: int) -> Path:
    """Save every head tensor and optimizer; export early/final standard HF policies."""
    from safetensors.torch import save_file

    folder = Path(cfg["output"]) / f"checkpoint-{step:06d}"
    if rank == 0:
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir()
        tensors = {
            k: v.detach().cpu().contiguous()
            for k, v in student.state_dict().items()
            if k.startswith("action_head.")
        }
        save_file(tensors, str(folder / "action_head.safetensors"))
        torch.save(cpu_tree(optimizer.state_dict()), folder / "optimizer.pt")
        if step in (cfg["initialization_update"], cfg["maximum_updates"]):
            destination = folder / "hf"
            student.save_pretrained(
                destination, max_shard_size="5GB", safe_serialization=True
            )
            shutil.copytree(
                Path(cfg["student"]) / "experiment_cfg", destination / "experiment_cfg"
            )
        receipt = {
            "optimizer_updates": step,
            "base_checkpoint": cfg["student"],
            "recipe_sha256": hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest(),
            "observation_pool_manifest_sha256": file_hash(
                Path(cfg["pool"]) / "manifest.json"
            ),
            "metadata_sha256": file_hash(
                Path(cfg["student"]) / "experiment_cfg/metadata.json"
            ),
            "head_sha256": file_hash(folder / "action_head.safetensors"),
            "head_fingerprint": fingerprint(student),
            "training_rollouts": 0,
            "full_hf_export": (folder / "hf").exists(),
        }
        (folder / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    dist.barrier()
    torch.save(
        {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        folder / f"rng_rank{rank}.pt",
    )
    dist.barrier()
    if rank == 0:
        (folder / "COMPLETE").write_text("OK\n")
    dist.barrier()
    return folder


def train(cfg: dict, updates: int, resume: Path | None) -> None:
    """Average equal per-rank gradients once per optimizer update on two GPUs."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world != 2 or cfg["global_batch_size"] % (world * len(SUITES)):
        raise ValueError("This recipe requires two ranks and an equal four-suite batch")
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
        validate_cross_metadata_flow_objective(
            "analytic_local_kl", "nft", m3_mode="student_coordinate_sde"
        )
        args = SimpleNamespace(
            execute=max(cfg["loss_prefixes"].values()),
            denoise_steps=4,
            noise_level=0.3,
            student=Path(cfg["student"]),
            student_head_precision="fp32",
            student_text_padding="native",
        )
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        student = load_model(args.student, args).cuda().eval()
        from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
            restore_opd_text_padding,
        )
        from rlinf.models.embodiment.gr00t.gr00t_n1d5.memory_utils import (
            enable_feature_only_backbone,
        )

        enable_feature_only_backbone(student)
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(name.startswith("action_head."))
        parameters = [p for p in student.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("Student master action-head parameters must be FP32")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg["learning_rate"],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
        )
        start = 0
        if resume is not None:
            from safetensors.torch import load_file

            if not (resume / "COMPLETE").exists():
                raise ValueError("Cannot resume an incomplete checkpoint")
            receipt = json.loads((resume / "manifest.json").read_text())
            expected_recipe = hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest()
            if receipt["recipe_sha256"] != expected_recipe or receipt[
                "head_sha256"
            ] != file_hash(resume / "action_head.safetensors"):
                raise ValueError("Resume recipe or checkpoint content changed")
            tensors = load_file(str(resume / "action_head.safetensors"), device="cpu")
            expected_keys = {
                k for k in student.state_dict() if k.startswith("action_head.")
            }
            if set(tensors) != expected_keys:
                raise ValueError("Resume requires the complete action head")
            student.load_state_dict(tensors, strict=False)
            optimizer.load_state_dict(
                torch.load(
                    resume / "optimizer.pt", map_location="cpu", weights_only=True
                )
            )
            start = receipt["optimizer_updates"]
        if not start < updates <= cfg["maximum_updates"]:
            raise ValueError("Invalid update range")
        teachers, affines = {}, {}
        for suite in SUITES:
            teacher = load_model(Path(cfg["teachers"][suite]), args)
            teacher.to(torch.bfloat16).cuda().eval().requires_grad_(False)
            enable_feature_only_backbone(teacher)
            teachers[suite] = teacher
            affines[suite] = teacher.opd_action_endpoint_affine_to(student.metadata)
        pool = ObservationPool(Path(cfg["pool"]))
        weights = flow_sde_kl_weights(4, 0.3).float().cuda()
        size = cfg["global_batch_size"] // world // len(SUITES)
        output = Path(cfg["output"])
        output.mkdir(exist_ok=True, parents=True)
        if rank == 0:
            import wandb

            run = wandb.init(
                project="libero_gr00t_pilot",
                entity="zhouxinyu2022-michigan-state-university",
                name=cfg["run_name"],
                id=cfg["wandb_id"],
                resume="allow" if resume else "never",
                group="offline_teacher_distillation",
                job_type="training",
                config=cfg,
                dir=str(output),
            )
            run.define_metric("optimizer_update")
            run.define_metric("offline/*", step_metric="optimizer_update")
        # Independent full-transform checks on one real cached decision per suite.
        for index, suite in enumerate(SUITES):
            raw = pool.sample(suite, 1, 1234 + index)[0]
            inputs = preprocess(student, raw)
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
            ):
                _, result = student.predict_action_batch(dict(raw), mode="train")
            for key in inputs:
                torch.testing.assert_close(
                    inputs[key].cpu(),
                    result["forward_inputs"][key].cpu(),
                    rtol=0,
                    atol=0,
                )
            audit_teacher_inputs(inputs, raw, teachers[suite])
        identity = fingerprint(student)
        identities = [None] * world
        dist.all_gather_object(identities, identity)
        if len(set(identities)) != 1:
            raise RuntimeError("Student initialization differs across ranks")
        (output / f"preflight_rank{rank}.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "rank": rank,
                    "world_size": world,
                    "resume_update": start,
                    "head_fingerprint": identity,
                    "physical_training_rollouts": 0,
                    "local_samples_per_suite": size,
                    "allocator_gib": cfg["allocator_gib"],
                },
                indent=2,
            )
            + "\n"
        )
        for step in range(start + 1, updates + 1):
            begun = time.perf_counter()
            seed = cfg["seed"] + rank * 1000003 + step * 7919
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            optimizer.zero_grad(set_to_none=True)
            lr = cfg["learning_rate"] * min(step / max(cfg["warmup_updates"], 1), 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            stats = torch.zeros((len(SUITES), 3), dtype=torch.float64, device="cuda")
            for suite_index, suite in enumerate(SUITES):
                teacher = teachers[suite]
                scale, shift = affines[suite]
                raw_rows = pool.sample(suite, size, seed + suite_index * 104729)
                rows = [preprocess(student, raw) for raw in raw_rows]
                for group in bucket_rows(rows, cfg["microbatch"]):
                    n = len(group)
                    inputs = concatenate(group)
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            "cuda", dtype=torch.bfloat16, cache_enabled=False
                        ),
                    ):
                        chains = student(
                            forward_type=ForwardType.FLOW_REFRESH, forward_inputs=inputs
                        )["chains"]
                        k = torch.randint(4, (n,), device="cuda")
                        x = chains[torch.arange(n, device="cuda"), k].detach()
                        source_x = map_target_flow_state_to_source(
                            x, k, 4, scale, shift
                        )
                        teacher_inputs = restore_opd_text_padding(
                            inputs, fixed_length=teacher.padding_value
                        )
                        state, mask = teacher.opd_normalize_state(
                            inputs["opd_raw_states"]
                        )
                        teacher_inputs = dict(
                            teacher_inputs,
                            state=state.to(device="cuda", dtype=inputs["state"].dtype),
                            state_mask=mask.cuda(),
                            embodiment_id=teacher.opd_teacher_embodiment_id(n).cuda(),
                        )
                        source_v = teacher(
                            forward_type=ForwardType.NFT,
                            forward_inputs=teacher_inputs,
                            nft_inputs={"x_t": source_x, "timesteps": k},
                        )["v_theta"]
                        target = transport_source_velocity_to_target(
                            source_x[:, : source_v.shape[1]],
                            source_v,
                            k,
                            4,
                            scale,
                            shift,
                        ).detach()
                    del chains, teacher_inputs, source_x, source_v
                    prefix = cfg["loss_prefixes"][suite]
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, cache_enabled=False
                    ):
                        prediction = student(
                            forward_type=ForwardType.NFT,
                            forward_inputs=inputs,
                            nft_inputs={"x_t": x, "timesteps": k},
                        )["v_theta"]
                        loss, metrics = compute_analytic_local_kl_loss(
                            prediction[:, :prefix], target[:, :prefix], weights[k]
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite offline distillation loss")
                    (loss * (n / (size * len(SUITES)))).backward()
                    stats[suite_index] += torch.tensor(
                        [
                            metrics["actor/velocity_mse"] * n,
                            metrics["actor/analytic_kl"] * n,
                            n,
                        ],
                        device="cuda",
                    )
                    del inputs, target, x, k, prediction, loss
                del rows, raw_rows
            # Local batches have equal size; average their accumulated gradients.
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite offline gradients")
            optimizer.step()
            dist.all_reduce(stats)
            if rank == 0:
                row = {
                    "optimizer_update": step,
                    "offline/velocity_mse": float((stats[:, 0] / stats[:, 2]).mean()),
                    "offline/analytic_kl": float((stats[:, 1] / stats[:, 2]).mean()),
                    "offline/grad_norm": float(norm),
                    "offline/lr": lr,
                    "offline/update_seconds": time.perf_counter() - begun,
                    "offline/cumulative_samples": step * cfg["global_batch_size"],
                    "offline/training_rollouts": 0,
                    "offline/training_simulation_steps": 0,
                    "offline/peak_allocated_gib": torch.cuda.max_memory_allocated()
                    / 2**30,
                    "offline/peak_reserved_gib": torch.cuda.max_memory_reserved()
                    / 2**30,
                }
                row.update(
                    {
                        f"offline/velocity_mse/{suite}": float(
                            stats[i, 0] / stats[i, 2]
                        )
                        for i, suite in enumerate(SUITES)
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
                    raise RuntimeError("Distributed student parameters diverged")
                save_checkpoint(student, optimizer, cfg, step, rank)
        if rank == 0:
            (output / f"completed_through_{updates}.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "optimizer_updates": updates,
                        "physical_training_rollouts": 0,
                        "physics_used_only_in_separate_collection_and_evaluation": True,
                    }
                )
                + "\n"
            )
    finally:
        if run is not None:
            run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    lock_path = args.config.parent / "protocol_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if (
            cfg["loss_prefixes"] != lock["executed_prefixes"]
            or cfg["latent_horizon"] != lock["latent_horizon"]
        ):
            raise ValueError("Offline recipe violates the frozen execution protocol")
        for suite, teacher in lock["teachers"].items():
            if (
                Path(cfg["teachers"][suite]).resolve()
                != Path(teacher["path"]).resolve()
            ):
                raise ValueError(
                    "Offline teacher path violates the frozen checkpoint set"
                )
            for name, checksum in teacher["files_sha256"].items():
                if file_hash(Path(teacher["path"]) / name) != checksum:
                    raise ValueError(f"Frozen teacher content changed: {suite}/{name}")
    train(cfg, args.updates, args.resume)


if __name__ == "__main__":
    main()
