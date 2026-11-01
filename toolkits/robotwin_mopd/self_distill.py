# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Single-specialist self-distillation control through the production trainer.

The place_container_plate student and frozen teacher start from the identical
PACE SFT checkpoint. This removes multi-task interference while preserving the
current full-50 M3, joint training forward, FSDP FP32 masters and BF16 AMP path.
No production worker is replaced. Run only after sourcing the isolated runtime::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m toolkits.robotwin_mopd.self_distill \
        --run-id place_self_m3_full50_r4 --output-root outputs/robotwin4/controls \
        --dry-run

Remove --dry-run only when GPUs are available. The existing owner-scoped launcher
handles GPU checks and cleanup. Default: T8, B128, micro2, R4, 20 optimizer updates
(5 physical rounds), LR1e-5 with four-update warmup. Save once at update20 to
avoid spending control-run time repeatedly writing a ~66 GB full checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path

TASK = "place_container_plate"


def self_distill_overrides(
    checkpoint: Path,
    *,
    trajectories: int = 8,
    updates: int = 20,
    batch_size: int = 128,
    micro_batch_size: int = 2,
    lr: float = 1e-5,
    warmup_updates: int = 4,
    save_interval: int = 20,
) -> list[str]:
    """Build explicit replacement maps; Hydra must not retain the other tasks."""
    if trajectories not in (4, 8):
        raise ValueError("The four-rank control supports T4 or T8")
    if updates < 4 or updates % 4 or save_interval < 4 or save_interval % 4:
        raise ValueError("Updates and save_interval must be positive multiples of R4")
    if batch_size < 1 or micro_batch_size < 1 or batch_size % (4 * micro_batch_size):
        raise ValueError("Global batch must divide by four ranks times microbatch")
    if lr < 0 or warmup_updates < 0:
        raise ValueError("LR and warmup must be nonnegative")
    checkpoint = checkpoint.expanduser().resolve()
    path = json.dumps(str(checkpoint))
    norm = json.dumps(str(checkpoint / "assets/robotwin/norm_stats.json"))
    overrides = [
        f"actor.model.model_path={path}",
        "~rollout.expert_models",
        f"+rollout.expert_models={{{TASK}:{{model_path:{path}}}}}",
        "~actor.model.opd_task_norm_stats",
        f"+actor.model.opd_task_norm_stats={{{TASK}:{norm}}}",
        "runner.rollout_allocation.mode=uniform",
        "runner.logger.wandb_group=robotwin_correctness_self_distillation",
        "runner.logger.project_name=RoboTwin_MultiTeacher_Pilots",
        f"runner.max_steps={updates}",
        f"runner.save_interval={save_interval}",
        "runner.val_check_interval=-1",
        "algorithm.flow_refresh_updates=4",
        "+actor.model.openpi.nft_full_horizon=true",
        "actor.model.openpi.nft_joint_forward=true",
        "actor.cast_trainable_params_to_fp32=true",
        "actor.model.require_fp32_trainable_params=true",
        "actor.fsdp_config.gradient_checkpointing=true",
        "actor.fsdp_config.mixed_precision.param_dtype=null",
        "actor.fsdp_config.mixed_precision.reduce_dtype=null",
        "actor.fsdp_config.mixed_precision.buffer_dtype=null",
        "actor.fsdp_config.amp_autocast.enabled=true",
        "actor.fsdp_config.amp_autocast.precision=bf16",
        f"actor.micro_batch_size={micro_batch_size}",
        f"actor.global_batch_size={batch_size}",
        f"actor.optim.lr={lr}",
        f"actor.optim.value_lr={lr}",
        f"actor.optim.lr_warmup_steps={warmup_updates}",
    ]
    for mode in ("train", "eval"):
        overrides.extend(
            [
                f"env.{mode}.task_names=[{TASK}]",
                f"~env.{mode}.task_horizons",
                f"+env.{mode}.task_horizons={{{TASK}:400}}",
                f"env.{mode}.total_num_envs={trajectories}",
                f"env.{mode}.max_episode_steps=400",
                f"env.{mode}.max_steps_per_rollout_epoch=400",
            ]
        )
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trajectories", type=int, choices=[4, 8], default=8)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint or (
        Path(os.environ["ROBOTWIN_RUNTIME_ROOT"]) / "models/pi05_pace_torch" / TASK
    )
    overrides = self_distill_overrides(
        checkpoint,
        trajectories=args.trajectories,
        updates=args.updates,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        warmup_updates=args.warmup_updates,
        save_interval=args.save_interval,
    )
    if (
        not args.dry_run
        and not (checkpoint / "assets/robotwin/norm_stats.json").is_file()
    ):
        parser.error(
            "The common SFT checkpoint must contain its matching norm_stats.json"
        )
    command = [
        sys.executable,
        "-m",
        "toolkits.robotwin_mopd.run",
        "--num-gpus",
        "4",
        "--run-id",
        args.run_id,
        "--output-root",
        str(args.output_root),
    ]
    if args.dry_run:
        command.append("--dry-run")
    os.execv(sys.executable, [*command, *overrides])


if __name__ == "__main__":
    main()
