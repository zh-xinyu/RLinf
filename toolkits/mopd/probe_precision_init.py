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
"""Validate MOPD master-weight and optimizer precision without running an env."""

import argparse
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="robocasa365_mopd_gr00t",
        choices=("robocasa365_mopd_gr00t", "robocasa365_mopd_m4_gr00t"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Student checkpoint override (defaults to the config or checkpoint root env).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-optimizer-state",
        action="store_true",
        help="Only inspect model parameters; do not allocate Adam moments.",
    )
    return parser.parse_args()


def compose_mopd_config(config_name: str) -> DictConfig:
    """Compose a complete MOPD config without launching the RLinf runner."""
    os.environ.setdefault("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    config_dir = REPO / "examples/embodiment/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def dtype_counts(named_parameters) -> dict[str, int]:
    """Count parameter elements by dtype."""
    counts: dict[str, int] = {}
    for _, parameter in named_parameters:
        if parameter.is_floating_point():
            key = str(parameter.dtype)
            counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def format_counts(counts: dict[str, int]) -> str:
    """Format dtype element counts for a concise probe log."""
    return ", ".join(f"{name}={count:,}" for name, count in sorted(counts.items()))


def validate_model_precision(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Validate the mixed checkpoint lifecycle before FSDP wrapping."""
    named = list(model.named_parameters())
    trainable = [
        (name, parameter) for name, parameter in named if parameter.requires_grad
    ]
    action_head = [
        (name, parameter)
        for name, parameter in named
        if name.startswith("action_head.")
    ]
    backbone = [
        (name, parameter)
        for name, parameter in named
        if not name.startswith("action_head.")
    ]

    print(f"trainable:   {format_counts(dtype_counts(trainable))}", flush=True)
    print(f"action head: {format_counts(dtype_counts(action_head))}", flush=True)
    print(f"backbone:    {format_counts(dtype_counts(backbone))}", flush=True)

    assert trainable, "The MOPD student has no trainable parameters."
    invalid_trainable = [name for name, p in trainable if p.dtype != torch.float32]
    assert not invalid_trainable, (
        "Trainable master weights must be FP32; non-FP32 examples: "
        f"{invalid_trainable[:3]}"
    )
    invalid_head = [name for name, p in action_head if p.dtype != torch.float32]
    assert not invalid_head, (
        "Every action-head parameter must be restored in FP32; non-FP32 examples: "
        f"{invalid_head[:3]}"
    )
    invalid_backbone = [name for name, p in backbone if p.dtype != torch.bfloat16]
    assert not invalid_backbone, (
        f"Frozen backbone parameters must remain BF16; examples: {invalid_backbone[:3]}"
    )
    assert all(name.startswith("action_head.") for name, _ in trainable), (
        "MOPD should only optimize the action head."
    )
    print("MODEL_PRECISION_OK", flush=True)
    return [parameter for _, parameter in trainable]


def validate_adam_state(
    parameters: list[torch.nn.Parameter], optim_cfg: DictConfig
) -> None:
    """Initialize AdamW exactly as configured and validate moment dtypes."""
    from rlinf.utils.utils import warmup_optimizer_state

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optim_cfg.lr),
        betas=(float(optim_cfg.adam_beta1), float(optim_cfg.adam_beta2)),
        eps=float(optim_cfg.adam_eps),
        weight_decay=float(optim_cfg.weight_decay),
    )
    warmup_optimizer_state(optimizer)

    moments = []
    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name, None)
            if isinstance(value, torch.Tensor):
                moments.append((name, value))
    assert moments, "AdamW did not initialize any moment tensors."
    invalid = [
        (name, value.dtype) for name, value in moments if value.dtype != torch.float32
    ]
    assert not invalid, f"Adam moments must be FP32; examples: {invalid[:3]}"
    moment_numel = sum(value.numel() for _, value in moments)
    print(
        f"adam moments: torch.float32={moment_numel:,} "
        f"({moment_numel * 4 / 2**30:.2f} GiB)",
        flush=True,
    )
    print("OPTIMIZER_PRECISION_OK", flush=True)
    del optimizer


def main() -> None:
    """Run the initialization-only precision probe."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The precision probe requires a CUDA device.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    cfg = compose_mopd_config(args.config_name)
    checkpoint = args.checkpoint
    if checkpoint is None and "ROBOCASA365_CHECKPOINT_ROOT" in os.environ:
        checkpoint_root = Path(os.environ["ROBOCASA365_CHECKPOINT_ROOT"])
        checkpoint = checkpoint_root / "pretraining/checkpoint-80000"
    if checkpoint is not None:
        with open_dict(cfg.actor.model):
            cfg.actor.model.model_path = str(checkpoint)

    print(f"config:     {args.config_name}", flush=True)
    print(f"checkpoint: {cfg.actor.model.model_path}", flush=True)
    print(f"device:     {device}", flush=True)
    print(
        "FSDP compute config: "
        f"param={cfg.actor.fsdp_config.mixed_precision.param_dtype}, "
        f"reduce={cfg.actor.fsdp_config.mixed_precision.reduce_dtype}, "
        "cast_forward_inputs="
        f"{cfg.actor.fsdp_config.mixed_precision.cast_forward_inputs}",
        flush=True,
    )

    from rlinf.models import get_model

    model = get_model(cfg.actor.model).to(device)
    trainable = validate_model_precision(model)
    if not args.skip_optimizer_state:
        validate_adam_state(trainable, cfg.actor.optim)

    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"peak CUDA allocation: {peak_gib:.2f} GiB", flush=True)
    print("PRECISION_INIT_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
