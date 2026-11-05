# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compare real GR00T NFT output/selected gradients before and after production FSDP wrapping.

One GPU, world size one, private file process group. This does not verify
four-rank gradient scaling, communication, optimization, or task success.
"""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

GRAD_KEYS = (
    "action_head.state_encoder.layer1.b",
    "action_head.action_encoder.W1.b",
    "action_head.action_decoder.layer2.b",
    "action_head.vlln.weight",
    "action_head.vl_self_attention.transformer_blocks.0.attn1.to_q.weight",
    "action_head.model.transformer_blocks.0.attn1.to_q.weight",
    "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight",
)


def canonical_name(name: str) -> str:
    """Remove only wrapper path components, preserving actual parameter names."""
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def selected_flat_gradients(prefix, flat, selected):
    """Read world-one flat gradients, including padding, with original names."""
    if flat.grad is None:
        return {}
    if flat.grad.numel() != sum(flat._numels_with_padding):
        raise ValueError("Parity reader requires full world-one flat gradients")
    result, offset, index = {}, 0, 0
    for size, padding in zip(
        flat._numels_with_padding, flat._is_padding_mask, strict=True
    ):
        if not padding:
            name = canonical_name(".".join(filter(None, (prefix, flat._fqns[index]))))
            if name in selected:
                result[name] = (
                    flat.grad.reshape(-1)[offset : offset + size]
                    .reshape(flat._shapes[index])
                    .detach()
                    .float()
                    .cpu()
                    .clone()
                )
            index += 1
        offset += size
    if index != len(flat._fqns):
        raise ValueError("Unexpected FSDP flat metadata")
    return result


def tensor_difference(actual, expected, atol, rtol):
    """Record numerical error without concealing a failed comparison."""
    actual, expected = actual.float().cpu(), expected.float().cpu()
    delta = actual - expected
    return {
        "close": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-20)),
        "reference_norm": float(expected.norm()),
        "reference_nonzero_elements": int(torch.count_nonzero(expected)),
        "elements": expected.numel(),
    }


def inv_freq_dtypes(model):
    return {
        canonical_name(name): str(module.inv_freq.dtype)
        for name, module in model.named_modules()
        if torch.is_tensor(getattr(module, "inv_freq", None))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=28)
    parser.add_argument("--minimum-free-gib", type=float, default=32)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Diagnostic control: request deterministic CUDA algorithms",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new parity output directory")
    cfg = OmegaConf.load(args.config)
    precision = cfg.actor.fsdp_config.mixed_precision
    if any(
        precision.get(key) is not None
        for key in ("param_dtype", "reduce_dtype", "buffer_dtype")
    ):
        raise ValueError(
            "Parity probe requires AMP-only FSDP with all three dtypes=null"
        )
    if precision.get("cast_forward_inputs", False):
        raise ValueError("Parity probe requires cast_forward_inputs=false")
    amp = cfg.actor.fsdp_config.amp_autocast
    if not amp.enabled or amp.precision != "bf16":
        raise ValueError("Parity probe requires BF16 autocast")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one CUDA_VISIBLE_DEVICES explicitly")
    free, total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(
            f"Need {args.minimum_free_gib} GiB free, found {free / 2**30:.2f}"
        )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(min(args.memory_gib * 2**30 / total, 1))
    os.environ["LOCAL_RANK"] = "0"

    from rlinf.hybrid_engines.fsdp import FSDP
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
    from toolkits.libero_gr00t.evaluate_full_checkpoint import (
        candidate_model_config,
        sha256,
    )
    from toolkits.libero_gr00t.probe import (
        create_env,
        observation_batch,
        renderer_index,
        tensors_to,
    )

    model_cfg, _ = candidate_model_config(args.config)
    egl = renderer_index()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env, task, states = create_env(suite, 0, args.seed)
    try:
        env.reset()
        observation = env.set_init_state(states[0])
        for _ in range(10):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        raw = observation_batch(observation, task.language)
    finally:
        env.close()
    model = get_model(model_cfg).cuda().eval()
    if cfg.actor.fsdp_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    original_inverse = inv_freq_dtypes(model)
    if any(dtype != "torch.float32" for dtype in original_inverse.values()):
        raise ValueError(f"Expected preserved FP32 inv_freq: {original_inverse}")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, payload = model.predict_action_batch(raw, mode="eval")
    inputs = tensors_to(payload["forward_inputs"], "cuda")
    if int(inputs["state_mask"].sum()) != 11:
        raise ValueError("Expected grouped 11D state")
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    x = torch.randn(1, 16, 32, generator=generator, device="cuda")
    index = torch.tensor([2], device="cuda")
    del payload
    args.output.mkdir(parents=True)
    (args.output / "status").write_text("RUNNING\n")
    torch.save(
        {
            "forward_inputs": tensors_to(inputs, "cpu"),
            "x_t": x.cpu(),
            "timesteps": index.cpu(),
        },
        args.output / "fixture.pt",
    )

    def forward_backward(policy):
        # Every pass starts from zero gradients and the identical RNG state.
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed + 2)
        policy.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = policy(
                forward_type=ForwardType.NFT,
                forward_inputs=inputs,
                nft_inputs={"x_t": x, "timesteps": index},
                compute_values=False,
            )["v_theta"]
            loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        return output.detach().float().cpu(), float(loss.detach())

    reference, reference_loss = forward_backward(model)
    named = dict(model.named_parameters())
    if any(key not in named or named[key].grad is None for key in GRAD_KEYS):
        raise ValueError("A selected reference action-head gradient is absent")
    reference_gradients = {
        key: named[key].grad.detach().float().cpu().clone() for key in GRAD_KEYS
    }
    repeated_reference, repeated_reference_loss = forward_backward(model)
    reference_repeat_errors = {
        key: tensor_difference(
            named[key].grad.detach().float().cpu(),
            reference_gradients[key],
            args.atol,
            args.rtol,
        )
        for key in GRAD_KEYS
    }
    reference_repeat_output = tensor_difference(
        repeated_reference, reference, args.atol, args.rtol
    )
    attention_processors = sorted(
        {
            type(module.processor).__name__
            for module in model.modules()
            if hasattr(module, "processor")
        }
    )
    gradient_checkpointing_modules = {
        canonical_name(name): bool(module.gradient_checkpointing)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "gradient_checkpointing", None), bool)
    }
    original_param_dtypes = {
        "head": sorted(
            {str(p.dtype) for n, p in named.items() if n.startswith("action_head.")}
        ),
        "backbone": sorted(
            {str(p.dtype) for n, p in named.items() if n.startswith("backbone.")}
        ),
    }
    model.zero_grad(set_to_none=True)
    del named
    gc.collect()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix="opdp-", dir="/tmp") as directory:
        torch.distributed.init_process_group(
            "nccl",
            init_method="file://" + str(Path(directory) / "group"),
            rank=0,
            world_size=1,
        )
        try:
            mesh = torch.distributed.device_mesh.init_device_mesh(
                "cuda", (1,), mesh_dim_names=("dp",)
            )
            wrapped = FSDPStrategy(cfg.actor, world_size=1).wrap_model(model, mesh)

            def wrapped_gradients():
                selected = {}
                for prefix, module in wrapped.named_modules():
                    if isinstance(module, FSDP) and module._handle is not None:
                        selected.update(
                            selected_flat_gradients(
                                prefix, module._handle.flat_param, GRAD_KEYS
                            )
                        )
                if set(selected) != set(GRAD_KEYS):
                    raise ValueError(
                        f"Missing wrapped gradients: {set(GRAD_KEYS) - set(selected)}"
                    )
                return selected

            actual, actual_loss = forward_backward(wrapped)
            selected = wrapped_gradients()
            repeated_actual, repeated_actual_loss = forward_backward(wrapped)
            repeated_selected = wrapped_gradients()
            wrapped_repeat_output = tensor_difference(
                repeated_actual, actual, args.atol, args.rtol
            )
            wrapped_repeat_errors = {
                key: tensor_difference(
                    repeated_selected[key], selected[key], args.atol, args.rtol
                )
                for key in GRAD_KEYS
            }
            output_error = tensor_difference(actual, reference, args.atol, args.rtol)
            gradient_errors = {
                key: tensor_difference(
                    selected[key], reference_gradients[key], args.atol, args.rtol
                )
                for key in GRAD_KEYS
            }
            wrapped_inverse = inv_freq_dtypes(wrapped)
            passed = (
                output_error["close"]
                and all(value["close"] for value in gradient_errors.values())
                and wrapped_inverse == original_inverse
                and reference_repeat_output["close"]
                and wrapped_repeat_output["close"]
                and all(value["close"] for value in reference_repeat_errors.values())
                and all(value["close"] for value in wrapped_repeat_errors.values())
            )
            report = {
                "passed": passed,
                "config": str(args.config.resolve()),
                "config_sha256": sha256(args.config),
                "world_size": 1,
                "production_strategy_factory": True,
                "use_orig_params": bool(cfg.actor.fsdp_config.use_orig_params),
                "fsdp_module_count": sum(
                    isinstance(m, FSDP) for m in wrapped.modules()
                ),
                "reference_loss": reference_loss,
                "wrapped_loss": actual_loss,
                "output": output_error,
                "selected_gradients": gradient_errors,
                "unwrapped_repeat": {
                    "output": reference_repeat_output,
                    "loss": repeated_reference_loss,
                    "selected_gradients": reference_repeat_errors,
                },
                "wrapped_repeat": {
                    "output": wrapped_repeat_output,
                    "loss": repeated_actual_loss,
                    "selected_gradients": wrapped_repeat_errors,
                },
                "numerical_environment": {
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cublas_workspace_config": os.environ.get(
                        "CUBLAS_WORKSPACE_CONFIG"
                    ),
                    "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                    "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "attention_processor_classes": attention_processors,
                    "gradient_checkpointing_modules": gradient_checkpointing_modules,
                    "atol": args.atol,
                    "rtol": args.rtol,
                },
                "parameter_dtypes_before_wrap": original_param_dtypes,
                "inv_freq_before": original_inverse,
                "inv_freq_after": wrapped_inverse,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "scope": "B1 NFT forward and selected gradients; no four-rank scaling, optimizer step, teacher loss or SR validation",
            }
            (args.output / "result.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            (args.output / "status").write_text("OK\n" if passed else "FAILED_PARITY\n")
            if not passed:
                raise RuntimeError("FSDP/unwrapped parity failed; see result.json")
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
