# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Audit explicit canonical fixed570 batches; this is not native-policy equivalence.

One model and one explicitly selected GPU; no Ray, simulator or optimizer. Initial
noise is injected only inside this diagnostic process by a temporary torch.randn
patch. Every intercepted draw must match the exact full action-noise shape.
Strict tolerances are declared before running and failures remain in the report.
"""

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from toolkits.libero_gr00t.checkpoint_action_diagnostic import (
    checkpoint_identity,
    raw_fixture,
    read_fixtures,
)
from toolkits.libero_gr00t.evaluate_full_checkpoint import (
    candidate_model_config,
    sha256,
)
from toolkits.libero_gr00t.fsdp_parity import GRAD_KEYS, tensor_difference


@contextmanager
def inject_initial_noise(noise: torch.Tensor, *, policy_call: bool = True):
    """Replace exactly one matching initial-noise draw; restore torch afterward."""
    original = torch.randn
    draws = 0

    def draw(*args, **kwargs):
        nonlocal draws
        shape = kwargs.get("size", args)
        if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
            shape = args[0]
        if tuple(shape) == tuple(noise.shape):
            caller = inspect.currentframe().f_back
            if policy_call and (
                caller.f_code.co_name != "get_rl_action"
                or caller.f_globals.get("__name__")
                != "rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model"
            ):
                raise RuntimeError(
                    "Unexpected same-shape random draw outside GR00T action-head initial noise"
                )
            draws += 1
            if draws > 1:
                raise RuntimeError("Multiple initial-noise draws in one policy call")
            device = kwargs.get("device", noise.device)
            dtype = kwargs.get("dtype", noise.dtype)
            value = noise.to(device=device, dtype=dtype).clone()
            if not torch.equal(value.cpu().float(), noise.cpu().float()):
                raise ValueError("Noise cast would change the injected values")
            return value
        return original(*args, **kwargs)

    with patch("torch.randn", draw):
        yield
    if draws != 1:
        raise RuntimeError("Policy did not consume exactly one initial-noise draw")


@contextmanager
def fp32_head_control(head):
    """Disable autocast only inside the FP32 head; preserve the BF16 backbone."""
    parameters = list(head.parameters())
    if not parameters or any(
        p.dtype != torch.float32 for p in parameters if p.is_floating_point()
    ):
        raise ValueError("Head-only control requires existing FP32 head parameters")
    device_type = parameters[0].device.type

    def wrap(original):
        def wrapped(*args, **kwargs):
            args = list(args)
            kwargs = dict(kwargs)
            for index, name in enumerate(("backbone_output", "action_input")):
                if len(args) > index:
                    args[index] = head._cast_batch_feature_floats(
                        args[index], torch.float32
                    )
                elif name in kwargs:
                    kwargs[name] = head._cast_batch_feature_floats(
                        kwargs[name], torch.float32
                    )
            with torch.autocast(device_type, enabled=False):
                return original(*args, **kwargs)

        return wrapped

    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with (
            patch.object(head, "forward", wrap(head.forward)),
            patch.object(head, "get_rl_action", wrap(head.get_rl_action)),
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def combine_observations(rows: list[dict]) -> dict:
    """Concatenate exactly four raw observations for genuine batched preprocessing."""
    return {
        key: torch.cat([row[key] for row in rows], dim=0)
        if torch.is_tensor(rows[0][key])
        else [item for row in rows for item in row[key]]
        for key in rows[0]
    }


def slice_inputs(inputs: dict, index: int) -> dict:
    """Slice the stored batch dimension without retokenizing the observation."""
    return {
        key: value[index : index + 1] if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def selected_gradients(model) -> dict:
    """Copy a fixed, predeclared subset of actual action-head gradients to CPU."""
    named = dict(model.named_parameters())
    if any(key not in named or named[key].grad is None for key in GRAD_KEYS):
        raise ValueError("A required selected gradient is missing")
    return {key: named[key].grad.detach().float().cpu().clone() for key in GRAD_KEYS}


def comparison_result(name, actual, expected, atol, rtol, deterministic):
    """Separate exact input/repeat checks from declared batch-kernel tolerances."""
    if name.startswith("preprocessing/"):
        category, exact_required = "preprocessing", True
    elif "_repeat" in name:
        category, exact_required = "within_mode_repeat", deterministic
    elif name.startswith("legacy_vs_canonical"):
        category, exact_required = "legacy_single", False
    elif name.startswith("gradient_b4_vs_b1"):
        category, exact_required = "batch_gradient", False
    else:
        category, exact_required = "batch_forward", False
    local_atol, local_rtol = (0.0, 0.0) if exact_required else (atol, rtol)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    # Do not allow NaN/Infinity JSON metrics or two equal infinities to pass.
    if finite:
        result = tensor_difference(actual, expected, local_atol, local_rtol)
    else:
        result = {"close": False, "max_abs": None, "relative_l2": None}
    result.update(
        category=category,
        exact=bool(torch.equal(actual.cpu(), expected.cpu())),
        exact_required=exact_required,
        finite=finite,
        atol=local_atol,
        rtol=local_rtol,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Existing native resolved training config; only the two padding flags are overridden",
    )
    fixtures = parser.add_mutually_exclusive_group(required=True)
    fixtures.add_argument("--observations", type=Path)
    fixtures.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=36)
    parser.add_argument("--minimum-free-gib", type=float, default=40)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--fp32-head-control",
        action="store_true",
        help="Diagnostic only: BF16 frozen VLM, FP32 head compute with head autocast/TF32 disabled",
    )
    args = parser.parse_args()
    if not args.run_gpu:
        print("No GPU work performed; pass --run-gpu with exactly one selected device")
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 8 <= args.memory_gib <= args.minimum_free_gib:
        raise ValueError("Require 8 <= memory-gib <= minimum-free-gib")
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    rows = read_fixtures(args.observations, args.fixture_manifest)
    # Four different prompts provide the exact mixed-length batching question.
    rows = rows[:4]
    if len(rows) != 4:
        raise ValueError("Provide at least four observation fixtures")
    cfg, _ = candidate_model_config(args.config)
    cfg.opd_fixed_text_padding = True
    cfg.opd_canonical_fixed_padding = True
    if cfg.rl_head_config.padding_value != 570:
        raise ValueError("This audit requires fixed570")
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly select exactly one CUDA_VISIBLE_DEVICES")
    free, total = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(
            f"Need {args.minimum_free_gib}GiB free, found {free / 2**30:.2f}"
        )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(min(args.memory_gib * 2**30 / total, 1))
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

    model = get_model(cfg).cuda().eval()
    model.backbone.requires_grad_(False)
    model.action_head.requires_grad_(True)
    captured_backbone = {}

    def capture_backbone(module, inputs, output):
        value = output["backbone_features"]
        captured_backbone["features"] = value.detach().cpu().clone()

    with ExitStack() as stack:
        if args.fp32_head_control:
            stack.enter_context(fp32_head_control(model.action_head))
        handle = model.backbone.register_forward_hook(capture_backbone)
        stack.callback(handle.remove)
        run_audit(args, cfg, rows, model, captured_backbone)


def run_audit(args, cfg, rows, model, captured_backbone):
    """Run one recorded precision condition without changing checkpoint weights."""
    from rlinf.models.embodiment.base_policy import ForwardType

    raw = [raw_fixture(row) for row in rows]
    # BF16 values are exactly representable in either requested inference dtype.
    noise = torch.randn(4, 16, 32, device="cpu", dtype=torch.bfloat16)
    report = {
        "complete": False,
        "passed": False,
        "scope": "Canonical fixed570 batching, not native-policy equivalence or task SR",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "model_config": OmegaConf.to_container(cfg, resolve=True),
        "checkpoint_identity": checkpoint_identity(Path(cfg.model_path)),
        "fixtures": rows,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "deterministic": args.deterministic,
        "fp32_head_control": args.fp32_head_control,
        "effective_compute": {
            "backbone": "BF16 parameters and BF16 autocast",
            "head": "FP32 without autocast"
            if args.fp32_head_control
            else "FP32 parameters with BF16 autocast",
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "physical_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "noise_sha256": hashlib.sha256(noise.float().numpy().tobytes()).hexdigest(),
        "comparisons": {},
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__),
                Path("rlinf/models/embodiment/gr00t/gr00t_n1d5/__init__.py"),
                Path("rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py"),
            )
        },
    }
    args.output.mkdir(parents=True)
    torch.save({"noise": noise, "fixtures": rows}, args.output / "fixture.pt")

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def compare(name, actual, expected):
        report["comparisons"][name] = comparison_result(
            name, actual, expected, args.atol, args.rtol, args.deterministic
        )
        save()

    save()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        individual = []
        individual_backbones = []
        for i, row in enumerate(raw):
            model.opd_canonical_fixed_padding = False
            with inject_initial_noise(noise[i : i + 1]):
                old_actions, old_payload = model.predict_action_batch(
                    copy.deepcopy(row), mode="eval"
                )
            legacy_backbone = captured_backbone["features"]
            model.opd_canonical_fixed_padding = True
            with inject_initial_noise(noise[i : i + 1]):
                actions, payload = model.predict_action_batch(
                    copy.deepcopy(row), mode="eval"
                )
            compare(
                f"legacy_vs_canonical_b1_backbone/{i}",
                captured_backbone["features"],
                legacy_backbone,
            )
            individual_backbones.append(captured_backbone["features"])
            compare(
                f"preprocessing/legacy_initial_noise/{i}",
                old_payload["forward_inputs"]["chains"][:, 0],
                noise[i : i + 1],
            )
            compare(
                f"preprocessing/canonical_initial_noise/{i}",
                payload["forward_inputs"]["chains"][:, 0],
                noise[i : i + 1],
            )
            compare(f"legacy_vs_canonical_b1_actions/{i}", actions, old_actions)
            compare(
                f"legacy_vs_canonical_b1_chains/{i}",
                payload["forward_inputs"]["chains"],
                old_payload["forward_inputs"]["chains"],
            )
            individual.append((actions, payload))
            del old_actions, old_payload
        lengths = [
            int(p["forward_inputs"]["eagle_attention_mask"].sum())
            for _, p in individual
        ]
        report["true_token_lengths"] = lengths
        if len(set(lengths)) < 2:
            raise ValueError("Fixtures must contain at least two true prompt lengths")
        with inject_initial_noise(noise):
            actions, batch_payload = model.predict_action_batch(
                combine_observations(raw), mode="eval"
            )
        inputs = batch_payload["forward_inputs"]
        report["backbone_output_dtype"] = str(captured_backbone["features"].dtype)
        compare(
            "backbone_b4_vs_b1",
            captured_backbone["features"],
            torch.cat(individual_backbones),
        )
        compare("preprocessing/b4_initial_noise", inputs["chains"][:, 0], noise)
        compare(
            "canonical_b4_vs_b1_actions", actions, torch.cat([a for a, _ in individual])
        )
        compare(
            "canonical_b4_vs_b1_chains",
            inputs["chains"],
            torch.cat([p["forward_inputs"]["chains"] for _, p in individual]),
        )
        for key in [
            "eagle_input_ids",
            "eagle_attention_mask",
            "eagle_pixel_values",
            "state",
            "state_mask",
            "embodiment_id",
        ]:
            compare(
                f"preprocessing/{key}",
                inputs[key],
                torch.cat([p["forward_inputs"][key] for _, p in individual]),
            )
        with inject_initial_noise(noise):
            refreshed = model(
                forward_type=ForwardType.FLOW_REFRESH, forward_inputs=inputs
            )["chains"]
        single_refresh = []
        for i in range(4):
            with inject_initial_noise(noise[i : i + 1]):
                single_refresh.append(
                    model(
                        forward_type=ForwardType.FLOW_REFRESH,
                        forward_inputs=slice_inputs(inputs, i),
                    )["chains"]
                )
        compare("preprocessing/refresh_b4_initial_noise", refreshed[:, 0], noise)
        compare(
            "preprocessing/refresh_b1_initial_noise",
            torch.cat(single_refresh)[:, 0],
            noise,
        )
        compare("flow_refresh_b4_vs_b1", refreshed, torch.cat(single_refresh))
        compare("rollout_vs_refresh_b4", inputs["chains"], refreshed)
        x = inputs["chains"][:, 2].detach().clone()
        del (
            individual,
            individual_backbones,
            legacy_backbone,
            actions,
            batch_payload,
            single_refresh,
            refreshed,
        )
    torch.cuda.empty_cache()

    def nft(policy_inputs, state):
        return model(
            forward_type=ForwardType.NFT,
            forward_inputs=policy_inputs,
            nft_inputs={
                "x_t": state,
                "timesteps": torch.full(
                    (state.shape[0],), 2, dtype=torch.long, device="cuda"
                ),
            },
            compute_values=False,
        )["v_theta"]

    # Compare the mean B4 loss with four B1 losses each divided by four.
    def backward(batched):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed + 1)
        outputs = []
        for i in [None] if batched else range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = nft(
                    inputs if i is None else slice_inputs(inputs, i),
                    x if i is None else x[i : i + 1],
                )
                loss = value.float().square().mean() / (1 if batched else 4)
            loss.backward()
            outputs.append(value.detach().cpu())
            del value, loss
        result = torch.cat(outputs), selected_gradients(model)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return result

    b1, g1 = backward(False)
    b1_repeat, g1_repeat = backward(False)
    b4, g4 = backward(True)
    b4_repeat, g4_repeat = backward(True)
    report["nft_output_dtype"] = str(b4.dtype)
    compare("nft_b4_vs_b1", b4, b1)
    compare("nft_b1_repeat", b1_repeat, b1)
    compare("nft_b4_repeat", b4_repeat, b4)
    for key in GRAD_KEYS:
        compare(f"gradient_b4_vs_b1/{key}", g4[key], g1[key])
        compare(f"gradient_b1_repeat/{key}", g1_repeat[key], g1[key])
        compare(f"gradient_b4_repeat/{key}", g4_repeat[key], g4[key])
    report["complete"] = True
    report["failed_comparisons_by_category"] = {
        category: [
            name
            for name, row in report["comparisons"].items()
            if row["category"] == category and not row["close"]
        ]
        for category in (
            "preprocessing",
            "legacy_single",
            "batch_forward",
            "within_mode_repeat",
            "batch_gradient",
        )
    }
    report["passed"] = all(row["close"] for row in report["comparisons"].values())
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    save()
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
