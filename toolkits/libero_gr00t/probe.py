# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded GR00T LIBERO evaluation and four-rank joint MOPD engineering probe.

MOPD uses real student rollouts, frozen suite teachers, fresh flow paths, and the
production M3 loss. Optional Gloo CPU gradient averaging gives one shared student
across four suite ranks. This is deliberately not the Ray production runner or an
effectiveness experiment. Training segments can be short and do not estimate SR.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from rlinf.algorithms.flow_opd import (
    map_target_flow_state_to_source,
    transport_source_velocity_to_target,
    validate_cross_metadata_flow_objective,
)
from rlinf.algorithms.losses import compute_analytic_local_kl_loss, flow_sde_kl_weights
from rlinf.models.embodiment.base_policy import ForwardType

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (240, 320, 320, 520), strict=True))


def renderer_index() -> int:
    """Find EGL's renderer for CUDA logical zero, without assuming equal order."""
    library = ctypes.CDLL("libEGL.so.1")
    library.eglGetProcAddress.restype = ctypes.c_void_p
    query = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(library.eglGetProcAddress(b"eglQueryDevicesEXT"))
    attribute = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
    )(library.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
    devices = (ctypes.c_void_p * 32)()
    count = ctypes.c_int()
    if not query(32, devices, ctypes.byref(count)):
        raise RuntimeError("Cannot enumerate EGL devices")
    for index in range(count.value):
        ordinal = ctypes.c_ssize_t(-1)
        if (
            attribute(devices[index], 0x323A, ctypes.byref(ordinal))
            and ordinal.value == 0
        ):
            return index
    raise RuntimeError("Cannot resolve EGL renderer for visible CUDA zero")


def observation_batch(obs: dict, language: str) -> dict:
    """Reproduce RLinf LIBERO image orientation and axis-angle proprioception."""
    quat = np.array(obs["robot0_eef_quat"], copy=True)
    quat[3] = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(1 - quat[3] ** 2)
    axis = (
        np.zeros(3)
        if math.isclose(denominator, 0.0)
        else quat[:3] * 2 * np.arccos(quat[3]) / denominator
    )
    return {
        "main_images": torch.from_numpy(obs["agentview_image"][::-1, ::-1].copy())[
            None
        ],
        "wrist_images": torch.from_numpy(
            obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        )[None],
        "states": torch.from_numpy(
            np.concatenate(
                [obs["robot0_eef_pos"], axis, obs["robot0_gripper_qpos"]]
            ).astype(np.float32)
        )[None],
        "task_descriptions": [language],
    }


def tensors_to(inputs: dict, device: str) -> dict:
    """Copy tensor payloads without mutating the cached rollout pool."""
    return {
        key: value.detach().clone().to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def load_model(path: Path, args):
    """Load a native GR00T policy with its own embodiment and metadata."""
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

    metadata = json.loads((path / "experiment_cfg/metadata.json").read_text())
    if len(metadata) != 1:
        raise ValueError(f"Expected one explicitly identifiable embodiment in {path}")
    tag = next(iter(metadata))
    grouped = "eef_pos_absolute" in metadata[tag]["statistics"]["state"]
    cfg = OmegaConf.load("examples/embodiment/config/model/gr00t.yaml")
    cfg.model_path = str(path.resolve())
    cfg.embodiment_tag = tag
    cfg.embodiment_projector_id = 31
    cfg.obs_converter_type = "libero"
    cfg.data_config_class = "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoFrankaDataConfig"
    if grouped:
        cfg.data_config_class = "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoGroupedDataConfig"
        cfg.obs_converter_type = "libero_grouped"
    cfg.num_action_chunks = args.execute
    cfg.denoising_steps = args.denoise_steps
    if path.resolve() == args.student.resolve():
        cfg.action_head_precision = args.student_head_precision
    cfg.add_value_head = False
    cfg.rl_head_config.add_value_head = False
    cfg.rl_head_config.noise_method = "flow_ode"
    cfg.rl_head_config.joint_logprob = True
    cfg.rl_head_config.noise_level = args.noise_level
    model = get_model(cfg).eval()
    if grouped:
        model._probe_expected_state_dim = 11
    if path.resolve() == args.student.resolve():
        model.opd_fixed_text_padding = args.student_text_padding == "fixed"
    for transform in model._modality_transform.transforms:
        if hasattr(transform, "eagle_processor"):
            processor = transform.eagle_processor.image_processor
            processor.min_dynamic_tiles = 1
            processor.max_dynamic_tiles = 1
            processor.use_thumbnail = False
    # Audit key/shape compatibility rather than accepting silently random policy layers.
    from safetensors import safe_open

    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import _safetensors_weight_map

    weight_map = _safetensors_weight_map(path)
    expected = model.state_dict()
    missing = set(expected) - set(weight_map)
    unexpected = {k for k in set(weight_map) - set(expected) if "value_head" not in k}
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint key mismatch: missing={sorted(missing)[:5]}, extra={sorted(unexpected)[:5]}"
        )
    for shard in set(weight_map.values()):
        with safe_open(path / shard, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name in expected and tuple(
                    weights.get_slice(name).get_shape()
                ) != tuple(expected[name].shape):
                    raise ValueError(f"Checkpoint shape mismatch: {name}")
    return model


def audit_teacher_inputs(student_inputs: dict, raw: dict, teacher) -> dict:
    """Compare lightweight adapters and reused vision inputs with full transforms.

    This fails rather than silently using a student's embodiment ID, state
    normalization, or incompatible image/prompt processing for its teacher.
    """
    raw = tensors_to(raw, "cpu")
    raw["states"] = raw["states"].to(torch.bfloat16).float()
    converted = teacher.obs_convert_fn(raw)
    converted = {
        key: value if isinstance(value, np.ndarray) else np.array(value)
        for key, value in converted.items()
    }
    full = teacher.apply_transforms(converted)
    state, mask = teacher.opd_normalize_state(raw["states"])
    torch.testing.assert_close(
        state.float().cpu(), full["state"].float().cpu(), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(mask.bool().cpu(), full["state_mask"].bool().cpu())
    embodiment = teacher.opd_teacher_embodiment_id(1)
    torch.testing.assert_close(
        embodiment.reshape(-1).cpu(), full["embodiment_id"].reshape(-1).cpu()
    )
    # Each policy retains its own text-padding contract. GR00T's downstream
    # attention does not consistently mask padding, so teacher tokens cannot
    # simply reuse a native-length student's tensors.
    expected_tokens = full["eagle_input_ids"].cpu()[
        full["eagle_attention_mask"].cpu().bool()
    ]
    observed_tokens = student_inputs["eagle_input_ids"].cpu()[
        student_inputs["eagle_attention_mask"].cpu().bool()
    ]
    torch.testing.assert_close(expected_tokens, observed_tokens, rtol=0, atol=0)
    for key in ("eagle_input_ids", "eagle_attention_mask"):
        expected = full[key].cpu()
        if getattr(teacher, "opd_fixed_text_padding", True):
            if expected.shape[-1] > teacher.padding_value:
                raise ValueError("Teacher prompt exceeds fixed padding length")
            full[key] = torch.nn.functional.pad(
                expected, (0, teacher.padding_value - expected.shape[-1])
            )
    for key in ("eagle_pixel_values", "eagle_image_sizes"):
        expected = full[key].cpu().reshape(-1).to(student_inputs[key].dtype)
        observed = student_inputs[key].cpu().reshape(-1)
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    adapted = dict(student_inputs)
    for key, value in full.items():
        if torch.is_tensor(value):
            adapted[key] = (
                value.to(torch.bfloat16) if value.dtype == torch.float32 else value
            )
    # NFT expects the rollout payload's batch/image layout.
    for key in ("eagle_pixel_values", "eagle_image_sizes"):
        adapted[key] = adapted[key].reshape(student_inputs[key].shape)
    adapted["state"] = state.to(dtype=student_inputs["state"].dtype)
    adapted["state_mask"] = mask
    adapted["embodiment_id"] = embodiment
    return adapted


def create_env(suite, task_id: int, seed: int):
    """Build one environment and load only published NumPy initial-state types."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    with torch.serialization.safe_globals(
        [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.float64)),
            type(np.dtype(np.float32)),
        ]
    ):
        initial_states = suite.get_task_init_states(task_id)
    return env, task, initial_states


def collect(model, suite, task_id: int, trial: int, args, *, max_chunks: int | None):
    """Run the current policy in physics and retain each visited observation."""
    env, task, states = create_env(suite, task_id, args.seed)
    if trial >= len(states):
        raise ValueError("Requested initial-state index is out of range")
    pool = []
    simulation_s = inference_s = 0.0
    steps = 0
    success = False
    start = time.perf_counter()
    try:
        env.reset()
        obs = env.set_init_state(states[trial])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        reset_s = time.perf_counter() - start
        while steps < HORIZONS[args.suite] and not success:
            raw = observation_batch(obs, task.language)
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                actions, payload = model.predict_action_batch(dict(raw), mode="eval")
            # These selected N1.5 checkpoints were trained with projector31;
            # loading a second teacher must not mutate this through global patches.
            if not torch.all(payload["forward_inputs"]["embodiment_id"] == 31):
                raise RuntimeError(
                    "Selected LIBERO N1.5 checkpoint must use embodiment ID31"
                )
            expected_state_dim = getattr(model, "_probe_expected_state_dim", 8)
            if int(payload["forward_inputs"]["state_mask"].sum()) != expected_state_dim:
                raise RuntimeError("Unexpected transformed state dimension")
            torch.cuda.synchronize()
            inference_s += time.perf_counter() - start
            pool.append(
                {"inputs": tensors_to(payload["forward_inputs"], "cpu"), "raw": raw}
            )
            actions = torch.as_tensor(actions).detach().cpu().numpy()[0]
            if not np.isfinite(actions).all():
                raise FloatingPointError("Nonfinite policy action")
            start = time.perf_counter()
            for action in actions[: args.execute]:
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                success = bool(done)
                if success or steps >= HORIZONS[args.suite]:
                    break
            simulation_s += time.perf_counter() - start
            if max_chunks is not None and len(pool) >= max_chunks:
                break
    finally:
        env.close()
    return pool, {
        "task_id": task_id,
        "trial": trial,
        "success_once": success,
        "steps": steps,
        "settling_steps": 10,
        "chunks": len(pool),
        "horizon": HORIZONS[args.suite],
        "reset_s": reset_s,
        "simulation_s": simulation_s,
        "inference_s": inference_s,
        "complete_episode": bool(success or steps >= HORIZONS[args.suite]),
    }


def average_gradients(parameters, world_size: int) -> None:
    """Average equal-suite gradients over Gloo without GPU collective buffers."""
    if world_size == 1:
        return
    for _, parameter in parameters:
        # Every rank participates even when a parameter has no local gradient.
        gradient = (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            if parameter.grad is None
            else parameter.grad.detach().float().cpu()
        )
        dist.all_reduce(gradient)
        gradient.div_(world_size)
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)


def parameter_fingerprint(parameters) -> str:
    """Hash sampled entries from every trainable tensor to audit rank agreement."""
    checksum = hashlib.sha256()
    for name, parameter in parameters:
        flat = parameter.detach().reshape(-1)
        sampled = torch.cat([flat[:64], flat[-64:]]).float().cpu().numpy()
        checksum.update(name.encode())
        checksum.update(sampled.tobytes())
    return checksum.hexdigest()


def run_mopd(student, teacher, suite, args, write, rank: int, world_size: int):
    """Run small genuine joint updates, refreshing paths after each update."""
    validate_cross_metadata_flow_objective(
        "analytic_local_kl", "nft", m3_mode=args.coordinate_mode
    )
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith("action_head."))
    teacher.requires_grad_(False)
    parameters = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    schema = [(n, tuple(p.shape), str(p.dtype)) for n, p in parameters]
    if world_size > 1:
        schemas = [None] * world_size
        dist.all_gather_object(schemas, schema)
        if any(item != schema for item in schemas):
            raise RuntimeError("Ranks disagree on trainable parameter schema")
    optimizer = torch.optim.AdamW(
        [p for _, p in parameters],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    weights = flow_sde_kl_weights(args.denoise_steps, args.noise_level)
    scale, shift = teacher.opd_action_endpoint_affine_to(student.metadata)
    write(
        {
            "kind": "coordinate_contract",
            "mode": args.coordinate_mode,
            "endpoint_scale": scale.tolist(),
            "endpoint_shift": shift.tolist(),
            "student_embodiment": str(student.embodiment_tag),
            "teacher_embodiment": str(teacher.embodiment_tag),
            "trainable_parameters": sum(p.numel() for _, p in parameters),
            "native_teacher_sde_pushforward": False,
        }
    )
    update = 0
    physical_steps = 0
    for round_index in range(args.rounds):
        task_id = args.tasks[round_index % len(args.tasks)]
        trial = (round_index // len(args.tasks)) % args.trials
        torch.manual_seed(args.seed + rank * 10000 + round_index)
        pool, metrics = collect(
            student, suite, task_id, trial, args, max_chunks=args.rollout_chunks
        )
        physical_steps += metrics["steps"]
        write(
            {
                "kind": "rollout",
                "round": round_index + 1,
                **metrics,
                "cumulative_simulation_steps": physical_steps,
            }
        )
        if not pool:
            raise RuntimeError("Rollout produced no observations")
        previous_fingerprint = parameter_fingerprint(parameters)
        for reuse in range(args.reuse):
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            # Sample from the whole newly collected observation pool, not a fixed
            # first batch. Batch size is one per suite rank for this engineering probe.
            row_index = int(torch.randint(len(pool), ()).item())
            cached = pool[row_index]
            inputs = tensors_to(cached["inputs"], "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                chains = student(
                    forward_type=ForwardType.FLOW_REFRESH, forward_inputs=inputs
                )["chains"]
            step = torch.randint(args.denoise_steps, (1,), device="cuda")
            target_x = chains[torch.arange(1, device="cuda"), step].detach()
            adapted = audit_teacher_inputs(cached["inputs"], cached["raw"], teacher)
            teacher_inputs = tensors_to(adapted, "cuda")
            source_x = map_target_flow_state_to_source(
                target_x, step, args.denoise_steps, scale, shift
            )
            teacher.cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                source_v = teacher(
                    forward_type=ForwardType.NFT,
                    forward_inputs=teacher_inputs,
                    nft_inputs={"x_t": source_x, "timesteps": step},
                )["v_theta"]
                target_v = transport_source_velocity_to_target(
                    source_x[:, : source_v.shape[1]],
                    source_v,
                    step,
                    args.denoise_steps,
                    scale,
                    shift,
                ).detach()
            teacher.cpu()
            del teacher_inputs, source_v, adapted
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_v = student(
                    forward_type=ForwardType.NFT,
                    forward_inputs=inputs,
                    nft_inputs={"x_t": target_x, "timesteps": step},
                )["v_theta"]
                loss, diagnostics = compute_analytic_local_kl_loss(
                    student_v, target_v, weights[step.cpu()].to("cuda")
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite M3 loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for _, p in parameters
            ):
                raise FloatingPointError("Nonfinite local student gradient")
            average_gradients(parameters, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0)
            optimizer.step()
            update += 1
            fingerprint = parameter_fingerprint(parameters)
            if fingerprint == previous_fingerprint:
                raise RuntimeError(
                    "Optimizer did not change sampled student parameters"
                )
            previous_fingerprint = fingerprint
            if world_size > 1:
                fingerprints = [None] * world_size
                dist.all_gather_object(fingerprints, fingerprint)
                if len(set(fingerprints)) != 1:
                    raise RuntimeError(
                        "Joint student parameters diverged across suite ranks"
                    )
            joint_metrics = torch.tensor(
                [diagnostics["actor/velocity_mse"], diagnostics["actor/analytic_kl"]],
                dtype=torch.float64,
            )
            if world_size > 1:
                dist.all_reduce(joint_metrics)
                joint_metrics.div_(world_size)
            torch.cuda.synchronize()
            write(
                {
                    "kind": "update",
                    "joint/velocity_mse": float(joint_metrics[0]),
                    "joint/analytic_kl": float(joint_metrics[1]),
                    "round": round_index + 1,
                    "reuse_index": reuse,
                    "optimizer_step": update,
                    "observation_pool_index": row_index,
                    "flow_step": int(step.item()),
                    "fresh_noise_sha256": hashlib.sha256(
                        chains[:, 0].float().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "teacher_inputs_audited": True,
                    "grad_norm": float(grad_norm),
                    "parameter_fingerprint": fingerprint,
                    "elapsed_s": time.perf_counter() - start,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    **diagnostics,
                }
            )
            del chains, inputs, student_v, target_v, target_x, source_x, loss
        del pool
    write(
        {
            "kind": "complete",
            "optimizer_steps": update,
            "rollout_rounds": args.rounds,
            "local_simulation_steps": physical_steps,
            "world_size": world_size,
            "formal_training": False,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eval", "mopd"), required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--student-text-padding", choices=("fixed", "native"), default="fixed"
    )
    parser.add_argument(
        "--eval-roles",
        nargs="+",
        choices=("student", "teacher"),
        default=["student", "teacher"],
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--reuse", type=int, default=2)
    parser.add_argument("--rollout-chunks", type=int, default=4)
    parser.add_argument("--execute", type=int, default=8)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=24.0)
    parser.add_argument(
        "--student-head-precision", choices=("bf16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("reject", "student_coordinate_sde"),
        default="reject",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if (
        min(
            args.trials,
            args.rounds,
            args.reuse,
            args.rollout_chunks,
            args.execute,
            args.denoise_steps,
        )
        < 1
    ):
        raise ValueError("Counts must be positive")
    if not args.tasks or any(task < 0 or task >= 10 for task in args.tasks):
        raise ValueError("Task IDs must be in [0, 10)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(args.memory_gib * 2**30 / total, 0.95)
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("gloo", timeout=timedelta(minutes=10))
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
            "world_size": world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "egl_index": egl,
            "global_batch_size": world_size if args.mode == "mopd" else None,
        }
    )
    try:
        if args.mode == "eval":
            if world_size != 1:
                raise ValueError(
                    "Run evaluation as independent suite processes, without WORLD_SIZE"
                )
            for role, path in (("student", args.student), ("teacher", args.teacher)):
                if role not in args.eval_roles:
                    continue
                model = load_model(path, args).cuda().eval()
                for task in args.tasks:
                    for trial in range(args.trials):
                        torch.manual_seed(args.seed + task * 100 + trial)
                        pool, row = collect(
                            model, suite, task, trial, args, max_chunks=None
                        )
                        write({"kind": "episode", "role": role, **row})
                        del pool
                del model
                torch.cuda.empty_cache()
            write({"kind": "complete", "formal_evaluation": False})
        else:
            student = load_model(args.student, args).cuda().eval()
            teacher = load_model(args.teacher, args).cpu().eval()
            run_mopd(student, teacher, suite, args, write, rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
