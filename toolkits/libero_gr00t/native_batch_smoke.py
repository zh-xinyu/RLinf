# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, single-GPU native B4/B1 parity audit using four saved observations.

Run from the worktree root with one explicitly selected free CUDA device. This
loads weights but creates no simulator and performs no updates. The same saved
RNG stream is replayed for automatic B4 splitting and four sequential B1 calls;
this gives equivalent random draws per item without changing policy code.

Example (do not run concurrently on an occupied GPU)::

    CUDA_VISIBLE_DEVICES=<free-device> outputs/runtime/bin/python -m \
      toolkits.libero_gr00t.native_batch_smoke --run-gpu \
      --student ../../libero_gr00t_20261104/models/student_fewshot \
      --observations ../../libero_feasibility_20261104 \
      --output ../../libero_gr00t_20261104/weak_student/native_batch_smoke.json

Optional --teachers takes Spatial, Object, Goal, Long checkpoint directories.
Their input audit compares the actor's lightweight state/text adapter against
independent full teacher preprocessing, followed by velocity parity. It does not
launch the Ray actor worker, so distributed actor integration remains separate.
"""

import argparse
import copy
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from toolkits.libero_gr00t.probe import SUITES, audit_teacher_inputs, observation_batch


def rng_state():
    """Snapshot every RNG used by preprocessing and action sampling."""
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


def cpu(value):
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value.detach().cpu().clone() if torch.is_tensor(value) else value


def device(value):
    return {
        key: item.cuda() if torch.is_tensor(item) else item
        for key, item in value.items()
    }


def compare(actual, expected, path="", atol=1e-6):
    """Check every output field, including cached flow chains and raw states."""
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            compare(actual[key], expected[key], f"{path}/{key}", atol)
    elif torch.is_tensor(actual):
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), rtol=0, atol=atol, msg=path
        )
    else:
        assert actual == expected, path


def load_policy(path, args, native):
    """Use production config hooks, without probe's instance-level overrides."""
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

    metadata = json.loads((path / "experiment_cfg/metadata.json").read_text())
    if len(metadata) != 1:
        raise ValueError("Require one explicit embodiment per checkpoint")
    tag = next(iter(metadata))
    grouped = "eef_pos_absolute" in metadata[tag]["statistics"]["state"]
    cfg = OmegaConf.load("examples/embodiment/config/model/gr00t.yaml")
    cfg.model_path = str(path.resolve())
    cfg.embodiment_tag = tag
    cfg.embodiment_projector_id = 31
    cfg.obs_converter_type = "libero_grouped" if grouped else "libero"
    class_name = "LiberoGroupedDataConfig" if grouped else "LiberoFrankaDataConfig"
    cfg.data_config_class = (
        f"rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:{class_name}"
    )
    cfg.opd_fixed_text_padding = not native
    cfg.disable_image_tiling = True
    cfg.action_head_precision = "fp32" if native else "bf16"
    cfg.num_action_chunks = args.execute
    cfg.denoising_steps = args.denoise_steps
    cfg.add_value_head = False
    cfg.rl_head_config.add_value_head = False
    cfg.rl_head_config.noise_method = "flow_ode"
    cfg.rl_head_config.joint_logprob = True
    cfg.rl_head_config.noise_level = 0.3
    model = get_model(cfg).eval().requires_grad_(False)
    assert model.opd_fixed_text_padding is (not native)
    processors = [
        t.eagle_processor.image_processor
        for t in model._modality_transform.transforms
        if hasattr(t, "eagle_processor")
    ]
    assert processors
    for processor in processors:
        assert processor.min_dynamic_tiles == processor.max_dynamic_tiles == 1
        assert not processor.use_thumbnail
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-gpu",
        action="store_true",
        help="Explicit opt-in; otherwise only validate arguments",
    )
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--memory-gib", type=float, default=24)
    parser.add_argument("--execute", type=int, default=8)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    if not args.run_gpu:
        print(
            "No GPU work performed. Add --run-gpu and select one free CUDA_VISIBLE_DEVICES to execute."
        )
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly expose exactly one free CUDA device")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.memory_gib * 2**30 / total)
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    observations = []
    for suite in SUITES:
        with np.load(
            args.observations / f"{suite}_obs.npz", allow_pickle=False
        ) as saved:
            observations.append(
                observation_batch(dict(saved), str(saved["task_description"].item()))
            )
    batch = {
        key: torch.cat([obs[key] for obs in observations])
        if torch.is_tensor(observations[0][key])
        else [item for obs in observations for item in obs[key]]
        for key in observations[0]
    }
    student = load_policy(args.student, args, native=True).cuda()
    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        restore_opd_text_padding,
    )

    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
    ):
        initial_rng = rng_state()
        actions, result = student.predict_action_batch(
            copy.deepcopy(batch), mode="train"
        )
        actions, result = cpu(actions), cpu(result)
        after_batch_rng = rng_state()
        restore_rng(initial_rng)
        single = [
            student.predict_action_batch(copy.deepcopy(obs), mode="train")
            for obs in observations
        ]
        single = [(cpu(action), cpu(payload)) for action, payload in single]
        assert torch.equal(torch.get_rng_state(), after_batch_rng[2])
        assert torch.equal(torch.cuda.get_rng_state(), after_batch_rng[3])
        assert random.getstate() == after_batch_rng[0]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == after_batch_rng[1][0]
        np.testing.assert_array_equal(numpy_state[1], after_batch_rng[1][1])
        assert numpy_state[2:] == after_batch_rng[1][2:]
        rows, lengths = [], []
        for row, (expected_action, expected) in enumerate(single):
            compare(actions[row : row + 1], expected_action, atol=args.atol)
            stored = {
                key: item[row : row + 1]
                for key, item in result["forward_inputs"].items()
            }
            trimmed = restore_opd_text_padding(stored)
            compare(trimmed, expected["forward_inputs"], atol=args.atol)
            for key in result.keys() - {"forward_inputs"}:
                value = result[key]
                compare(
                    value[row : row + 1] if torch.is_tensor(value) else value,
                    expected[key],
                    key,
                    args.atol,
                )
            lengths.append(trimmed["eagle_input_ids"].shape[-1])
            rows.append(stored)
            # Both NFT and FLOW_REFRESH must repair storage padding internally.
            nft = {
                "x_t": stored["chains"][:, 1].cuda(),
                "timesteps": torch.ones(1, dtype=torch.long, device="cuda"),
            }
            for kind in (ForwardType.NFT, ForwardType.FLOW_REFRESH):
                state = rng_state()
                a = student(
                    forward_type=kind, forward_inputs=device(stored), nft_inputs=nft
                )
                restore_rng(state)
                b = student(
                    forward_type=kind,
                    forward_inputs=device(expected["forward_inputs"]),
                    nft_inputs=nft,
                )
                compare(a, b, str(kind), args.atol)
                del a, b
        if len(set(lengths)) < 2:
            raise AssertionError("Fixture must exercise multiple native token lengths")
        student.cpu()
        torch.cuda.empty_cache()
        teacher_checks = []
        if args.teachers:
            for index, path in enumerate(args.teachers):
                teacher = load_policy(path, args, native=False).cuda()
                inputs = rows[index]
                reference = audit_teacher_inputs(inputs, observations[index], teacher)
                adapted = restore_opd_text_padding(
                    inputs, fixed_length=teacher.padding_value
                )
                state, mask = teacher.opd_normalize_state(inputs["opd_raw_states"])
                adapted.update(
                    state=state.to(inputs["state"].dtype),
                    state_mask=mask,
                    embodiment_id=teacher.opd_teacher_embodiment_id(1),
                )
                compare(adapted, reference, "teacher_inputs", args.atol)
                scale, shift = teacher.opd_action_endpoint_affine_to(student.metadata)
                from rlinf.algorithms.flow_opd import map_target_flow_state_to_source

                step = torch.ones(1, dtype=torch.long)
                source = map_target_flow_state_to_source(
                    inputs["chains"][:, 1], step, args.denoise_steps, scale, shift
                )
                nft = {"x_t": source.cuda(), "timesteps": step.cuda()}
                a = teacher(
                    forward_type=ForwardType.NFT,
                    forward_inputs=device(adapted),
                    nft_inputs=nft,
                )
                b = teacher(
                    forward_type=ForwardType.NFT,
                    forward_inputs=device(reference),
                    nft_inputs=nft,
                )
                compare(a, b, "teacher_velocity", args.atol)
                teacher_checks.append(
                    {"suite": SUITES[index], "checkpoint": str(path), "passed": True}
                )
                del a, b, nft, teacher
                gc.collect()
                torch.cuda.empty_cache()
    report = {
        "passed": True,
        "student": str(args.student),
        "seed": args.seed,
        "native_token_lengths": lengths,
        "batch_size": 4,
        "student_checks": [
            "rollout actions",
            "all cached fields",
            "per-item RNG stream",
            "NFT",
            "FLOW_REFRESH",
        ],
        "teacher_checks": teacher_checks,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "atol": args.atol,
        "scope": "single GPU; no simulator, training update, Ray or FSDP integration",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
