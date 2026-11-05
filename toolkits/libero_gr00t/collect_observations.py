# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect fixed initial observations for LIBERO action-agreement diagnostics.

No model, teacher, or policy action is used. Each official initial state receives
exactly the same ten zero-motion/open-gripper settling calls as pilot.evaluate.
The output manifest is accepted by checkpoint_action_diagnostic.py. These are
initial-state observations, not a student-rollout or held-out trajectory bank.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
OBSERVATION_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def file_hash(path: Path) -> str:
    """Hash an observation artifact without loading it as Python objects."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_one(suite, task_id: int, initial_state_id: int, seed: int, create_env):
    """Reset and settle one environment; always release its renderer and physics."""
    env, task, states = create_env(suite, task_id, seed)
    try:
        if not 0 <= initial_state_id < len(states):
            raise ValueError("Initial-state index outside the official state array")
        state = np.asarray(states[initial_state_id])
        if state.dtype.hasobject or not np.isfinite(state).all():
            raise ValueError("Invalid official initial state")
        state_identity = {
            "initial_state_array_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "initial_state_array_dtype": str(state.dtype),
            "initial_state_array_shape": list(state.shape),
        }
        env.reset()
        observation = env.set_init_state(states[initial_state_id])
        for _ in range(10):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        payload = {}
        for key in OBSERVATION_KEYS:
            value = np.asarray(observation[key])
            if value.dtype.hasobject or not np.isfinite(value).all():
                raise ValueError(f"Invalid observation array: {key}")
            payload[key] = value.copy()
        payload["task_description"] = np.asarray(task.language)
        return payload, state_identity
    finally:
        env.close()


def collect_bank(
    output: Path,
    suite_objects: dict,
    tasks: list[int],
    initial_state_id: int,
    seed: int,
    create_env,
    runtime_protocol: dict | None = None,
) -> list[dict]:
    """Write a complete fixture manifest only after every requested task succeeds."""
    if output.exists():
        raise FileExistsError(output)
    if not suite_objects or any(suite not in SUITES for suite in suite_objects):
        raise ValueError("Unknown or empty suite selection")
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(type(task) is not int or task not in range(10) for task in tasks)
    ):
        raise ValueError("Tasks must be unique integers in0..9")
    if type(initial_state_id) is not int or initial_state_id not in range(50):
        raise ValueError("Initial-state index must be in0..49")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Seed must be a nonnegative integer below2**31")
    output.mkdir(parents=True)
    protocol = {
        "complete": False,
        "scope": "Official initial observations after settling; no policy/model actions",
        "suites": list(suite_objects),
        "tasks": tasks,
        "initial_state_id": initial_state_id,
        "reset_seed": seed,
        "settling_steps_per_fixture": 10,
        "settling_action": [0.0] * 6 + [-1.0],
        "camera_height": 256,
        "camera_width": 256,
        "expected_fixtures": len(suite_objects) * len(tasks),
        **(runtime_protocol or {}),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for suite_name, suite in suite_objects.items():
        for task in tasks:
            payload, identity = collect_one(
                suite, task, initial_state_id, seed, create_env
            )
            name = f"{suite_name}_task{task}_init{initial_state_id}.npz"
            np.savez_compressed(
                output / name,
                **payload,
                suite=np.asarray(suite_name),
                task_id=np.asarray(task),
                initial_state_id=np.asarray(initial_state_id),
                reset_seed=np.asarray(seed),
                settling_steps=np.asarray(10),
            )
            rows.append(
                {
                    "suite": suite_name,
                    "task_id": task,
                    "initial_state_id": initial_state_id,
                    "reset_seed": seed,
                    "settling_steps": 10,
                    "path": name,
                    "sha256": file_hash(output / name),
                    **identity,
                }
            )
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(rows[-1]) + "\n")
    (output / "fixtures.json").write_text(json.dumps(rows, indent=2) + "\n")
    protocol.update(
        complete=True,
        completed_fixtures=len(rows),
        simulator_settling_calls=10 * len(rows),
        simulator_policy_calls=0,
        environment_resets=len(rows),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--initial-state", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--minimum-free-gib", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_gpu:
        print("No simulator work performed; select one CUDA device and pass --run-gpu")
        return
    if len(set(args.suites)) != len(args.suites):
        raise ValueError("Duplicate suites")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.minimum_free_gib < 1:
        raise ValueError("Require at least1 GiB free for bounded EGL collection")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import torch

    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("Explicitly select exactly one CUDA_VISIBLE_DEVICES")
    free, _ = torch.cuda.mem_get_info()
    if free < args.minimum_free_gib * 2**30:
        raise RuntimeError(f"Insufficient free GPU memory: {free / 2**30:.2f} GiB")
    torch.set_num_threads(1)
    from toolkits.libero_gr00t.probe import create_env, renderer_index

    egl = renderer_index()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    from libero.libero import benchmark

    suites = {name: benchmark.get_benchmark_dict()[name]() for name in args.suites}
    rows = collect_bank(
        args.output,
        suites,
        args.tasks,
        args.initial_state,
        args.seed,
        create_env,
        runtime_protocol={
            "physical_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "egl_index": egl,
        },
    )
    print(
        json.dumps(
            {"complete": True, "fixtures": len(rows), "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
