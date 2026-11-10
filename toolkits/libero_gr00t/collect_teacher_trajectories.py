# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect a fixed bank of successful teacher decision observations for offline KD."""

import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from toolkits.libero_gr00t.offline_observations import file_hash
from toolkits.libero_gr00t.probe import (
    create_env,
    load_model,
    observation_batch,
    renderer_index,
)

PROTOCOLS = {
    "libero_spatial": (5, 240),
    "libero_object": (5, 240),
    "libero_goal": (5, 320),
    "libero_10": (10, 480),
}


def task_collection_complete(successes: int, target: int, minimum: int) -> bool:
    """Accept a bounded task with fewer than its target only above explicit minimum."""
    if not 1 <= minimum <= target:
        raise ValueError("Require1 <= minimum <= target successes")
    return successes >= minimum


def save_episode(folder: Path, row: dict, images, states, actions, counts) -> Path:
    """Persist one successful episode atomically before appending its outcome log."""
    folder.mkdir(exist_ok=True)
    path = folder / f"task{row['task_id']:02d}_state{row['initial_state_id']:02d}.npz"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            images=np.stack(images),
            states=np.stack(states),
            actions=np.stack(actions),
            execution_counts=np.asarray(counts, dtype=np.int32),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)
    return path


def read_attempts(folder: Path) -> list[dict]:
    """Require saved data for every accepted outcome; refuse unsafe partial resume."""
    path = folder / "attempts.jsonl"
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    seen = set()
    for row in rows:
        key = (row["task_id"], row["initial_state_id"])
        if key in seen:
            raise ValueError("Duplicate persisted teacher attempt")
        seen.add(key)
        if row["accepted"] and not (folder / row["episode_file"]).exists():
            raise ValueError("Accepted attempt has no persisted observation file")
    return rows


def collect(args) -> dict:
    """Retain only successful episodes; account for every failed attempt as interaction."""
    resuming = args.output.exists()
    if resuming and not args.resume:
        raise FileExistsError(args.output)
    if torch.cuda.device_count() != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Expose exactly one selected GPU")
    if (
        args.successes_per_task < 1
        or not args.successes_per_task <= args.maximum_attempts <= 40
        or not 1 <= args.minimum_successes_per_task <= args.successes_per_task
    ):
        raise ValueError("Require1..40 ordered initial-state attempts per task")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if resuming:
        previous = json.loads((args.output / "protocol.json").read_text())
        for key, value in {
            "suite": args.suite,
            "teacher": str(args.teacher),
            "successes_per_task": args.successes_per_task,
            "maximum_attempts": args.maximum_attempts,
            "minimum_successes_per_task": args.minimum_successes_per_task,
            "seed": args.seed,
        }.items():
            if previous.get(key) != value:
                raise ValueError(f"Resume collection protocol differs: {key}")
    attempts = read_attempts(args.output)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        16 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    execute, horizon = PROTOCOLS[args.suite]
    settings = SimpleNamespace(
        execute=execute,
        denoise_steps=4,
        noise_level=0.5,
        student=Path("/nonexistent/offline_student_placeholder"),
        student_head_precision="fp32",
        student_text_padding="native",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args.teacher, settings)
    # Reproduce the BF16 rotary behavior tested in the official-protocol calibration.
    model.to(torch.bfloat16).cuda().eval().requires_grad_(False)
    model.action_head.rl_config.noise_method = "flow_sde"
    model.action_head.rl_config.noise_level = 0.5
    model.action_head.rl_config.joint_logprob = False
    egl = renderer_index()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    protocol = {
        "suite": args.suite,
        "teacher": str(args.teacher),
        "execute": execute,
        "horizon": horizon,
        "successes_per_task": args.successes_per_task,
        "maximum_attempts": args.maximum_attempts,
        "minimum_successes_per_task": args.minimum_successes_per_task,
        "seed": args.seed,
        "policy_seed": "seed+task_id*100+initial_state_id",
        "mode": "eval",
        "text_padding": 570,
        "precision": "bf16_including_rotary",
        "full_flow_shape": [16, 32],
        "denoise_steps": 4,
        "observation_stride": "Every actual teacher decision, no intermediate fake decisions",
        "image_orientation": "Same as rollout observation_batch (both image axes reversed)",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "egl_index": egl,
        "started_at": time.time(),
    }
    if not resuming:
        (args.output / "protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n"
        )
    accepted = sum(row["accepted"] for row in attempts)
    for task_id in range(10):
        previous_attempts = {
            row["initial_state_id"]: row
            for row in attempts
            if row["task_id"] == task_id
        }
        successes = sum(row["accepted"] for row in previous_attempts.values())
        for trial in range(args.maximum_attempts):
            if successes >= args.successes_per_task:
                break
            if trial in previous_attempts:
                continue
            seed = args.seed + task_id * 100 + trial
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env, task, states = create_env(suite, task_id, args.seed)
            images, raw_states, action_chunks, execution_counts = [], [], [], []
            steps = calls = 0
            success = False
            started = time.perf_counter()
            try:
                env.reset()
                obs = env.set_init_state(states[trial])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                while steps < horizon and not success:
                    raw = observation_batch(obs, task.language)
                    images.append(
                        np.stack(
                            [
                                raw["main_images"][0].numpy(),
                                raw["wrist_images"][0].numpy(),
                            ]
                        )
                    )
                    raw_states.append(raw["states"][0].numpy().copy())
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        actions, payload = model.predict_action_batch(
                            dict(raw), mode="eval"
                        )
                    if tuple(actions.shape) != (1, execute, 7) or tuple(
                        payload["forward_inputs"]["chains"].shape
                    ) != (1, 5, 16, 32):
                        raise ValueError("Unexpected teacher action/flow shape")
                    actions = actions.float().cpu().numpy()[0]
                    if not np.isfinite(actions).all():
                        raise FloatingPointError("Nonfinite teacher actions")
                    action_chunks.append(actions)
                    del payload, raw
                    calls += 1
                    executed = 0
                    for action in actions:
                        obs, _, done, _ = env.step(action.tolist())
                        steps += 1
                        executed += 1
                        success = bool(done)
                        if success or steps == horizon:
                            break
                    execution_counts.append(executed)
            finally:
                env.close()
            row = {
                "task_id": task_id,
                "initial_state_id": trial,
                "success": success,
                "policy_steps": steps,
                "settling_steps": 10,
                "policy_calls": calls,
                "wall_s": time.perf_counter() - started,
                "accepted": success,
                "task_description": task.language,
            }
            if success:
                if not calls or sum(execution_counts) != steps:
                    raise ValueError("Decision/action alignment mismatch")
                path = save_episode(
                    args.output / "episodes",
                    row,
                    images,
                    raw_states,
                    action_chunks,
                    execution_counts,
                )
                row["episode_file"] = str(path.relative_to(args.output))
                row["episode_sha256"] = file_hash(path)
                successes += 1
                accepted += 1
            attempts.append(row)
            with (args.output / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            progress = {
                "suite": args.suite,
                "accepted_trajectories": accepted,
                "attempted_trajectories": len(attempts),
                "task_id": task_id,
                "task_successes": successes,
                "simulation_steps": sum(r["policy_steps"] for r in attempts),
                "settling_steps": 10 * len(attempts),
            }
            (args.output / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n"
            )
            print(json.dumps(progress), flush=True)
            if successes == args.successes_per_task:
                break
        if not task_collection_complete(
            successes, args.successes_per_task, args.minimum_successes_per_task
        ):
            raise RuntimeError(
                f"Task{task_id} lacks minimum{args.minimum_successes_per_task} successful episodes after bounded attempts"
            )
    attempts = read_attempts(args.output)
    all_images, all_states, all_actions, all_counts = [], [], [], []
    task_ids, episode_ids, tasks = [], [], {}
    for row in attempts:
        tasks[row["task_id"]] = row["task_description"]
        if not row["accepted"]:
            continue
        path = args.output / row["episode_file"]
        if file_hash(path) != row["episode_sha256"]:
            raise ValueError("Persisted episode changed")
        with np.load(path, allow_pickle=False) as episode:
            all_images.append(episode["images"])
            all_states.append(episode["states"])
            all_actions.append(episode["actions"])
            all_counts.append(episode["execution_counts"])
            n = len(episode["states"])
        task_ids.extend([row["task_id"]] * n)
        episode_ids.extend([row["task_id"] * 100 + row["initial_state_id"]] * n)
    for name, value in {
        "images": np.concatenate(all_images),
        "states": np.concatenate(all_states),
        "actions": np.concatenate(all_actions),
        "execution_counts": np.concatenate(all_counts),
        "task_ids": np.asarray(task_ids, dtype=np.int32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int32),
    }.items():
        np.save(args.output / f"{name}.npy", value)
    (args.output / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    summary = {
        "complete": True,
        "suite": args.suite,
        "episodes": accepted,
        "frames": len(task_ids),
        "tasks": 10,
        "accepted_per_task": {
            str(t): sum(row["accepted"] for row in attempts if row["task_id"] == t)
            for t in range(10)
        },
        "attempted_trajectories": len(attempts),
        "simulation_steps": sum(r["policy_steps"] for r in attempts),
        "settling_steps": 10 * len(attempts),
        "policy_calls": sum(r["policy_calls"] for r in attempts),
        "files_sha256": {p.name: file_hash(p) for p in args.output.glob("*.npy")},
        "teacher_metadata_sha256": file_hash(
            args.teacher / "experiment_cfg/metadata.json"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--suite", choices=list(PROTOCOLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successes-per-task", type=int, default=5)
    parser.add_argument("--maximum-attempts", type=int, default=30)
    parser.add_argument("--minimum-successes-per-task", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
