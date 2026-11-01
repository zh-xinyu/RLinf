# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Read pinned official Aloha clean demonstrations for joint flow matching.

Alignment follows XPolicyLab 0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3,
policy/Pi_05/openpi/scripts/process_data.py: action[t] = state[t + 1], with
the last state repeated. The revised official files already store that action.
Only marked standard RGB JPEGs are accepted; legacy unmarked bytes fail closed.
"""

import hashlib
import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

TASKS = ("adjust_bottle", "handover_mic", "place_container_plate", "stack_bowls_two")
PARTS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def decode_rgb_jpeg(value: bytes) -> np.ndarray:
    """Decode only the official revised XPL-RGB1 standard JPEG representation.

    An unmarked legacy RoboTwin JPEG has reversed channels relative to the JPEG
    standard. Reject it rather than silently training on the wrong colors.
    """
    with Image.open(io.BytesIO(bytes(value).rstrip(b"\0"))) as image:
        if image.format != "JPEG" or image.info.get("comment") != b"XPL-RGB1":
            raise ValueError("Expected a marked XPL-RGB1 standard RGB JPEG")
        return np.asarray(image.convert("RGB")).copy()


def concatenate_joints(group: h5py.Group) -> np.ndarray:
    """Concatenate left 6 joints/gripper and right 6 joints/gripper."""
    parts = [np.asarray(group[key], dtype=np.float32) for key in PARTS]
    values = np.concatenate([part.reshape(len(part), -1) for part in parts], axis=1)
    if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
        raise ValueError("Expected finite [T, 14] Aloha joint targets")
    if not ((values[:, [6, 13]] >= -1e-5) & (values[:, [6, 13]] <= 1 + 1e-5)).all():
        raise ValueError("Aloha grippers must be normalized to [0, 1]")
    return values


def validate_action_alignment(states: np.ndarray, actions: np.ndarray) -> None:
    """Require the official one-frame target shift, including terminal padding."""
    expected = np.concatenate([states[1:], states[-1:]], axis=0)
    if states.shape != actions.shape or not np.allclose(
        actions, expected, atol=1e-6, rtol=0
    ):
        raise ValueError(
            "Demo action is not next-state joint target (terminal repeated)"
        )


def action_chunk(actions: np.ndarray, frame: int, horizon: int = 50) -> np.ndarray:
    """Take consecutive target commands and repeat the final action at the end."""
    if not 0 <= frame < len(actions):
        raise IndexError(frame)
    return actions[
        np.minimum(np.arange(frame, frame + horizon), len(actions) - 1)
    ].copy()


@dataclass
class Episode:
    """Small in-memory state/action metadata; images stay in their HDF5 files."""

    path: Path
    states: np.ndarray
    actions: np.ndarray
    instructions: tuple[str, ...]


class JointDemoDataset(Dataset):
    """Deterministic task-balanced, episode-balanced random frame sampling."""

    def __init__(
        self,
        root: str | Path,
        *,
        seed: int = 0,
        samples: int = 64000,
        split: str = "train",
        holdout_episodes: int = 5,
    ):
        self.root = Path(root)
        self.seed = seed
        self.samples = samples
        if (
            split not in {"train", "validation", "all"}
            or not 0 <= holdout_episodes < 50
        ):
            raise ValueError("Invalid split or holdout_episodes")
        if split == "validation" and holdout_episodes == 0:
            raise ValueError("Validation requires held-out episodes")
        self.episodes = []
        terminal_adjustments = {}
        self._handles = OrderedDict()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["revision"] != "3dc3b798668feb99ac61cc9086d84cbcc3d79186":
            raise ValueError("Dataset revision was not audited for this reader")
        for task in TASKS:
            directory = self.root / task / "aloha_agilex"
            paths = sorted(
                (directory / "data").glob("*.hdf5"),
                key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            )
            episode_ids = [int(path.stem.rsplit("_", 1)[-1]) for path in paths]
            if episode_ids != list(range(50)):
                raise ValueError(f"Expected 50 episodes for {task}, found {len(paths)}")
            if holdout_episodes and split != "all":
                paths = (
                    paths[-holdout_episodes:]
                    if split == "validation"
                    else paths[:-holdout_episodes]
                )
            task_episodes = []
            terminal_differences = []
            for path in paths:
                with h5py.File(path, "r") as data:
                    states = concatenate_joints(data["state"])
                    stored_actions = concatenate_joints(data["action"])
                    if states.shape != stored_actions.shape or not np.allclose(
                        stored_actions[:-1], states[1:], atol=1e-6, rtol=0
                    ):
                        raise ValueError(
                            f"Nonterminal action/state alignment mismatch: {path}"
                        )
                    terminal_differences.append(
                        float(np.abs(stored_actions[-1] - states[-1]).max())
                    )
                    # The official Pi_05 converter derives targets from states,
                    # explicitly repeating the terminal state. Some raw files
                    # retain a tiny final target movement; record but do not use it.
                    actions = np.concatenate([states[1:], states[-1:]], axis=0)
                    validate_action_alignment(states, actions)
                    for camera in CAMERAS:
                        images = data[f"vision/{camera}/colors"]
                        if len(images) != len(states):
                            raise ValueError(f"Image/state mismatch: {path}, {camera}")
                        decode_rgb_jpeg(images[0])
                instruction_path = directory / "instruction" / f"{path.stem}.json"
                instructions = tuple(json.loads(instruction_path.read_text())["seen"])
                if not instructions or not all(
                    isinstance(x, str) and x.strip() for x in instructions
                ):
                    raise ValueError(f"Missing seen instructions: {instruction_path}")
                task_episodes.append(Episode(path, states, actions, instructions))
            self.episodes.append(task_episodes)
            terminal_adjustments[task] = {
                "episodes_with_nonzero_terminal_difference": sum(
                    value > 1e-6 for value in terminal_differences
                ),
                "maximum_absolute_terminal_difference": max(terminal_differences),
            }
        self.protocol = {
            "dataset_revision": self.manifest["revision"],
            "split": split,
            "holdout_episodes": holdout_episodes,
            "episodes": {
                task: len(eps) for task, eps in zip(TASKS, self.episodes, strict=True)
            },
            "episode_ids": {
                task: [int(ep.path.stem.rsplit("_", 1)[-1]) for ep in eps]
                for task, eps in zip(TASKS, self.episodes, strict=True)
            },
            "frames": {
                task: sum(len(ep.states) for ep in eps)
                for task, eps in zip(TASKS, self.episodes, strict=True)
            },
            "sampling": "uniform task -> uniform episode -> uniform frame; seen instruction sampled per item",
            "alignment": "action[t]=state[t+1], terminal action repeated; full 50-step chunk",
            "stored_action_terminal_difference": terminal_adjustments,
            "manifest_sha256": hashlib.sha256(
                (self.root / "manifest.json").read_bytes()
            ).hexdigest(),
        }

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        task_id = index % len(TASKS)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        episodes = self.episodes[task_id]
        episode = episodes[int(rng.integers(len(episodes)))]
        frame = int(rng.integers(len(episode.states)))
        if episode.path not in self._handles:
            if len(self._handles) >= 8:
                _, handle = self._handles.popitem(last=False)
                handle.close()
            self._handles[episode.path] = h5py.File(episode.path, "r")
        self._handles.move_to_end(episode.path)
        data = self._handles[episode.path]
        images = [
            decode_rgb_jpeg(data[f"vision/{camera}/colors"][frame])
            for camera in CAMERAS
        ]
        return {
            "observation/image": images[0],
            "observation/wrist_image": np.stack(images[1:]),
            "observation/state": episode.states[frame].copy(),
            "actions": action_chunk(episode.actions, frame),
            "prompt": episode.instructions[
                int(rng.integers(len(episode.instructions)))
            ],
            "opd_task_ids": task_id,
        }


class RankBatchIndices(Sampler):
    """Partition each global batch without duplicate rank samples."""

    def __init__(self, global_batch: int, world_size: int, rank: int, steps: int):
        if global_batch % world_size or global_batch % 4:
            raise ValueError(
                "Global batch must be divisible by world size and four tasks"
            )
        self.global_batch = global_batch
        self.local_batch = global_batch // world_size
        self.rank = rank
        self.steps = steps

    def __iter__(self):
        for step in range(self.steps):
            start = step * self.global_batch + self.rank * self.local_batch
            yield from range(start, start + self.local_batch)

    def __len__(self) -> int:
        return self.steps * self.local_batch


def collate_demo_rows(rows: list[dict]) -> dict:
    """Stack arrays while preserving the string prompt list."""
    return {
        key: [row[key] for row in rows]
        if key == "prompt"
        else torch.from_numpy(np.stack([row[key] for row in rows]))
        for key in rows[0]
    }
