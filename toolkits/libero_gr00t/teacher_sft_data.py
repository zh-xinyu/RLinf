# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Behavior-cloning examples from immutable, successful teacher rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from toolkits.libero_gr00t.offline_observations import (
    SUITES,
    file_hash,
    sample_frame_indices,
)


def executed_action_targets(
    actions: np.ndarray, counts: np.ndarray, horizon: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Join executed prefixes within ONE episode; mask everything after termination.

    A recorded decision contains only K controller commands. Future labels come
    from actual subsequent replans, never unexecuted suffixes of a prediction.
    End padding repeats the last executed action and contributes no loss.
    """
    if actions.ndim != 3 or actions.shape[-1] != 7 or horizon < 1:
        raise ValueError("Expected nonempty decision x K x 7 controller actions")
    if not len(actions) or counts.shape != (len(actions),):
        raise ValueError("Execution counts must match the decisions")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Execution counts must be integers")
    if np.any(counts < 1) or np.any(counts > actions.shape[1]):
        raise ValueError("Invalid executed prefix length")
    if np.any(counts[:-1] != actions.shape[1]):
        raise ValueError("Only the terminal decision may have a partial prefix")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite teacher action")
    stream = np.concatenate([a[:n] for a, n in zip(actions, counts, strict=True)])
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    targets = np.empty((len(actions), horizon, 7), dtype=np.float32)
    valid = np.zeros((len(actions), horizon), dtype=bool)
    for i, start in enumerate(offsets):
        future = stream[start : start + horizon]
        targets[i] = future[-1]
        targets[i, : len(future)] = future
        valid[i, : len(future)] = True
    return targets, valid


def prepare(source: Path, output: Path) -> dict:
    """Verify teacher data hashes and derive only labels, without copying RGB data."""
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Teacher collection is incomplete")
    output.mkdir(parents=True)
    receipt = {
        "complete": False,
        "source": str(source.resolve()),
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "label_rule": "Next16 executed controller actions within each successful episode",
        "terminal_rule": "Repeat last real action; mask padded timesteps out of the loss",
        "observation_rule": "Actual teacher decision frames only; no invented intermediate images",
        "new_collection_trajectories": 0,
        "new_simulation_steps": 0,
        "inherited_collection_cost": {
            k: manifest[k]
            for k in (
                "accepted_trajectories",
                "attempted_trajectories",
                "simulation_steps",
                "settling_steps",
            )
        },
        "suites": {},
    }
    for suite in SUITES:
        original = source / suite
        summary = manifest["suites"][suite]
        for filename, expected in summary["files_sha256"].items():
            if file_hash(original / filename) != expected:
                raise ValueError(f"Teacher data changed: {suite}/{filename}")
        arrays = {
            key: np.load(original / f"{key}.npy", mmap_mode="r")
            for key in ("actions", "execution_counts", "episode_ids", "task_ids")
        }
        n = len(arrays["episode_ids"])
        targets = np.empty((n, 16, 7), dtype=np.float32)
        valid = np.zeros((n, 16), dtype=bool)
        episode_ids = arrays["episode_ids"]
        for episode in np.unique(episode_ids):
            indices = np.flatnonzero(episode_ids == episode)
            if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                raise ValueError("Episode decision rows must be contiguous")
            if len(np.unique(arrays["task_ids"][indices])) != 1:
                raise ValueError("Episode crosses a task boundary")
            targets[indices], valid[indices] = executed_action_targets(
                arrays["actions"][indices], arrays["execution_counts"][indices]
            )
        if not np.isin(targets[..., -1], [-1.0, 0.0, 1.0]).all():
            raise ValueError("Expected binarized simulator gripper commands")
        destination = output / suite
        destination.mkdir()
        np.save(destination / "actions.npy", targets)
        np.save(destination / "valid.npy", valid)
        receipt["suites"][suite] = {
            "decision_observations": n,
            "episodes": len(np.unique(episode_ids)),
            "valid_future_action_labels": int(valid.sum()),
            "physical_steps_in_accepted_episodes": int(
                arrays["execution_counts"].sum()
            ),
            "files_sha256": {
                p.name: file_hash(p) for p in sorted(destination.iterdir())
            },
        }
    receipt["complete"] = True
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


class TeacherActionPool:
    """Sample suites externally, then local tasks and decision frames uniformly."""

    def __init__(self, folder: Path):
        self.manifest = json.loads((folder / "manifest.json").read_text())
        if not self.manifest["complete"]:
            raise ValueError("Derived action pool is incomplete")
        source = Path(self.manifest["source"])
        if (
            file_hash(source / "manifest.json")
            != self.manifest["source_manifest_sha256"]
        ):
            raise ValueError("Source observation manifest changed")
        self.arrays, self.tasks = {}, {}
        for suite in SUITES:
            for name, checksum in self.manifest["suites"][suite][
                "files_sha256"
            ].items():
                if file_hash(folder / suite / name) != checksum:
                    raise ValueError("Derived action labels changed")
            self.arrays[suite] = {
                name: np.load(source / suite / f"{name}.npy", mmap_mode="r")
                for name in ("images", "states", "task_ids")
            }
            self.arrays[suite].update(
                {
                    name: np.load(folder / suite / f"{name}.npy", mmap_mode="r")
                    for name in ("actions", "valid")
                }
            )
            self.tasks[suite] = json.loads((source / suite / "tasks.json").read_text())

    def sample(self, suite: str, size: int, seed: int) -> list[tuple]:
        """Return raw observations, future physical commands and temporal masks."""
        import torch

        arrays = self.arrays[suite]
        indices = sample_frame_indices(arrays["task_ids"], size, seed)
        rows = []
        for i in indices:
            raw = {
                "main_images": torch.from_numpy(np.array(arrays["images"][i, 0]))[None],
                "wrist_images": torch.from_numpy(np.array(arrays["images"][i, 1]))[
                    None
                ],
                "states": torch.from_numpy(np.array(arrays["states"][i]))[None],
                "task_descriptions": [
                    self.tasks[suite][str(int(arrays["task_ids"][i]))]
                ],
            }
            rows.append(
                (raw, np.array(arrays["actions"][i]), np.array(arrays["valid"][i]))
            )
        return rows


def apply_label_transforms(model, converted: dict) -> dict:
    """Enable action packing alone while leaving image transforms deterministic."""
    packers = [
        t for t in model._modality_transform.transforms if hasattr(t, "eagle_processor")
    ]
    if len(packers) != 1 or packers[0].language_dropout_prob != 0:
        raise ValueError("Require one native GR00T packer without language dropout")
    packer = packers[0]
    previous = packer.training
    try:
        # In eval mode the native packer intentionally omits action/action_mask.
        # Do not train() the whole pipeline: that would enable image augmentation.
        packer.training = True
        return model.apply_transforms(converted)
    finally:
        packer.training = previous


def preprocess_sft(model, raw: dict, actions: np.ndarray, valid: np.ndarray) -> dict:
    """Apply the checkpoint's full native transforms to observations AND labels."""
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        restore_opd_text_padding,
    )

    if actions.shape != (16, 7) or valid.shape != (16,) or not valid.any():
        raise ValueError("Expected a nonempty16-step physical action target")
    raw = dict(raw)
    raw["states"] = raw["states"].to(torch.bfloat16).float()
    converted = model.obs_convert_fn(raw)
    converted.update(
        {
            "action.eef_pos_delta": actions[None, :, :3],
            "action.eef_rot_delta": actions[None, :, 3:6],
            # Invert the runtime converter: simulator command =1-2*RLDS label.
            "action.gripper_close": (1.0 - actions[None, :, 6:7]) / 2.0,
        }
    )
    converted = {
        k: v if isinstance(v, np.ndarray) else np.asarray(v)
        for k, v in converted.items()
    }
    normalized = apply_label_transforms(model, converted)
    normalized["action_mask"] = (
        normalized["action_mask"].bool() & torch.as_tensor(valid)[None, :, None]
    )
    for key, value in normalized.items():
        if value.dtype == torch.float32:
            normalized[key] = value.to(torch.bfloat16)
    normalized = restore_opd_text_padding(normalized)
    for key in ("eagle_pixel_values", "eagle_image_sizes"):
        normalized[key] = normalized[key].reshape(
            1, model.image_nums, *normalized[key].shape[1:]
        )
    return normalized


def main() -> None:
    """Derive a hashed, offline action-label pool from recorded teacher episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
