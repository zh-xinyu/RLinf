# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed observation pools from the pinned few-shot demonstrations, without physics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def file_hash(path: Path) -> str:
    """Hash one dataset asset without materializing the file in memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def sample_frame_indices(task_ids: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Sample tasks uniformly, then frames uniformly within each selected task."""
    if task_ids.ndim != 1 or size < 1:
        raise ValueError("Expected one-dimensional task IDs and positive size")
    tasks = np.unique(task_ids)
    if not np.array_equal(tasks, np.arange(10)):
        raise ValueError("Each suite must cover exactly ten local tasks")
    generator = np.random.default_rng(seed)
    selected_tasks = generator.integers(0, 10, size=size)
    indices = np.empty(size, dtype=np.int64)
    for task in tasks:
        selected = np.flatnonzero(selected_tasks == task)
        pool = np.flatnonzero(task_ids == task)
        indices[selected] = generator.choice(pool, size=len(selected), replace=True)
    return indices


def prepare(source: Path, output: Path) -> dict:
    """Pack deterministic, unaugmented RGB frames and raw8D states into mmap arrays."""
    import av
    import pandas as pd

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    receipt = {
        "complete": False,
        "source": str(source.resolve()),
        "suites": {},
        "physical_rollouts": 0,
        "simulation_steps": 0,
        "orientation": "RGB exactly as stored in the existing SFT dataset; no additional image flip",
        "targets": "No demonstration action labels are loaded; teacher velocity supervision only",
    }
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for suite in SUITES:
        folder = source / suite
        provenance = json.loads((folder / "SOURCE.json").read_text())
        for name, expected in provenance["asset_sha256"].items():
            if file_hash(folder / name) != expected:
                raise ValueError(f"Dataset asset mismatch: {suite}/{name}")
        tasks = {
            int(row["task_index"]): row["task"]
            for row in map(
                json.loads, (folder / "meta/tasks.jsonl").read_text().splitlines()
            )
        }
        episodes = provenance["selected_episode_ids"]
        total = provenance["frames"]
        destination = output / suite
        destination.mkdir()
        images = np.lib.format.open_memmap(
            destination / "images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total, 2, 256, 256, 3),
        )
        states = np.lib.format.open_memmap(
            destination / "states.npy", mode="w+", dtype=np.float32, shape=(total, 8)
        )
        task_ids = np.lib.format.open_memmap(
            destination / "task_ids.npy", mode="w+", dtype=np.int32, shape=(total,)
        )
        episode_ids = np.lib.format.open_memmap(
            destination / "episode_ids.npy", mode="w+", dtype=np.int32, shape=(total,)
        )
        offset = 0
        for episode in episodes:
            table = pd.read_parquet(
                folder / f"data/chunk-000/episode_{episode:06d}.parquet",
                columns=["timestamp", "observation.state", "task_index"],
            )
            n = len(table)
            state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
            if state.shape != (n, 8) or not np.isfinite(state).all():
                raise ValueError("Expected finite raw8D LIBERO proprioception")
            timestamps = table["timestamp"].to_numpy()
            for view, key in enumerate(("image", "wrist_image")):
                video = (
                    folder
                    / f"videos/chunk-000/observation.images.{key}/episode_{episode:06d}.mp4"
                )
                frames, times = [], []
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    for frame in container.decode(stream):
                        if frame.pts is None:
                            raise ValueError("Video frame lacks presentation timestamp")
                        frames.append(frame.to_ndarray(format="rgb24"))
                        times.append(float(frame.pts * frame.time_base))
                times = np.asarray(times)
                nearest = np.abs(times[:, None] - timestamps[None, :]).argmin(axis=0)
                if np.max(np.abs(times[nearest] - timestamps)) > 0.026:
                    raise ValueError(
                        "Video/state timestamp alignment exceeds half a frame"
                    )
                rgb = np.stack(frames)[nearest]
                if rgb.shape != (n, 256, 256, 3):
                    raise ValueError("Unexpected demonstration image shape")
                images[offset : offset + n, view] = rgb
            states[offset : offset + n] = state
            task_ids[offset : offset + n] = table["task_index"].to_numpy()
            episode_ids[offset : offset + n] = episode
            offset += n
        if offset != total or set(np.unique(task_ids)) != set(range(10)):
            raise ValueError("Incomplete demonstration coverage")
        for array in (images, states, task_ids, episode_ids):
            array.flush()
        (destination / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
        receipt["suites"][suite] = {
            "episodes": len(episodes),
            "frames": total,
            "tasks": 10,
            "source_receipt_sha256": file_hash(folder / "SOURCE.json"),
            "files_sha256": {
                p.name: file_hash(p) for p in sorted(destination.iterdir())
            },
        }
        (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps({"suite": suite, "frames": total, "complete": True}), flush=True
        )
        del images, states, task_ids, episode_ids
    receipt["complete"] = True
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


class ObservationPool:
    """Read fixed frames without loading all image arrays into each learner process."""

    def __init__(self, folder: Path):
        manifest = json.loads((folder / "manifest.json").read_text())
        if not manifest["complete"]:
            raise ValueError("Observation pool is incomplete")
        self.arrays = {}
        self.tasks = {}
        for suite in SUITES:
            self.arrays[suite] = {
                name: np.load(folder / suite / f"{name}.npy", mmap_mode="r")
                for name in ("images", "states", "task_ids")
            }
            self.tasks[suite] = json.loads((folder / suite / "tasks.json").read_text())

    def sample(self, suite: str, size: int, seed: int) -> list[dict]:
        """Return policy-ready raw observations; no normalization is shared across models."""
        import torch

        arrays = self.arrays[suite]
        indices = sample_frame_indices(arrays["task_ids"], size, seed)
        return [
            {
                "main_images": torch.from_numpy(np.array(arrays["images"][i, 0]))[None],
                "wrist_images": torch.from_numpy(np.array(arrays["images"][i, 1]))[
                    None
                ],
                "states": torch.from_numpy(np.array(arrays["states"][i]))[None],
                "task_descriptions": [
                    self.tasks[suite][str(int(arrays["task_ids"][i]))]
                ],
            }
            for i in indices
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
