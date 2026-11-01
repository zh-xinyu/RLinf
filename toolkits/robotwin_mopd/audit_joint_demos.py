# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only compatibility audit of official demos and fixed PACE teacher norms."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from toolkits.robotwin_mopd.joint_sft_data import TASKS, JointDemoDataset, action_chunk


def summarize(values: np.ndarray) -> dict:
    """Report scales and out-of-range fractions without clipping labels."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "finite": bool(np.isfinite(values).all()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "rms": float(np.sqrt(np.square(values).mean())),
        "abs_gt_2_fraction": float((np.abs(values) > 2).mean()),
        "abs_gt_10_fraction": float((np.abs(values) > 10).mean()),
        "absolute_p99": float(np.quantile(np.abs(values), 0.99)),
    }


def audit(data_root: Path, norm_root: Path, samples_per_episode: int) -> dict:
    """Use the production OpenPI DeltaActions/Normalize implementations on CPU."""
    from openpi.shared import normalize
    from openpi.transforms import DeltaActions, Normalize

    dataset = JointDemoDataset(data_root, split="all")
    report = {
        "protocol": dataset.protocol,
        "samples_per_episode": samples_per_episode,
        "tasks": {},
    }
    mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False])
    for task_id, task in enumerate(TASKS):
        norm_path = norm_root / task / "assets/robotwin/norm_stats.json"
        stats = normalize.load(norm_path.parent)
        delta = DeltaActions(mask)
        normalizer = Normalize(stats, use_quantiles=True)
        (
            raw_states,
            raw_actions,
            delta_actions,
            normalized_states,
            normalized_actions,
        ) = [], [], [], [], []
        for episode in dataset.episodes[task_id]:
            for frame in np.linspace(
                0, len(episode.states) - 1, samples_per_episode, dtype=int
            ):
                state = episode.states[frame].copy()
                actions = action_chunk(episode.actions, int(frame))
                inputs = delta({"state": state.copy(), "actions": actions.copy()})
                np.testing.assert_array_equal(
                    inputs["actions"][:, [6, 13]], actions[:, [6, 13]]
                )
                np.testing.assert_allclose(
                    inputs["actions"][:, mask],
                    actions[:, mask] - state[None, mask],
                    atol=1e-6,
                )
                output = normalizer(inputs)
                raw_states.append(state)
                raw_actions.append(actions)
                delta_actions.append(inputs["actions"].copy())
                normalized_states.append(output["state"])
                normalized_actions.append(output["actions"])
        normalized_actions = np.stack(normalized_actions)
        padded = np.pad(normalized_actions, ((0, 0), (0, 0), (0, 18)))
        report["tasks"][task] = {
            "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
            "sampled_chunks": len(normalized_actions),
            "raw_state": summarize(np.stack(raw_states)),
            "raw_action": summarize(np.stack(raw_actions)),
            "delta_action": summarize(np.stack(delta_actions)),
            "normalized_state": summarize(np.stack(normalized_states)),
            "normalized_action_physical14": summarize(padded[..., :14]),
            "normalized_action_padded18": summarize(padded[..., 14:]),
            "normalized_action_per_dimension": [
                summarize(padded[..., dim]) for dim in range(14)
            ],
            "delta_joint_absolute_gripper_verified": True,
        }
    report["all_finite"] = all(
        row["normalized_action_physical14"]["finite"]
        and row["normalized_state"]["finite"]
        for row in report["tasks"].values()
    )
    report["norms_recomputed"] = False
    report["action_values_clipped"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--norm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-episode", type=int, default=16)
    args = parser.parse_args()
    report = audit(args.data, args.norm_root, args.samples_per_episode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                task: row["normalized_action_physical14"]
                for task, row in report["tasks"].items()
            },
            indent=2,
        )
    )
    if not report["all_finite"]:
        raise ValueError("Nonfinite labels; do not launch training")


if __name__ == "__main__":
    main()
