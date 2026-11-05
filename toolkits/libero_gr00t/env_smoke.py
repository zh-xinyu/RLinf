# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real four-suite reset, labels and physical counters without a policy."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def main() -> None:
    """Run two balanced resets and three actual simulator steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from toolkits.libero_gr00t.probe import SUITES, renderer_index

    egl = renderer_index()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    from rlinf.envs.libero.libero_env import LiberoEnv

    base = Path("examples/embodiment/config/env")
    extra = OmegaConf.load(base / "libero_four_suite.yaml")
    del extra["defaults"]
    cfg = OmegaConf.merge(OmegaConf.load(base / "libero_spatial.yaml"), extra)
    cfg.total_num_envs = 4
    cfg.video_cfg.video_base_dir = str(args.output.parent / "unused_video")
    env = LiberoEnv(cfg, 4, 0, 1, None)
    try:
        obs, _ = env.reset()
        first_tasks = env.task_ids.tolist()
        assert sorted(obs["task_groups"]) == sorted(SUITES)
        assert tuple(obs["main_images"].shape) == (4, 256, 256, 3)
        action = np.zeros((4, 7), dtype=np.float32)
        action[:, -1] = -1
        for _ in range(3):
            env.step(action)
        assert env.physics_step_counts == {
            "policy_simulation_steps": 12,
            "settling_simulation_steps": 40,
            "total_simulation_steps": 52,
            "reset_episodes": 4,
        }
        env.update_reset_state_ids()
        env.is_start = True
        obs, _ = env.reset()
        assert sorted(obs["task_groups"]) == sorted(SUITES)
        counts = env.physics_step_counts
        assert counts == {
            "policy_simulation_steps": 12,
            "settling_simulation_steps": 80,
            "total_simulation_steps": 92,
            "reset_episodes": 8,
        }
        result = {
            "passed": True,
            "egl_index": egl,
            "first_tasks": first_tasks,
            "second_tasks": env.task_ids.tolist(),
            "task_groups": obs["task_groups"],
            "physics_step_counts": counts,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
