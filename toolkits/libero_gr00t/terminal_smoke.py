# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check real multi-suite simulator slots stop at their individual timeouts."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from toolkits.libero_gr00t.probe import SUITES, renderer_index

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(renderer_index())
    from rlinf.envs.libero.libero_env import LiberoEnv

    base = Path("examples/embodiment/config/env")
    extra = OmegaConf.load(base / "libero_four_suite.yaml")
    del extra["defaults"]
    cfg = OmegaConf.merge(OmegaConf.load(base / "libero_spatial.yaml"), extra)
    cfg.total_num_envs = 8
    cfg.max_episode_steps = 8
    cfg.max_steps_per_rollout_epoch = 8
    cfg.task_suite_horizons = dict(zip(SUITES, (2, 4, 6, 8), strict=True))
    cfg.video_cfg.video_base_dir = str(args.output.parent / "unused_video")
    env = LiberoEnv(cfg, 8, 0, 1, None)
    try:
        env.reset()
        action = np.zeros((8, 7), dtype=np.float32)
        action[:, -1] = -1
        counts = []
        snapshot = None
        for step in range(1, 11):
            obs, _, terminal, truncated, _ = env.step(action)
            expected = 2 * sum(min(step, horizon) for horizon in (2, 4, 6, 8))
            assert env.physics_step_counts["policy_simulation_steps"] == expected
            assert not terminal.any(), "Zero-action fixture unexpectedly solved a task"
            counts.append(expected)
            if step == 8:
                assert truncated.all()
                snapshot = obs["states"].clone()
            elif step > 8:
                torch.testing.assert_close(obs["states"], snapshot, rtol=0, atol=0)
        assert env.physics_step_counts["settling_simulation_steps"] == 80
        result = {
            "passed": True,
            "requested_steps": 10,
            "per_call_cumulative_policy_steps": counts,
            "physics_step_counts": env.physics_step_counts,
            "scope": "eight real environments with shortened test horizons; no policy",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
