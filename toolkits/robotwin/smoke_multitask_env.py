# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise four native policy steps without loading a VLA model."""

import json

import torch
from omegaconf import OmegaConf

from rlinf.envs.robotwin.multitask_env import RoboTwinMultiTaskEnv


def main():
    cfg = OmegaConf.load("examples/embodiment/config/env/robotwin_four_task_clean.yaml")
    cfg.video_cfg.video_base_dir = "outputs/robotwin4/smoke_video"
    env = RoboTwinMultiTaskEnv(cfg, 4, 0, 1, None)
    rows = [
        {
            "slot": i,
            "task_id": i,
            "task": name,
            "suite": name,
            "reset_seed": 100000 + i,
            "round": 0,
        }
        for i, name in enumerate(cfg.task_names)
    ]
    env.set_rollout_tasks(rows)
    try:
        obs, _ = env.reset()
        print("RESET", obs["task_groups"], obs["states"].shape, flush=True)
        assert obs["wrist_images"].shape[:2] == (4, 2)
        # Hold current joints/grippers instead of issuing arbitrary targets.
        actions = obs["states"].to(torch.float32)[:, None, :]
        obs, rewards, terms, truncs, _ = env.chunk_step(actions)
        print("STEP", json.dumps(env.rollout_episode_records()), flush=True)
        assert all(r["simulation_steps"] == 1 for r in env.rollout_episode_records())
        assert obs[-1]["task_groups"] == list(cfg.task_names)
        assert rewards.shape == terms.shape == truncs.shape == (4, 1)
        print("MULTITASK_SMOKE_PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
