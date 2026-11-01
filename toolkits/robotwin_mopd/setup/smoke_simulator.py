# Copyright 2026 The RLinf Authors.
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
"""Exercise a real Aloha RoboTwin reset, cameras and policy action without models."""

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from robotwin.envs.vector_env import VectorEnv


def main():
    repo = Path(os.environ["REPO_PATH"])
    config = yaml.safe_load(
        (
            repo / "examples/embodiment/config/env/robotwin_adjust_bottle.yaml"
        ).read_text()
    )["task_config"]
    config.update(
        embodiment=["aloha-agilex"],
        planner_backend="mplib",
        save_path="/tmp/robotwin4-smoke",
        collect_data=False,
        save_data=False,
    )
    config["camera"]["collect_wrist_camera"] = True
    config["domain_randomization"].update(
        random_background=False,
        cluttered_table=False,
        clean_background_rate=1,
        random_head_camera_dis=0,
        random_table_height=0,
        random_light=False,
        crazy_random_light_rate=0,
    )
    t0 = time.monotonic()
    env = VectorEnv(config, 1, env_seeds=[100000])
    env.reset(env_seeds=[100000])
    obs = env.envs[0].get_obs()
    result = {
        "task": "adjust_bottle",
        "init_seconds": time.monotonic() - t0,
        "observation_shapes": {
            k: list(v.shape) for k, v in obs.items() if hasattr(v, "shape")
        },
        "instruction": obs["instruction"],
    }
    # Native policy step matches the original RoboTwin evaluation semantics.
    task = env.envs[0].task
    t1 = time.monotonic()
    task.take_action(np.asarray(obs["state"]), action_type="qpos")
    result["action_seconds"] = time.monotonic() - t1
    result["success"] = bool(task.check_success())
    result["status"] = "PASS"
    print(json.dumps(result), flush=True)
    env.close()
    print("CLOSED", flush=True)


if __name__ == "__main__":
    main()
