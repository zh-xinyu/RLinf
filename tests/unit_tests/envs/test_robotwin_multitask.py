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

"""CPU-only tests for quota routing and exact terminal padding accounting."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.envs.robotwin.multitask_env import TASK_HORIZONS, RoboTwinMultiTaskEnv


class FakeSlot:
    def __init__(self, name, horizon):
        self.name = name
        self.horizon = horizon
        self.calls = 0
        self.closed = False

    def reset(self, seed):
        self.actual_seed = seed
        self.elapsed = 0
        self.obs = {
            "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "state": np.zeros(14, dtype=np.float32),
            "instruction": self.name,
        }
        return self.obs

    def step_chunk(self, actions):
        self.calls += 1
        end = 2 if self.name == "adjust_bottle" else self.horizon
        executed = min(len(actions), end - self.elapsed)
        self.elapsed += executed
        success = self.name == "adjust_bottle" and self.elapsed == end
        return self.obs, executed, success, self.elapsed == end and not success

    def close(self):
        self.closed = True


@pytest.fixture
def env(monkeypatch):
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "auto_reset": False,
            "ignore_terminations": False,
            "group_size": 1,
            "video_cfg": {},
            "task_names": list(TASK_HORIZONS),
            "task_horizons": dict.fromkeys(TASK_HORIZONS, 5),
            "max_steps_per_rollout_epoch": 5,
            "reward_coef": 1.0,
        }
    )
    monkeypatch.setattr(
        RoboTwinMultiTaskEnv,
        "_make_slot",
        lambda self, task_id: FakeSlot(self.task_names[task_id], 5),
    )
    result = RoboTwinMultiTaskEnv(cfg, 4, 0, 1, None)
    yield result
    result.close()


def rows(ids=(0, 1, 2, 3)):
    names = list(TASK_HORIZONS)
    return [
        {
            "slot": i,
            "task_id": task_id,
            "task": names[task_id],
            "suite": names[task_id],
            "reset_seed": 100 + i,
            "round": 3,
        }
        for i, task_id in enumerate(ids)
    ]


def test_routing_terminal_masks_and_actual_budget(env):
    env.set_rollout_tasks(rows())
    obs, _ = env.reset()
    assert obs["task_groups"] == list(TASK_HORIZONS)
    assert obs["wrist_images"].shape == (4, 2, 8, 8, 3)
    actions = np.zeros((4, 4, 14))
    last, rewards, terminated, truncated, _ = env.chunk_step(actions)
    assert terminated[0].tolist() == [False, True, True, True]
    assert rewards[0].tolist() == [0, 1, 0, 0]
    assert not truncated.any()
    assert last[-1]["task_groups"] == list(TASK_HORIZONS)
    _, rewards, terminated, truncated, _ = env.chunk_step(actions)
    assert terminated[0].all()
    assert truncated[1:].all()
    assert env._slots[0].calls == 1  # Frozen terminal slot never calls simulator.
    assert rewards.sum() == 0
    records = env.rollout_episode_records()
    assert [r["simulation_steps"] for r in records] == [2, 5, 5, 5]
    assert sum(r["success_once"] for r in records) == 1
    assert all(r["round"] == 3 for r in records)


def test_reallocation_and_offload_recreate_correct_tasks(env):
    env.set_rollout_tasks(rows())
    env.reset()
    old = env._slots[0]
    env.set_rollout_tasks(rows((3, 2, 1, 0)))
    assert old.closed
    obs, _ = env.reset()
    assert obs["task_groups"] == list(reversed(TASK_HORIZONS))
    env.offload()
    assert env._slots == [None] * 4
    assert [r["actual_reset_seed"] for r in env.rollout_episode_records()] == [
        100,
        101,
        102,
        103,
    ]
    obs, _ = env.reset()
    assert obs["task_groups"] == list(reversed(TASK_HORIZONS))


def test_reject_invalid_quota_identity_and_slots(env):
    invalid = rows()
    invalid[0]["suite"] = "wrong_teacher"
    with pytest.raises(ValueError, match="disagree"):
        env.set_rollout_tasks(invalid)
    with pytest.raises(ValueError, match="every local slot"):
        env.set_rollout_tasks(rows()[:-1])
