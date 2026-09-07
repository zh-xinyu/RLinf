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
"""Test real wrapper methods with a CPU fake backend; no RoboCasa, Ray, or rendering."""

from collections import Counter
from types import MethodType

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env


def task_sampler(rank=0, num_groups=6, group_size=1, strategy="suite_balanced"):
    env = Robocasa365Env.__new__(Robocasa365Env)
    env.cfg = OmegaConf.create({})
    env.seed_offset = rank
    env.total_num_processes = 8
    env.num_group = num_groups
    env.group_size = group_size
    env.num_envs = num_groups * group_size
    env.task_sampling_strategy = strategy
    env.task_specs = [
        {"metadata_view": {"task_soup": suite}}
        for suite, count in (("atomic", 18), ("seen", 16), ("unseen", 16))
        for _ in range(count)
    ]
    env.num_tasks = len(env.task_specs)
    env._ordered_task_cursor = rank * num_groups
    env._generator = np.random.default_rng(0)
    env._init_suite_balanced_sampling()
    return env


def test_full_50_tasks_cover_each_rank_and_all_tasks_in_two_epochs():
    visited = set()
    for rank in range(8):
        env = task_sampler(rank)
        for _ in range(2):
            ids = env._sample_task_ids(6)
            assert Counter(
                env.task_specs[i]["metadata_view"]["task_soup"] for i in ids
            ) == {"atomic": 2, "seen": 2, "unseen": 2}
            visited.update(ids)
    assert visited == set(range(50))


def test_partial_reset_preserves_suite_slots_and_other_task_cursors():
    env = task_sampler()
    env._sample_task_ids(6)
    before = env._suite_reset_counts.copy()
    selected = env._sample_task_ids(2, np.asarray([1, 5]))
    assert [env.task_specs[i]["metadata_view"]["task_soup"] for i in selected] == [
        "seen",
        "unseen",
    ]
    np.testing.assert_array_equal(env._suite_reset_counts - before, [0, 1, 0, 0, 0, 1])


@pytest.mark.parametrize("groups", [1, 2, 4, 5])
def test_invalid_per_worker_suite_coverage_fails_early(groups):
    with pytest.raises(ValueError, match="positive multiple of the suite count"):
        task_sampler(num_groups=groups)


def test_group_sampling_keeps_paired_envs_on_same_task():
    env = task_sampler(group_size=2)
    env.use_fixed_reset_state_ids = False
    env._refresh_task_context = lambda: None
    env._set_next_task_ids()
    np.testing.assert_array_equal(env.task_ids[::2], env.task_ids[1::2])


def test_eval_keeps_ordered_task_assignment():
    env = task_sampler(rank=1)
    env.cfg.is_eval = True
    np.testing.assert_array_equal(env._sample_task_ids(6), np.arange(6, 12))


class FakeBackend:
    def __init__(self, success_steps):
        self.success_steps = np.asarray(success_steps)
        self.steps = np.zeros(len(success_steps), dtype=int)
        self.calls = []

    def reset(self, id):
        self.steps[id] = 0
        return [{"step": 0} for _ in id], [{"success": False} for _ in id]

    def step(self, actions, id):
        assert len(actions) == len(id)
        self.calls.append(id.tolist())
        self.steps[id] += 1
        obs = [{"step": int(self.steps[i])} for i in id]
        info = [{"success": bool(self.steps[i] == self.success_steps[i])} for i in id]
        return obs, np.zeros(len(id)), np.zeros(len(id), dtype=bool), info


def episode_env(
    horizons=(2, 5), success_steps=(3, 2), *, ignore=False, auto_reset=False
):
    env = Robocasa365Env.__new__(Robocasa365Env)
    env.cfg = OmegaConf.create({"reward_coef": 1.0, "is_eval": False})
    env.num_envs = len(horizons)
    env.group_size = 1
    env._is_start = True
    env.auto_reset = auto_reset
    env.ignore_terminations = ignore
    env.use_rel_reward = False
    env.rotate_tasks_on_auto_reset = False
    env.task_horizons = np.asarray(horizons)
    env.task_metadata = [{"task_soup": "suite"} for _ in horizons]
    env.prev_step_reward = np.zeros(env.num_envs)
    env._elapsed_steps = np.zeros(env.num_envs, dtype=np.int32)
    env.current_raw_obs = None
    env.current_info_list = None
    env.env = FakeBackend(success_steps)
    env._wrap_obs = MethodType(
        lambda self, obs, info: {"states": torch.tensor([o["step"] for o in obs])}, env
    )
    env._init_metrics()
    env.reset()
    return env


def test_first_done_freezes_physics_rewards_and_success_metrics():
    env = episode_env()
    action = np.zeros((2, 1))
    env.step(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward.tolist() == [0, 1]
    assert terminated.tolist() == [False, True]
    assert truncated.tolist() == [True, False]
    terminal_info = info["episode"]
    for _ in range(6):
        later_obs, reward, term, trunc, info = env.step(action)
        assert reward.tolist() == [0, 0]
        assert torch.equal(later_obs["states"], obs["states"])
        assert torch.equal(term, terminated) and torch.equal(trunc, truncated)
        for key, value in terminal_info.items():
            torch.testing.assert_close(info["episode"][key], value)
    assert env.env.calls == [[0, 1], [0, 1]]
    assert info["episode"]["success_once"].tolist() == [False, True]
    assert info["episode"]["episode_len"].tolist() == [2, 2]


def test_only_active_envs_are_stepped_and_partial_reset_reactivates_one():
    env = episode_env(horizons=(1, 4), success_steps=(99, 99))
    action = np.zeros((2, 1))
    env.step(action)
    env.step(action)
    assert env.env.calls == [[0, 1], [1]]
    env.reset(env_idx=0)
    _, _, _, truncated, _ = env.step(action)
    assert env.env.calls[-1] == [0, 1]
    assert truncated.tolist() == [True, False]
    assert env.elapsed_steps.tolist() == [1, 3]


def test_ignore_terminations_continues_until_horizon_but_not_beyond():
    env = episode_env(horizons=(3,), success_steps=(1,), ignore=True)
    for _ in range(5):
        _, _, term, trunc, info = env.step(np.zeros((1, 1)))
    assert len(env.env.calls) == 3
    assert term.tolist() == [False] and trunc.tolist() == [True]
    assert info["episode"]["success_once"].tolist() == [True]
    assert info["episode"]["success_at_end"].tolist() == [False]
    assert info["episode"]["return"].tolist() == [1]


@pytest.mark.parametrize("auto_reset", [False, True])
def test_done_inside_action_chunk_keeps_first_done_stats(auto_reset):
    env = episode_env(horizons=(2, 5), success_steps=(3, 1), auto_reset=auto_reset)
    _, rewards, term, trunc, infos = env.chunk_step(np.zeros((2, 4, 1)))
    assert env.env.calls == [[0, 1], [0]]
    assert rewards.sum(dim=1).tolist() == [0, 1]
    episode = infos[-1]["final_info"]["episode"] if auto_reset else infos[-1]["episode"]
    assert episode["success_once"].tolist() == [False, True]
    assert episode["episode_len"].tolist() == [2, 1]
    if auto_reset:
        assert not term[:, :-1].any() and not trunc[:, :-1].any()
        assert env.elapsed_steps.tolist() == [0, 0]
    else:
        assert term[1].tolist() == [True, True, True, True]
        assert trunc[0].tolist() == [False, True, True, True]
