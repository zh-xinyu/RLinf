# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for multi-suite identity, balanced resets, and step accounting."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.envs.libero.multi_suite import (
    CombinedLiberoBenchmark,
    SuiteBalancedResetSampler,
)


class FakeBenchmark:
    def __init__(self, name, count=2):
        self.name = name
        self.count = count

    def get_num_tasks(self):
        return self.count

    def get_task(self, task_id):
        return SimpleNamespace(name="shared_name", problem_folder=self.name, id=task_id)

    def get_task_init_states(self, task_id):
        return [(self.name, task_id, trial) for trial in range(3)]


def test_catalog_preserves_suite_identity_and_native_task_folder():
    catalog = CombinedLiberoBenchmark(["spatial", "long"], FakeBenchmark)
    assert catalog.get_num_tasks() == 4
    assert catalog.get_task(0).name == catalog.get_task(2).name
    assert catalog.get_task(2).problem_folder == "long"
    assert catalog.suite_for_task(2) == "long"
    assert catalog.get_task_init_states(3)[0] == ("long", 1, 0)
    with pytest.raises(IndexError):
        catalog.get_task(-1)
    with pytest.raises(IndexError):
        catalog.get_task(4)


@pytest.mark.parametrize("names", [[], ["spatial", "spatial"]])
def test_catalog_rejects_ambiguous_suites(names):
    with pytest.raises(ValueError):
        CombinedLiberoBenchmark(names, FakeBenchmark)


def test_balanced_sampling_is_not_biased_by_task_or_trial_counts():
    labels = ["a", "a", "b", "c", "c", "c", "d"]
    counts = [1, 20, 2, 3, 9, 1, 100]
    sampler = SuiteBalancedResetSampler(labels, counts, np.random.default_rng(10))
    ids = sampler.sample(48)
    task_ids = np.searchsorted(np.cumsum(counts), ids, side="right")
    assert Counter(labels[i] for i in task_ids) == dict.fromkeys("abcd", 12)
    assert Counter(task_ids)[0] == Counter(task_ids)[1] == 6
    assert all(Counter(task_ids)[i] == 4 for i in [3, 4, 5])
    twin = SuiteBalancedResetSampler(labels, counts, np.random.default_rng(10))
    np.testing.assert_array_equal(ids, twin.sample(48))
    with pytest.raises(ValueError, match="divisible"):
        sampler.sample(6)


def test_balanced_task_filter_retains_all_suites_and_valid_ids():
    labels = ["a", "a", "b", "b"]
    sampler = SuiteBalancedResetSampler(
        labels, [2] * 4, np.random.default_rng(4), [1, 2]
    )
    assert set(sampler.sample(12) // 2) == {1, 2}
    with pytest.raises(ValueError, match="retain every suite"):
        SuiteBalancedResetSampler(labels, [2] * 4, np.random.default_rng(), [0])
    with pytest.raises(ValueError, match="invalid global"):
        SuiteBalancedResetSampler(labels, [2] * 4, np.random.default_rng(), [4])


@pytest.fixture
def env_module():
    pytest.importorskip("libero")
    from rlinf.envs.libero import libero_env

    return libero_env


def make_cfg(**overrides):
    cfg = {
        "seed": 1,
        "group_size": 1,
        "use_fixed_reset_state_ids": False,
        "ignore_terminations": False,
        "auto_reset": False,
        "task_suite_name": "a",
        "task_suite_names": ["a", "b", "c", "d"],
        "task_sampling_strategy": "suite_balanced",
        "use_ordered_reset_state_ids": False,
        "max_steps_per_rollout_epoch": 8,
        "max_episode_steps": 8,
        "use_rel_reward": False,
        "video_cfg": {},
        "reset_gripper_open": True,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def make_env(env_module, monkeypatch, cfg=None, num_envs=4):
    monkeypatch.setattr(env_module, "get_libero_type", lambda: "standard")
    monkeypatch.setattr(
        env_module, "get_benchmark_overridden", lambda name: lambda: FakeBenchmark(name)
    )
    monkeypatch.setattr(env_module.LiberoEnv, "_init_env", lambda self: None)
    return env_module.LiberoEnv(cfg or make_cfg(), num_envs, 0, 1, None)


def test_every_worker_has_all_suites_and_rejects_partial_quota(env_module, monkeypatch):
    env = make_env(env_module, monkeypatch, num_envs=8)
    assert Counter(env._task_suite_labels[i] for i in env.task_ids) == dict.fromkeys(
        "abcd", 2
    )
    with pytest.raises(ValueError, match="per-worker"):
        make_env(env_module, monkeypatch, num_envs=6)
    with pytest.raises(ValueError, match="full-round"):
        make_env(env_module, monkeypatch, make_cfg(auto_reset=True))


def test_single_suite_default_and_observation_labels(env_module, monkeypatch):
    cfg = make_cfg()
    del cfg.task_suite_names
    del cfg.task_sampling_strategy
    env = make_env(env_module, monkeypatch, cfg)
    assert env._balanced_reset_sampler is None
    assert env._reset_settle_steps == 15
    env.task_descriptions = ["do task"] * 4
    monkeypatch.setattr(
        env,
        "_extract_image_and_state",
        lambda obs: {
            "full_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "state": np.zeros(8),
        },
    )
    assert env._wrap_obs([{}] * 4)["task_groups"] == ["a"] * 4


def test_simulation_counters_include_settling_and_post_done_steps(
    env_module, monkeypatch
):
    env = make_env(
        env_module,
        monkeypatch,
        make_cfg(
            reset_settle_steps=2,
            task_suite_horizons={"a": 1, "b": 2, "c": 3, "d": 4},
        ),
    )
    calls = []

    def step(actions, ids=None):
        count = len(actions)
        calls.append(count)
        return [{}] * count, np.zeros(count), np.zeros(count, dtype=bool), [{}] * count

    env.env = SimpleNamespace(step=step)
    monkeypatch.setattr(env, "_wrap_obs", lambda obs: {"obs": obs})
    monkeypatch.setattr(env, "_reconfigure", lambda *args: None)
    monkeypatch.setattr(env, "_calc_step_reward", lambda terms: np.zeros(4))
    env.reset()
    assert env.physics_step_counts == {
        "policy_simulation_steps": 0,
        "settling_simulation_steps": 8,
        "total_simulation_steps": 8,
        "reset_episodes": 4,
    }
    _, _, _, truncations, _ = env.step(np.zeros((4, 7)))
    assert truncations.tolist() == [True, False, False, False]
    _, _, _, truncations, _ = env.step(np.zeros((4, 7)))
    assert truncations.tolist() == [True, True, False, False]
    assert calls == [4, 4, 4, 4]
    assert env.physics_step_counts["policy_simulation_steps"] == 8
    env.reset()
    assert env.physics_step_counts["policy_simulation_steps"] == 8
    assert env.physics_step_counts["settling_simulation_steps"] == 16
    assert env.physics_step_counts["reset_episodes"] == 8


def test_horizon_mapping_must_cover_all_suites(env_module, monkeypatch):
    with pytest.raises(ValueError, match="every selected suite"):
        make_env(env_module, monkeypatch, make_cfg(task_suite_horizons={"a": 2}))
    with pytest.raises(ValueError, match="fit within"):
        make_env(
            env_module,
            monkeypatch,
            make_cfg(task_suite_horizons={"a": 1, "b": 2, "c": 3, "d": 9}),
        )


def test_round_boundary_uses_prepared_ids_without_skipping_tasks(
    env_module, monkeypatch
):
    env = make_env(env_module, monkeypatch, make_cfg(reset_settle_steps=1))
    env.env = SimpleNamespace(step=lambda actions, ids: ([{}] * 4, None, None, None))
    monkeypatch.setattr(env, "_wrap_obs", lambda obs: {})
    consumed = []
    monkeypatch.setattr(
        env, "_reconfigure", lambda ids, slots: consumed.append(ids.copy())
    )
    first_ids = env.reset_state_ids.copy()
    env.reset()
    env.update_reset_state_ids()  # EnvWorker.finish_rollout.
    second_ids = env.reset_state_ids.copy()
    env.is_start = True  # EnvWorker.interact before the next round.
    env.reset()
    np.testing.assert_array_equal(consumed[0], first_ids)
    np.testing.assert_array_equal(consumed[1], second_ids)
    assert set(first_ids // 3) | set(second_ids // 3) == set(range(8))
