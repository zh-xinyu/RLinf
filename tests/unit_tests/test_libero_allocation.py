# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate quota collection and terminal accounting without GPU simulation."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_libero_multi_suite import make_cfg, make_env

from rlinf.config import validate_rollout_allocation_cfg
from rlinf.envs.libero.multi_suite import validate_rollout_plan
from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.utils.rollout_allocation import RolloutAllocator, training_signature


@pytest.fixture
def env_module():
    pytest.importorskip("libero")
    from rlinf.envs.libero import libero_env

    return libero_env


def quota_env(env_module, monkeypatch, num_envs=4):
    return make_env(
        env_module,
        monkeypatch,
        make_cfg(
            task_sampling_strategy="suite_quota",
            freeze_finished_episodes=True,
            reset_settle_steps=1,
            task_suite_horizons=dict(zip("abcd", [2, 2, 4, 4])),
        ),
        num_envs,
    )


def controller(env, workers=1):
    return RolloutAllocator(
        env.rollout_task_catalog,
        workers,
        env.num_envs,
        1,
        {
            "mode": "uniform",
            "seed": 42,
        },
    )


def test_frozen_outcomes_records_and_mask_count(env_module, monkeypatch):
    env = quota_env(env_module, monkeypatch)
    allocation = controller(env)
    rows = allocation.plan()
    env.set_rollout_tasks(rows)
    env.task_ids = np.asarray([row["task_id"] for row in rows])
    calls = []
    steps = np.zeros(4, dtype=int)

    def step(actions, ids):
        ids = np.asarray(ids)
        calls.append(ids.tolist())
        steps[ids] += 1
        # Slot 0 succeeds immediately. Slot 1 would succeed after its timeout,
        # but must never receive that extra step.
        success = (ids == 0) | ((ids == 1) & (steps[ids] >= 3))
        return [{"tick": int(steps[i])} for i in ids], None, success, [{} for _ in ids]

    env.env = SimpleNamespace(step=step)
    monkeypatch.setattr(
        env, "_wrap_obs", lambda raw: {"ticks": [x["tick"] for x in raw]}
    )
    monkeypatch.setattr(env, "_reconfigure", lambda *args: None)
    monkeypatch.setattr(env, "_calc_step_reward", lambda terms: terms.astype(float))
    env.reset()
    steps[:] = 0  # Settling calls are not policy steps.
    with pytest.raises(ValueError):
        env.rollout_episode_records()
    with pytest.raises(ValueError):
        env.set_rollout_tasks(rows)
    _, _, terms, truncs, infos = env.chunk_step(torch.zeros(4, 8, 7))
    done = terms | truncs
    mask, _ = compute_loss_mask(
        torch.cat([torch.zeros(1, 4, 8, dtype=torch.bool), done[None]], dim=0)
    )
    records = env.rollout_episode_records()
    assert [r["simulation_steps"] for r in records] == [1, 2, 4, 4]
    assert [r["success_once"] for r in records] == [True, False, False, False]
    assert mask.sum().item() == env.physics_step_counts["policy_simulation_steps"] == 11
    assert env.physics_step_counts["settling_simulation_steps"] == 4
    assert infos[-1]["episode"]["episode_len"].tolist() == [1, 2, 4, 4]
    assert env.current_raw_obs[1]["tick"] == 2
    assert calls == [[0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3], [2, 3], [2, 3]]
    allocation.observe(records)
    env.set_rollout_tasks(allocation.plan())
    env.reset()
    assert not env._episode_terminated.any() and not env._episode_truncated.any()


def test_plan_identity_occurrences_and_reset_seeds(env_module, monkeypatch):
    env = quota_env(env_module, monkeypatch)
    allocator = controller(env)
    rows = allocator.plan()
    assert (
        len({t["task"] for t in env.rollout_task_catalog}) == 8
    )  # Native names duplicate.
    for row in rows:
        row["occurrence"] = 5
    env.set_rollout_tasks(list(reversed(rows)))
    prepared = env.reset_state_ids.copy()
    env.update_reset_state_ids()
    np.testing.assert_array_equal(prepared, env.reset_state_ids)
    seeds = []
    env.env = SimpleNamespace(
        reconfigure_env_fns=lambda *args: None,
        seed=lambda seed: seeds.append(seed),
        reset=lambda **kwargs: None,
        set_init_state=lambda **kwargs: None,
        step=lambda actions, ids: ([{} for _ in ids], None, None, [{} for _ in ids]),
    )
    monkeypatch.setattr(env, "get_env_fn_params", lambda ids: [])
    monkeypatch.setattr(env, "_wrap_obs", lambda raw: {})
    monkeypatch.setattr(env, "_get_reset_states", lambda **kwargs: [])
    env.reset()
    assert env.trial_ids.tolist() == [2] * 4
    assert env.task_ids.tolist() == [row["task_id"] for row in rows]
    assert seeds == [[row["reset_seed"] for row in rows]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("slot", 3),
        ("task", "wrong"),
        ("suite", "wrong"),
        ("horizon", 99),
        ("reset_seed", -1),
        ("reset_seed", 2**32),
        ("reset_seed", True),
        ("worker", 99),
    ],
)
def test_reject_invalid_plan(env_module, monkeypatch, field, value):
    env = quota_env(env_module, monkeypatch)
    rows = controller(env).plan()
    rows[0][field] = value
    with pytest.raises(ValueError):
        env.set_rollout_tasks(rows)


@pytest.mark.parametrize("counts", [[5, 7, 9, 11], [6, 8, 14, 20]])
def test_non_divisible_global_quotas_keep_all_suites_per_worker(
    env_module, monkeypatch, counts
):
    env = quota_env(env_module, monkeypatch, num_envs=sum(counts) // 4)
    allocation = RolloutAllocator(
        env.rollout_task_catalog,
        4,
        env.num_envs,
        1,
        {
            "mode": "static",
            "warmup_rounds": 20,
            "static_quotas": dict(zip("abcd", counts)),
        },
    )
    allocation.rounds = 20
    rows = allocation.plan()
    assert Counter(row["suite"] for row in rows) == dict(zip("abcd", counts))
    for worker in range(4):
        local = [row for row in rows if row["worker"] == worker]
        assert (
            len(
                validate_rollout_plan(
                    local, env.rollout_task_catalog, env.num_envs, list("abcd")
                )
            )
            == env.num_envs
        )


def protocol_cfg():
    return OmegaConf.create(
        {
            "runner": {
                "rollout_allocation": {"enabled": True, "mode": "uniform"},
                "step_unit": "optimizer_update",
            },
            "env": {
                "train": {
                    "env_type": "libero",
                    "task_sampling_strategy": "suite_quota",
                    "group_size": 1,
                    "task_suite_names": [
                        "libero_spatial",
                        "libero_object",
                        "libero_goal",
                        "libero_10",
                    ],
                    "task_suite_horizons": {
                        "libero_spatial": 240,
                        "libero_object": 320,
                        "libero_goal": 320,
                        "libero_10": 520,
                    },
                    "freeze_finished_episodes": True,
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
            "actor": {"model": {"opd_fixed_text_padding": False}},
            "rollout": {
                "model": {},
                "expert_models": {"a": {"opd_fixed_text_padding": True}},
            },
        }
    )


def test_native_horizon_validation_and_calibration_signature():
    cfg = protocol_cfg()
    validate_rollout_allocation_cfg(cfg)
    original = training_signature(cfg)
    assert (
        original["libero_protocol"]["teacher_input"]["a"]["opd_fixed_text_padding"]
        is True
    )
    cfg.actor.model.inference_autocast_dtype = "bf16"
    assert training_signature(cfg) != original
    del cfg.actor.model.inference_autocast_dtype
    cfg.actor.model.opd_fixed_text_padding = True
    assert training_signature(cfg) != original
    cfg.env.train.task_suite_horizons.libero_10 = 32
    with pytest.raises(ValueError, match="native horizons"):
        validate_rollout_allocation_cfg(cfg)
    cfg.env.train.env_type = "robocasa365"
    cfg.env.train.episode_horizon_source = "task_horizon"
    validate_rollout_allocation_cfg(cfg)
    assert "libero_protocol" not in training_signature(cfg)


def test_frozen_eval_resets_new_trials_and_leaves_exhausted_slots_idle(
    env_module, monkeypatch
):
    env = make_env(
        env_module,
        monkeypatch,
        make_cfg(
            is_eval=True,
            auto_reset=True,
            task_sampling_strategy="ordered",
            freeze_finished_episodes=True,
            reset_settle_steps=1,
            task_suite_horizons=dict.fromkeys("abcd", 2),
        ),
    )
    calls = []

    def step(actions, ids):
        ids = list(ids)
        calls.append(ids)
        return [{} for _ in ids], None, np.ones(len(ids), dtype=bool), [{} for _ in ids]

    env.env = SimpleNamespace(step=step)
    monkeypatch.setattr(env, "_reconfigure", lambda *args: None)
    monkeypatch.setattr(env, "_wrap_obs", lambda raw: {"obs": list(raw)})
    monkeypatch.setattr(env, "_calc_step_reward", lambda terms: terms.astype(float))
    env.reset()
    # One new trial remains after the first four finish; no trials after that.
    plans = iter([np.array([0, -1, -1, -1]), np.full(4, -1)])
    monkeypatch.setattr(env, "_get_ordered_reset_state_ids", lambda n: next(plans))
    first = env.chunk_step(torch.zeros(4, 8, 7))
    assert env.physics_step_counts["policy_simulation_steps"] == 4
    assert first[-1][-1]["final_info"]["episode"]["episode_len"].tolist() == [1] * 4
    assert env._episode_terminated.tolist() == [False, True, True, True]
    # The fake reset does not change task/trial IDs, so metric deduplication
    # suppresses its repeated identity, independently of physics accounting.
    env.chunk_step(torch.zeros(4, 8, 7))
    assert env.physics_step_counts["policy_simulation_steps"] == 5
    assert calls == [[0, 1, 2, 3], [0, 1, 2, 3], [0], [0]]
    assert env._episode_terminated.all()


def test_generic_worker_catalog_and_robocasa_fallback():
    from rlinf.workers.env.env_worker import EnvWorker

    task = {"task_id": 0, "task": "libero_10/0", "suite": "libero_10", "horizon": 520}
    worker = SimpleNamespace(
        stage_num=1,
        actor_split_num=1,
        _rank=0,
        train_num_envs_per_stage=8,
        rollout_epoch=1,
        env_list=[
            SimpleNamespace(
                episode_horizon_source="task_horizon", rollout_task_catalog=[task]
            )
        ],
    )
    assert EnvWorker.describe_rollout_allocation(worker)["tasks"] == [task]
    worker.env_list = [
        SimpleNamespace(
            episode_horizon_source="task_horizon",
            task_specs=[
                {
                    "task_name": "robo_task",
                    "metadata_view": {"task_soup": "atomic"},
                    "horizon": 200,
                }
            ],
        )
    ]
    assert EnvWorker.describe_rollout_allocation(worker)["tasks"] == [
        {"task_id": 0, "task": "robo_task", "suite": "atomic", "horizon": 200}
    ]
