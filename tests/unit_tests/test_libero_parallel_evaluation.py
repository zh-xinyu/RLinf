# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Parallel evaluation must preserve per-episode inference and exact coverage."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from toolkits.libero_gr00t.parallel_evaluation import (
    EpisodePolicy,
    audit_episodes,
    validate_selection,
)


class RandomPolicy:
    training = False

    def predict_action_batch(self, obs, mode):
        assert mode == "eval"
        # Include the random draws used between ODE steps as well as initial x.
        action = torch.randn(1, 10, 7) + obs["states"][:, None, :1]
        for _ in range(4):
            torch.randn(1, 16, 32)
        return action, {}


def test_episode_streams_match_sequential_seed_and_ignore_schedule(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = RandomPolicy()
    keys = [[0, 4], [30, 7]]
    engine = EpisodePolicy(1234, trace=True)
    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    first = engine.predict(
        model, {"states": torch.tensor([[1.0], [2.0]])}, keys, [True, True]
    )
    assert torch.equal(before, torch.get_rng_state())
    second = engine.predict(
        model, {"states": torch.tensor([[2.0], [1.0]])}, keys[::-1], [True, False]
    )
    assert torch.equal(second[1], first[0])
    for i, key in enumerate(keys):
        torch.manual_seed(1234 + (key[0] % 10) * 100 + key[1])
        expected, _ = model.predict_action_batch(
            {"states": torch.tensor([[float(i + 1)]])}, "eval"
        )
        assert torch.equal(expected[0], first[i])
        if i == 1:
            expected, _ = model.predict_action_batch(
                {"states": torch.tensor([[2.0]])}, "eval"
            )
            assert torch.equal(expected[0], second[0])
    assert engine.calls == 3
    assert len(engine.trace) == 3
    with pytest.raises(ValueError, match="Frozen slot"):
        engine.predict(model, {"states": torch.ones(1, 1)}, [(0, 9)], [False])


def test_pair_coverage_rejects_duplicates_and_withholds_partial_score():
    row = {
        "task_id": 0,
        "trial_id": 0,
        "suite": "libero_spatial",
        "horizon": 240,
        "simulation_steps": 10,
        "success_once": True,
    }
    assert audit_episodes([row], [0], [0])["macro_success_rate"] == 1.0
    assert audit_episodes([row], [0], [0, 1])["macro_success_rate"] is None
    with pytest.raises(ValueError, match="duplicate"):
        audit_episodes([row, row], [0], [0])
    with pytest.raises(ValueError, match="incomplete"):
        audit_episodes([dict(row, success_once=False)], [0], [0])
    with pytest.raises(ValueError, match="divide"):
        validate_selection([0], list(range(50)), 8)
    with pytest.raises(ValueError, match="unique"):
        validate_selection([0, 0], [0], 1)


def test_config_uses_current_protocol_and_no_training_workers(tmp_path, monkeypatch):
    from toolkits.libero_gr00t import launch_production as production
    from toolkits.libero_gr00t.launch_parallel_evaluation import task_config

    paths = {
        key: tmp_path / key
        for key in (
            "student",
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        )
    }
    recipe_args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / "run",
        gpus=[0, 1, 2, 3],
        config_name="libero_mopd_formal_m1",
        updates=200,
        save_every_updates=50,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
        seed=0,
    )
    config = tmp_path / "training.yaml"
    monkeypatch.setenv("OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    OmegaConf.save(production.build_config(recipe_args, paths), config)
    args = SimpleNamespace(
        config=config,
        student=None,
        weights=None,
        seed=1234,
        envs_per_task=5,
        trials=list(range(50)),
        trace=False,
        python=Path("/tmp/python"),
    )
    cfg = task_config(args, 30, tmp_path)
    assert "actor" not in cfg and "train" not in cfg.env
    assert cfg.runner.only_eval and cfg.env.eval.rollout_epoch == 10
    assert cfg.rollout.model.num_action_chunks == 10
    assert cfg.env.eval.task_suite_execution_lengths.libero_10 == 10
    assert cfg.env.eval.task_suite_horizons.libero_object == 240
    assert cfg.evaluation.decisions == 48
    assert cfg.rollout.model.opd_native_rollout_batch_size == 1
    assert not cfg.rollout.model.opd_skip_finished_rollout
    cfg = task_config(args, 20, tmp_path)
    assert cfg.evaluation.decisions == 64
    import rlinf.config as config_module

    monkeypatch.setattr(config_module, "Cluster", lambda: None)
    monkeypatch.setattr(
        config_module,
        "HybridComponentPlacement",
        lambda *args: SimpleNamespace(get_world_size=lambda _: 1),
    )
    config_module.validate_embodied_cfg(cfg)


def test_explicit_eval_plan_selects_exact_states_without_suite_quotas():
    pytest.importorskip("libero")
    from rlinf.envs.libero.libero_env import LiberoEnv

    env = object.__new__(LiberoEnv)
    env.is_eval, env.auto_reset = True, False
    env.freeze_finished_episodes, env.ignore_terminations = True, False
    env.group_size, env.num_envs = 1, 2
    env.task_sampling_strategy = "ordered"
    env._allocation_episode_plan = None
    env.task_suite = SimpleNamespace()
    env.task_suite_names = ["libero_10"]
    env.task_id_filter = [30]
    env._task_suite_labels = (
        ["libero_spatial"] * 10
        + ["libero_object"] * 10
        + ["libero_goal"] * 10
        + ["libero_10"] * 10
    )
    env._suite_horizons = {"libero_10": 480}
    env.trial_id_bins = [50] * 40
    env.cumsum_trial_id_bins = np.cumsum(env.trial_id_bins)
    env.cfg = OmegaConf.create({"seed": 1234, "max_episode_steps": 480})
    env.set_evaluation_episodes(30, [1, 49])
    assert env.reset_state_ids.tolist() == [1501, 1549]
    assert [row["reset_seed"] for row in env._allocation_episode_plan] == [1234, 1234]
    env._episode_terminated, env._episode_truncated = (
        np.array([True, False]),
        np.array([False, False]),
    )
    with pytest.raises(ValueError, match="unfinished"):
        env.set_evaluation_episodes(30, [0, 2])
    env._episode_truncated[1] = True
    with pytest.raises(ValueError, match="outside"):
        env.set_evaluation_episodes(0, [0, 2])


def test_evaluation_packet_reset_bypasses_legacy_ordered_cursor(monkeypatch):
    pytest.importorskip("libero")
    from rlinf.envs.libero.libero_env import LiberoEnv

    env = object.__new__(LiberoEnv)
    env.is_eval = True
    env.task_sampling_strategy = "ordered"
    env._allocation_episode_plan = [{"trial_id": 42}]
    env.num_envs = 1
    env._is_start = True
    env.reset_state_ids = np.array([42])
    env._reset_settle_steps = 1
    env._settling_steps = env._reset_episodes = 0
    env.current_raw_obs = None
    env._last_raw_infos = [None]
    env.cfg = SimpleNamespace(reset_gripper_open=True)
    configured = []
    monkeypatch.setattr(
        env, "_reconfigure", lambda ids, slots: configured.append(ids.copy())
    )
    env.env = SimpleNamespace(step=lambda *args: ([{"a": 1}], None, None, [{}]))
    monkeypatch.setattr(env, "_wrap_obs", lambda rows: rows)
    monkeypatch.setattr(env, "_reset_metrics", lambda slots: None)
    env.reset()
    assert configured[0].tolist() == [42]
    assert not env.is_start
    assert env._settling_steps == 1


def test_packet_adapter_records_each_completed_packet_once(tmp_path):
    from toolkits.libero_gr00t.parallel_evaluation_workers import EvaluationPackets

    class FakeEnv:
        num_envs = 2
        cfg = SimpleNamespace(reset_settle_steps=10)

        def set_evaluation_episodes(self, task, trials):
            self.trials = trials

        def reset(self):
            return {}, {}

        def chunk_step(self, actions):
            done = torch.ones(2, 10, dtype=torch.bool)
            return [], None, done, ~done, []

        def rollout_episode_records(self):
            return [{"trial_id": trial} for trial in self.trials]

    cfg = SimpleNamespace(trials=[0, 1, 2, 3], task_id=0, output=str(tmp_path))
    wrapper = EvaluationPackets(FakeEnv(), cfg)
    for _ in range(2):
        wrapper.reset()
        wrapper.chunk_step(None)
        wrapper.chunk_step(None)
    assert [row["trial_id"] for row in wrapper.rows] == [0, 1, 2, 3]
