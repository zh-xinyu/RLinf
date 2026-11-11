# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for fixed scenes through the standard environment factory."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.robotwin import fixed_eval_env as module
from rlinf.envs.robotwin.multitask_env import RoboTwinMultiTaskEnv
from toolkits.robotwin_mopd.worker_evaluate import checked_metrics


@pytest.fixture
def fixed_env(monkeypatch, tmp_path):
    # The fake CPU environment has no installed simulator; do not inherit a live runtime.
    monkeypatch.delenv("OPD_ROBOTWIN_PROFILE", raising=False)
    entries = [
        {
            "scene_seed": seed,
            "instruction": f"prompt {seed}",
            "episode_info": {"side": "left"},
        }
        for seed in (101, 205)
    ]
    pool = {"protocol": {"selection_start": 20}, "tasks": {"open_laptop": entries}}
    monkeypatch.setattr(module, "load_pool", lambda *a, **kw: pool)
    monkeypatch.setattr(module, "validate_pool_protocol", lambda *a, **kw: {})
    monkeypatch.setenv("ROBOTWIN_PATH", str(tmp_path))
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "auto_reset": False,
            "ignore_terminations": False,
            "group_size": 1,
            "video_cfg": {},
            "task_names": ["open_laptop"],
            "task_horizons": {"open_laptop": 5},
            "max_steps_per_rollout_epoch": 5,
            "reward_coef": 1.0,
            "is_eval": True,
            "enable_offload": False,
            "rollout_epoch": 2,
            "task_config": {},
            "fixed_evaluation_pool": "pool.json",
            "fixed_evaluation_tasks": ["open_laptop"],
            "fixed_evaluation_records": str(tmp_path / "records.jsonl"),
        }
    )

    class Slot:
        task = SimpleNamespace(get_info=lambda: {"side": "left"})

        def reset(self, seed, *, instruction, strict_seed):
            assert strict_seed
            self.actual_seed = seed
            self.calls = 0
            self.obs = {
                "full_image": np.zeros((8, 8, 3), np.uint8),
                "left_wrist_image": np.zeros((8, 8, 3), np.uint8),
                "right_wrist_image": np.zeros((8, 8, 3), np.uint8),
                "state": np.zeros(14, np.float32),
                "instruction": instruction,
            }
            return self.obs

        def step_chunk(self, actions):
            self.calls += 1
            return self.obs, 2, True, False

        def close(self):
            pass

    monkeypatch.setattr(module.FixedSceneRoboTwinEnv, "_make_slot", lambda *a: Slot())
    env = module.FixedSceneRoboTwinEnv(cfg, 1, 0, 1, None)
    yield env
    env.close()


def test_strict_replay_identity_and_terminal_padding(fixed_env):
    env = fixed_env
    obs, _ = env.reset()
    assert env.reset_state_ids.tolist() == [101]
    assert env._elapsed_steps.tolist() == [0]
    with pytest.raises(RuntimeError, match="incomplete"):
        env.reset()
    result = env.chunk_step(np.zeros((1, 4, 14)))
    assert result[2].any()
    env.chunk_step(np.zeros((1, 4, 14)))
    assert env._slots[0].calls == 1
    obs, _ = env.reset()
    assert env.reset_state_ids.tolist() == [205]
    assert obs["task_descriptions"] == ["prompt 205"]
    env.chunk_step(np.zeros((1, 4, 14)))
    with pytest.raises(RuntimeError, match="exhausted"):
        env.reset()
    records = [json.loads(line) for line in env.record_path.read_text().splitlines()]
    assert [r["simulation_steps"] for r in records] == [2, 2]
    assert [r["pool_index"] for r in records] == [20, 21]
    checked_metrics(
        {"num_trajectories": 2, "success_once": 1, "episode_len": 2}, records, 2
    )
    with pytest.raises(RuntimeError, match="success metric"):
        checked_metrics(
            {"num_trajectories": 2, "success_once": 0, "episode_len": 2}, records, 2
        )


def test_scene_mismatch_stops_before_inference(fixed_env):
    fixed_env.entries[0]["episode_info"] = {"side": "right"}
    with pytest.raises(RuntimeError, match="placeholders"):
        fixed_env.reset()
    assert not fixed_env.record_path.exists()


def test_factory_keeps_training_unchanged():
    assert (
        get_env_cls("robotwin", {"task_names": ["open_laptop"]}) is RoboTwinMultiTaskEnv
    )
    assert (
        get_env_cls("robotwin", {"fixed_evaluation_pool": "pool.json"})
        is module.FixedSceneRoboTwinEnv
    )


def test_eval_config_pins_runtime_and_physical_gpu(monkeypatch, tmp_path):
    import sys

    from rlinf.scheduler.cluster.config import ClusterConfig
    from toolkits.robotwin_mopd.worker_evaluate import compose_config

    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("ASSETS_PATH", str(tmp_path / "assets"))
    args = SimpleNamespace(
        output=tmp_path / "eval",
        checkpoint=tmp_path / "student",
        task="open_laptop",
        gpu=1,
        episodes=2,
        override=[],
        norm_stats={"open_laptop": str(tmp_path / "norm_stats.json")},
    )
    pool = {
        "protocol": {
            "task_horizons": {"open_laptop": 700},
            "task_names": ["open_laptop"],
        }
    }
    args.pool = tmp_path / "pool.json"
    cfg = compose_config(args, pool)
    cluster = ClusterConfig.from_dict_cfg(cfg.cluster)
    assert (
        cluster.node_groups[0].env_configs[0].python_interpreter_path == sys.executable
    )
    assert cfg.cluster.component_placement["env,rollout"].placement == "1"
    assert cfg.cluster.component_placement["env,rollout"].node_group == "robotwin_eval"
    assert cfg.runner.only_eval and "train" not in cfg.env
    assert "expert_models" not in cfg.rollout
    assert cfg.rollout.model.num_action_chunks == 25
    assert cfg.rollout.model.num_steps == 10
    assert cfg.env.eval.is_eval and cfg.env.eval.total_num_envs == 1
    assert cfg.env.eval.rollout_epoch == 2
    assert cfg.env.eval.max_steps_per_rollout_epoch == 700
