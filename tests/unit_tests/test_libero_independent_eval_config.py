# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compose and validate the standalone LIBERO GR00T evaluator without Ray/GPU."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]


def composed(monkeypatch, tmp_path, overrides=()):
    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    monkeypatch.setenv("OPD_STUDENT_CHECKPOINT", str(tmp_path / "base_grouped"))
    monkeypatch.setenv(
        "OPD_EVAL_WEIGHTS",
        str(tmp_path / "global_step_50/actor/model_state_dict/full_weights.pt"),
    )
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "eval"))
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "evaluations/libero")
    ):
        cfg = compose(
            config_name="libero_four_suite_gr00t_mopd_eval", overrides=list(overrides)
        )
    OmegaConf.resolve(cfg)
    return cfg


def validate_without_ray(monkeypatch, cfg):
    import rlinf.config as config
    from rlinf.scheduler.cluster.config import ClusterConfig
    from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo
    from rlinf.scheduler.hardware import Accelerator, HardwareInfo, HardwareResource

    node = NodeInfo(
        node_labels=[],
        node_rank=0,
        ray_id="fake",
        node_ip="127.0.0.1",
        num_cpus=32,
        python_interpreter_path="/fixture/python",
        default_env_vars={},
        env_vars={},
        hardware_resources=[
            HardwareResource(
                type=Accelerator.HW_TYPE,
                infos=[HardwareInfo(type=Accelerator.HW_TYPE, model="NV_GPU:Mock")] * 8,
            )
        ],
    )
    group = NodeGroupInfo(label="eval", nodes=[node], hardware_type=Accelerator.HW_TYPE)
    cluster = SimpleNamespace(
        get_node_group=lambda label=None: group, get_node_info=lambda rank: node
    )
    monkeypatch.setattr(config, "Cluster", lambda *args, **kwargs: cluster)
    ClusterConfig.from_dict_cfg(cfg.cluster)
    return config.validate_cfg(cfg)


def test_actual_eval_entrypoint_config_has_complete_grouped_checkpoint_contract(
    monkeypatch, tmp_path
):
    cfg = validate_without_ray(monkeypatch, composed(monkeypatch, tmp_path))
    assert cfg.runner.only_eval and cfg.runner.task_type == "embodied_eval"
    assert "actor" not in cfg and "train" not in cfg.env
    assert cfg.runner.ckpt_path.endswith("/actor/model_state_dict/full_weights.pt")
    model = cfg.rollout.model
    assert model.obs_converter_type == "libero_grouped"
    assert model.data_config_class.endswith(":LiberoGroupedDataConfig")
    assert (
        model.embodiment_tag == "new_embodiment" and model.embodiment_projector_id == 31
    )
    assert model.opd_fixed_text_padding is False and model.disable_image_tiling is True
    assert model.action_head_precision == "fp32" and model.precision == "bf16"
    assert model.inference_autocast_dtype == "bf16"
    assert model.num_action_chunks == 8 and model.denoising_steps == 4
    assert model.rl_head_config.disable_dropout is True
    assert cfg.env.eval.reset_settle_steps == 10 and cfg.env.eval.reset_gripper_open
    assert cfg.runner.per_worker_log and cfg.runner.per_worker_log_path.endswith(
        "worker_logs"
    )


def test_four_suite_order_and_worst_case_budget_cover_2000_unique_trials(
    monkeypatch, tmp_path
):
    from rlinf.envs.libero.multi_suite import CombinedLiberoBenchmark

    cfg = composed(monkeypatch, tmp_path)
    names = list(cfg.env.eval.task_suite_names)
    catalog = CombinedLiberoBenchmark(
        names,
        lambda name: SimpleNamespace(
            get_num_tasks=lambda: 10,
            get_task=lambda task: (name, task),
            get_task_init_states=lambda task: np.zeros((50, 1)),
        ),
    )
    assert catalog.get_num_tasks() == 40
    assert [catalog.get_task(task) for task in (0, 10, 20, 30)] == [
        (name, 0) for name in names
    ]
    # Exact eval ordering: interleaved task IDs, then round-robin worker ownership.
    pairs = [(task, trial) for trial in range(50) for task in range(40)]
    rank_pools = [pairs[rank::4] for rank in range(4)]
    assert len({pair for pool in rank_pools for pair in pool}) == 2000
    for pool in rank_pools:
        assert len(pool) == 500
        worst_case = math.ceil(len(pool) / 4) * max(
            cfg.env.eval.task_suite_horizons.values()
        )
        assert worst_case <= cfg.env.eval.max_steps_per_rollout_epoch
    assert cfg.env.eval.freeze_finished_episodes is True
    assert cfg.env.eval.auto_reset and not cfg.env.eval.ignore_terminations
    assert cfg.env.eval.task_sampling_strategy == "ordered"
    assert (
        cfg.env.eval.max_steps_per_rollout_epoch % cfg.rollout.model.num_action_chunks
        == 0
    )


def test_base_only_override_and_invalid_chunk_budget(monkeypatch, tmp_path):
    cfg = composed(monkeypatch, tmp_path, ["runner.ckpt_path=null"])
    assert validate_without_ray(monkeypatch, cfg).runner.ckpt_path is None
    broken = composed(
        monkeypatch, tmp_path, ["env.eval.max_steps_per_rollout_epoch=65001"]
    )
    with pytest.raises(AssertionError, match="divisible"):
        validate_without_ray(monkeypatch, broken)


def test_real_eval_ordered_pool_exhaustion_and_unique_completion_count():
    pytest.importorskip("libero")
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import (
        build_interleaved_eval_reset_state_ids,
        distribute_reset_state_ids_round_robin,
        record_completed_episode_task_stats,
    )

    trial_counts = [50] * 40
    ids = build_interleaved_eval_reset_state_ids(trial_counts, np.cumsum(trial_counts))
    pools = distribute_reset_state_ids_round_robin(ids, 4)
    assert pools.shape == (4, 500)
    logger = SimpleNamespace(info=lambda message: None, warning=lambda message: None)
    combined_counts = np.zeros(40, dtype=int)
    observed = set()
    for pool in pools:
        env = SimpleNamespace(
            specific_reset_id=None, is_eval=True, _eval_reset_pool=pool, start_idx=0
        )
        seen, stats = set(), {}
        for _ in range(125):
            allocated = LiberoEnv._get_ordered_reset_state_ids(env, 4)
            assert np.all(allocated >= 0)
            tasks, trials = allocated // 50, allocated % 50
            counted = record_completed_episode_task_stats(
                np.arange(4),
                {"episode": {"success_once": np.ones(4, dtype=bool)}},
                tasks,
                trials,
                4,
                seen,
                stats,
                logger,
            )
            assert counted.all()
            observed.update(zip(tasks.tolist(), trials.tolist(), strict=True))
        assert (LiberoEnv._get_ordered_reset_state_ids(env, 4) == -1).all()
        # Completed slots can be observed again by fixed-budget workers, but no
        # episode may be counted twice after their reset pool is exhausted.
        duplicate = record_completed_episode_task_stats(
            np.arange(4),
            {"episode": {"success_once": np.ones(4, dtype=bool)}},
            tasks,
            trials,
            4,
            seen,
            stats,
            logger,
        )
        assert not duplicate.any() and len(seen) == 500
        for task, result in stats.items():
            combined_counts[task] += result["total"]
    assert len(observed) == 2000
    np.testing.assert_array_equal(combined_counts, np.full(40, 50))


def test_exhausted_eval_auto_reset_retains_terminal_observation_without_restart():
    pytest.importorskip("libero")
    import torch

    from rlinf.envs.libero.libero_env import LiberoEnv

    env = SimpleNamespace(
        num_envs=2,
        task_ids=np.array([0, 1]),
        trial_ids=np.array([49, 49]),
        _eval_seen_trials=set(),
        _task_success_stats={},
        _get_ordered_reset_state_ids=lambda count: np.full(count, -1),
    )

    def forbidden_reset(**kwargs):
        pytest.fail("Exhausted evaluation must not restart terminal environments")

    env.reset = forbidden_reset
    obs = {"states": torch.zeros(2, 8)}
    infos = {"episode": {"success_once": torch.tensor([True, False])}}
    first_obs, first_info, first_mask = LiberoEnv._handle_eval_auto_reset(
        env, np.ones(2, dtype=bool), obs, infos
    )
    assert first_obs is obs and first_mask.all() and first_info["_final_info"].all()
    second_obs, second_info, second_mask = LiberoEnv._handle_eval_auto_reset(
        env, np.ones(2, dtype=bool), obs, infos
    )
    assert (
        second_obs is obs
        and not second_mask.any()
        and not second_info["_final_info"].any()
    )
