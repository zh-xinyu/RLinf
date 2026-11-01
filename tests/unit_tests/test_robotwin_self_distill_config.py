# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compose self-distillation controls without Ray, CUDA, or model loading."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_flow_refresh_cfg, validate_rollout_allocation_cfg
from rlinf.utils.rollout_allocation import RolloutAllocator
from toolkits.robotwin_mopd.self_distill import TASK, self_distill_overrides


@pytest.mark.parametrize("trajectories", [4, 8])
def test_single_task_control_is_exact_and_keeps_training_path(
    monkeypatch, tmp_path, trajectories
):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(repo / "examples/embodiment"))
    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("ASSETS_PATH", str(tmp_path / "assets"))
    checkpoint = tmp_path / "models" / TASK
    overrides = self_distill_overrides(checkpoint, trajectories=trajectories)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05", overrides=overrides)
    OmegaConf.resolve(cfg)
    validate_flow_refresh_cfg(cfg)
    validate_rollout_allocation_cfg(cfg)
    assert list(cfg.rollout.expert_models) == [TASK]
    assert list(cfg.actor.model.opd_task_norm_stats) == [TASK]
    assert (
        cfg.actor.model.model_path
        == cfg.rollout.expert_models[TASK].model_path
        == str(checkpoint)
    )
    assert cfg.rollout.model.model_path == str(checkpoint)
    for mode in ("train", "eval"):
        assert list(cfg.env[mode].task_names) == [TASK]
        assert dict(cfg.env[mode].task_horizons) == {TASK: 400}
        assert cfg.env[mode].max_steps_per_rollout_epoch == 400
        assert cfg.env[mode].total_num_envs == trajectories
    assert cfg.algorithm.loss_type == "analytic_local_kl"
    assert cfg.algorithm.flow_refresh_updates == 4
    assert cfg.actor.model.openpi.nft_full_horizon
    assert cfg.actor.model.openpi.nft_joint_forward
    assert cfg.actor.model.openpi.action_horizon == 50
    assert cfg.actor.model.num_action_chunks == 25
    assert cfg.actor.cast_trainable_params_to_fp32
    assert cfg.actor.fsdp_config.amp_autocast.enabled
    assert all(
        cfg.actor.fsdp_config.mixed_precision[key] is None
        for key in ("param_dtype", "reduce_dtype", "buffer_dtype")
    )
    assert cfg.actor.global_batch_size == 128
    assert cfg.actor.micro_batch_size == 2
    assert cfg.actor.optim.lr == 1e-5
    assert cfg.actor.optim.lr_warmup_steps == 4
    assert cfg.runner.max_steps // cfg.algorithm.flow_refresh_updates == 5
    allocator = RolloutAllocator(
        [{"task_id": 0, "task": TASK, "suite": TASK, "horizon": 400}],
        workers=4,
        envs_per_worker=trajectories // 4,
        epochs=1,
        config=cfg.runner.rollout_allocation,
    )
    plan = allocator.plan()
    assert len(plan) == trajectories
    assert {row["task_id"] for row in plan} == {0}
    assert {row["worker"] for row in plan} == set(range(4))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"updates": 19},
        {"batch_size": 127},
        {"micro_batch_size": 0},
        {"save_interval": 5},
        {"lr": -1},
    ],
)
def test_invalid_control_shape_fails_before_launcher(tmp_path, kwargs):
    with pytest.raises(ValueError):
        self_distill_overrides(tmp_path, **kwargs)
