# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compose the isolated pilot without launching Ray or loading model weights."""

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_flow_refresh_cfg, validate_rollout_allocation_cfg


@pytest.fixture
def pilot_cfg(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(repo / "examples/embodiment"))
    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("ASSETS_PATH", str(tmp_path / "runtime/repos/RoboTwin"))
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("OPD_TEACHER_REFERENCE", raising=False)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05")
    OmegaConf.resolve(cfg)
    return cfg


def test_pilot_config_matches_task_routing_and_refresh_contract(pilot_cfg):
    validate_flow_refresh_cfg(pilot_cfg)
    validate_rollout_allocation_cfg(pilot_cfg)
    tasks = list(pilot_cfg.env.train.task_names)
    assert tasks == list(pilot_cfg.rollout.expert_models)
    assert tasks == list(pilot_cfg.actor.model.opd_task_norm_stats)
    assert set(tasks) == set(pilot_cfg.env.train.task_horizons)
    assert len(tasks) == 4
    assert pilot_cfg.env.train.total_num_envs % len(tasks) == 0
    assert (
        pilot_cfg.actor.global_batch_size % (8 * pilot_cfg.actor.micro_batch_size) == 0
    )
    assert pilot_cfg.actor.model.openpi.train_expert_only is False
    assert pilot_cfg.actor.model.openpi.action_horizon == 50
    assert pilot_cfg.actor.model.openpi.action_chunk == 25
    assert pilot_cfg.actor.model.openpi.action_env_dim == 14


def test_subset_launcher_replaces_all_physical_gpu_placement(
    pilot_cfg, monkeypatch, tmp_path
):
    """Composed child config must not schedule the other physical GPUs."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(Path(sys.prefix).parent))
    # run.py intentionally enforces the isolated runtime path contract.
    if Path(sys.prefix).name != "venv":
        pytest.skip("Launcher integration requires the isolated runtime venv")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolkits.robotwin_mopd.run",
            "--run-id",
            "subset_check",
            "--output-root",
            str(tmp_path),
            "--num-gpus",
            "4",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cfg = OmegaConf.create(result.stdout)
    assert dict(cfg.cluster.component_placement) == {
        "actor": "4,5,6,7",
        "env": "4,5,6,7",
        "rollout": "4,5,6,7",
    }
    assert not (tmp_path / "subset_check").exists()


def test_smoke_overrides_retain_valid_round_boundaries(pilot_cfg):
    pilot_cfg.runner.max_steps = 4
    pilot_cfg.runner.save_interval = 4
    pilot_cfg.env.train.total_num_envs = 32
    validate_flow_refresh_cfg(pilot_cfg)
    validate_rollout_allocation_cfg(pilot_cfg)


def test_adaptive_requires_teacher_calibration(pilot_cfg):
    pilot_cfg.runner.rollout_allocation.mode = "adaptive"
    with pytest.raises(ValueError, match="teacher_reference_file"):
        validate_rollout_allocation_cfg(pilot_cfg)
    pilot_cfg.runner.rollout_allocation.teacher_reference_file = "/test/reference.json"
    validate_rollout_allocation_cfg(pilot_cfg)


@pytest.mark.parametrize("field", ["max_steps", "save_interval"])
def test_save_and_stop_cannot_split_refresh_round(pilot_cfg, field):
    pilot_cfg.runner[field] = 3
    with pytest.raises(ValueError, match="divisible"):
        validate_flow_refresh_cfg(pilot_cfg)


def test_quota_plan_requires_enabled_allocation(pilot_cfg):
    pilot_cfg.runner.rollout_allocation.enabled = False
    with pytest.raises(ValueError, match="suite_quota requires"):
        validate_rollout_allocation_cfg(pilot_cfg)


def test_checkpointed_openpi_requires_joint_training_path(pilot_cfg):
    pilot_cfg.actor.model.openpi.nft_joint_forward = False
    with pytest.raises(ValueError, match="nft_joint_forward"):
        validate_flow_refresh_cfg(pilot_cfg)


def test_full_horizon_opt_in_keeps_execution_config(pilot_cfg):
    assert pilot_cfg.actor.model.openpi.get("nft_full_horizon", False) is False
    OmegaConf.update(
        pilot_cfg, "actor.model.openpi.nft_full_horizon", True, force_add=True
    )
    validate_flow_refresh_cfg(pilot_cfg)
    assert pilot_cfg.actor.model.openpi.action_horizon == 50
    assert pilot_cfg.actor.model.openpi.action_chunk == 25
    assert pilot_cfg.actor.model.num_action_chunks == 25
    assert pilot_cfg.actor.model.openpi.action_env_dim == 14


@pytest.mark.parametrize(
    "key,value",
    [
        ("algorithm.reward_type", "action_level"),
        ("algorithm.loss_type", "nft"),
        ("actor.model.openpi.nft_full_horizon", "yes"),
    ],
)
def test_full_horizon_rejects_incompatible_masks_or_loss(pilot_cfg, key, value):
    OmegaConf.update(
        pilot_cfg, "actor.model.openpi.nft_full_horizon", True, force_add=True
    )
    OmegaConf.update(pilot_cfg, key, value)
    with pytest.raises(ValueError, match="nft_full_horizon"):
        validate_flow_refresh_cfg(pilot_cfg)


@pytest.mark.parametrize("reuse", [1, 2])
def test_explicit_checkpoint_schedule_respects_complete_refresh_rounds(
    pilot_cfg, reuse
):
    pilot_cfg.algorithm.flow_refresh_updates = reuse
    pilot_cfg.runner.max_steps = 100 * reuse
    pilot_cfg.runner.save_interval = -1
    steps = [r * reuse for r in (10, 20, 40, 60, 80, 100)]
    OmegaConf.update(pilot_cfg, "runner.save_steps", steps, force_add=True)
    validate_flow_refresh_cfg(pilot_cfg)


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [0, 80],
        [20, 20, 80],
        [40, 20, 80],
        [20, 40],
        [3, 80],
        [20, 82],
        [True, 80],
        [20.0, 80],
        "20,80",
    ],
)
def test_explicit_checkpoint_schedule_rejects_invalid_steps(pilot_cfg, steps):
    pilot_cfg.runner.max_steps = 80
    pilot_cfg.runner.save_interval = -1
    OmegaConf.update(pilot_cfg, "runner.save_steps", steps, force_add=True)
    with pytest.raises(ValueError, match="save_steps"):
        validate_flow_refresh_cfg(pilot_cfg)


def test_smoke_replaces_long_formal_checkpoint_schedule(
    pilot_cfg, monkeypatch, tmp_path
):
    if Path(sys.prefix).name != "venv":
        pytest.skip("Launcher integration requires the isolated runtime venv")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(Path(sys.prefix).parent))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolkits.robotwin_mopd.run",
            "--run-id",
            "formal_schedule_smoke",
            "--output-root",
            str(tmp_path),
            "--num-gpus",
            "4",
            "--dry-run",
            "--smoke",
            "runner.max_steps=200",
            "runner.save_interval=-1",
            "+runner.save_steps=[20,40,80,120,160,200]",
            "actor.global_batch_size=256",
            "actor.micro_batch_size=16",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cfg = OmegaConf.create(result.stdout)
    assert cfg.runner.max_steps == 2
    assert list(cfg.runner.save_steps) == [2]
    assert cfg.runner.save_interval == -1
    assert cfg.actor.global_batch_size == 256
    assert cfg.actor.micro_batch_size == 16
    assert not (tmp_path / "formal_schedule_smoke").exists()
