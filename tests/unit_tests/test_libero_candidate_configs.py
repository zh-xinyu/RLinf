# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate candidate matrix invariants without Ray, GPU, or checkpoints."""

from itertools import combinations
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError
from test_libero_independent_eval_config import validate_without_ray

REPO = Path(__file__).resolve().parents[2]


def candidate(monkeypatch, tmp_path, arm):
    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    for key in (
        "STUDENT_CHECKPOINT",
        "RUN_DIR",
        "LIBERO_SPATIAL_TEACHER",
        "LIBERO_OBJECT_TEACHER",
        "LIBERO_GOAL_TEACHER",
        "LIBERO_LONG_TEACHER",
    ):
        monkeypatch.setenv(f"OPD_{key}", str(tmp_path / key.lower()))
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "examples/embodiment/config")
    ):
        return compose(config_name=f"libero_mopd_candidate_m{arm}")


@pytest.mark.parametrize("arm", range(4))
def test_all_candidates_validate_and_preserve_round_budget(monkeypatch, tmp_path, arm):
    monkeypatch.setenv(
        "OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "dummy_reference.json")
    )
    cfg = candidate(monkeypatch, tmp_path, arm)
    OmegaConf.resolve(cfg)
    cfg = validate_without_ray(monkeypatch, cfg)
    repeats = 1 if arm in (0, 3) else 2
    assert cfg.algorithm.flow_refresh_updates == repeats
    assert cfg.runner.max_steps == 150 * repeats
    assert cfg.runner.save_interval == 25 * repeats
    assert cfg.env.train.total_num_envs == 32 and cfg.env.train.rollout_epoch == 1
    assert cfg.env.train.total_num_envs // 4 == 8
    assert cfg.env.train.task_sampling_strategy == "suite_quota"
    assert cfg.env.train.freeze_finished_episodes
    assert cfg.actor.global_batch_size == 64 and cfg.actor.micro_batch_size == 1
    assert cfg.actor.optim.lr == cfg.actor.optim.value_lr == 3e-6
    assert cfg.actor.optim.lr_warmup_steps == 4
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    assert cfg.rollout.model.action_head_precision == cfg.actor.model.action_head_precision == "fp32"
    assert cfg.weight_syncer.type == "bucket"
    assert cfg.weight_syncer.bucket.bucket_size == 32 * 1024**2
    assert cfg.weight_syncer.bucket.bucket_dtype is None
    assert cfg.weight_syncer.bucket.load_instant
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
    assert cfg.runner.logger.project_name == "LIBERO_MOPD_Candidates"
    assert list(cfg.runner.logger.logger_backends) == ["tensorboard"]
    adaptive = arm in (2, 3)
    assert cfg.algorithm.sync_suite_loss_weights is adaptive
    assert cfg.runner.rollout_allocation.mode == ("adaptive" if adaptive else "uniform")
    assert cfg.actor.seed == cfg.env.train.seed == cfg.runner.rollout_allocation.seed == 0
    assert cfg.runner.rollout_allocation.enabled
    assert cfg.runner.rollout_allocation.max_quota_change == 3
    assert cfg.runner.val_check_interval == -1  # Separate independent evaluator.


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    return result


def test_pairwise_differences_are_only_matrix_factors(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "dummy_reference.json")
    )
    configs = [
        flatten(
            OmegaConf.to_container(candidate(monkeypatch, tmp_path, arm), resolve=True)
        )
        for arm in range(4)
    ]
    for left, right in combinations(range(4), 2):
        differing = {
            key
            for key in configs[left] | configs[right]
            if configs[left].get(key) != configs[right].get(key)
        }
        expected = {"runner.logger.experiment_name"}
        if (left in (0, 3)) != (right in (0, 3)):
            expected |= {
                "algorithm.flow_refresh_updates",
                "runner.max_steps",
                "runner.save_interval",
            }
        if (left in (2, 3)) != (right in (2, 3)):
            expected |= {
                "algorithm.sync_suite_loss_weights",
                "runner.rollout_allocation.mode",
                "runner.rollout_allocation.teacher_reference_file",
            }
        assert differing == expected


@pytest.mark.parametrize("arm", range(4))
def test_only_adaptive_arms_require_reference_environment_variable(
    monkeypatch, tmp_path, arm
):
    monkeypatch.delenv("OPD_LIBERO_TEACHER_REFERENCE", raising=False)
    cfg = candidate(monkeypatch, tmp_path, arm)
    if arm in (2, 3):
        with pytest.raises(
            InterpolationResolutionError, match="OPD_LIBERO_TEACHER_REFERENCE"
        ):
            OmegaConf.resolve(cfg)
    else:
        OmegaConf.resolve(cfg)
        assert cfg.runner.rollout_allocation.teacher_reference_file is None
