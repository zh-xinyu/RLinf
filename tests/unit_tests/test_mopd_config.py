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
"""Configuration tests for GR00T RoboCasa365 MOPD pilots."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]


def test_adaptive_sync_config(monkeypatch):
    from rlinf.config import validate_suite_distillation_cfg

    monkeypatch.setenv("OPD_PROBE_CHECKPOINT", "/checkpoint/u40")
    monkeypatch.setenv("OPD_TEACHER_REFERENCE", "/calibration/reference.json")
    monkeypatch.setenv("OPD_STUDENT_CHECKPOINT", "/checkpoint/c10")
    cfg = compose_mopd_config("robocasa365_mopd_adaptive_sync_r2_gr00t_full")
    validate_suite_distillation_cfg(cfg)
    assert cfg.algorithm.sync_suite_loss_weights
    assert cfg.algorithm.get("suite_loss_weights") is None
    assert cfg.runner.rollout_allocation.rho == 0.5
    assert cfg.actor.global_batch_size == 2304 and cfg.actor.micro_batch_size == 16
    assert cfg.runner.max_steps == 200 and cfg.runner.save_interval == 10
    OmegaConf.set_struct(cfg, False)
    cfg.algorithm.suite_loss_weights = dict.fromkeys(cfg.rollout.expert_models, 1.0)
    with pytest.raises(ValueError, match="exclusive"):
        validate_suite_distillation_cfg(cfg)
    del cfg.algorithm.suite_loss_weights
    cfg.runner.rollout_allocation.enabled = False
    with pytest.raises(ValueError, match="synchronous allocation"):
        validate_suite_distillation_cfg(cfg)


def compose_mopd_config(config_name, overrides=None):
    with initialize_config_dir(
        config_dir=str(REPO / "examples/embodiment/config"), version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture(autouse=True)
def isolated_paths(monkeypatch, tmp_path):
    """Keep config composition independent of a developer's machine paths."""
    monkeypatch.setenv("OPD_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    monkeypatch.delenv("OPD_STUDENT_CHECKPOINT", raising=False)


@pytest.mark.parametrize(
    "config_name",
    ("robocasa365_mopd_gr00t", "robocasa365_mopd_m4_gr00t"),
)
def test_mopd_constant_lr_is_independent_of_runner_budget(config_name):
    cfg = compose_mopd_config(config_name)

    assert cfg.runner.max_steps == 25
    assert cfg.actor.optim.lr == pytest.approx(2.0e-6)
    assert cfg.actor.optim.lr_scheduler == "constant"
    assert cfg.actor.optim.lr_warmup_steps == 0
    assert "total_training_steps" not in cfg.actor.optim
    assert "lr_warmup_steps_ratio" not in cfg.actor.optim


@pytest.mark.parametrize(
    "config_name", ("robocasa365_mopd_gr00t", "robocasa365_mopd_m4_gr00t")
)
def test_mopd_paths_resolve_from_local_runtime(config_name, tmp_path):
    cfg = compose_mopd_config(config_name)
    root = tmp_path / "models"
    student = str(root / "pretraining/checkpoint-80000")
    assert cfg.actor.model.model_path == student
    assert cfg.rollout.model.model_path == student
    assert cfg.runner.logger.log_path == str(tmp_path / "run")
    for suite, teacher in cfg.rollout.expert_models.items():
        assert teacher.model_path == str(
            root / "target_posttraining" / suite / "checkpoint-60000"
        )


def test_full_mopd_requires_explicit_student_and_uses_full_suite_recipe():
    cfg = compose_mopd_config(
        "robocasa365_mopd_gr00t_full", ["actor.model.model_path=/cold-start/checkpoint"]
    )
    assert (
        cfg.actor.model.model_path
        == cfg.rollout.model.model_path
        == "/cold-start/checkpoint"
    )
    assert cfg.env.train.task_filter is None
    assert len(cfg.env.train.task_soup) == 3
    assert cfg.env.train.task_sampling_strategy == "suite_balanced"
    assert cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch == 96
    assert cfg.actor.global_batch_size == 8192 and cfg.actor.micro_batch_size == 32
    assert cfg.actor.optim.lr_scheduler == "constant"
    assert cfg.runner.val_check_interval == -1 and cfg.runner.save_interval == 25
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None


def test_controlled_r2_recipe_uses_update_budget():
    from rlinf.config import validate_flow_refresh_cfg

    cfg = compose_mopd_config(
        "robocasa365_mopd_refresh_r2_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint"],
    )
    validate_flow_refresh_cfg(cfg)
    assert cfg.runner.step_unit == "optimizer_update"
    assert cfg.runner.max_steps == 200
    assert cfg.algorithm.flow_refresh_updates == 2
    assert cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch == 48
    assert cfg.actor.global_batch_size == 2048
    assert cfg.actor.optim.lr_warmup_steps == 10
    assert cfg.actor.optim.lr == 3e-5
    cfg.runner.max_steps = 199
    with pytest.raises(ValueError, match="divisible"):
        validate_flow_refresh_cfg(cfg)
    cfg.runner.max_steps = 200
    cfg.runner.resume_dir = "/legacy/global_step_25"
    with pytest.raises(ValueError, match="fresh run"):
        validate_flow_refresh_cfg(cfg)


def test_controlled_base_matches_refresh_except_r():
    from rlinf.config import validate_flow_refresh_cfg

    overrides = ["actor.model.model_path=/cold-start/checkpoint"]
    base = compose_mopd_config("robocasa365_mopd_base_controlled_gr00t_full", overrides)
    refresh = compose_mopd_config("robocasa365_mopd_refresh_r2_gr00t_full", overrides)
    validate_flow_refresh_cfg(base)
    assert base.algorithm.flow_refresh_updates == 1
    base.algorithm.flow_refresh_updates = refresh.algorithm.flow_refresh_updates
    assert OmegaConf.to_container(base) == OmegaConf.to_container(refresh)


def test_full_mopd_has_no_silent_foundation_checkpoint_fallback():
    from omegaconf.errors import InterpolationResolutionError

    with pytest.raises(InterpolationResolutionError, match="OPD_STUDENT_CHECKPOINT"):
        compose_mopd_config("robocasa365_mopd_gr00t_full")


def test_refresh_recipe_keeps_base_optimization_and_counts_physical_rounds():
    from rlinf.config import validate_flow_refresh_cfg

    cfg = compose_mopd_config(
        "robocasa365_mopd_refresh_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint"],
    )
    validate_flow_refresh_cfg(cfg)
    assert cfg.algorithm.flow_refresh_updates == 4
    assert cfg.algorithm.update_epoch == 1
    assert cfg.runner.max_steps == 50 and cfg.runner.save_interval == 10
    assert cfg.actor.global_batch_size == 8192 and cfg.actor.micro_batch_size == 32
    assert cfg.actor.optim.lr_warmup_steps == 3
    assert cfg.actor.optim.lr == pytest.approx(3e-5)


@pytest.mark.parametrize(
    "override",
    [
        "algorithm.flow_refresh_updates=0",
        "algorithm.flow_refresh_updates=1.5",
        "algorithm.update_epoch=4",
        "algorithm.suite_balanced_sampling=false",
        "actor.model.rl_head_config.noise_method=flow_sde",
        "actor.model.rl_head_config.joint_logprob=false",
        "env.train.auto_reset=true",
        "+runner.weight_sync_interval=2",
    ],
)
def test_refresh_rejects_unsupported_combinations(override):
    from rlinf.config import validate_flow_refresh_cfg

    cfg = compose_mopd_config(
        "robocasa365_mopd_refresh_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint", override],
    )
    with pytest.raises(ValueError, match="flow_refresh_updates|Flow refresh requires"):
        validate_flow_refresh_cfg(cfg)


@pytest.mark.parametrize("repeats", [1, 2, 4, 8])
def test_t96_mainline_recipes_share_optimization(repeats):
    from rlinf.config import validate_flow_refresh_cfg

    overrides = ["actor.model.model_path=/cold-start/checkpoint"]
    base = compose_mopd_config(
        "robocasa365_mopd_base_t96_controlled_gr00t_full", overrides
    )
    refresh = compose_mopd_config(
        "robocasa365_mopd_refresh_r2_t96_gr00t_full", overrides
    )
    assert base.algorithm.flow_refresh_updates == 1
    assert refresh.algorithm.flow_refresh_updates == 2
    base.algorithm.flow_refresh_updates = refresh.algorithm.flow_refresh_updates
    assert OmegaConf.to_container(base) == OmegaConf.to_container(refresh)
    refresh.algorithm.flow_refresh_updates = repeats
    validate_flow_refresh_cfg(refresh)
    assert refresh.env.train.total_num_envs * refresh.env.train.rollout_epoch == 96
    assert refresh.actor.global_batch_size == 2048
    assert refresh.runner.max_steps == 200 and refresh.runner.save_interval == 40
    assert refresh.actor.optim.lr_warmup_steps == 10
    assert refresh.algorithm.flow_observation_sampling == "full_pool"


@pytest.mark.parametrize(
    "override, message",
    [
        ("algorithm.flow_reuse_mode=cached", "refresh only"),
        ("algorithm.flow_observation_sampling=fixed_batch", "full_pool"),
        ("+algorithm.flow_probe.enabled=true", "flow_probe"),
        ("+runner.use_training_pipeline=true", "synchronous runner"),
    ],
)
def test_mainline_rejects_experimental_modes(override, message):
    from rlinf.config import validate_flow_refresh_cfg

    cfg = compose_mopd_config(
        "robocasa365_mopd_refresh_r2_t96_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint", override],
    )
    with pytest.raises(ValueError, match=message):
        validate_flow_refresh_cfg(cfg)


def test_adaptive_r2_recipe_and_required_collection_contract(monkeypatch):
    from rlinf.config import validate_flow_refresh_cfg, validate_rollout_allocation_cfg

    monkeypatch.setenv("OPD_TEACHER_REFERENCE", "/calibration/reference.json")
    cfg = compose_mopd_config(
        "robocasa365_mopd_adaptive_r2_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint"],
    )
    validate_flow_refresh_cfg(cfg)
    validate_rollout_allocation_cfg(cfg)
    assert cfg.actor.global_batch_size == 2304
    assert cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch == 96
    assert cfg.algorithm.flow_refresh_updates == 2
    assert cfg.runner.max_steps == 200 and cfg.runner.save_interval == 40
    assert cfg.actor.optim.lr == pytest.approx(3e-5)
    assert cfg.actor.optim.lr_warmup_steps == 10
    assert cfg.runner.logger.project_name == "RoboCasa_MultiTeacher_Diagnostics"
    cfg.runner.overlap_env_bootstrap = True
    with pytest.raises(ValueError, match="no bootstrap prefetch"):
        validate_rollout_allocation_cfg(cfg)
    cfg.runner.overlap_env_bootstrap = False
    cfg.runner.rollout_allocation.teacher_reference_file = None
    with pytest.raises(ValueError, match="calibrated"):
        validate_rollout_allocation_cfg(cfg)
    cfg.runner.rollout_allocation.mode = "uniform"
    validate_rollout_allocation_cfg(cfg)


def test_suite_diagnostics_defaults_and_mixed_compatibility():
    from rlinf.config import validate_suite_distillation_cfg

    cfg = compose_mopd_config(
        "robocasa365_mopd_adaptive_r2_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint"],
    )
    assert cfg.algorithm.suite_batch_order == "grouped"
    assert cfg.algorithm.log_suite_velocity_metrics
    validate_suite_distillation_cfg(cfg)
    cfg.algorithm.suite_batch_order = "mixed"
    validate_suite_distillation_cfg(cfg)
    # Metrics-only mode also works with an action-level validity mask.
    cfg.algorithm.reward_type = "action_level"
    validate_suite_distillation_cfg(cfg)
    cfg.algorithm.suite_balanced_sampling = False
    with pytest.raises(ValueError, match="one batch per update"):
        validate_suite_distillation_cfg(cfg)


@pytest.mark.parametrize(
    "override, message",
    [
        ("algorithm.suite_batch_order=unknown", "suite_batch_order"),
        ("algorithm.log_suite_velocity_metrics=1", "boolean"),
        ("algorithm.loss_type=opd", "routed FSDP M1/M3"),
        ("actor.training_backend=megatron", "routed FSDP M1/M3"),
        ("algorithm.update_epoch=2", "update_epoch=1"),
        ("algorithm.suite_balanced_sampling=false", "balanced sampling"),
        ("algorithm.reward_type=action_level", "chunk_level"),
        ("algorithm.filter_rewards=true", "reward filtering"),
    ],
)
def test_suite_diagnostics_reject_incompatible_config(override, message):
    from rlinf.config import validate_suite_distillation_cfg

    cfg = compose_mopd_config(
        "robocasa365_mopd_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint", override],
    )
    with pytest.raises(ValueError, match=message):
        validate_suite_distillation_cfg(cfg)


def test_suite_weight_probe_config(monkeypatch):
    from rlinf.config import validate_suite_distillation_cfg

    monkeypatch.setenv("OPD_PROBE_CHECKPOINT", "/common/checkpoints/global_step_40")
    cfg = compose_mopd_config(
        "robocasa365_mopd_suite_weight_probe_gr00t_full",
        ["actor.model.model_path=/cold-start/checkpoint"],
    )
    validate_suite_distillation_cfg(cfg)
    assert cfg.algorithm.suite_loss_weights.atomic_seen == 0.5
    assert cfg.runner.max_steps == 120
    assert cfg.actor.global_batch_size == 2304
    assert cfg.actor.micro_batch_size == 32
    cfg.algorithm.log_suite_velocity_metrics = False
    with pytest.raises(ValueError, match="raw suite velocity metrics"):
        validate_suite_distillation_cfg(cfg)
