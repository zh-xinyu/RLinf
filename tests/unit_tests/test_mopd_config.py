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


def test_full_mopd_has_no_silent_foundation_checkpoint_fallback():
    from omegaconf.errors import InterpolationResolutionError

    with pytest.raises(InterpolationResolutionError, match="OPD_STUDENT_CHECKPOINT"):
        compose_mopd_config("robocasa365_mopd_gr00t_full")
