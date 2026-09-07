# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compose public pi0.5 OPD examples without personal paths or local launchers."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

REPO = Path(__file__).resolve().parents[2]
TRAIN_CONFIGS = (
    "maniskill_veldistill_openpi_pi05",
    "maniskill_m3_openpi_pi05",
    "maniskill_m3_hybrid_openpi_pi05",
    "maniskill_m4_openpi_pi05",
)


@pytest.fixture(autouse=True)
def portable_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    monkeypatch.setenv("OPD_PI05_STUDENT_CHECKPOINT", str(tmp_path / "student"))
    monkeypatch.setenv("OPD_PI05_TEACHER_CHECKPOINT", str(tmp_path / "teacher"))
    monkeypatch.delenv("OPD_RUN_DIR", raising=False)


def compose_config(name, directory="examples/embodiment/config", overrides=None):
    with initialize_config_dir(config_dir=str(REPO / directory), version_base=None):
        cfg = compose(config_name=name, overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize("name", TRAIN_CONFIGS)
def test_training_paths_are_external_inputs(name, tmp_path):
    cfg = compose_config(name)
    assert cfg.actor.model.model_path == str(tmp_path / "student")
    assert cfg.rollout.model.model_path == str(tmp_path / "student")
    assert cfg.rollout.expert_model.model_path == str(tmp_path / "teacher")
    assert cfg.runner.logger.log_path == "./outputs"


def test_eval_needs_only_student_and_honors_output_override(monkeypatch, tmp_path):
    monkeypatch.delenv("OPD_PI05_TEACHER_CHECKPOINT")
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "eval-run"))
    cfg = compose_config(
        "maniskill_veldistill_openpi_pi05_eval", "evaluations/maniskill"
    )
    assert cfg.rollout.model.model_path == str(tmp_path / "student")
    assert cfg.runner.logger.log_path == str(tmp_path / "eval-run")


@pytest.mark.parametrize("variable", ["STUDENT", "TEACHER"])
def test_missing_path_cannot_silently_use_a_developer_machine(monkeypatch, variable):
    key = f"OPD_PI05_{variable}_CHECKPOINT"
    monkeypatch.delenv(key)
    with pytest.raises(InterpolationResolutionError, match=key):
        compose_config(TRAIN_CONFIGS[0])


def test_explicit_paths_do_not_require_environment_variables(monkeypatch):
    monkeypatch.delenv("OPD_PI05_STUDENT_CHECKPOINT")
    monkeypatch.delenv("OPD_PI05_TEACHER_CHECKPOINT")
    cfg = compose_config(
        TRAIN_CONFIGS[0],
        overrides=[
            "actor.model.model_path=/model/student",
            "rollout.model.model_path=/model/student",
            "rollout.expert_model.model_path=/model/teacher",
        ],
    )
    assert cfg.actor.model.model_path == "/model/student"
