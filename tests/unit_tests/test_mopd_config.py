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

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from toolkits.mopd.probe_precision_init import REPO, compose_mopd_config


@pytest.fixture(autouse=True)
def isolated_paths(monkeypatch, tmp_path):
    """Keep config composition independent of a developer's machine paths."""
    monkeypatch.setenv("OPD_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("EMBODIED_PATH", str(REPO / "examples/embodiment"))


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


def test_standalone_eval_paths_resolve_from_local_runtime(tmp_path):
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "evaluations/robocasa365")
    ):
        cfg = compose(config_name="robocasa365_mopd_gr00t_eval")
    OmegaConf.resolve(cfg)
    assert cfg.rollout.model.model_path == str(
        tmp_path / "models/pretraining/checkpoint-80000"
    )
    assert cfg.runner.logger.log_path == str(tmp_path / "run")
