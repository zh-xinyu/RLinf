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
"""Check that experiment grouping reaches the W&B initialization call."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.utils.metric_logger import MetricLogger


def test_wandb_group_passed_to_init(monkeypatch, tmp_path):
    calls, definitions, logs = [], [], []
    fake = SimpleNamespace(
        init=lambda **kw: calls.append(kw),
        define_metric=lambda *a, **kw: definitions.append((a, kw)),
        log=lambda **kw: logs.append(kw),
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = OmegaConf.create(
        {
            "runner": {
                "step_unit": "optimizer_update",
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "Formal_Robocasa_Groot_MOPD",
                    "experiment_name": "refresh-test",
                    "wandb_group": "mopd-refresh",
                    "logger_backends": ["wandb"],
                },
            }
        }
    )
    logger = MetricLogger(cfg)
    assert calls[0]["group"] == "mopd-refresh"
    assert calls[0]["project"] == "Formal_Robocasa_Groot_MOPD"
    assert (("rollout_success/*",), {"step_metric": "rollout_count"}) in definitions
    logger.log({"train/actor/velocity_mse": 0.1}, 2)
    logger.log({"rollout_count": 1}, 2)
    assert len(logs) == 2
    assert all(call["step"] == 2 and call["commit"] is False for call in logs)
    logger.finish()


@pytest.mark.parametrize("override", [None, "pilot_action11_clean"])
def test_wandb_run_name_preserves_worker_names(monkeypatch, tmp_path, override):
    calls = []
    fake = SimpleNamespace(
        init=lambda **kw: calls.append(kw),
        log=lambda **kw: None,
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    logger_config = {
        "log_path": str(tmp_path),
        "project_name": "libero_gr00t_pilot",
        "experiment_name": "original-experiment",
        "wandb_group": "clean_runtime_lr_probe",
        "logger_backends": ["wandb"],
    }
    if override is not None:
        logger_config["wandb_run_name"] = override
    cfg = OmegaConf.create(
        {"runner": {"per_worker_log": True, "logger": logger_config}}
    )
    logger = MetricLogger(cfg)
    logger.log({"loss": 0.1}, 1, worker_group_name="actor", rank=0)
    logger.log({"loss": 0.2}, 1, worker_group_name="actor", rank=1)
    logger.log({"loss": 0.3}, 2, worker_group_name="actor", rank=0)
    name = override or "original-experiment"
    assert [call["name"] for call in calls] == [
        name,
        f"{name}-actor-rank_0",
        f"{name}-actor-rank_1",
    ]
    assert all(call["project"] == "libero_gr00t_pilot" for call in calls)
    assert all(call["group"] == "clean_runtime_lr_probe" for call in calls)
    assert logger.experiment_name == "original-experiment"
    assert calls[1]["dir"].endswith("worker_logs/actor/rank_0/wandb/")
    logger.finish()


@pytest.mark.parametrize("use_override", [False, True])
def test_libero_candidate_wandb_defaults(monkeypatch, use_override):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/libero_mopd_candidate_common.yaml"
    )
    for name in ("OPD_WANDB_RUN_NAME", "OPD_WANDB_GROUP"):
        monkeypatch.delenv(name, raising=False)
    if use_override:
        monkeypatch.setenv("OPD_WANDB_RUN_NAME", "pilot_action11_clean")
        monkeypatch.setenv("OPD_WANDB_GROUP", "clean_runtime_lr_probe")
    logger_cfg = OmegaConf.to_container(
        OmegaConf.load(config_path).runner.logger, resolve=True
    )
    assert logger_cfg["project_name"] == "libero_gr00t_pilot"
    assert logger_cfg["logger_backends"] == ["tensorboard", "wandb"]
    assert logger_cfg["wandb_run_name"] == (
        "pilot_action11_clean" if use_override else "libero_gr00t_pilot"
    )
    assert logger_cfg["wandb_group"] == (
        "clean_runtime_lr_probe" if use_override else "libero_gr00t_pilot"
    )
