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
from types import SimpleNamespace

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
