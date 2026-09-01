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

from toolkits.mopd.probe_precision_init import compose_mopd_config


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
