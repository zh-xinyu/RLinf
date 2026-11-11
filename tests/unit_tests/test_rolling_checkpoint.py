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
"""CPU recovery tests covering interrupted saves and preservation of policy files."""

import copy
import json
import warnings
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.distributed.checkpoint import (
    FileSystemReader,
    load_state_dict,
    save_state_dict,
)

from rlinf.utils.rolling_checkpoint import publish_rolling_checkpoint


def make_checkpoint(root, step, *, managed=True, weights_only=False):
    checkpoint = root / f"global_step_{step}"
    weights = checkpoint / "actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fsdp_checkpoint": {
            "model": {"weight": torch.full((2,), float(step))},
            "optimizers": [{"state": {"weight": {"exp_avg": torch.ones(2)}}}],
            "lr_schedulers": [{"last_epoch": step}],
            "rng": {"cpu": torch.get_rng_state()},
        }
    }
    torch.save(payload["fsdp_checkpoint"]["model"], weights)
    if not weights_only:
        from torch.distributed.checkpoint import FileSystemWriter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_state_dict(
                payload,
                FileSystemWriter(str(checkpoint / "actor/dcp_checkpoint")),
                no_dist=True,
            )
    (checkpoint / "runner_state.json").write_text(
        json.dumps(
            {
                "optimizer_updates": step,
                "rollout_count": step // 2,
                "env_steps": step * 10,
                "step_unit": "optimizer_update",
                "weights_only": weights_only,
                "rolling_recovery_managed": managed,
            }
        )
    )
    (checkpoint / "rollout_allocation_state.json").write_text(
        json.dumps({"rounds": step // 2, "simulation_steps": step * 10})
    )
    return checkpoint, payload


def test_replaces_recovery_but_preserves_all_policy_weights(tmp_path):
    foreign, _ = make_checkpoint(tmp_path, 5, managed=False)
    first, _ = make_checkpoint(tmp_path, 10)
    before = (first / "actor/model_state_dict/full_weights.pt").read_bytes()
    allocator_before = (first / "rollout_allocation_state.json").read_bytes()
    (first / "runner_state.json").chmod(0o640)
    publish_rolling_checkpoint(first, require_allocation=True)
    second, expected = make_checkpoint(tmp_path, 20)
    removed = publish_rolling_checkpoint(second, require_allocation=True)
    assert removed == [first / "actor/dcp_checkpoint"]
    assert (foreign / "actor/dcp_checkpoint/.metadata").exists()
    assert (first / "actor/model_state_dict/full_weights.pt").read_bytes() == before
    assert (first / "rollout_allocation_state.json").read_bytes() == allocator_before
    assert json.loads((first / "runner_state.json").read_text())["weights_only"]
    assert (first / "runner_state.json").stat().st_mode & 0o777 == 0o640
    assert (
        json.loads((tmp_path / "latest_full_checkpoint.json").read_text())["checkpoint"]
        == second.name
    )
    restored = copy.deepcopy(expected)
    restored["fsdp_checkpoint"]["model"]["weight"].zero_()
    restored["fsdp_checkpoint"]["optimizers"][0]["state"]["weight"]["exp_avg"].zero_()
    restored["fsdp_checkpoint"]["lr_schedulers"][0]["last_epoch"] = -1
    restored["fsdp_checkpoint"]["rng"]["cpu"].zero_()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_state_dict(
            restored,
            FileSystemReader(str(second / "actor/dcp_checkpoint")),
            no_dist=True,
        )
    assert torch.equal(
        restored["fsdp_checkpoint"]["model"]["weight"],
        expected["fsdp_checkpoint"]["model"]["weight"],
    )
    assert torch.equal(
        restored["fsdp_checkpoint"]["optimizers"][0]["state"]["weight"]["exp_avg"],
        torch.ones(2),
    )
    assert restored["fsdp_checkpoint"]["lr_schedulers"][0]["last_epoch"] == 20
    assert torch.equal(
        restored["fsdp_checkpoint"]["rng"]["cpu"],
        expected["fsdp_checkpoint"]["rng"]["cpu"],
    )


def test_truncated_new_save_keeps_previous_complete_recovery(tmp_path):
    first, _ = make_checkpoint(tmp_path, 10)
    publish_rolling_checkpoint(first, require_allocation=True)
    pointer = (tmp_path / "latest_full_checkpoint.json").read_bytes()
    second, _ = make_checkpoint(tmp_path, 20)
    shard = next((second / "actor/dcp_checkpoint").glob("*.distcp"))
    with shard.open("r+b") as stream:
        stream.truncate(shard.stat().st_size - 1)
    with pytest.raises(ValueError, match="truncated"):
        publish_rolling_checkpoint(second, require_allocation=True)
    assert (tmp_path / "latest_full_checkpoint.json").read_bytes() == pointer
    assert (first / "actor/dcp_checkpoint/.metadata").exists()
    assert not json.loads((first / "runner_state.json").read_text())["weights_only"]


def test_mismatched_allocator_does_not_replace_recovery(tmp_path):
    first, _ = make_checkpoint(tmp_path, 10)
    publish_rolling_checkpoint(first, require_allocation=True)
    second, _ = make_checkpoint(tmp_path, 20)
    (second / "rollout_allocation_state.json").write_text(
        json.dumps({"rounds": 9, "simulation_steps": 200})
    )
    with pytest.raises(ValueError, match="allocation counters"):
        publish_rolling_checkpoint(second, require_allocation=True)
    assert (
        json.loads((tmp_path / "latest_full_checkpoint.json").read_text())["checkpoint"]
        == first.name
    )


@pytest.mark.parametrize("rolling", [False, True])
def test_runner_opt_in_preserves_legacy_weights_only_mode(tmp_path, rolling):
    from rlinf.runners.embodied_runner import EmbodiedRunner

    calls = []

    def save_actor(actor_path, step, weights_only=False):
        calls.append(weights_only)
        return SimpleNamespace(
            wait=lambda: make_checkpoint(
                actor_path.parent.parent, step, weights_only=weights_only
            )
        )

    from pathlib import Path

    runner = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "runner": {
                    "logger": {"log_path": str(tmp_path), "experiment_name": "run"},
                    "intermediate_weights_only": True,
                    "rolling_full_checkpoint": rolling,
                }
            }
        ),
        logger=SimpleNamespace(info=lambda message: None),
        actor=SimpleNamespace(
            save_checkpoint=lambda path, step, **kwargs: save_actor(
                Path(path), step, **kwargs
            )
        ),
        optimizer_update_steps=True,
        updates_per_rollout=2,
        max_steps=100,
    )
    for step in [10, 20]:
        runner.global_step = step
        runner.cumulative_env_steps = step * 10
        runner.rollout_allocator = SimpleNamespace(
            state_dict=lambda: {"rounds": step // 2, "simulation_steps": step * 10}
        )
        EmbodiedRunner._save_checkpoint(runner)
    assert calls == [not rolling, not rolling]
    root = tmp_path / "run/checkpoints"
    assert (root / "global_step_10/actor/model_state_dict/full_weights.pt").exists()
    assert (root / "global_step_20/actor/dcp_checkpoint/.metadata").exists() == rolling
    assert (root / "latest_full_checkpoint.json").exists() == rolling


@pytest.mark.parametrize(
    "rolling,step_unit,backend,match",
    [
        ("true", "optimizer_update", "fsdp", "boolean"),
        (True, "rollout", "fsdp", "optimizer-update"),
        (True, "optimizer_update", "megatron", "FSDP"),
    ],
)
def test_recovery_config_rejects_unsupported_modes(rolling, step_unit, backend, match):
    from rlinf.config import validate_flow_refresh_cfg

    cfg = OmegaConf.create(
        {
            "runner": {"rolling_full_checkpoint": rolling, "step_unit": step_unit},
            "actor": {"model": {"model_type": "openpi"}, "training_backend": backend},
            "algorithm": {"flow_refresh_updates": 1},
        }
    )
    with pytest.raises(ValueError, match=match):
        validate_flow_refresh_cfg(cfg)
