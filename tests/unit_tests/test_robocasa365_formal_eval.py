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

"""CPU-only checks for the frozen RoboCasa365 GR00T evaluation protocol."""

import json

import pytest

from evaluations.robocasa365 import formal_eval


def test_manifest_has_exact_target_50_membership_and_native_horizons():
    tasks = formal_eval.manifest_tasks()
    assert len(tasks) == 50
    assert len({task["task"] for task in tasks}) == 50
    for suite, count in formal_eval.SUITE_COUNTS.items():
        assert sum(task["suite"] == suite for task in tasks) == count
    assert (
        next(task["horizon"] for task in tasks if task["task"] == "OpenDrawer") == 750
    )


def test_task_config_keeps_native_physics_horizon_with_rounded_tensor_budget(tmp_path):
    task = next(
        task for task in formal_eval.manifest_tasks() if task["task"] == "OpenDrawer"
    )
    plan = {
        "output": str(tmp_path),
        "base_checkpoint": str(tmp_path / "base"),
        "weights": str(tmp_path / "full_weights.pt"),
        "episodes_per_task": 50,
        "envs_per_task": 5,
        "seed": 7,
        "save_video": False,
    }
    cfg = formal_eval.compose_task_config(plan, task, gpu=1)
    assert cfg.env.eval.task_soup == ["atomic_seen"]
    assert cfg.env.eval.task_filter == ["OpenDrawer"]
    assert cfg.env.eval.episode_horizon_source == "task_horizon"
    assert cfg.env.eval.max_steps_per_rollout_epoch == 752
    assert cfg.env.eval.max_episode_steps == 752
    assert cfg.env.eval.rollout_epoch == 10
    assert cfg.env.eval.total_num_envs == 5
    assert cfg.env.eval.auto_reset is False
    assert cfg.env.eval.ignore_terminations is True
    assert cfg.env.eval.use_fixed_reset_state_ids is True
    assert cfg.runner.ckpt_path == str(tmp_path / "full_weights.pt")
    assert cfg.cluster.component_placement["env,rollout"] == "1"


def test_metric_validation_rejects_missing_or_fractional_episodes():
    good = {
        "num_trajectories": 50,
        "success_once": 0.4,
        "success_at_end": 0.3,
        "episode_len": 750.0,
    }
    assert formal_eval.checked_metrics(good, 50, 750)["success_once_count"] == 20
    with pytest.raises(ValueError, match="native episode length"):
        formal_eval.checked_metrics({**good, "episode_len": 752.0}, 50, 750)
    with pytest.raises(ValueError, match="Expected 50"):
        formal_eval.checked_metrics({**good, "num_trajectories": 49}, 50)
    with pytest.raises(ValueError, match="Invalid binary metric"):
        formal_eval.checked_metrics({**good, "success_once": 0.413}, 50)


def test_smoke_summary_requires_complete_task(tmp_path, monkeypatch):
    monkeypatch.setattr(formal_eval, "REPO", tmp_path)
    monkeypatch.setattr(formal_eval, "source_hashes", lambda: {})
    output = tmp_path / "outputs" / "smoke" / "mini"
    task = next(
        task for task in formal_eval.manifest_tasks() if task["task"] == "OpenDrawer"
    )
    plan = {
        "protocol": formal_eval.PROTOCOL,
        "mode": "smoke",
        "output": str(output),
        "source_sha256": {},
        "manifest_sha256": formal_eval.sha256(formal_eval.MANIFEST),
        "tasks": [task],
        "episodes_per_task": 1,
        "envs_per_task": 1,
        "seed": formal_eval.SEED,
        "gpus": [1],
    }
    task_dir = output / "tasks" / "OpenDrawer"
    task_dir.mkdir(parents=True)
    (task_dir / "status").write_text("RUNNING\n")
    with pytest.raises(RuntimeError, match="incomplete"):
        formal_eval.summarize(plan)
    (task_dir / "status").write_text("OK\n")
    (task_dir / "metrics.json").write_text(
        json.dumps(
            {
                "task": "OpenDrawer",
                "suite": "atomic_seen",
                "metrics": {
                    "num_trajectories": 1,
                    "success_once": 1.0,
                    "success_at_end": 0.0,
                    "episode_len": 750.0,
                },
            }
        )
    )
    summary = formal_eval.summarize(plan)
    assert summary["mode"] == "smoke"
    assert summary["episodes"] == 1
    assert summary["success_once"] == 1.0
