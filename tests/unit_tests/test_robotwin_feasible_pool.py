# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of expert-screened scene acceptance and evaluation-pool integrity."""

import copy
import json

import pytest

from toolkits.robotwin_mopd import feasible_pool as module


class FakeTask:
    def __init__(self, plan=True, success=True, error=None):
        self.plan_success = plan
        self.success = success
        self.error = error
        self.closed = False

    def setup_demo(self, **kwargs):
        self.setup_kwargs = kwargs

    def get_info(self):
        return {"{A}": "initial", "{a}": "left"}

    def play_once(self):
        if self.error:
            raise self.error
        return {
            "info": {"{A}": "expert", "{a}": "right"},
            "actions": ["must not retain"],
        }

    def check_success(self):
        return self.success

    def close_env(self, clear_cache):
        assert clear_cache
        self.closed = True


def screen(task):
    return module.screen_candidate(
        task_name="adjust_bottle",
        seed=123,
        episode_index=0,
        setup_args=dict(module.SCREENING_OVERRIDES),
        task_factory=lambda _: task,
        instruction_sampler=lambda name, info, seed: f"unseen {info['{A}']} {seed}",
    )


def test_accepted_prompt_uses_successful_expert_info_only():
    task = FakeTask()
    row, audit = screen(task)
    assert task.closed
    assert task.setup_kwargs["seed"] == 123
    assert task.setup_kwargs["is_test"] is True
    assert task.setup_kwargs["save_data"] is False
    assert row == {
        "scene_seed": 123,
        "episode_info": {"{A}": "expert", "{a}": "right"},
        "instruction": "unseen expert 123",
    }
    assert audit["initial_info_matches_expert"] is False
    assert audit["reason"] == "accepted"
    assert "actions" not in json.dumps(row)


@pytest.mark.parametrize("plan,success", [(False, False), (False, True), (True, False)])
def test_both_planner_and_task_must_succeed(plan, success):
    task = FakeTask(plan, success)
    row, audit = screen(task)
    assert row is None and audit["reason"] == "expert_failed"
    assert task.closed


def test_expert_failure_closes_environment():
    task = FakeTask(error=RuntimeError("planning failed"))
    row, audit = screen(task)
    assert row is None and audit["reason"] == "expert_error"
    assert task.closed


def test_screening_and_fixed_replay_share_test_flags(monkeypatch):
    import sys
    from types import SimpleNamespace

    from rlinf.envs.robotwin.multitask_env import NativeTaskSlot

    screened = FakeTask()
    row, _ = screen(screened)
    replayed = FakeTask()
    replayed.set_instruction = lambda instruction: setattr(
        replayed, "instruction", instruction
    )
    monkeypatch.setitem(
        sys.modules,
        "description.utils.generate_episode_instructions",
        SimpleNamespace(
            generate_episode_descriptions=lambda *args: pytest.fail("Use cached prompt")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.utils.create_actor",
        SimpleNamespace(UnStableError=RuntimeError),
    )
    monkeypatch.setitem(
        sys.modules,
        "robotwin.envs.vector_env",
        SimpleNamespace(class_decorator=lambda _: replayed),
    )
    slot = NativeTaskSlot.__new__(NativeTaskSlot)
    slot.task = None
    slot.task_name = "adjust_bottle"
    slot.args = dict(module.SCREENING_OVERRIDES)
    slot.horizon = 400
    slot.observe = lambda: {"instruction": replayed.instruction}
    slot.reset(row["scene_seed"], instruction=row["instruction"], strict_seed=True)
    assert screened.setup_kwargs["seed"] == replayed.setup_kwargs["seed"] == 123
    for key, expected in module.SCREENING_OVERRIDES.items():
        assert screened.setup_kwargs[key] == replayed.setup_kwargs[key] == expected
    assert replayed.setup_kwargs["is_test"] is True


def sample_pool():
    return {
        "schema_version": 1,
        "complete": True,
        "protocol": {
            "name": module.PROTOCOL_NAME,
            "instruction_type": "unseen",
            "task_names": ["adjust_bottle"],
            "episodes_per_task": 1,
            "task_horizons": {"adjust_bottle": 400},
            "task_config_sha256": module.canonical_sha256(
                {"embodiment": ["aloha-agilex"]}
            ),
        },
        "tasks": {
            "adjust_bottle": [
                {
                    "scene_seed": 123,
                    "episode_info": {"{A}": "bottle"},
                    "instruction": "unseen prompt",
                }
            ]
        },
    }


def test_pool_integrity_and_runtime_protocol(tmp_path):
    path = tmp_path / "pool.json"
    module.write_pool(path, sample_pool())
    loaded = module.load_pool(path)
    module.validate_pool_protocol(
        loaded,
        task_config={"embodiment": ["aloha-agilex"]},
        task_horizons={"adjust_bottle": 400},
    )
    with pytest.raises(ValueError, match="task_config"):
        module.validate_pool_protocol(
            loaded, task_config={}, task_horizons={"adjust_bottle": 400}
        )
    loaded["tasks"]["adjust_bottle"][0]["instruction"] = "tampered"
    path.write_text(json.dumps(loaded))
    with pytest.raises(ValueError, match="SHA256"):
        module.load_pool(path)


def test_partial_or_duplicate_pools_cannot_be_used(tmp_path):
    path = tmp_path / "pool.json"
    pool = sample_pool()
    pool["complete"] = False
    module.write_pool(path, pool)
    with pytest.raises(ValueError, match="incomplete"):
        module.load_pool(path)
    pool["complete"] = True
    pool["tasks"]["adjust_bottle"].append(
        copy.deepcopy(pool["tasks"]["adjust_bottle"][0])
    )
    module.write_pool(path, pool)
    with pytest.raises(ValueError, match="duplicate"):
        module.load_pool(path)


def test_idle_check_rejects_other_compute_process(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    replies = iter(["0, GPU-a, 1\n2, GPU-b, 100\n", "GPU-b, 456\n"])
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *args, **kwargs: next(replies)
    )
    with pytest.raises(RuntimeError, match="occupied"):
        module.require_idle_visible_gpus()
