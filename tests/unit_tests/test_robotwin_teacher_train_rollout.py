# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the frozen-teacher training-path diagnostic."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from toolkits.robotwin_mopd.teacher_train_rollout import (
    assert_gpu_allowed,
    diagnostic_placement,
    routed_predict,
    select_rows,
    summarize_records,
)


class FakeTeacher:
    def __init__(self, task, offset):
        self.task = task
        self.offset = offset
        self.calls = []

    def predict_action_batch(self, env_obs, mode):
        assert not torch.is_grad_enabled()
        assert set(env_obs["task_groups"]) == {self.task}
        self.calls.append((mode, env_obs))
        values = env_obs["states"] + self.offset
        return values[:, None], {
            "prev_logprobs": values,
            "prev_values": None,
            "forward_inputs": {
                "action": values,
                "chains": values[:, None, None],
            },
        }


def test_mixed_teacher_routes_restore_actions_and_replay_order():
    labels = ["b", "a", "d", "c", "a", "b", "d", "c"]
    models = {task: FakeTeacher(task, i * 100) for i, task in enumerate("abcd")}
    obs = {
        "states": torch.arange(8, dtype=torch.float32)[:, None],
        "task_groups": labels,
        "task_descriptions": [f"prompt{i}" for i in range(8)],
        "main_images": np.arange(8)[:, None, None],
        "wrist_images": None,
    }
    activated = []
    actions, result = routed_predict(models, obs, activate=activated.append)
    expected = torch.tensor([100, 1, 302, 203, 4, 105, 306, 207.0])
    torch.testing.assert_close(actions.flatten(), expected)
    torch.testing.assert_close(result["forward_inputs"]["chains"].flatten(), expected)
    assert result["prev_values"] is None
    assert activated == list("abcd")
    for model in models.values():
        assert len(model.calls) == 1
        assert model.calls[0][0] == "train"
    assert models["a"].calls[0][1]["task_descriptions"] == ["prompt1", "prompt4"]


@pytest.mark.parametrize("labels", [None, ["a"], ["a", "unknown"]])
def test_missing_or_unknown_routing_fails_before_any_teacher_call(labels):
    model = FakeTeacher("a", 0)
    with pytest.raises(ValueError):
        routed_predict(
            {"a": model}, {"states": torch.zeros(2, 1), "task_groups": labels}
        )
    assert not model.calls


def test_subset_does_not_invoke_absent_teachers():
    models = {task: FakeTeacher(task, 0) for task in "abcd"}
    routed_predict(models, {"states": torch.zeros(2, 1), "task_groups": ["c", "c"]})
    assert [len(model.calls) for model in models.values()] == [0, 0, 1, 0]


def test_batch_shape_mismatch_is_explicit():
    with pytest.raises(ValueError, match="batch-first"):
        select_rows(torch.ones(3), [0], batch_size=4)


def test_gpu_guard_rejects_wrong_uuid_without_model_load(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid="GPU-forbidden"),
    )
    with pytest.raises(RuntimeError, match="Refusing GPU"):
        assert_gpu_allowed(["GPU-allowed"])
    assert assert_gpu_allowed(["GPU-forbidden"])["gpu_uuid"] == "GPU-forbidden"


def test_episode_summary_uses_each_episode_once():
    rows = [
        {"task": "a", "success_once": True, "simulation_steps": 100},
        {"task": "a", "success_once": False, "simulation_steps": 400},
        {"task": "b", "success_once": False, "simulation_steps": 600},
    ]
    result = summarize_records(rows)
    assert result["a"] == {
        "episodes": 2,
        "successes": 1,
        "success_rate": 0.5,
        "simulation_steps": 500,
    }
    assert result["b"]["success_rate"] == 0


def test_physical_placement_never_uses_discovery_all():
    assert diagnostic_placement([4, 5, 6, 7]) == {"actor,env,rollout": "4,5,6,7"}
    with pytest.raises(ValueError):
        diagnostic_placement([4, 4, 6, 7])


def test_gpu_guard_accepts_torch_uuid_without_nvidia_prefix(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid="3fbe-ABCD"),
    )
    assert assert_gpu_allowed(["GPU-3fbe-abcd"])["gpu_uuid"] == "3fbe-ABCD"


@pytest.mark.parametrize("bad_actions", [False, True])
def test_student_diagnostic_calls_original_predict_and_checks_finite(
    monkeypatch, bad_actions
):
    from collections import Counter

    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from toolkits.robotwin_mopd.teacher_train_rollout import make_worker_classes

    cls, _ = make_worker_classes()
    worker = object.__new__(cls)
    worker.student_checkpoint = "/test/specialist"
    worker.predict_seconds = 0.0
    worker.predict_calls = 0
    worker.routed_samples = Counter()
    worker.action_min = float("inf")
    worker.action_max = float("-inf")
    worker.first_batch_prompts = None
    worker._attach_teacher_ids = lambda obs, result: None
    calls = []

    def original_predict(self, env_obs, mode="train"):
        calls.append((self, env_obs, mode))
        value = float("nan") if bad_actions else 0.5
        return torch.full((4, 25, 14), value), {"forward_inputs": {}}

    monkeypatch.setattr(MultiStepRolloutWorker, "predict", original_predict)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    observations = {"task_groups": list("abcd"), "task_descriptions": ["do task"] * 4}
    if bad_actions:
        with pytest.raises(FloatingPointError):
            worker.predict(observations)
    else:
        actions, _ = worker.predict(observations)
        assert actions.shape == (4, 25, 14)
        assert worker.action_min == worker.action_max == 0.5
        assert worker.first_batch_prompts == observations["task_descriptions"]
    assert calls == [(worker, observations, "train")]
    assert worker.routed_samples == Counter("abcd")
