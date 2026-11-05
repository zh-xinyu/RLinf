# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Observation-only collection follows the evaluator's reset/settling protocol."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.libero_gr00t.checkpoint_action_diagnostic import read_fixtures
from toolkits.libero_gr00t.collect_observations import collect_bank, collect_one


class FakeEnvironment:
    def __init__(self, fail=False):
        self.calls = []
        self.closed = False
        self.fail = fail

    def reset(self):
        self.calls.append("reset")

    def set_init_state(self, state):
        self.calls.append(("state", state.copy()))
        return self.observation()

    def step(self, action):
        self.calls.append(("step", action))
        if self.fail:
            raise RuntimeError("injected step failure")
        return self.observation(), 0, True, {}

    def close(self):
        self.closed = True

    def observation(self):
        return {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.01, -0.01]),
        }


def factory(env):
    def create(suite, task, seed):
        assert seed == 1234
        return (
            env,
            SimpleNamespace(language="fixed language"),
            np.arange(12.0).reshape(3, 4),
        )

    return create


def test_exact_reset_state_and_ten_open_gripper_steps_then_close():
    env = FakeEnvironment()
    obs, identity = collect_one(None, 0, 1, 1234, factory(env))
    assert env.closed
    assert env.calls[0] == "reset"
    np.testing.assert_equal(env.calls[1][1], [4.0, 5.0, 6.0, 7.0])
    assert env.calls[2:] == [("step", [0.0] * 6 + [-1.0])] * 10
    assert obs["task_description"].item() == "fixed language"
    assert len(identity["initial_state_array_sha256"]) == 64


@pytest.mark.parametrize("failure", ["step", "index"])
def test_environment_closed_on_failure(failure):
    env = FakeEnvironment(fail=failure == "step")
    with pytest.raises((RuntimeError, ValueError)):
        collect_one(None, 0, 4 if failure == "index" else 1, 1234, factory(env))
    assert env.closed


def test_manifest_is_accepted_by_diagnostic_and_npz_needs_no_pickle(tmp_path):
    env = FakeEnvironment()
    output = tmp_path / "bank"
    rows = collect_bank(output, {"libero_spatial": None}, [0], 1, 1234, factory(env))
    assert len(rows) == 1
    loaded = read_fixtures(None, output / "fixtures.json")
    assert loaded[0]["initial_state_id"] == 1
    assert loaded[0]["reset_seed"] == 1234
    with np.load(output / rows[0]["path"], allow_pickle=False) as saved:
        assert saved["task_description"].item() == "fixed language"
        assert saved["agentview_image"].shape == (256, 256, 3)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["complete"]
    assert protocol["simulator_settling_calls"] == 10
    assert protocol["simulator_policy_calls"] == 0
    with pytest.raises(FileExistsError):
        collect_bank(output, {"libero_spatial": None}, [0], 1, 1234, factory(env))


def test_partial_failure_does_not_publish_complete_manifest(tmp_path):
    output = tmp_path / "bank"
    env = FakeEnvironment(fail=True)
    with pytest.raises(RuntimeError):
        collect_bank(output, {"libero_spatial": None}, [0], 1, 1234, factory(env))
    assert not (output / "fixtures.json").exists()
    assert not json.loads((output / "protocol.json").read_text())["complete"]


def test_invalid_task_selection_fails_before_creating_output(tmp_path):
    output = tmp_path / "bank"
    with pytest.raises(ValueError):
        collect_bank(
            output,
            {"libero_spatial": None},
            [0, 0],
            1,
            1234,
            factory(FakeEnvironment()),
        )
    assert not output.exists()
