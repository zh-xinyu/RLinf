# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only flow diagnostic tests; no policy checkpoint or simulator is loaded."""

from types import SimpleNamespace

import pytest
import torch

from toolkits.robotwin_mopd import flow_probe as probe


def test_quadrants_retain_unexecuted_horizon_and_padded_coordinates():
    teacher = torch.zeros(1, 50, 32)
    student = torch.zeros_like(teacher)
    student[:, :25, :14] = 1
    student[:, :25, 14:] = 2
    student[:, 25:, :14] = 3
    student[:, 25:, 14:] = 4
    result = probe.partition_statistics(student, teacher)
    assert result["first25/physical14"]["mse"] == 1
    assert result["first25/padded18"]["mse"] == 4
    assert result["last25/physical14"]["mse"] == 9
    assert result["last25/padded18"]["mse"] == 16
    assert result["all50/all32"]["elements"] == 1600
    assert result["all50/all32"]["mse"] == pytest.approx(student.square().mean().item())


def test_aggregate_uses_element_counts_not_averaged_means():
    first = probe.partition_statistics(torch.ones(1, 50, 32), torch.zeros(1, 50, 32))
    second = probe.partition_statistics(
        torch.full((2, 50, 32), 3.0), torch.zeros(2, 50, 32)
    )
    result = probe.aggregate_statistics([first, second])
    assert result["all50/all32"]["mse"] == pytest.approx(19 / 3)
    assert result["all50/all32"]["elements"] == 4800


class FakeModel:
    def __init__(self, velocity):
        self.velocity = velocity
        self.config = SimpleNamespace(num_steps=10, action_horizon=50, action_dim=32)
        self.action_in_proj = SimpleNamespace(weight=torch.zeros(1))
        self.received_latents = []
        self.received_times = []

    def _get_timesteps(self, steps, device):
        return torch.cat(
            [
                torch.linspace(1, 1 / steps, steps, device=device),
                torch.zeros(1, device=device),
            ]
        )

    def _sample_actions_with_prefix_cache(self, *args, noise, **kwargs):
        self.initial_noise = noise.clone()
        chains = torch.stack([noise - self.velocity * i / 10 for i in range(11)], dim=1)
        return {"chains": chains, "actions": chains[:, -1]}

    def get_velocity(self, state, x_t, timestep, *args):
        self.received_latents.append(x_t.clone())
        self.received_times.append(timestep.clone())
        return torch.full_like(x_t, self.velocity), None

    def output_transform(self, values, **kwargs):
        return {"actions": values["actions"][:, :25, :14]}


def test_probe_same_noise_same_student_path_and_true_grid(monkeypatch):
    context = {
        "state": torch.zeros(1, 32),
        "prefix": None,
        "prefix_masks": None,
        "cache": None,
        "task_ids": None,
        "observation": SimpleNamespace(state=torch.zeros(1, 32)),
    }
    monkeypatch.setattr(probe, "prepare_context", lambda *_: context)
    student, teacher = FakeModel(1.0), FakeModel(3.0)
    result, actions = probe.probe_observation(
        student, teacher, {}, scene_seed=42, probe_index=2, flow_indices=[0, 4, 9]
    )
    assert torch.equal(student.initial_noise, teacher.initial_noise)
    assert result["noise_seed"] == probe.probe_noise_seed(42, 2)
    assert result["noise_seed"] != probe.probe_noise_seed(42, 1)
    assert actions.shape == (1, 25, 14)
    assert [sample["time"] for sample in result["velocity_samples"]] == pytest.approx(
        [1.0, 0.6, 0.1]
    )
    for left, right in zip(
        student.received_latents, teacher.received_latents, strict=True
    ):
        assert torch.equal(left, right)
        assert left.shape == (1, 50, 32)
    assert result["same_noise_endpoint"]["last25/padded18"]["mse"] == pytest.approx(4)
    repeated, _ = probe.probe_observation(
        FakeModel(1),
        FakeModel(3),
        {},
        scene_seed=42,
        probe_index=2,
        flow_indices=[0, 4, 9],
    )
    assert repeated["initial_noise_sha256"] == result["initial_noise_sha256"]


class FakeEnv:
    def __init__(self, terminal_after=None):
        self.executed = 0
        self.calls = 0
        self.terminal_after = terminal_after

    def reset(self, evaluation_episode):
        self.scene = evaluation_episode
        return {"task_descriptions": [self.scene["instruction"]]}, {}

    def rollout_episode_records(self):
        return [
            {
                "actual_reset_seed": self.scene["scene_seed"],
                "simulation_steps": self.executed,
            }
        ]

    def chunk_step(self, actions):
        self.executed += 25
        self.calls += 1
        return (
            [{"task_descriptions": [self.scene["instruction"]]}],
            None,
            torch.tensor([self.calls == self.terminal_after]),
            torch.tensor([False]),
            None,
        )


def test_four_observations_execute_only_three_chunks(monkeypatch):
    monkeypatch.setattr(
        probe,
        "probe_observation",
        lambda *_args, **_kwargs: ({"velocity_samples": []}, torch.zeros(1, 25, 14)),
    )
    env = FakeEnv()
    results, executed, flags = probe.collect_probe_observations(
        None,
        None,
        env,
        {"scene_seed": 42, "instruction": "fixed"},
        max_probes=4,
        flow_indices=[0, 4, 9],
    )
    assert len(results) == 4 and env.calls == 3 and executed == 75
    assert [row["executed_control_targets_before_observation"] for row in results] == [
        0,
        25,
        50,
        75,
    ]
    assert flags == {"terminated": False, "truncated": False}


def test_native_terminal_stops_diagnostic_without_padding(monkeypatch):
    monkeypatch.setattr(
        probe,
        "probe_observation",
        lambda *_args, **_kwargs: ({"velocity_samples": []}, torch.zeros(1, 25, 14)),
    )
    env = FakeEnv(terminal_after=1)
    results, executed, flags = probe.collect_probe_observations(
        None,
        None,
        env,
        {"scene_seed": 42, "instruction": "fixed"},
        max_probes=4,
        flow_indices=[0, 4, 9],
    )
    assert len(results) == 1 and env.calls == 1 and executed == 25
    assert flags["terminated"]
