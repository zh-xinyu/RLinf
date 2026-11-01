# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for fixed observation/noise/latent comparisons; no models loaded."""

from types import SimpleNamespace

import pytest
import torch

from toolkits.robotwin_mopd import fixed_flow_probe as probe


class FakeModel:
    def __init__(self, offset=0):
        self.action_in_proj = SimpleNamespace(weight=torch.zeros(1))
        self.offset = offset
        self.received = []

    def _get_timesteps(self, steps, device):
        return torch.linspace(1, 0, steps + 1, device=device)

    def output_transform(self, values, **kwargs):
        return {"actions": values["actions"][:, :25, :14]}

    def get_velocity(self, state, x_t, timestep, *unused):
        self.received.append((x_t.clone(), timestep.clone()))
        return x_t + self.offset, None


def patch_reference(monkeypatch):
    def prepare(_model, obs):
        return {
            "state": obs["states"],
            "prefix": None,
            "prefix_masks": None,
            "cache": None,
            "task_ids": None,
            "observation": SimpleNamespace(state=obs["states"]),
        }

    def sample(_model, _context, noise):
        chain = torch.stack([noise + index for index in range(11)], dim=1)
        return {"chains": chain, "actions": chain[:, -1]}

    monkeypatch.setattr(probe, "prepare_context", prepare)
    monkeypatch.setattr(probe, "sample_complete_flow", sample)


def observation():
    return {
        "states": torch.zeros(1, 32),
        "main_images": torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
        "task_descriptions": ["fixed seen prompt"],
        "task_groups": ["adjust_bottle"],
    }


def test_reference_cache_repeats_noise_and_preserves_all_ten_points(monkeypatch):
    patch_reference(monkeypatch)
    obs = observation()
    sample, actions = probe.make_reference_sample(
        FakeModel(), obs, scene_seed=200001, index=0
    )
    repeated, _ = probe.make_reference_sample(
        FakeModel(), obs, scene_seed=200001, index=0
    )
    assert sample["noise_sha256"] == repeated["noise_sha256"]
    assert sample["x_t_sha256"] == repeated["x_t_sha256"]
    assert sample["x_t"].shape == (1, 10, 50, 32)
    assert actions.shape == (1, 25, 14)
    obs["states"].fill_(9)
    assert sample["observation"]["states"].count_nonzero() == 0
    assert probe.input_hash(sample["observation"]) == sample["observation_sha256"]


def test_all_checkpoints_receive_identical_states_times_and_prompt(monkeypatch):
    patch_reference(monkeypatch)
    sample, _ = probe.make_reference_sample(
        FakeModel(), observation(), scene_seed=42, index=0
    )
    first, later = FakeModel(), FakeModel(3)
    a = probe.score_fixed_inputs(first, [sample], amp="none")
    b = probe.score_fixed_inputs(later, [sample], amp="none")
    assert len(first.received) == len(later.received) == 10
    for index, (left, right) in enumerate(
        zip(first.received, later.received, strict=True)
    ):
        torch.testing.assert_close(left[0], sample["x_t"][:, index], rtol=0, atol=0)
        torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
        torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
    torch.testing.assert_close(b[0] - a[0], torch.full_like(a[0], 3))


@pytest.mark.parametrize("changed", ["prompt", "x_t", "noise", "normalized_state"])
def test_cache_changes_are_rejected(monkeypatch, changed):
    patch_reference(monkeypatch)
    sample, _ = probe.make_reference_sample(
        FakeModel(), observation(), scene_seed=42, index=0
    )
    if changed == "prompt":
        sample["observation"]["task_descriptions"][0] = "wrong"
    else:
        sample[changed] = sample[changed] + 1
    with pytest.raises(ValueError):
        probe.score_fixed_inputs(FakeModel(), [sample], amp="none")


def test_ten_bin_means_use_element_denominator_not_weight_sum():
    values = torch.arange(10.0).reshape(1, 10, 1, 1).expand(1, 10, 50, 32)
    weights = torch.linspace(1.0, 2.0, 10)
    report = probe.summarize_errors(
        [values, values],
        [torch.zeros_like(values)] * 2,
        weights=weights,
        grid=torch.linspace(1, 0, 11),
    )
    summary = report["aggregate"]["all50/all32"]
    assert summary["elements"] == 2 * 10 * 50 * 32
    assert summary["unweighted_mse"] == pytest.approx(
        torch.arange(10.0).square().mean().item()
    )
    assert summary["m3_weighted_mse"] == pytest.approx(
        (weights * torch.arange(10.0).square()).mean().item()
    )
    assert len(report["per_time_bin"]) == 10
    assert (
        report["per_time_bin"][9]["statistics"]["last25/padded18"]["elements"]
        == 2 * 25 * 18
    )


def test_reference_collection_tracks_retry_seed_and_executes_only_intermediate_chunk(
    monkeypatch,
):
    patch_reference(monkeypatch)

    class Env:
        executed = 0

        def reset(self):
            return observation(), {}

        def rollout_episode_records(self):
            return [{"actual_reset_seed": 200001, "simulation_steps": self.executed}]

        def chunk_step(self, actions):
            assert actions.shape == (1, 25, 14)
            self.executed += 25
            return (
                [observation()],
                None,
                torch.tensor([False]),
                torch.tensor([False]),
                None,
            )

    samples, metadata = probe.collect_reference(
        FakeModel(), Env(), requested_seed=200000, probes=2
    )
    assert len(samples) == 2 and metadata["executed_control_targets"] == 25
    assert metadata["actual_reset_seed"] == 200001
    assert metadata["requested_scene_seed"] == 200000
    assert metadata["instruction"] == "fixed seen prompt"
    assert samples[1]["noise_seed"] == probe.probe_noise_seed(200001, 1)
    assert samples[1]["executed_control_targets_before_observation"] == 25


def test_hash_covers_dtype_shape_and_bfloat16():
    assert probe.tensor_hash(torch.zeros(2, dtype=torch.bfloat16))
    assert probe.tensor_hash(torch.zeros(2)) != probe.tensor_hash(torch.zeros(1, 2))
    assert probe.tensor_hash(torch.zeros(2)) != probe.tensor_hash(
        torch.zeros(2, dtype=torch.float64)
    )


def test_saved_cache_can_be_reloaded_without_recollecting(monkeypatch, tmp_path):
    patch_reference(monkeypatch)
    sample, _ = probe.make_reference_sample(
        FakeModel(), observation(), scene_seed=42, index=0
    )
    path = tmp_path / "reference.pt"
    torch.save({"samples": [sample]}, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)["samples"]
    values = probe.score_fixed_inputs(FakeModel(), reloaded, amp="none")
    assert values[0].shape == (1, 10, 50, 32)
    assert reloaded[0]["x_t_sha256"] == sample["x_t_sha256"]
