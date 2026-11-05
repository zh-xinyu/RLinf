# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-fixture diagnostics must retain time and physical-action semantics."""

import json

import pytest
import torch

from toolkits.libero_gr00t.checkpoint_action_diagnostic import (
    compare_sample,
    read_fixtures,
)


def test_time_weights_and_gripper_do_not_contaminate_motion_rmse():
    sv = torch.zeros(4, 8, 7)
    tv = torch.arange(1, 5).view(4, 1, 1).expand_as(sv).float()
    sa, ta = torch.zeros(8, 7), torch.zeros(8, 7)
    ta[:, :6] = 2
    ta[:4, 6] = 1
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = compare_sample(sv, tv, sa, ta, weights)
    assert result["velocity_mse_per_time"] == [1.0, 4.0, 9.0, 16.0]
    assert result["velocity_mse_uniform_time_mean"] == 7.5
    assert result["m3_weighted_velocity_mse"] == 25.0
    assert result["physical_action_rmse_first6"] == 2.0
    assert result["gripper_disagreement_fraction"] == 0.5


def test_identity_is_zero_at_all_times():
    v, a = torch.randn(4, 8, 7), torch.randn(8, 7)
    result = compare_sample(v, v, a, a, torch.ones(4))
    assert result["velocity_mse_per_time"] == [0.0] * 4
    assert result["physical_action_rmse_first6"] == 0
    assert result["gripper_disagreement_fraction"] == 0


@pytest.mark.parametrize("fault", ["shape", "nonfinite"])
def test_bad_samples_fail(fault):
    v, a = torch.zeros(4, 8, 7), torch.zeros(8, 7)
    if fault == "shape":
        v = v[:, :7]
    else:
        v[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        compare_sample(v, v, a, a, torch.ones(4))


def test_fixture_manifest_resolves_relative_paths_and_hashes(tmp_path):
    fixture = tmp_path / "obs.npz"
    fixture.write_bytes(b"fixture bytes")
    manifest = tmp_path / "fixtures.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "suite": "libero_goal",
                    "path": "obs.npz",
                    "task_id": 3,
                    "initial_state_id": 2,
                }
            ]
        )
    )
    row = read_fixtures(None, manifest)[0]
    assert row["path"] == str(fixture)
    assert len(row["sha256"]) == 64
    assert row["task_id"] == 3 and row["initial_state_id"] == 2


def test_unknown_state_is_not_invented_for_legacy_directory(tmp_path):
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        (tmp_path / f"{suite}_obs.npz").write_bytes(b"fixture")
    rows = read_fixtures(tmp_path, None)
    assert len(rows) == 4
    assert all("initial_state_id" not in row for row in rows)


@pytest.mark.parametrize("fault", ["duplicate", "suite", "task"])
def test_bad_manifest_fails(tmp_path, fault):
    (tmp_path / "obs.npz").write_bytes(b"fixture")
    row = {"suite": "libero_goal", "path": "obs.npz", "task_id": 0}
    rows = [row]
    if fault == "duplicate":
        rows.append(row)
    elif fault == "suite":
        row["suite"] = "unknown"
    else:
        row["task_id"] = 10
    manifest = tmp_path / "fixtures.json"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        read_fixtures(None, manifest)
