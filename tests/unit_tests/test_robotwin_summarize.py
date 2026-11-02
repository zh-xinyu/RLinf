# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Offline pilot windows use recorded rollout rounds, never an assumed R."""

import json

import pytest

from toolkits.robotwin_mopd.summarize import main, mse_windows, read_rows


def updates_for(repeats):
    rows = []
    for rollout, count in enumerate(repeats, start=1):
        for _ in range(count):
            rows.append(
                {
                    "update_step": len(rows) + 1,
                    "rollout_count": rollout,
                    "train/actor/velocity_mse": float(rollout),
                }
            )
    return rows


@pytest.mark.parametrize("repeats", [2, 4])
def test_two_round_windows_match_recorded_r2_and_r4(repeats):
    result = mse_windows(updates_for([repeats] * 4), 2)
    assert result["mse_first_window"] == 1.5
    assert result["mse_last_window"] == 3.5
    assert result["mse_first_window_details"]["rollout_counts"] == [1, 2]
    assert result["mse_last_window_details"]["rollout_counts"] == [3, 4]
    for label in ("first", "last"):
        details = result[f"mse_{label}_window_details"]
        assert details["round_count"] == 2
        assert details["update_count"] == 2 * repeats
        assert details["metric_counts"] == {
            "finite": 2 * repeats,
            "missing": 0,
            "invalid": 0,
        }
    assert result["loss_all_finite"] is True


def test_resumed_nonconsecutive_rounds_and_partial_last_round_are_counted():
    rows = updates_for([4, 4, 1])
    for row in rows:
        row["rollout_count"] = {1: 11, 2: 12, 3: 15}[row["rollout_count"]]
        row["update_step"] += 20
    result = mse_windows(rows, 2)
    details = result["mse_last_window_details"]
    assert details["rollout_counts"] == [12, 15]
    assert details["update_count"] == 5
    assert details["update_steps"] == [25, 26, 27, 28, 29]
    assert result["mse_last_window"] == pytest.approx((2 * 4 + 3) / 5)


def test_missing_mse_keeps_round_and_update_membership():
    rows = updates_for([2, 2, 2])
    del rows[0]["train/actor/velocity_mse"]
    rows[1]["train/actor/velocity_mse"] = None
    result = mse_windows(rows, 1)
    assert result["mse_first_window"] is None
    assert result["mse_first_window_details"]["rollout_counts"] == [1]
    assert result["mse_first_window_details"]["update_count"] == 2
    assert result["mse_first_window_details"]["metric_counts"]["missing"] == 2
    assert result["mse_last_window"] == 3.0
    assert result["loss_all_finite"] is None


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -float("inf"), "bad", True]
)
def test_invalid_mse_is_visible_and_window_does_not_report_finite_mean(invalid):
    rows = updates_for([2, 2])
    rows[0]["train/actor/velocity_mse"] = invalid
    result = mse_windows(rows, 1)
    assert result["loss_all_finite"] is False
    assert result["loss_metric_counts"]["invalid"] == 1
    assert result["mse_first_window"] is None
    assert result["mse_first_window_details"]["metric_counts"] == {
        "finite": 1,
        "missing": 0,
        "invalid": 1,
    }
    assert result["mse_last_window"] == 2.0
    json.dumps(result, allow_nan=False)


def test_missing_rollout_count_is_reported_without_guessing_r():
    rows = updates_for([2, 2])
    del rows[0]["rollout_count"]
    rows[1]["rollout_count"] = "1"
    result = mse_windows(rows, 1)
    assert result["updates_without_valid_rollout_count"] == 2
    assert result["mse_first_window_details"]["rollout_counts"] == [2]
    assert result["mse_first_window_details"]["update_count"] == 2
    assert result["loss_metric_counts"]["finite"] == 4


def test_empty_updates_produce_explicit_empty_windows():
    result = mse_windows([], 5)
    assert result["loss_all_finite"] is None
    assert result["mse_first_window"] is None
    assert result["mse_last_window_details"]["update_count"] == 0
    assert result["mse_last_window_details"]["round_count"] == 0


def test_partial_trailing_json_is_ignored_but_interior_corruption_raises(tmp_path):
    path = tmp_path / "updates.jsonl"
    path.write_text('{"update_step": 1}\n{"update_step":')
    assert read_rows(path) == [{"update_step": 1}]
    path.write_text('{broken}\n{"update_step": 1}\n')
    with pytest.raises(json.JSONDecodeError):
        read_rows(path)


def test_cli_preserves_training_sr_and_writes_counts_only_to_requested_dir(
    tmp_path, monkeypatch, capsys
):
    updates = updates_for([4, 4])
    (tmp_path / "optimizer_updates.jsonl").write_text(
        "\n".join(map(json.dumps, updates)) + '\n{"partial":'
    )
    episodes = [
        {"round": 1, "task": "a", "success_once": False, "simulation_steps": 10},
        {"round": 2, "task": "a", "success_once": True, "simulation_steps": 20},
        {"round": 2, "task": "b", "success_once": False, "simulation_steps": 30},
    ]
    (tmp_path / "rollout_episodes.jsonl").write_text(
        "\n".join(map(json.dumps, episodes))
    )
    monkeypatch.setattr(
        "sys.argv", ["summarize", str(tmp_path), "--window-rounds", "1"]
    )
    main()
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["logged_updates"] == 8
    assert result["mse_last_window_details"]["update_count"] == 4
    assert result["recent_training_sr"] == {"a": 1.0, "b": 0.0}
    assert result["recent_training_macro_sr"] == 0.5
    assert result["recent_episode_counts"] == {"a": 1, "b": 1}
    assert result["executed_control_targets"] == 60
    assert "Training rollout SR, not independent evaluation" in result["note"]


def test_available_missing_metrics_do_not_expand_window_or_assume_fixed_reuse():
    rows = updates_for([2, 4, 1])
    rows[2]["train/actor/velocity_mse"] = None
    result = mse_windows(rows, 2)
    assert result["mse_last_window_details"]["rollout_counts"] == [2, 3]
    assert result["mse_last_window_details"]["update_count"] == 5
    assert result["mse_last_window_details"]["metric_counts"] == {
        "finite": 4,
        "missing": 1,
        "invalid": 0,
    }
    assert result["mse_last_window"] == pytest.approx((2 * 3 + 3) / 4)
    assert result["loss_all_finite"] is None
