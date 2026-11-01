# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Elapsed calibration budget must respect overlapping parallel evaluations."""

import os

import pytest

from toolkits.robotwin_mopd.calibrate import calibration_wall_seconds


def test_parallel_evaluations_use_elapsed_span(tmp_path):
    results = [
        {"started_unix": 100, "finished_unix": 140},
        {"started_unix": 110, "finished_unix": 150},
    ]
    seconds, source = calibration_wall_seconds(tmp_path, results)
    assert seconds == 50
    assert source == "recorded_evaluation_timestamps"


def test_legacy_file_timestamp_estimate_is_explicitly_labeled(tmp_path):
    launch = tmp_path / "teacher_pilot_launch.json"
    launch.write_text("[]")
    result = tmp_path / "teacher_pilot_example.json"
    result.write_text("{}")
    os.utime(launch, (100, 100))
    os.utime(result, (125, 125))
    seconds, source = calibration_wall_seconds(tmp_path, [{"task": "example"}])
    assert seconds == 25
    assert source.startswith("approximate_")


def test_missing_or_invalid_budget_fails(tmp_path):
    with pytest.raises(ValueError, match="No elapsed"):
        calibration_wall_seconds(tmp_path, [{"task": "example"}])
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Invalid"):
            calibration_wall_seconds(tmp_path, [], invalid)
