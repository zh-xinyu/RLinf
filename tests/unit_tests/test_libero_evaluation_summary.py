# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reject misleading evaluation summaries from incomplete or mixed episode logs."""

import pytest

from toolkits.libero_gr00t.summarize_evaluation import audit


def test_full_2000_episode_coverage_and_duplicate_logs(tmp_path):
    log = tmp_path / "worker.log"
    lines = [
        f"[libero eval] task_id={task}, trial_id={trial}, success={task < 10}\n"
        for task in range(40)
        for trial in range(50)
    ]
    log.write_text(
        "".join(lines)
        + lines[0]
        + "[libero eval] duplicate episode skipped: task_id=0, trial_id=0\n"
    )
    report, rows = audit([log])
    assert report["complete"] and len(rows) == report["episodes"] == 2000
    assert report["macro_success_rate"] == 0.25
    assert report["identical_log_duplicates"] == 1


def test_incomplete_coverage_never_reports_partial_sr(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("[libero eval] task_id=0, trial_id=0, success=True\n")
    report, rows = audit([log])
    assert not report["complete"] and len(rows) == 1
    assert report["macro_success_rate"] is report["suite_success_rates"] is None
    assert len(report["missing_pairs"]) == 1999


@pytest.mark.parametrize(
    "second,reason",
    [
        ("task_id=0, trial_id=0, success=False", "Conflicting"),
        ("task_id=40, trial_id=0, success=True", "Unexpected"),
        ("task_id=0, trial_id=50, success=True", "Unexpected"),
    ],
)
def test_conflicting_or_out_of_protocol_pairs_fail(tmp_path, second, reason):
    log = tmp_path / "worker.log"
    log.write_text(
        "[libero eval] task_id=0, trial_id=0, success=True\n[libero eval] " + second
    )
    with pytest.raises(ValueError, match=reason):
        audit([log])
