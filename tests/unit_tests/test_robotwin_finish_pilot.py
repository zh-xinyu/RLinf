# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Safety gates for the detached pilot evaluation queue (no GPU required)."""

import json
from pathlib import Path

from toolkits.robotwin_mopd import finish_pilot as queue


def write_reference(path, task, count, seed=200000):
    path.write_text(
        json.dumps(
            {
                "task": task,
                "completed_episodes": count,
                "episodes": [
                    {"actual_reset_seed": seed + i, "success_once": False}
                    for i in range(count)
                ],
            }
        )
    )


def test_live_launcher_blocks_even_with_complete_status_and_checkpoint(
    tmp_path, monkeypatch
):
    weights = tmp_path / "checkpoint/actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"complete")
    (tmp_path / "status").write_text("COMPLETE")
    monkeypatch.setattr(queue, "same_process_alive", lambda *_: True)
    assert (
        queue.training_gate(
            tmp_path, {"pid": 12, "create_time": 42}, tmp_path / "checkpoint"
        )
        == "WAITING_TRAINING"
    )


def test_failed_or_missing_checkpoint_never_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "same_process_alive", lambda *_: False)
    identity = {"pid": 12, "create_time": 42}
    (tmp_path / "status").write_text("FAILED")
    assert (
        queue.training_gate(tmp_path, identity, tmp_path / "checkpoint")
        == "BLOCKED_TRAINING_FAILED"
    )
    (tmp_path / "status").write_text("COMPLETE")
    assert (
        queue.training_gate(tmp_path, identity, tmp_path / "checkpoint")
        == "BLOCKED_MISSING_FINAL_CHECKPOINT"
    )
    weights = tmp_path / "checkpoint/actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"complete")
    assert queue.training_gate(tmp_path, identity, tmp_path / "checkpoint") == "READY"


def test_pid_reuse_does_not_match_original(monkeypatch):
    class DifferentProcess:
        def create_time(self):
            return 43.0

        def status(self):
            return "running"

    monkeypatch.setattr(queue.psutil, "Process", lambda _: DifferentProcess())
    assert not queue.same_process_alive(12, 42.0)
    assert queue.same_process_alive(12, 43.0)


def test_plan_only_extends_base_and_never_runs_teacher(tmp_path):
    references = tmp_path / "references"
    references.mkdir()
    for task in queue.TASKS:
        write_reference(references / f"base_pilot_{task}.json", task, 3)
        write_reference(references / f"teacher_pilot_{task}.json", task, 5)
    jobs = queue.build_jobs(references, tmp_path / "new", Path("final_checkpoint"))
    assert len(jobs) == 8
    assert [job["episodes"] for job in jobs] == [5] * 4 + [2] * 4
    assert [job["seed"] for job in jobs] == [200000] * 4 + [200003] * 4
    assert all("teacher" not in job["command"] for job in jobs)
    assert all(str(references) not in job["output"] for job in jobs)


def test_idle_gpu_excludes_memory_utilization_and_any_compute_process(monkeypatch):
    outputs = iter(
        [
            "0, GPU-a, 2, 0\n1, GPU-b, 600, 0\n2, GPU-c, 2, 8\n3, GPU-d, 2, 0\n",
            "GPU-d, 1234\n",
        ]
    )
    monkeypatch.setattr(
        queue.subprocess, "check_output", lambda *_args, **_kwargs: next(outputs)
    )
    assert queue.idle_gpus(["0", "1", "2", "3"]) == ["0"]


def test_summary_keeps_actual_seed_mismatch_explicit(tmp_path):
    references = tmp_path / "references"
    output = tmp_path / "output"
    references.mkdir()
    output.mkdir()
    for task in queue.TASKS:
        write_reference(references / f"base_pilot_{task}.json", task, 3)
        write_reference(references / f"teacher_pilot_{task}.json", task, 5, seed=200001)
        write_reference(
            output / f"base_pilot_extension_{task}.json", task, 2, seed=200003
        )
        write_reference(output / f"student_{task}.json", task, 5)
    queue.summarize(references, output, [])
    summary = json.loads((output / "summary.json").read_text())
    task = summary["tasks"]["adjust_bottle"]
    assert task["base"]["episodes"] == task["student"]["episodes"] == 5
    assert task["student_vs_base_seeds"]["same_ordered_five_seeds"]
    assert not task["student_vs_teacher_seeds"]["same_ordered_five_seeds"]
    assert len(task["student_vs_teacher_seeds"]["overlap"]) == 4
