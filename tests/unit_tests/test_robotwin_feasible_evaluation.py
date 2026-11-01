# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fixed-seed, fixed-prompt evaluation and its continuation queue."""

import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from toolkits.robotwin_mopd import feasible_pool
from toolkits.robotwin_mopd import finish_feasible_pilot as queue


def make_pool(path):
    config = {"embodiment": ["aloha-agilex"]}
    pool = {
        "schema_version": 1,
        "complete": True,
        "protocol": {
            "name": feasible_pool.PROTOCOL_NAME,
            "instruction_type": "unseen",
            "task_names": list(queue.TASKS),
            "episodes_per_task": 5,
            "task_config_sha256": feasible_pool.canonical_sha256(config),
            "task_horizons": dict.fromkeys(queue.TASKS, 400),
            "robotwin_revision": "test_revision",
            "source_files_sha256": {},
        },
        "tasks": {
            task: [
                {
                    "scene_seed": 100 + i,
                    "instruction": f"fixed {task} {i}",
                    "episode_info": {},
                }
                for i in range(5)
            ]
            for task in queue.TASKS
        },
    }
    feasible_pool.write_pool(path, pool)
    return pool


def result_for(pool, task):
    return {
        "task": task,
        "completed_episodes": 5,
        "evaluation_pool": {"pool_content_sha256": pool["pool_content_sha256"]},
        "episodes": [
            {
                "actual_reset_seed": row["scene_seed"],
                "reset_seed": row["scene_seed"],
                "instruction": row["instruction"],
                "success_once": True,
                "evaluation_pool_sha256": pool["pool_content_sha256"],
            }
            for row in pool["tasks"][task]
        ],
    }


def test_live_upstream_always_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "same_process_alive", lambda *_: True)
    assert (
        queue.upstream_gate(tmp_path / "not_needed.json", {"pid": 12, "create_time": 1})
        == "WAITING_UNFILTERED_QUEUE"
    )


def test_upstream_requires_complete_and_exact_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "same_process_alive", lambda *_: False)
    upstream = tmp_path / "upstream.json"
    upstream.write_text(
        json.dumps(
            {"status": "PARTIAL_FAILURE", "checkpoint": str(tmp_path / "checkpoint")}
        )
    )
    identity = {"pid": 12, "create_time": 1}
    assert queue.upstream_gate(upstream, identity) == "BLOCKED_UPSTREAM_PARTIAL_FAILURE"
    upstream.write_text(
        json.dumps({"status": "COMPLETE", "checkpoint": str(tmp_path / "checkpoint")})
    )
    assert (
        queue.upstream_gate(upstream, identity) == "BLOCKED_MISSING_STUDENT_CHECKPOINT"
    )
    weights = tmp_path / "checkpoint/actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    assert queue.upstream_gate(upstream, identity) == "READY"


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_reset_seed", 999),
        ("reset_seed", 999),
        ("instruction", "different"),
        ("evaluation_pool_sha256", "wrong"),
    ],
)
def test_result_rejects_any_scene_prompt_or_pool_drift(tmp_path, field, value):
    pool_path = tmp_path / "pool.json"
    pool = make_pool(pool_path)
    task = queue.TASKS[0]
    result = result_for(pool, task)
    result["episodes"][0][field] = value
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        queue.validate_result({"kind": "evaluation", "task": task}, path, pool_path)


def test_plan_is_one_pool_and_twelve_matched_evaluations(tmp_path):
    jobs = queue.build_jobs(tmp_path, tmp_path / "checkpoint")
    assert len(jobs) == 13 and jobs[0]["kind"] == "pool"
    for job in jobs[1:]:
        attempt = queue.make_attempt(job, tmp_path, "0")
        index = attempt["command"].index("--evaluation-pool")
        assert attempt["command"][index + 1] == str(tmp_path / "evaluation_pool.json")
        job["attempts"].append(attempt)
        later = queue.make_attempt(job, tmp_path, "1")
        assert later["output"] != attempt["output"]


def test_summary_requires_exact_three_policy_scene_prompt_match(tmp_path):
    pool = make_pool(tmp_path / "evaluation_pool.json")
    jobs = queue.build_jobs(tmp_path, tmp_path / "checkpoint")
    for job in jobs[1:]:
        result = result_for(pool, job["task"])
        queue.atomic_json(Path(job["output"]), result)
        job["status"] = "COMPLETE"
    queue.summarize(tmp_path, jobs)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert all(
        task["identical_five_scenes_and_prompts"] for task in summary["tasks"].values()
    )


def test_evaluator_reads_fixed_entries_and_rejects_other_task_or_config(
    tmp_path, monkeypatch
):
    from toolkits.robotwin_mopd.evaluate import read_evaluation_pool

    pool_path = tmp_path / "pool.json"
    pool = make_pool(pool_path)
    monkeypatch.setenv("ROBOTWIN_PATH", str(tmp_path))
    monkeypatch.setattr(
        feasible_pool.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "test_revision\n",
    )
    cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "task_config": {"embodiment": ["aloha-agilex"]},
                    "task_horizons": pool["protocol"]["task_horizons"],
                }
            }
        }
    )
    _, selected = read_evaluation_pool(pool_path, queue.TASKS[0], 5, cfg)
    assert selected == pool["tasks"][queue.TASKS[0]]
    with pytest.raises(ValueError, match="not part"):
        read_evaluation_pool(pool_path, "other_task", 5, cfg)
    with pytest.raises(ValueError, match="fewer"):
        read_evaluation_pool(pool_path, queue.TASKS[0], 6, cfg)
    changed = copy.deepcopy(cfg)
    changed.env.eval.task_config.embodiment = ["wrong_robot"]
    with pytest.raises(ValueError, match="task_config"):
        read_evaluation_pool(pool_path, queue.TASKS[0], 5, changed)


def test_resume_recovers_unrecorded_child_without_duplicate_spawn(monkeypatch):
    import os
    from types import SimpleNamespace

    command = ["python", "evaluate.py", "--output", "/isolated/attempt_1/result.json"]
    process = SimpleNamespace(
        pid=123,
        info={
            "uids": SimpleNamespace(real=os.getuid()),
            "cmdline": command,
            "create_time": 456.0,
            "status": "running",
        },
    )
    monkeypatch.setattr(queue.psutil, "process_iter", lambda *_: [process])
    attempt = {"command": command}
    assert queue.attempt_alive(attempt)
    assert attempt["pid"] == 123 and attempt["create_time"] == 456.0


def test_resume_never_adopts_reused_pid(monkeypatch):
    monkeypatch.setattr(queue, "same_process_alive", lambda *_: False)
    assert not queue.attempt_alive({"pid": 123, "create_time": 456.0, "command": []})


def test_evaluator_summary_timestamps_partial_and_complete(tmp_path, monkeypatch):
    from toolkits.robotwin_mopd import evaluate

    monkeypatch.setattr(evaluate.time, "time", lambda: 150.0)
    output = tmp_path / "evaluation.json"
    metadata = {"requested_episodes": 2, "started_unix": 100.0}
    episode = {"success_once": True, "simulation_steps": 10}
    evaluate.write_summary(output, metadata, [episode])
    partial = json.loads(output.read_text())
    assert partial["started_unix"] == 100.0
    assert partial["finished_unix"] is None
    evaluate.write_summary(output, metadata, [episode, episode])
    complete = json.loads(output.read_text())
    assert complete["started_unix"] == 100.0
    assert complete["finished_unix"] == 150.0
