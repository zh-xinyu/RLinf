# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Unified CSV exports preserve coverage, task means and experiment identities."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from toolkits.libero_gr00t import collect_results as results


@pytest.fixture(autouse=True)
def prepared_csv_directory(tmp_path):
    """Explicit CSV destinations are prepared by the caller, including on NAS."""
    (tmp_path / "results").mkdir()


def evaluation(path, *, repeats=1, round_=0, checkpoint="weights", base="base"):
    path.mkdir(parents=True)
    manifest = {
        "mode": "run",
        "tasks": list(range(40)),
        "trials": list(range(50)),
        "base_checkpoint_identity": {"content_sha256": base},
        "metadata_sha256": "metadata",
        "weights": None
        if not round_
        else f"/checkpoints/global_step_{round_ * repeats}/actor/model_state_dict/full_weights.pt",
        "weights_sha256": checkpoint if round_ else None,
        "execution_lengths": dict(zip(results.SUITES, results.EXECUTION)),
        "horizons": dict(zip(results.SUITES, results.HORIZONS)),
        "rng_protocol": "episode-private",
        "seed": 1234,
        "inference_model_config": {"precision": "bf16"},
        "source_sha256": {"eval": "source"},
    }
    for task in range(40):
        directory = path / f"task_{task:02d}"
        directory.mkdir()
        horizon = results.HORIZONS[task // 10]
        rows = [
            {
                "task_id": task,
                "trial_id": trial,
                "suite": results.SUITES[task // 10],
                "success_once": trial < task % 10,
                "horizon": horizon,
                "simulation_steps": 5 if trial < task % 10 else horizon,
            }
            for trial in range(50)
        ]
        (directory / "episodes.json").write_text(json.dumps(rows))
        (directory / "status").write_text("OK\n")
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "episodes": 2000,
                "expected_episodes": 2000,
                "missing_pairs": [],
                "suite_success_rates": dict.fromkeys(results.SUITES, 0.09),
                "macro_success_rate": 0.09,
            }
        )
    )
    return path


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_shared_initial_and_three_controls_merge_without_duplicates(tmp_path):
    root = tmp_path / "results"
    initial = evaluation(tmp_path / "initial")
    for repeats in (1, 2, 4):
        results.record_evaluation(root, initial, repeats, 0, 0)
        output = evaluation(tmp_path / f"r{repeats}", repeats=repeats, round_=100)
        results.record_evaluation(root, output, repeats, 2, 100)
        results.record_evaluation(root, output, repeats, 2, 100)
    summary = rows(root / "checkpoint_summary.csv")
    tasks = rows(root / "checkpoint_results.csv")
    assert len(summary) == 4 and len(tasks) == 160
    assert [r for r in summary if r["arm"] == "INITIAL"][0]["seed"] == ""
    assert all(float(row["macro_sr"]) == pytest.approx(0.09) for row in summary)
    assert {row["optimizer_update"] for row in summary} == {"0", "100", "200", "400"}
    assert {int(row["episodes"]) for row in tasks} == {50}
    assert all(
        float(row["success_rate"]) == (int(row["task_id"]) % 10) / 50 for row in tasks
    )
    assert all(
        row["training_wall_seconds"] == "" for row in summary if row["round"] != "0"
    )


def test_export_uses_actual_rollout_simulation_counter(tmp_path):
    root = tmp_path / "results"
    run = tmp_path / "run"
    (run / "libero_uniform").mkdir(parents=True)
    (run / "libero_uniform/rollout_allocation.jsonl").write_text(
        json.dumps({"rounds": 10, "trajectories": 640, "simulation_steps": 12345})
        + "\n"
    )
    output = evaluation(tmp_path / "eval", repeats=4, round_=10)
    results.record_evaluation(root, output, 4, 1, 10, run)
    row = rows(root / "checkpoint_summary.csv")[0]
    assert row["training_simulation_steps"] == "12345"
    assert row["training_trajectories"] == "640"


@pytest.mark.parametrize("change", ["missing", "duplicate", "summary", "task_failed"])
def test_partial_or_inconsistent_raw_results_are_not_exported(tmp_path, change):
    output = evaluation(tmp_path / "eval")
    task = output / "task_00/episodes.json"
    episodes = json.loads(task.read_text())
    if change == "missing":
        task.write_text(json.dumps(episodes[:-1]))
    elif change == "duplicate":
        task.write_text(json.dumps(episodes + episodes[:1]))
    elif change == "summary":
        summary = json.loads((output / "summary.json").read_text())
        summary["macro_success_rate"] = 0.5
        (output / "summary.json").write_text(json.dumps(summary))
    else:
        (output / "task_00/status").write_text("FAILED\n")
    with pytest.raises(ValueError):
        results.record_evaluation(tmp_path / "results", output, 1, 0, 0)
    assert not (tmp_path / "results/checkpoint_summary.csv").exists()


@pytest.mark.parametrize("changed_base", [False, True])
def test_different_checkpoint_or_initialization_cannot_replace_existing_score(
    tmp_path, changed_base
):
    root = tmp_path / "results"
    first = evaluation(tmp_path / "first", round_=10)
    results.record_evaluation(root, first, 1, 0, 10)
    old = (root / "checkpoint_summary.csv").read_bytes()
    different = evaluation(
        tmp_path / "other",
        round_=10,
        checkpoint="weights" if changed_base else "new-weights",
        base="new-base" if changed_base else "base",
    )
    with pytest.raises(ValueError):
        results.record_evaluation(root, different, 1, 0, 10)
    assert (root / "checkpoint_summary.csv").read_bytes() == old


def test_concurrent_exports_keep_all_experiments(tmp_path):
    root = tmp_path / "results"
    outputs = [
        (r, evaluation(tmp_path / f"r{r}", repeats=r, round_=10)) for r in (1, 2, 4)
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(
            pool.map(
                lambda item: results.record_evaluation(root, item[1], item[0], 0, 10),
                outputs,
            )
        )
    assert len(rows(root / "checkpoint_summary.csv")) == 3
    assert len(rows(root / "checkpoint_results.csv")) == 120
