# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Private evaluation launch, artifact binding, and exact coverage checks."""

import json
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from toolkits.libero_gr00t import launch_evaluation as launch


def arguments(tmp_path, all_tasks=False):
    python = tmp_path / "runtime/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture")
    (python.parent.parent / "env.sh").write_text("fixture")
    student = tmp_path / "student"
    (student / "experiment_cfg").mkdir(parents=True)
    (student / "config.json").write_text("{}")
    (student / "model.safetensors").write_bytes(b"fixture model weights")
    (student / "experiment_cfg/metadata.json").write_text("{}")
    return SimpleNamespace(
        python=python,
        student=student,
        weights=None,
        validation_protocol=None,
        output=tmp_path / "out",
        gpus=[7, 0, 6, 1],
        all_tasks=all_tasks,
    )


@pytest.mark.parametrize(
    "all_tasks,budget,count", [(False, 6760, 4), (True, 65000, 40)]
)
def test_config_preserves_inference_and_sets_complete_coverage(
    tmp_path, all_tasks, budget, count
):
    args = arguments(tmp_path, all_tasks)
    cfg = launch.build_config(args)
    assert len(cfg.env.eval.task_id_filter) == count
    assert cfg.env.eval.max_steps_per_rollout_epoch == budget
    assert cfg.env.eval.total_num_envs == 16
    assert list(cfg.env.eval.task_suite_horizons.values()) == [240, 320, 320, 520]
    assert cfg.runner.ckpt_path is None
    assert cfg.rollout.model.action_head_precision == "fp32"
    assert cfg.cluster.component_placement["env,rollout"].placement == "0,1,6,7"
    assert cfg.runner.logger.log_path == str(args.output)


def test_checkpoint_protocol_binds_hash_metadata_and_contract(tmp_path):
    args = arguments(tmp_path)
    args.weights = tmp_path / "weights.pt"
    args.weights.write_bytes(b"validated checkpoint fixture")
    args.validation_protocol = tmp_path / "protocol.json"
    cfg = launch.build_config(args)
    protocol = {
        "strict": True,
        "state_tensor_count": 899,
        "base_comparison": {"unchanged_backbone_tensors": 585},
        "weights_sha256": launch.sha256(args.weights),
        "metadata_sha256": launch.sha256(args.student / "experiment_cfg/metadata.json"),
        "base_checkpoint": str(args.student),
        "model_config": OmegaConf.to_container(cfg.rollout.model, resolve=True),
    }
    args.validation_protocol.write_text(json.dumps(protocol))
    assert not launch.validate_checkpoint(args, cfg)["baseline"]
    protocol["model_config"]["num_action_chunks"] = 16
    args.validation_protocol.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="num_action_chunks"):
        launch.validate_checkpoint(args, cfg)
    args.weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="weights hash"):
        launch.validate_checkpoint(args, cfg)


@pytest.mark.parametrize("tasks", [[0, 10, 20, 30], list(range(40))])
def test_exact_selected_coverage_and_identical_duplicates(tmp_path, tasks):
    log = tmp_path / "env.log"
    lines = [
        f"[libero eval] task_id={task}, trial_id={trial}, success={task == 0}\n"
        for task in tasks
        for trial in range(50)
    ]
    log.write_text("".join(lines) + lines[0])
    report, rows = launch.audit_selected([log], tasks)
    assert report["complete"] and len(rows) == len(tasks) * 50
    assert report["identical_log_duplicates"] == 1
    assert report["suite_success_rates"]["libero_spatial"] == 1 / (len(tasks) / 4)
    log.write_text("".join(lines[:-1]))
    report, _ = launch.audit_selected([log], tasks)
    assert not report["complete"] and report["macro_success_rate"] is None


@pytest.mark.parametrize(
    "second,reason",
    [
        ("task_id=1, trial_id=0, success=True", "Unexpected"),
        ("task_id=0, trial_id=50, success=True", "Unexpected"),
        ("task_id=0, trial_id=0, success=False", "Conflicting"),
    ],
)
def test_reject_invalid_results(tmp_path, second, reason):
    log = tmp_path / "env.log"
    log.write_text(
        "[libero eval] task_id=0, trial_id=0, success=True\n[libero eval] " + second
    )
    with pytest.raises(ValueError, match=reason):
        launch.audit_selected([log], [0, 10, 20, 30])


@pytest.mark.parametrize("mode", ["dry", "success", "failure", "incomplete"])
def test_owned_lifecycle_and_failure_cleanup(tmp_path, monkeypatch, mode):
    args = arguments(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_evaluation",
            "--python",
            str(args.python),
            "--student",
            str(args.student),
            "--baseline",
            "--output",
            str(args.output),
            "--gpus",
            "0",
            "1",
            "6",
            "7",
        ]
        + ([] if mode == "dry" else ["--run"]),
    )
    monkeypatch.setenv("RAY_ADDRESS", "someone-elses-cluster:6379")
    monkeypatch.setattr(
        launch,
        "read_runtime_environment",
        lambda _: {"LIBERO_CONFIG_PATH": "/fixture", "PYTHONPATH": str(launch.REPO)},
    )
    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setattr(launch.tempfile, "mkdtemp", lambda **_: str(private))
    monkeypatch.setattr(launch, "choose_ports", lambda: {"port": 19001})
    monkeypatch.setattr(launch.signal, "signal", lambda *_: None)
    checks, commands, stops = [], [], []
    monkeypatch.setattr(
        launch,
        "check_gpus",
        lambda count, env: checks.append(env["GPU_FREE_MB"]) or "ok",
    )
    monkeypatch.setattr(launch, "stop_owned", stops.append)

    def run(command, env, log, **kwargs):
        commands.append(command)
        if "--head" in command:
            assert "RAY_ADDRESS" not in env and launch.OWNER_KEY in env
            (private / "ray/ray_current_cluster").write_text("127.0.0.1:19001")
        else:
            assert env["RAY_ADDRESS"] == "127.0.0.1:19001"
            assert launch.OWNER_KEY not in env
            if mode == "failure":
                raise RuntimeError("simulated driver failure")
            d = args.output / "workers/EnvGroup"
            d.mkdir(parents=True)
            for rank, task in enumerate([0, 10, 20, 30]):
                (d / f"rank_{rank}.log").write_text(
                    "".join(
                        f"[libero eval] task_id={task}, trial_id={trial}, success=True\n"
                        for trial in range(49 if mode == "incomplete" else 50)
                    )
                )

    monkeypatch.setattr(launch, "run_logged", run)
    if mode in ("failure", "incomplete"):
        with pytest.raises(RuntimeError):
            launch.main()
    else:
        launch.main()
    manifest = json.loads((args.output / "launch.json").read_text())
    if mode == "dry":
        assert not commands and not stops and not checks
        assert (args.output / "status").read_text() == "DRY_RUN\n"
    else:
        assert len(commands) == 2 and stops == [manifest["owner"]]
        assert checks == ["20480"]
        assert (args.output / "status").read_text() == (
            "OK\n" if mode == "success" else "FAILED\n"
        )
    if mode == "incomplete":
        assert (
            json.loads((args.output / "summary.json").read_text())["macro_success_rate"]
            is None
        )


def test_baseline_identity_tracks_config_and_every_weight_shard(tmp_path):
    args = arguments(tmp_path)
    first = launch.baseline_identity(args.student)
    assert {record["path"] for record in first["files"]} == {
        "config.json",
        "experiment_cfg/metadata.json",
        "model.safetensors",
    }
    (args.student / "model.safetensors").write_bytes(b"different weights")
    second = launch.baseline_identity(args.student)
    assert first["content_sha256"] != second["content_sha256"]
    (args.student / "config.json").write_text('{"different": true}')
    third = launch.baseline_identity(args.student)
    assert second["content_sha256"] != third["content_sha256"]
    (args.student / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="unavailable weights"):
        launch.baseline_identity(args.student)
    (args.student / "missing.safetensors").write_bytes(b"second shard")
    final = launch.baseline_identity(args.student)
    assert len(final["files"]) == 5
    (args.student / "model.safetensors").unlink()
    (args.student / "missing.safetensors").unlink()
    with pytest.raises(ValueError, match="no safetensors"):
        launch.baseline_identity(args.student)


@pytest.mark.parametrize("tasks", [[], [0, 0], [-1], [40], [True], [1.0]])
def test_invalid_task_sets_rejected(tasks):
    with pytest.raises(ValueError, match="nonempty unique integer"):
        launch.audit_selected([], tasks)
