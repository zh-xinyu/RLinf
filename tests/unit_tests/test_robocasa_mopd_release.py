# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Check the collaborator matrix, portable calibration and launch isolation."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from evaluations.robocasa365 import formal_eval
from toolkits.robocasa_mopd import owned_ray, resources, run


@pytest.fixture
def release_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("OPD_MODEL_ROOT", str(tmp_path / "teachers"))
    monkeypatch.setenv("OPD_STUDENT_CHECKPOINT", str(tmp_path / "Joint-SFT-Full"))
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("OPD_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    monkeypatch.setenv("EMBODIED_PATH", str(run.REPO / "examples/embodiment"))


def test_matrix_preserves_controls_and_round_budget(release_paths):
    configs = [run.training_config(f"m{i}", f"test-m{i}")[0] for i in range(4)]
    for i, cfg in enumerate(configs):
        r = 1 if i in (0, 3) else 2
        assert cfg.algorithm.flow_refresh_updates == r
        assert cfg.runner.max_steps == r * 150
        assert cfg.runner.save_interval == r * 25
        assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
        assert cfg.runner.rollout_allocation.mode == (
            "uniform" if i < 2 else "adaptive"
        )
        assert cfg.algorithm.sync_suite_loss_weights == (i >= 2)
        assert cfg.env.train.task_sampling_strategy == "suite_quota"
        assert cfg.env.train.task_filter is None
        assert cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch == 96
        assert cfg.actor.global_batch_size == 2304
        assert cfg.actor.micro_batch_size == 32
        assert cfg.actor.global_batch_size // (8 * cfg.actor.micro_batch_size) == 9
        assert cfg.actor.optim.lr == 3e-5
        assert cfg.actor.optim.lr_warmup_steps == 10
        assert cfg.actor.optim.lr_scheduler == "constant"
        assert cfg.actor.cuda_cache_retention_gib == 0
        assert cfg.actor.model.skip_backbone_logits
        assert cfg.algorithm.flow_observation_sampling == "full_pool"
    # Ignore only the intended experimental factors and unique output names.
    normalized = []
    for cfg in configs:
        value = OmegaConf.to_container(cfg, resolve=True)
        for key in ("max_steps", "save_interval"):
            value["runner"].pop(key)
        value["runner"]["logger"].pop("experiment_name")
        value["runner"]["rollout_allocation"].pop("mode")
        for key in ("flow_refresh_updates", "sync_suite_loss_weights"):
            value["algorithm"].pop(key)
        normalized.append(
            json.loads(
                json.dumps(value).replace(cfg.runner.logger.experiment_name, "RUN")
            )
        )
    assert all(c == normalized[0] for c in normalized)


def test_smoke_only_changes_budget_output_and_logging(release_paths):
    cfg, output, _ = run.training_config("m2", "smoke-test", smoke=True)
    assert cfg.runner.max_steps == 4 and cfg.runner.save_interval == 4
    assert cfg.env.train.task_filter is None
    assert cfg.runner.resume_dir is None
    assert cfg.runner.logger.logger_backends == ["tensorboard"]
    assert output.parent.name == "smoke"


def test_resource_binding_verifies_bytes_before_relocating(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = {"Joint-SFT-Full": tmp_path / "Joint-SFT-Full"}
    suites = ("atomic_seen", "composite_seen", "composite_unseen")
    for suite in suites:
        paths[suite] = (
            tmp_path / "teachers" / "target_posttraining" / suite / "checkpoint-60000"
        )
    manifest = {"checkpoints": {}}
    for name, path in paths.items():
        path.mkdir(parents=True)
        data = name.encode()
        (path / "model.safetensors").write_bytes(data)
        manifest["checkpoints"][name] = {
            "model.safetensors": {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        }
    reference = {
        "teacher_checkpoints": dict.fromkeys(suites, "old"),
        "tasks": [{"episodes": 20}],
        "training_signature": {"invariant": 1},
    }
    (inputs / "checkpoints.json").write_text(json.dumps(manifest))
    (inputs / "teacher_reference.portable.json").write_text(json.dumps(reference))
    monkeypatch.setattr(resources, "INPUTS", inputs)
    output = tmp_path / "bound.json"
    resources.prepare(paths["Joint-SFT-Full"], tmp_path / "teachers", output)
    bound = json.loads(output.read_text())
    assert bound["tasks"] == reference["tasks"]
    assert bound["training_signature"] == reference["training_signature"]
    assert bound["teacher_checkpoints"] == {s: str(paths[s]) for s in suites}
    assert bound["checkpoint_manifest_sha256"] == formal_eval.sha256(
        inputs / "checkpoints.json"
    )
    corrupted = paths["Joint-SFT-Full"] / "model.safetensors"
    corrupted.write_bytes(b"x" * corrupted.stat().st_size)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resources.prepare(
            paths["Joint-SFT-Full"], tmp_path / "teachers", tmp_path / "bad.json"
        )
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("mode,budget", [("formal", 50), ("intermediate", 20)])
def test_eval_mode_locks_task_set_and_budget(tmp_path, monkeypatch, mode, budget):
    (tmp_path / "examples").symlink_to(run.REPO / "examples", target_is_directory=True)
    monkeypatch.setattr(formal_eval, "REPO", tmp_path)
    monkeypatch.setattr(formal_eval, "source_hashes", lambda: {})
    plan = {
        "protocol": formal_eval.PROTOCOL,
        "mode": mode,
        "source_sha256": {},
        "manifest_sha256": formal_eval.sha256(formal_eval.MANIFEST),
        "seed": 7,
        "tasks": formal_eval.manifest_tasks(),
        "episodes_per_task": budget,
        "envs_per_task": 5,
        "output": str(tmp_path / "outputs/eval/run"),
        "gpus": [0],
    }
    formal_eval.validate_plan(plan)
    cfg = formal_eval.compose_task_config(
        {**plan, "base_checkpoint": "/base", "weights": None, "save_video": False},
        plan["tasks"][0],
        0,
    )
    assert cfg.env.eval.rollout_epoch == budget // 5
    with pytest.raises(RuntimeError, match="budget"):
        formal_eval.validate_plan(
            {**plan, "episodes_per_task": 20 if budget == 50 else 50}
        )
    with pytest.raises(RuntimeError, match="budget"):
        formal_eval.validate_plan({**plan, "tasks": plan["tasks"][:-1]})


def test_owned_cleanup_excludes_peer_and_other_user(monkeypatch):
    def process(pid, uid, owner):
        p = Mock(pid=pid)
        p.uids.return_value.real = uid
        p.status.return_value = "running"
        p.environ.return_value = {owned_ray.OWNER_KEY: owner}
        return p

    ours = process(10, os.getuid(), "ours")
    peer = process(11, os.getuid(), "peer")
    other_user = process(12, os.getuid() + 1, "ours")
    monkeypatch.setattr(
        owned_ray.psutil, "process_iter", lambda: [ours, peer, other_user]
    )
    assert owned_ray.owned_processes("ours") == [ours]


def test_gpu_gate_refuses_busy_cards(monkeypatch):
    monkeypatch.setattr(
        owned_ray.subprocess, "check_output", lambda *a, **kw: "0, 25000, 49140\n"
    )
    with pytest.raises(RuntimeError, match="busy"):
        owned_ray.check_gpus(1, {"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="selected 1"):
        owned_ray.check_gpus(8, {"CUDA_VISIBLE_DEVICES": "0"})


def test_driver_failure_is_propagated_without_killing_finished_process(
    tmp_path, monkeypatch
):
    child = Mock()
    child.wait.return_value = 2
    child.poll.return_value = 2
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **kw: child)
    kill = Mock()
    monkeypatch.setattr(run.os, "killpg", kill)
    with pytest.raises(run.subprocess.CalledProcessError):
        run.run_command(["false"], tmp_path / "driver.log", {})
    kill.assert_not_called()


def test_shared_reference_cannot_silently_change(release_paths, tmp_path):
    cfg, _, _ = run.training_config("m2", "reference-test")
    reference = json.loads((run.INPUTS / "teacher_reference.portable.json").read_text())
    reference["teacher_checkpoints"] = {
        name: str(Path(teacher.model_path).resolve())
        for name, teacher in cfg.rollout.expert_models.items()
    }
    reference["checkpoint_manifest_sha256"] = formal_eval.sha256(
        run.INPUTS / "checkpoints.json"
    )
    path = Path(cfg.runner.rollout_allocation.teacher_reference_file)
    path.write_text(json.dumps(reference))
    run.check_reference(cfg)
    reference["tasks"][0]["successes"] += 1
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="Shared teacher reference changed"):
        run.check_reference(cfg)
