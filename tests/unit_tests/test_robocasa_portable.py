# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Four/eight GPU controls, Slurm visibility and matching smoke requirements."""

import argparse
import hashlib
import json

import pytest
from omegaconf import OmegaConf

from toolkits.robocasa_mopd import experiment, hardware, resources, run


def test_slurm_machine_can_be_prepared_on_login_node(tmp_path, monkeypatch):
    value = json.loads(
        (
            experiment.REPO / "toolkits/robocasa_mopd/machine.slurm.example.json"
        ).read_text()
    )
    for key in ("runtime_root", "resource_root", "student_checkpoint", "output_root"):
        value[key] = str(tmp_path / key)
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(value))
    machine = experiment.load_machine(path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,6,7")
    env = experiment.environment(machine)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3,6,7"
    assert env["OPD_NUM_GPUS"] == "4" and env["OPD_RESOURCE_MODE"] == "dedicated"
    assert hardware.gpu_ids(env) == [2, 3, 6, 7]
    env.pop("SLURM_JOB_ID", None)
    with pytest.raises(ValueError, match="allocation"):
        hardware.gpu_ids(env, require_allocation=True)
    env.update(SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="32", SLURM_JOB_NUM_NODES="1")
    assert hardware.gpu_ids(env, require_allocation=True) == [2, 3, 6, 7]
    env["SLURM_CPUS_PER_TASK"] = "16"
    with pytest.raises(ValueError, match="cpus-per-task"):
        hardware.gpu_ids(env, require_allocation=True)
    commands = experiment.commands_for(argparse.Namespace(command="smoke"), machine)
    assert commands[0][-1] == "--gpu"
    assert machine["runtime_profile"] in commands[0]
    assert commands[1][-1] == "toolkits.robocasa_mopd.check_logging"
    assert commands[-1][-2] == "--weights"
    assert "global_step_4/actor/model_state_dict/full_weights.pt" in commands[-1][-1]


@pytest.mark.parametrize("visible", ["0,1", "0,0,1,2", "", "MIG-abc,1,2,3"])
def test_bad_allocations_fail_closed(visible):
    env = {"OPD_NUM_GPUS": "4", "CUDA_VISIBLE_DEVICES": visible}
    with pytest.raises(ValueError):
        hardware.gpu_ids(env, require_allocation=True)


def test_uuid_conversion_preserves_only_allocated_devices(monkeypatch):
    mapping = {"GPU-aa": "2", "GPU-bb": "3", "GPU-cc": "6", "GPU-dd": "7"}
    calls = []

    def query(command, **kwargs):
        token = command[-1].split("=", 1)[1]
        calls.append(token)
        return mapping[token]

    monkeypatch.setattr(hardware.subprocess, "check_output", query)
    assert hardware.gpu_ids(
        {"OPD_NUM_GPUS": "4", "CUDA_VISIBLE_DEVICES": ",".join(mapping)}
    ) == [2, 3, 6, 7]
    assert calls == list(mapping)


@pytest.mark.parametrize("arm", ["m0", "m1", "m2", "m3"])
def test_hardware_selection_does_not_change_scientific_recipe(
    monkeypatch, tmp_path, arm
):
    for key in (
        "OPD_MODEL_ROOT",
        "OPD_STUDENT_CHECKPOINT",
        "OPD_RUN_DIR",
        "OPD_TEACHER_REFERENCE",
    ):
        monkeypatch.setenv(key, str(tmp_path / key))
    monkeypatch.setenv("EMBODIED_PATH", str(run.REPO / "examples/embodiment"))
    configs = []
    for world in (4, 8):
        monkeypatch.setenv("OPD_NUM_GPUS", str(world))
        monkeypatch.setenv(
            "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(world))
        )
        monkeypatch.setenv("OPD_RESOURCE_MODE", "dedicated")
        cfg, _, _ = run.training_config(arm, "test")
        assert cfg.actor.global_batch_size == 2304
        assert cfg.actor.micro_batch_size == 32
        assert 2304 // (world * 32) == (18 if world == 4 else 9)
        assert (
            cfg.actor.cuda_cache_preallocate_gib
            == cfg.actor.cuda_cache_retention_gib
            == 0
        )
        value = OmegaConf.to_container(cfg, resolve=True)
        value["cluster"].pop("component_placement")
        configs.append(value)
    assert configs[0] == configs[1]
    monkeypatch.setenv("OPD_NUM_GPUS", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,6,7")
    cfg, _, _ = run.training_config(arm, "test")
    assert dict(cfg.cluster.component_placement) == dict.fromkeys(
        ("actor", "env", "rollout"), "2,3,6,7"
    )


def test_smoke_must_match_current_commit_and_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REPO", tmp_path)
    monkeypatch.setattr(run.subprocess, "check_output", lambda *a, **k: "abc\n")
    env = {"OPD_ENV_PROFILE": hardware.PROFILES[1], "OPD_RESOURCE_MODE": "dedicated"}
    record = {
        "commit": "abc",
        "profile": env["OPD_ENV_PROFILE"],
        "num_gpus": 4,
        "resource_mode": "dedicated",
        "checkpoint_manifest_sha256": run.sha256(run.INPUTS / "checkpoints.json"),
    }
    with pytest.raises(ValueError, match="smoke"):
        run.require_completed_smoke(env, 4)
    for name in ("setup-m2-smoke", "setup-joint-sft-full-eval-smoke"):
        p = tmp_path / "outputs/launches" / name
        p.mkdir(parents=True)
        (p / "status").write_text("OK\n")
        (p / "launch.json").write_text(json.dumps(record))
    run.require_completed_smoke(env, 4)
    with pytest.raises(ValueError, match="mismatch"):
        run.require_completed_smoke(env, 8)


def test_student_download_verifies_bytes_and_never_overwrites(tmp_path, monkeypatch):
    import huggingface_hub

    data = b"inference weights"
    cached = tmp_path / "cached"
    cached.write_bytes(data)
    manifest = {
        "student_initialization": {
            "hf_repo": "owner/model",
            "hf_revision": "fixed",
            "hf_prefix": "prefix",
        },
        "checkpoints": {
            "Joint-SFT-Full": {
                "model.safetensors": {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            }
        },
    }
    (tmp_path / "checkpoints.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(resources, "INPUTS", tmp_path)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **kwargs: str(cached)
    )
    target = tmp_path / "student"
    resources.download_student(target)
    resources.download_student(target)
    (target / "model.safetensors").write_bytes(b"other checkpoint")
    with pytest.raises(ValueError):
        resources.download_student(target)
    assert (target / "model.safetensors").read_bytes() == b"other checkpoint"


def test_slurm_submission_matrix_includes_all_independent_evaluations(tmp_path):
    from toolkits.robocasa_mopd import submit_slurm

    jobs = submit_slurm.stages()
    assert jobs[:4] == [["train", a] for a in ("m0", "m1", "m2", "m3")]
    assert len(jobs) == 29
    assert sum(j[0] == "eval" for j in jobs) == 25
    args = argparse.Namespace(
        partition="confirmed-partition",
        account="account",
        time_limit="14-00:00:00",
        machine=tmp_path / "machine.json",
    )
    cmd = submit_slurm.submission(
        args, {"num_gpus": 4, "ray_cpus": 32}, jobs[-1], "123"
    )
    assert "--gpus=4" in cmd and "--dependency=afterok:123" in cmd
    assert cmd[-4:] == ["eval", "m3", "--round", "150"]


@pytest.mark.parametrize(
    "profile,torch_version",
    [(hardware.PROFILES[0], "2.6.0"), (hardware.PROFILES[1], "2.7.1+cu128")],
)
def test_evaluation_verifies_editable_and_wheel_versions(
    tmp_path, monkeypatch, profile, torch_version
):
    from types import SimpleNamespace

    from evaluations.robocasa365 import formal_eval

    monkeypatch.setenv("OPD_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("OPD_ENV_PROFILE", profile)
    for package in formal_eval.EXPECTED_COMMITS:
        (tmp_path / "repos" / package).mkdir(parents=True)
    monkeypatch.setattr(
        formal_eval.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(tmp_path / "repos" / name / name / "__init__.py")
        ),
    )
    monkeypatch.setattr(
        formal_eval,
        "git_output",
        lambda path, *args: formal_eval.EXPECTED_COMMITS[path.name],
    )
    versions = dict(formal_eval.EXPECTED_PACKAGES, torch=torch_version)
    monkeypatch.setattr(
        formal_eval.importlib.metadata, "version", lambda name: versions[name]
    )
    report = formal_eval.validate_runtime()
    assert report["profile"] == profile
    assert report["package_versions"]["robosuite"] == "1.5.2"
    versions["torch"] = "unexpected"
    with pytest.raises(RuntimeError, match="Frozen package versions changed"):
        formal_eval.validate_runtime()
