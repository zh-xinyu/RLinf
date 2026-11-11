# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Calibration relocation must preserve content and inference protocol."""

import json
import shutil

import pytest
from omegaconf import OmegaConf

from rlinf.utils.rollout_allocation import load_teacher_reference, training_signature
from toolkits.libero_gr00t import reference_bundle as bundle
from toolkits.libero_gr00t import run_uniform as entry


def fixture(tmp_path):
    machine = tmp_path / "machine.json"
    settings = {key: str(tmp_path / key) for key in entry.REQUIRED}
    machine.write_text(json.dumps(settings))
    for path in [
        tmp_path / "spatial_teacher",
        *[tmp_path / "models" / s for s in ("object", "goal", "long")],
    ]:
        (path / "experiment_cfg").mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "experiment_cfg/metadata.json").write_text("{}")
        (path / "model.safetensors").write_bytes(path.name.encode())
    cfg = bundle.bind_config(machine, 2, tmp_path / "bound.json")
    config = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, config)
    reference = {
        "schema_version": 1,
        "status": "COMPLETE",
        "training_signature": training_signature(cfg),
        "teacher_checkpoints": {
            s: str(cfg.rollout.expert_models[s].model_path) for s in bundle.SUITES
        },
        "tasks": [
            {
                **task,
                "episodes": 50,
                "successes": 25,
                "simulation_steps": 500,
                "settling_steps": 500,
            }
            for task in bundle.tasks()
        ],
        "wall_seconds": 10,
        "calibration_budget": {"trajectories": 2000},
        "provenance": {
            "capture": "/private/author/path",
            "episode_files_sha256": {"capture": "verified"},
        },
    }
    source = tmp_path / "source_reference.json"
    source.write_text(json.dumps(reference))
    exported = tmp_path / "bundle.json"
    bundle.export_bundle(config, source, exported)
    return cfg, config, source, exported


def test_portable_reference_export_and_content_verified_relocation(tmp_path):
    cfg, config, source, exported = fixture(tmp_path)
    text = exported.read_text()
    assert "/private/author/path" not in text
    assert str(tmp_path / "spatial_teacher") not in text
    assert str(tmp_path / "models") not in text
    for suite in bundle.SUITES:
        original = cfg.rollout.expert_models[suite].model_path
        relocated = tmp_path / "other_node" / suite
        shutil.copytree(original, relocated)
        cfg.rollout.expert_models[suite].model_path = str(relocated)
    output = tmp_path / "relocated_reference.json"
    result = bundle.bind_bundle(exported, cfg, output)
    assert bundle.bind_bundle(exported, cfg, output) == result
    assert result["calibration_budget"] == {"trajectories": 2000}
    assert result["tasks"] == json.loads(source.read_text())["tasks"]
    assert load_teacher_reference(str(output), cfg, bundle.tasks()) == dict.fromkeys(
        bundle.SUITES, 0.5
    )


def test_changed_teacher_metadata_cannot_be_rebound(tmp_path):
    cfg, config, source, exported = fixture(tmp_path)
    path = tmp_path / "models/object/experiment_cfg/metadata.json"
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="content differs"):
        bundle.bind_bundle(exported, cfg, tmp_path / "bound.json")


def test_relocation_cannot_change_execution_protocol(tmp_path):
    cfg, config, source, exported = fixture(tmp_path)
    cfg.env.train.reset_settle_steps = 11
    with pytest.raises(ValueError, match="protocol differs"):
        bundle.bind_bundle(exported, cfg, tmp_path / "bound.json")


def test_release_export_requires_requested_calibration_coverage(tmp_path):
    cfg, config, source, exported = fixture(tmp_path)
    with pytest.raises(ValueError, match="at least 51"):
        bundle.export_bundle(config, source, tmp_path / "insufficient.json", minimum=51)


def test_incomplete_task_set_cannot_be_rebound(tmp_path):
    cfg, config, source, exported = fixture(tmp_path)
    data = json.loads(exported.read_text())
    data["reference"]["tasks"].pop()
    exported.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="task set"):
        bundle.bind_bundle(exported, cfg, tmp_path / "bound.json")


def test_reference_without_measured_budget_is_not_published(tmp_path):
    cfg, config, source, exported = fixture(tmp_path)
    reference = json.loads(source.read_text())
    reference.pop("wall_seconds")
    source.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="wall_seconds"):
        bundle.export_bundle(config, source, tmp_path / "invalid_bundle.json")
