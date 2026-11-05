# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete, relabelled, or changed specialist calibration artifacts."""

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_libero_candidate_configs import candidate

from toolkits.libero_gr00t import teacher_reference as reference


def prepared(monkeypatch, tmp_path):
    cfg = candidate(monkeypatch, tmp_path, 1)
    OmegaConf.resolve(cfg)
    Path(cfg.actor.model.model_path).mkdir()
    for expert in cfg.rollout.expert_models.values():
        root = Path(expert.model_path)
        (root / "experiment_cfg").mkdir(parents=True)
        (root / "config.json").write_text("{}")
        (root / "experiment_cfg/metadata.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"fake unit-test weights")
    config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config)
    folder = tmp_path / "capture"
    reference.prepare(config, Path(sys.executable), folder, 20, 1234)
    return cfg, folder


def populate(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for suite in reference.SUITES:
        protocol = {
            "kind": "protocol",
            "mode": "eval",
            "suite": suite,
            "teacher": manifest["teacher_checkpoints"][suite],
            "eval_roles": ["teacher"],
            "tasks": list(range(10)),
            "trials": 20,
            "seed": 1234,
            "execute": 8,
            "denoise_steps": 4,
            "noise_level": 0.3,
            "world_size": 1,
        }
        rows = [
            {
                "kind": "episode",
                "role": "teacher",
                "suite": suite,
                "task_id": t,
                "trial": i,
                "success_once": i % 2 == 0,
                "complete_episode": True,
                "horizon": reference.HORIZONS[suite],
                "settling_steps": 10,
                "steps": reference.HORIZONS[suite],
            }
            for t in range(10)
            for i in range(20)
        ]
        path = folder / f"{suite}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in [protocol, *rows, {"kind": "complete"}])
            + "\n"
        )
        state = {
            "status": "COMPLETE",
            "started_at": 1000.0,
            "ended_at": 2000.0,
            "command": reference.probe_command(manifest, suite, path),
            "episode_file_sha256": reference.digest(path),
        }
        (folder / f"{suite}.status.json").write_text(json.dumps(state))


def mutate_rows(folder, mutation):
    path = folder / "libero_spatial.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    state_path = folder / "libero_spatial.status.json"
    state = json.loads(state_path.read_text())
    state["episode_file_sha256"] = reference.digest(path)
    state_path.write_text(json.dumps(state))


def test_complete_reference_passes_shared_loader(monkeypatch, tmp_path):
    cfg, folder = prepared(monkeypatch, tmp_path)
    populate(folder)
    result = reference.finalize(folder)
    assert len(result["tasks"]) == 40
    assert result["calibration_budget"]["trajectories"] == 800
    assert result["calibration_budget"]["settling_steps"] == 8000
    assert result["wall_seconds"] == 1000.0
    assert (
        sum(t["simulation_steps"] for t in result["tasks"])
        == result["calibration_budget"]["simulation_steps"]
    )
    values = reference.load_teacher_reference(
        str(folder / "teacher_reference.json"), cfg, result["tasks"]
    )
    assert values == dict.fromkeys(reference.SUITES, 0.5)
    with pytest.raises(FileExistsError):
        reference.finalize(folder)


@pytest.mark.parametrize("episodes", [1, 19, 51, True])
def test_small_screen_cannot_be_prepared(tmp_path, episodes):
    with pytest.raises(ValueError, match="20..50"):
        reference.prepare(
            tmp_path / "missing.yaml",
            Path(sys.executable),
            tmp_path / "out",
            episodes,
            0,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(1),
        lambda rows: rows.insert(1, dict(rows[1])),
        lambda rows: rows[1].update(role="student"),
        lambda rows: rows[1].update(complete_episode=False),
        lambda rows: rows[1].update(success_once=1),
        lambda rows: rows[1].update(horizon=520),
        lambda rows: rows[2].update(steps=20),
        lambda rows: rows[0].update(execute=16),
        lambda rows: rows[0].update(teacher="/wrong/checkpoint"),
    ],
)
def test_bad_records_do_not_export(monkeypatch, tmp_path, mutation):
    _, folder = prepared(monkeypatch, tmp_path)
    populate(folder)
    mutate_rows(folder, mutation)
    with pytest.raises(ValueError):
        reference.finalize(folder)
    assert not (folder / "teacher_reference.json").exists()


def test_changed_checkpoint_rejected(monkeypatch, tmp_path):
    cfg, folder = prepared(monkeypatch, tmp_path)
    populate(folder)
    path = (
        Path(cfg.rollout.expert_models.libero_spatial.model_path) / "model.safetensors"
    )
    path.write_bytes(b"new checkpoint at same path")
    with pytest.raises(ValueError, match="Teacher content changed"):
        reference.finalize(folder)


def test_incomplete_status_and_changed_source_rejected(monkeypatch, tmp_path):
    _, folder = prepared(monkeypatch, tmp_path)
    populate(folder)
    path = folder / "libero_spatial.status.json"
    state = json.loads(path.read_text())
    state["status"] = "RUNNING"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="did not complete"):
        reference.finalize(folder)
    monkeypatch.setattr(reference, "source_identity", lambda: {"other": "changed"})
    with pytest.raises(ValueError, match="implementation changed"):
        reference.finalize(folder)


def test_contract_rejects_teacher_padding_and_truncated_horizon(monkeypatch, tmp_path):
    cfg = candidate(monkeypatch, tmp_path, 1)
    reference.validate_contract(cfg)
    cfg.rollout.expert_models.libero_object.opd_fixed_text_padding = False
    with pytest.raises(ValueError, match="teacher opd_fixed_text_padding"):
        reference.validate_contract(cfg)
    cfg.rollout.expert_models.libero_object.opd_fixed_text_padding = True
    cfg.env.train.task_suite_horizons.libero_10 = 32
    with pytest.raises(ValueError, match="native suite horizons"):
        reference.validate_contract(cfg)
