# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for author-cache provenance and strict fixed-pool replay."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from toolkits.robotwin_mopd import author_pool as module


def pool_fixture():
    return {
        "schema_version": 2,
        "complete": True,
        "protocol": {
            "name": module.PROTOCOL_NAME,
            "instruction_type": "unseen",
            "task_names": list(module.TASKS),
            "episodes_per_task": 2,
            "selection_count": 2,
            "selection_start": 0,
            "author_cohort": 0,
            "author_revision": module.AUTHOR_REVISION,
            "author_cache_sha256": module.CACHE_SHA256,
            "online_expert_revalidation": False,
            "allow_reseed": False,
            "task_config_sha256": module.canonical_sha256({"is_test": True}),
            "task_horizons": dict.fromkeys(module.TASKS, 400),
            "robotwin_revision": "local-revision",
            "source_files_sha256": {"envs/_base_task.py": "filehash"},
            "runtime_profile": {"sapien_version": "3.0.0b1"},
            "author_reference_root": "/reference",
        },
        "tasks": {
            task: [
                {
                    "scene_seed": seed,
                    "episode_info": {"{a}": "left"},
                    "instruction": f"prompt {seed}",
                }
                for seed in (100001, 100002)
            ]
            for task in module.TASKS
        },
    }


def save_load(tmp_path, pool):
    path = tmp_path / "pool.json"
    module.write_pool(path, pool)
    return module.load_pool(path)


def test_valid_pool_and_tampering(tmp_path):
    pool = save_load(tmp_path, pool_fixture())
    pool["tasks"][module.TASKS[0]][0]["instruction"] = "tampered"
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="SHA256"):
        module.load_pool(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(schema_version=1), "schema"),
        (lambda p: p.update(complete=False), "incomplete"),
        (lambda p: p["protocol"].update(allow_reseed=True), "reseeding"),
        (
            lambda p: p["protocol"].update(online_expert_revalidation=True),
            "local-screening",
        ),
        (lambda p: p["protocol"].update(author_cache_sha256="wrong"), "provenance"),
        (lambda p: p["protocol"].update(author_cohort=3), "cohort"),
        (lambda p: p["tasks"][module.TASKS[0]].pop(), "count"),
        (
            lambda p: p["tasks"][module.TASKS[0]][1].update(scene_seed=100001),
            "duplicate",
        ),
    ],
)
def test_invalid_structure_even_with_recomputed_hash(tmp_path, mutation, match):
    pool = pool_fixture()
    mutation(pool)
    with pytest.raises(ValueError, match=match):
        save_load(tmp_path, pool)


def runtime_stubs(monkeypatch, pool):
    monkeypatch.setattr(module, "revision", lambda _: "local-revision")
    monkeypatch.setattr(
        module, "source_hashes", lambda _: {"envs/_base_task.py": "filehash"}
    )
    cache = {
        "tasks": {
            task: {
                "0": [
                    {k: row[k] for k in ("scene_seed", "episode_info")} for row in rows
                ]
            }
            for task, rows in pool["tasks"].items()
        }
    }
    monkeypatch.setattr(module, "author_cache", lambda _: cache)
    monkeypatch.setattr(
        module, "author_instruction", lambda ref, task, info, seed: f"prompt {seed}"
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.robotwin.runtime_profile",
        SimpleNamespace(
            validate_robotwin4_runtime=lambda: {"sapien_version": "3.0.0b1"}
        ),
    )


def validate(pool, tmp_path, **kwargs):
    params = {
        "task_config": {"is_test": True},
        "task_horizons": dict.fromkeys(module.TASKS, 400),
        "robotwin_root": tmp_path,
    }
    params.update(kwargs)
    return module.validate_pool_protocol(pool, **params)


def test_runtime_and_cache_validation(monkeypatch, tmp_path):
    pool = pool_fixture()
    runtime_stubs(monkeypatch, pool)
    assert validate(pool, tmp_path) == {"sapien_version": "3.0.0b1"}
    with pytest.raises(ValueError, match="task_config"):
        validate(pool, tmp_path, task_config={})
    with pytest.raises(ValueError, match="horizons"):
        validate(pool, tmp_path, task_horizons=dict.fromkeys(module.TASKS, 900))
    changed = copy.deepcopy(pool)
    changed["protocol"]["runtime_profile"]["sapien_version"] = "3.0.1"
    with pytest.raises(ValueError, match="runtime profile"):
        validate(changed, tmp_path)
    changed = copy.deepcopy(pool)
    changed["protocol"]["source_files_sha256"] = {}
    with pytest.raises(ValueError, match="source-file"):
        validate(changed, tmp_path)
    changed = copy.deepcopy(pool)
    changed["tasks"][module.TASKS[0]][0]["scene_seed"] = 999
    with pytest.raises(ValueError, match="predeclared"):
        validate(changed, tmp_path)
    changed = copy.deepcopy(pool)
    changed["tasks"][module.TASKS[0]][0]["instruction"] = "wrong"
    with pytest.raises(ValueError, match="instruction mismatch"):
        validate(changed, tmp_path)


def test_provider_cannot_reseed_or_change_prompt():
    entry = {"scene_seed": 123, "instruction": "fixed"}
    module.validate_replay_identity(entry, 123, ["fixed"])
    with pytest.raises(RuntimeError, match="reseeded"):
        module.validate_replay_identity(entry, 124, ["fixed"])
    with pytest.raises(RuntimeError, match="instruction"):
        module.validate_replay_identity(entry, 123, ["changed"])


def test_author_guard_checks_every_released_source(monkeypatch, tmp_path):
    hashes = {f"envs/task_{i}.py": "good" for i in range(54)}
    cache_path = tmp_path / "pace/configs/expert_scenes.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"compatible_source_files_sha256": hashes}))
    monkeypatch.setattr(module, "revision", lambda _: module.AUTHOR_REVISION)
    seen = []

    def digest(path):
        seen.append(path)
        return module.CACHE_SHA256 if path == cache_path else "good"

    monkeypatch.setattr(module, "file_sha256", digest)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: None)
    module.author_cache(tmp_path)
    assert len(seen) == 55
    monkeypatch.setattr(
        module,
        "file_sha256",
        lambda p: module.CACHE_SHA256 if p == cache_path else "bad",
    )
    with pytest.raises(ValueError, match="source guard"):
        module.author_cache(tmp_path)


@pytest.mark.parametrize("provider_name", ["author_pool", "feasible_pool"])
def test_evaluator_dispatch_preserves_each_validator(
    monkeypatch, tmp_path, provider_name
):
    from omegaconf import OmegaConf

    from toolkits.robotwin_mopd import evaluate, feasible_pool

    provider = module if provider_name == "author_pool" else feasible_pool
    name = provider.PROTOCOL_NAME
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"protocol": {"name": name}}))
    calls = []
    loaded = {
        "protocol": {"task_names": ["adjust_bottle"]},
        "tasks": {"adjust_bottle": [{"scene_seed": 1}]},
    }
    monkeypatch.setattr(provider, "load_pool", lambda path: loaded)
    monkeypatch.setattr(
        provider, "validate_pool_protocol", lambda pool, **kw: calls.append(kw)
    )
    monkeypatch.setenv("ROBOTWIN_PATH", str(tmp_path))
    cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "task_config": {"is_test": True},
                    "task_horizons": {"adjust_bottle": 400},
                }
            }
        }
    )
    pool, entries = evaluate.read_evaluation_pool(path, "adjust_bottle", 1, cfg)
    assert pool is loaded and entries == [{"scene_seed": 1}]
    assert calls[0]["task_config"] == {"is_test": True}
    with pytest.raises(ValueError, match="fewer scenes"):
        evaluate.read_evaluation_pool(path, "adjust_bottle", 2, cfg)


def test_runtime_identity_allows_relocation_but_rejects_changed_content():
    a = {
        "profile": "pi05-four-task",
        "sapien_version": "3.0.0b1",
        "sapien_module_path": "/node-a/sapien/__init__.py",
        "sapien_module_sha256": "hash",
        "sapien_vendor_manifest": {"wheel_sha256": "wheel", "path": "/node-a"},
    }
    b = copy.deepcopy(a)
    b["sapien_module_path"] = "/node-b/sapien/__init__.py"
    b["sapien_vendor_manifest"]["path"] = "/node-b"
    assert module.runtime_identity(a) == module.runtime_identity(b)
    for key in ("sapien_version", "sapien_module_sha256"):
        changed = copy.deepcopy(b)
        changed[key] = "different"
        assert module.runtime_identity(a) != module.runtime_identity(changed)
    b["sapien_vendor_manifest"]["wheel_sha256"] = "different"
    assert module.runtime_identity(a) != module.runtime_identity(b)
