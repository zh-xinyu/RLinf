# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the verified simulator and new-process path propagation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rlinf.envs.robotwin import runtime_profile as profile

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    root = tmp_path / "robotwin"
    for name in (
        "envs/_base_task.py",
        "envs/robot/robot.py",
        "envs/camera/camera.py",
        "task_config/demo_clean.yml",
        "task_config/_camera_config.yml",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source fixture\n")
    vendor = tmp_path / "vendor"
    package = vendor / "sapien"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("__version__ = '3.0.0b1'\n")
    fake = SimpleNamespace(__version__=profile.SAPIEN_VERSION, __file__=str(module))
    monkeypatch.setenv("ROBOTWIN_PATH", str(root))
    monkeypatch.setenv("OPD_ROBOTWIN_SAPIEN_VENDOR", str(vendor))
    monkeypatch.delenv("OPD_ROBOTWIN_PROFILE", raising=False)
    monkeypatch.setattr(profile.importlib, "import_module", lambda name: fake)
    monkeypatch.setattr(
        profile.importlib.metadata, "version", lambda name: profile.SAPIEN_VERSION
    )
    monkeypatch.setattr(
        profile.subprocess,
        "check_output",
        lambda *args, **kwargs: profile.ROBOTWIN_REVISION,
    )
    profile._validate.cache_clear()
    yield fake
    profile._validate.cache_clear()


def test_valid_runtime_records_actual_module_and_source(fake_runtime):
    result = profile.validate_robotwin4_runtime()
    assert result["sapien_module_path"] == fake_runtime.__file__
    assert result["robotwin_revision"] == profile.ROBOTWIN_REVISION
    assert len(result["robotwin_source_sha256"]) == 5


def test_loaded_wrong_module_rejected_even_if_metadata_is_correct(fake_runtime):
    fake_runtime.__version__ = "3.0.1"
    with pytest.raises(RuntimeError, match="loaded module 3.0.1"):
        profile.validate_robotwin4_runtime()


def test_wrong_distribution_rejected(fake_runtime, monkeypatch):
    monkeypatch.setattr(profile.importlib.metadata, "version", lambda name: "3.0.1")
    with pytest.raises(RuntimeError, match="distribution 3.0.1"):
        profile.validate_robotwin4_runtime()


def test_vendor_shadowing_rejected(fake_runtime, monkeypatch, tmp_path):
    monkeypatch.setenv("OPD_ROBOTWIN_SAPIEN_VENDOR", str(tmp_path / "other"))
    with pytest.raises(RuntimeError, match="requested SAPIEN vendor"):
        profile.validate_robotwin4_runtime()


def test_source_revision_rejected(fake_runtime, monkeypatch):
    monkeypatch.setattr(
        profile.subprocess, "check_output", lambda *args, **kwargs: "wrong-revision"
    )
    with pytest.raises(RuntimeError, match="requires source"):
        profile.validate_robotwin4_runtime()


def test_missing_environment_flag_does_not_bypass_required_config(fake_runtime):
    fake_runtime.__version__ = "3.0.1"
    with pytest.raises(RuntimeError, match="requires SAPIEN"):
        profile.validate_requested_profile(profile.PROFILE)


def test_generic_environment_does_not_import_simulator(monkeypatch):
    monkeypatch.delenv("OPD_ROBOTWIN_PROFILE", raising=False)

    def unexpected_import(name):
        raise AssertionError("Generic environment must not import SAPIEN")

    monkeypatch.setattr(profile.importlib, "import_module", unexpected_import)
    assert profile.validate_requested_profile() is None


def test_dedicated_config_requires_profile():
    config = yaml.safe_load(
        (
            REPO / "examples/embodiment/config/env/robotwin_four_task_clean.yaml"
        ).read_text()
    )
    assert config["required_runtime_profile"] == profile.PROFILE


def test_generated_profile_propagates_vendor_to_child(tmp_path):
    runtime = tmp_path / "runtime"
    vendor = runtime / "vendor/sapien-3.0.0b1"
    package = vendor / "sapien"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '3.0.0b1'\n")
    env = dict(os.environ, ROBOTWIN_RUNTIME_ROOT=str(runtime))
    env.pop("OPD_ROBOTWIN_SAPIEN_VENDOR", None)
    writer = REPO / "toolkits/robotwin_mopd/setup/write_environment.sh"
    subprocess.run(["bash", str(writer)], env=env, check=True)
    code = (
        "import json,os,sapien; print(json.dumps({"
        "'module':sapien.__file__, 'profile':os.environ['OPD_ROBOTWIN_PROFILE'],"
        "'vendor':os.environ['OPD_ROBOTWIN_SAPIEN_VENDOR']}))"
    )
    result = subprocess.check_output(
        [
            "bash",
            "-ec",
            'source "$1"; "$2" -c "$3"',
            "bash",
            str(runtime / "env.sh"),
            sys.executable,
            code,
        ],
        env=env,
        text=True,
    )
    result = json.loads(result)
    assert Path(result["module"]).parent == package
    assert result["vendor"] == str(vendor)
    assert result["profile"] == profile.PROFILE


def test_worker_enforces_config_before_environment_creation(monkeypatch):
    from rlinf.envs.robotwin.multitask_env import RoboTwinMultiTaskEnv

    monkeypatch.delenv("OPD_ROBOTWIN_PROFILE", raising=False)

    def reject_wrong_worker():
        raise RuntimeError("wrong worker SAPIEN")

    monkeypatch.setattr(profile, "validate_robotwin4_runtime", reject_wrong_worker)
    with pytest.raises(RuntimeError, match="wrong worker SAPIEN"):
        RoboTwinMultiTaskEnv(
            cfg={"required_runtime_profile": profile.PROFILE},
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
