# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable profile contract and existing-runtime protection tests."""

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "opd_profile", REPO / "requirements/opd_profile.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_fixed_sources_and_package_snapshot():
    profile = PROFILE.load_profile()
    assert {item["name"] for item in profile["sources"]} == {
        "gr00t",
        "robocasa",
        "robosuite",
        "lerobot",
    }
    for item in profile["sources"]:
        assert re.fullmatch(r"[0-9a-f]{40}", item["commit"])
        assert item["url"].startswith("https://github.com/")
    pins = PROFILE.package_pins()
    assert len(pins) == 275
    assert pins["torch"] == "2.6.0" and pins["mujoco"] == "3.3.1"
    assert all(not re.search(r"[/;<>*]", value) for value in pins.values())
    wheel = profile["flash_attention"]["url"]
    assert "cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" in wheel


def test_existing_runtime_is_never_modified(tmp_path):
    sentinel = tmp_path / "existing-training-data"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        PROFILE.install(tmp_path, dry_run=False)
    assert sentinel.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_dry_run_without_any_private_files(tmp_path, monkeypatch, capsys):
    # Only public installer + profile are available in this simulated clone.
    public = tmp_path / "clone"
    shutil.copytree(
        PROFILE.PROFILE, public / "requirements/profiles" / PROFILE.PROFILE.name
    )
    monkeypatch.setattr(PROFILE, "REPO", public)
    monkeypatch.setattr(
        PROFILE, "PROFILE", public / "requirements/profiles" / PROFILE.PROFILE.name
    )
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\necho 'uv 0.12.0'\n")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(PROFILE.shutil, "which", lambda _: str(fake_uv))
    runtime = tmp_path / "new-runtime"
    PROFILE.install(runtime, dry_run=True)
    assert not runtime.exists()
    assert json.loads(capsys.readouterr().out)["pypi_packages"] == 275


def test_incorrect_uv_fails_before_creating_runtime(tmp_path, monkeypatch):
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\necho 'uv 0.0.1'\n")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(PROFILE.shutil, "which", lambda _: str(fake_uv))
    runtime = tmp_path / "new-runtime"
    with pytest.raises(ValueError, match="Expected uv"):
        PROFILE.install(runtime, dry_run=False)
    assert not runtime.exists()


@pytest.mark.parametrize("mode,sync_expected", [("create-only", False), ("", True)])
def test_shared_venv_helper_keeps_normal_sync_behavior(tmp_path, mode, sync_expected):
    log = tmp_path / "commands"
    script = r"""
source "$1"
VENV_DIR="$2"
install_uv() { :; }
# Fake only the venv creation/activation; no environment or package installation.
uv() {
    printf '%s\n' "$*" >> "$OPD_TEST_LOG"
    if [ "$1" = venv ]; then
        mkdir -p "$2/bin"
        printf ':\n' > "$2/bin/activate"
    fi
}
export OPD_TEST_LOG="$3"
create_and_sync_venv "$4"
"""
    subprocess.run(
        [
            "bash",
            "-c",
            script,
            "test",
            str(REPO / "requirements/install.sh"),
            str(tmp_path / "venv"),
            str(log),
            mode,
        ],
        check=True,
    )
    assert ("sync --active" in log.read_text()) == sync_expected


def test_private_python_override_does_not_leak_into_install(tmp_path, monkeypatch):
    monkeypatch.setenv("RLINF_PYTHON_EXECUTABLE", "/private/runtime/python")
    monkeypatch.setenv("PYTHONPATH", "/private/overlay")
    monkeypatch.setattr(PROFILE.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(PROFILE.subprocess, "check_output", lambda *a, **k: "uv 0.12.0")
    captured = {}

    def stop_before_install(command, **kwargs):
        captured.update(kwargs["env"])
        raise RuntimeError("stop before creating venv")

    monkeypatch.setattr(PROFILE, "run", stop_before_install)
    with pytest.raises(RuntimeError, match="stop before creating venv"):
        PROFILE.install(tmp_path / "new-runtime", dry_run=False)
    assert "RLINF_PYTHON_EXECUTABLE" not in captured
    assert "PYTHONPATH" not in captured
    assert captured["UV_PYTHON_PREFERENCE"] == "only-managed"
