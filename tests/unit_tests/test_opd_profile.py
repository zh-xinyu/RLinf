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
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setenv("BASH_ENV", "/private/shell-hook")
    monkeypatch.setenv("ENV", "/private/shell-hook")
    monkeypatch.setenv("GIT_LFS_SKIP_SMUDGE", "1")
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
    assert "BASH_ENV" not in captured
    assert "ENV" not in captured
    assert "GIT_LFS_SKIP_SMUDGE" not in captured
    assert captured["UV_PYTHON_PREFERENCE"] == "only-managed"


LIBERO_PROFILE = "gr00t-n15-libero-20261105"


def test_libero_profile_is_explicit_and_preserves_robocasa_default():
    default = PROFILE.load_profile()
    directory = PROFILE.resolve_profile(LIBERO_PROFILE)
    libero = PROFILE.load_profile(directory)
    assert PROFILE.load_profile() == default
    assert default["id"] == "gr00t-n15-robocasa365-20261019"
    assert libero["id"] == LIBERO_PROFILE
    assert {item["name"] for item in libero["sources"]} == {"gr00t", "libero"}
    assert all(re.fullmatch(r"[0-9a-f]{40}", x["commit"]) for x in libero["sources"])
    pins = PROFILE.package_pins(directory)
    assert len(pins) == 262
    for name, expected in {
        "ray": "2.56.1",
        "pyarrow": "25.0.0",
        "sympy": "1.13.1",
        "numpy": "1.26.4",
        "robosuite": "1.4.1",
        "mujoco": "3.3.7",
    }.items():
        assert pins[name] == expected
    assert "robocasa" not in libero["verification_imports"]
    for path in directory.iterdir():
        assert not re.search(r"/egr/|/localscratch|file://", path.read_text())


@pytest.mark.parametrize(
    "name", ["../escape", "/tmp/profile", ".", "..", "", "missing"]
)
def test_profile_selector_rejects_unbundled_paths(name):
    with pytest.raises(ValueError, match="profile"):
        PROFILE.resolve_profile(name)


def test_profile_selector_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(PROFILE, "PROFILE", tmp_path / "default")
    external = tmp_path / "external"
    external.mkdir()
    (external / "profile.json").write_text("{}")
    (tmp_path / "alias").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="Unknown bundled"):
        PROFILE.resolve_profile("alias")


def test_libero_install_passes_profile_to_child_and_isolates_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(PROFILE.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(PROFILE.subprocess, "check_output", lambda *a, **k: "uv 0.10.4")
    global_config = tmp_path / "user-libero"
    global_config.mkdir()
    sentinel = global_config / "config.yaml"
    sentinel.write_text("keep user config")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(global_config))
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "6")
    monkeypatch.setenv("OPD_LIBERO_AUTOMAP_EGL", "1")
    calls = []
    monkeypatch.setattr(
        PROFILE, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    runtime = tmp_path / "new-runtime"
    PROFILE.install(runtime, dry_run=False, profile_name=LIBERO_PROFILE)
    verify_command, kwargs = next((cmd, kw) for cmd, kw in calls if "verify" in cmd)
    assert verify_command[-2:] == ["--profile", LIBERO_PROFILE]
    assert kwargs["env"]["LIBERO_CONFIG_PATH"] == str(runtime / "libero-config")
    assert kwargs["env"]["UV_LINK_MODE"] == "copy"
    assert "MUJOCO_EGL_DEVICE_ID" not in kwargs["env"]
    assert "OPD_LIBERO_AUTOMAP_EGL" not in kwargs["env"]
    config = json.loads((runtime / "libero-config/config.yaml").read_text())
    assert config == PROFILE.libero_config(runtime)
    assert all(Path(path).is_relative_to(runtime) for path in config.values())
    assert sentinel.read_text() == "keep user config"
    activation = (runtime / "activate.sh").read_text()
    assert f"export LIBERO_CONFIG_PATH={runtime}/libero-config" in activation
    assert "unset MUJOCO_EGL_DEVICE_ID OPD_LIBERO_AUTOMAP_EGL" in activation
    requirements = PROFILE.resolve_profile(LIBERO_PROFILE) / "requirements.txt"
    assert any(requirements in command for command, _ in calls)
    for cmd, _ in calls:
        if cmd[0] == "bash":
            assert cmd[:3] == ["bash", "--noprofile", "--norc"]
    clone_calls = [(cmd, kw) for cmd, kw in calls if "opd-profile" in cmd]
    for name, expected in [("gr00t", "1"), ("libero", None)]:
        _, clone_kwargs = next(
            (cmd, kw) for cmd, kw in clone_calls if runtime / "repos" / name in cmd
        )
        assert clone_kwargs["env"].get("GIT_LFS_SKIP_SMUDGE") == expected
    # The source-specific choice never reaches package installation or verification.
    assert "GIT_LFS_SKIP_SMUDGE" not in kwargs["env"]


def test_libero_verify_rejects_foreign_config_before_imports(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    profile = PROFILE.load_profile(PROFILE.resolve_profile(LIBERO_PROFILE))
    runtime.mkdir()
    PROFILE.prepare_profile_config(runtime, profile)
    path = runtime / "libero-config/config.yaml"
    config = json.loads(path.read_text())
    config["assets"] = "/another/runtime/assets"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(PROFILE.sys, "prefix", str(runtime / "venv"))
    monkeypatch.setattr(PROFILE.platform, "python_version", lambda: profile["python"])
    monkeypatch.setattr(PROFILE.os, "environ", {})
    with pytest.raises(ValueError, match="private config"):
        PROFILE.verify(runtime, profile_name=LIBERO_PROFILE)


def test_libero_existing_runtime_is_never_modified(tmp_path):
    sentinel = tmp_path / "existing-training-data"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        PROFILE.install(tmp_path, dry_run=False, profile_name=LIBERO_PROFILE)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize(
    "profile_name,uv_version,pin_count",
    [(None, "0.12.0", 275), (LIBERO_PROFILE, "0.10.4", 262)],
)
def test_python39_bootstrap_dry_run(tmp_path, profile_name, uv_version, pin_count):
    """The host bootstrap interpreter must parse and execute both profile paths."""
    python39 = os.environ.get("OPD_TEST_BOOTSTRAP_PYTHON") or shutil.which("python3.9")
    if not python39:
        pytest.skip("Python 3.9 not available; set OPD_TEST_BOOTSTRAP_PYTHON")
    version = subprocess.check_output(
        [python39, "-c", "import sys; print(sys.version_info[:2])"], text=True
    ).strip()
    assert version == "(3, 9)", (
        f"Bootstrap regression requires Python 3.9, got {version}"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(f"#!/bin/sh\necho 'uv {uv_version}'\n")
    uv.chmod(0o755)
    runtime = tmp_path / "new-runtime"
    command = [
        python39,
        str(REPO / "requirements/opd_profile.py"),
        "install",
        "--runtime",
        str(runtime),
        "--dry-run",
    ]
    if profile_name:
        command += ["--profile", profile_name]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    report = json.loads(result.stdout)
    assert report["pypi_packages"] == pin_count
    assert report["profile"]["id"] == (profile_name or PROFILE.PROFILE.name)
    assert not runtime.exists()


@pytest.mark.parametrize("invalid", [None, "missing", "foreign", "not_editable"])
def test_editable_source_egg_info_cannot_shadow_installed_provenance(
    tmp_path, monkeypatch, invalid
):
    checkout = tmp_path / "repos/libero"
    site = tmp_path / "venv/site-packages"
    metadata = {"url": checkout.as_uri(), "dir_info": {"editable": True}}
    if invalid == "foreign":
        metadata["url"] = (tmp_path / "another-checkout").as_uri()
    if invalid == "not_editable":
        metadata["dir_info"]["editable"] = False
    installed = SimpleNamespace(
        metadata={"Name": "libero"},
        version="0.1.0",
        read_text=lambda _: None if invalid == "missing" else json.dumps(metadata),
    )
    monkeypatch.setattr(
        PROFILE.sysconfig,
        "get_paths",
        lambda: {"purelib": str(site), "platlib": str(site)},
    )
    # Simulate source egg-info winning the unscoped lookup, with no direct URL.
    shadow = SimpleNamespace(read_text=lambda _: None)
    monkeypatch.setattr(PROFILE.importlib.metadata, "distribution", lambda _: shadow)
    calls = []

    def installed_distributions(*, path):
        calls.append(path)
        return [installed]

    monkeypatch.setattr(
        PROFILE.importlib.metadata, "distributions", installed_distributions
    )
    source = {"name": "libero", "version": "0.1.0"}
    if invalid:
        with pytest.raises(ValueError, match="direct_url|provenance"):
            PROFILE.verify_source_install(source, checkout)
    else:
        PROFILE.verify_source_install(source, checkout)
    assert calls == [[str(site)]]
