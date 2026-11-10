# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the LIBERO recipe while changing the hardware-specific GPU stack."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "libero_blackwell_profile", REPO / "requirements/opd_profile.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)
BLACKWELL = "gr00t-n15-libero-blackwell-cu128"
BASE = "gr00t-n15-libero-20261105-r2"


def test_blackwell_preserves_scientific_dependencies():
    baseline_dir = PROFILE.resolve_profile(BASE)
    blackwell_dir = PROFILE.resolve_profile(BLACKWELL)
    baseline = PROFILE.load_profile(baseline_dir)
    blackwell = PROFILE.load_profile(blackwell_dir)
    assert baseline["sources"] == blackwell["sources"]
    for field in ("python", "uv", "bootstrap", "environment", "verification_imports"):
        assert baseline[field] == blackwell[field]
    old = PROFILE.package_pins(baseline_dir)
    new = PROFILE.package_pins(blackwell_dir)
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    assert changed <= {
        "torch",
        "torchvision",
        "triton",
        "sympy",
        *[key for key in new if key.startswith("nvidia-")],
    }
    assert new["torch"] == "2.7.1+cu128"
    assert new["sympy"] == "1.14.0"
    assert blackwell["torch_cuda"] == "12.8"
    assert blackwell["torch_cxx11_abi"] is True
    assert "torch2.7cxx11abiTRUE" in blackwell["flash_attention"]["url"]
    assert all(
        url.startswith("https://download.pytorch.org/")
        for url in blackwell["package_wheels"]
    )


@pytest.mark.parametrize("name", [BASE, BLACKWELL])
def test_official_wheels_precede_no_deps_pins(tmp_path, monkeypatch, name):
    directory = PROFILE.resolve_profile(name)
    definition = PROFILE.load_profile(directory)
    monkeypatch.setattr(PROFILE.shutil, "which", lambda _: "/test/uv")
    monkeypatch.setattr(PROFILE.subprocess, "check_output", lambda *a, **k: "uv 0.10.4")
    commands = []

    def capture(command, **kwargs):
        commands.append(list(map(str, command)))
        if "-r" in command:
            raise RuntimeError("Stop before dependency installation")

    monkeypatch.setattr(PROFILE, "run", capture)
    with pytest.raises(RuntimeError, match="Stop before dependency installation"):
        PROFILE.install(tmp_path / "runtime", False, name)
    wheel_commands = [
        command
        for command in commands
        if any("https://download.pytorch.org/" in arg for arg in command)
    ]
    if definition.get("package_wheels"):
        assert len(wheel_commands) == 1
        assert commands.index(wheel_commands[0]) < len(commands) - 1
        assert "--no-deps" in wheel_commands[0]
    else:
        assert not wheel_commands
