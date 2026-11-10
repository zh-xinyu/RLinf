# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Portable launch setup must preserve allocation and fail on runtime conflicts."""

import ast
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from toolkits.libero_gr00t.prepare_runtime import (
    REPO,
    launch_environment,
    prepare,
    profile_module,
)

BASE = "gr00t-n15-libero-20261105-r2"
BLACKWELL = "gr00t-n15-libero-blackwell-cu128"


@pytest.mark.parametrize("name,safe", [(BASE, "0"), (BLACKWELL, "1")])
def test_process_environment_preserves_allocation_and_selects_backend(
    tmp_path, name, safe
):
    module = profile_module()
    env = launch_environment(
        tmp_path, module.load_profile(module.resolve_profile(name))
    )
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["OPD_BLACKWELL_SDPA_SAFE"] == safe
    assert env["OPD_LIBERO_AUTOMAP_EGL"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[0].endswith("runtime_adapter")
    assert env["LIBERO_CONFIG_PATH"] == str(tmp_path / "libero-config")


def test_generated_wrapper_handles_spaces_and_existing_files(tmp_path):
    runtime = tmp_path / "runtime with spaces"
    (runtime / "venv/bin").mkdir(parents=True)
    (runtime / "venv/bin/python").symlink_to(sys.executable)
    for name in ("libero", "gr00t"):
        (runtime / "repos" / name).mkdir(parents=True)
    report = prepare(runtime, BASE)
    assert prepare(runtime, BASE) == report
    code = (
        "import os, sys; "
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='7,3'; "
        "assert os.environ['OPD_LIBERO_AUTOMAP_EGL']=='1'; "
        "assert 'torch' not in sys.modules; "
        "print('scoped-runtime-ok')"
    )
    result = subprocess.run(
        [report["python"], "-c", code],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "7,3"},
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "scoped-runtime-ok"
    (runtime / "libero_env.sh").write_text("active runtime: do not overwrite")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(runtime, BASE)


def test_egl_container_mapping_precedes_physical_id_parsing(monkeypatch):
    """Exercise only the pure selection function; no EGL or CUDA is imported."""
    path = REPO / "toolkits/libero_gr00t/runtime_adapter/egl_context.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "create_initialized_egl_device_display"
    )
    monkeypatch.setenv("OPD_LIBERO_AUTOMAP_EGL", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-container-uuid")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "99")
    selected = []
    egl = SimpleNamespace(
        eglQueryDevicesEXT=lambda: ["device0", "device1"],
        eglGetPlatformDisplayEXT=lambda _, device, attrs: selected.append(device) or 1,
        eglGetError=lambda: 0,
        eglInitialize=lambda display, major, minor: 1,
        EGL_PLATFORM_DEVICE_EXT=0,
        EGL_NO_DISPLAY=0,
        EGL_SUCCESS=0,
        EGL_TRUE=1,
    )
    namespace = {
        "os": os,
        "EGL": egl,
        "_opd_egl_index_for_cuda_zero": lambda: 1,
        "error": SimpleNamespace(GLError=RuntimeError),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    assert namespace[function.name]() == 1
    assert selected == ["device1"]


def test_blackwell_startup_disables_only_mem_efficient_sdpa(tmp_path):
    """Stub Torch in a fresh interpreter so the hook never queries a real GPU."""
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "calls=[]\n"
        "backends=SimpleNamespace(cuda=SimpleNamespace(\n"
        "enable_mem_efficient_sdp=lambda value: calls.append(value)))\n"
    )
    adapter = REPO / "toolkits/libero_gr00t/runtime_adapter"
    result = subprocess.run(
        [sys.executable, "-c", "import torch; assert torch.calls == [False]"],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(adapter))),
            "OPD_BLACKWELL_SDPA_SAFE": "1",
        },
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
