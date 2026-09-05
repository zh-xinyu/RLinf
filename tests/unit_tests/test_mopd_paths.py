# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only tests for MOPD launch path resolution; never start Ray or CUDA."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "toolkits/mopd"


@pytest.fixture
def path_env(tmp_path):
    """Use an explicit nonexistent config, never the developer's credentials."""
    env = dict(os.environ)
    for key in ("REPO", "OPD_RUN_ID", "OPD_RUN_KIND"):
        env.pop(key, None)
    env.update(
        OPD_LOCAL_ENV_FILE=str(tmp_path / "missing.env"),
        OPD_RUNTIME_ROOT=str(tmp_path / "runtime with spaces"),
        OPD_SHARED_ROOT=str(tmp_path / "shared"),
        WANDB_ENV_FILE=str(tmp_path / "credentials-not-read"),
        OPD_DRY_RUN="1",
    )
    return env


def run_script(script, args, env):
    """Run a launcher with no side effects in dry-run mode."""
    return subprocess.run(
        ["bash", str(TOOLS / script), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
        cwd="/tmp",
    )


@pytest.mark.parametrize(
    ("script", "args", "kind", "name"),
    [
        (
            "run_mopd.sh",
            ["robocasa365_mopd_gr00t", "path-test"],
            "train",
            "robocasa365_mopd_gr00t.path-test",
        ),
        ("run_eval_mopd.sh", ["path-test"], "eval", "eval.path-test"),
        ("run_probe.sh", ["probe_precision_init"], "smoke", "probe_precision_init"),
    ],
)
def test_launch_dry_run_uses_main_outputs(script, args, kind, name, path_env):
    result = run_script(script, args, path_env)
    assert result.returncode == 0, result.stderr
    assert f"runtime={path_env['OPD_RUNTIME_ROOT']}" in result.stdout
    assert f"run={REPO}/outputs/{kind}/{name}" in result.stdout
    assert not Path(path_env["OPD_RUNTIME_ROOT"]).exists()
    assert not (REPO / "outputs" / kind / name).exists()


def test_relative_runtime_is_rejected(path_env):
    path_env["OPD_RUNTIME_ROOT"] = "relative/path"
    result = run_script("local_env.sh", [], path_env)
    assert result.returncode != 0
    assert "absolute OPD_RUNTIME_ROOT" in result.stderr


def test_escaping_run_id_is_rejected(path_env):
    path_env["OPD_RUN_ID"] = "../outside"
    result = run_script("run_mopd.sh", ["config", "tag"], path_env)
    assert result.returncode != 0
    assert "single safe directory" in result.stderr


def test_environment_overrides_local_file(path_env, tmp_path):
    config = tmp_path / "local.env"
    config.write_text("export OPD_RUNTIME_ROOT=/some/other/runtime\n")
    path_env["OPD_LOCAL_ENV_FILE"] = str(config)
    result = run_script("local_env.sh", [], path_env)
    assert result.returncode == 0, result.stderr
    assert f"runtime={path_env['OPD_RUNTIME_ROOT']}" in result.stdout


def test_dry_run_preserves_override_arguments_and_never_loads_credentials(
    path_env, tmp_path
):
    credentials = tmp_path / "credentials.env"
    credentials.write_text("echo CREDENTIALS_WERE_EXECUTED; exit 9\n")
    path_env["WANDB_ENV_FILE"] = str(credentials)
    override = "runner.logger.experiment_name=two words;literal"
    result = run_script("run_mopd.sh", ["config", "tag", override], path_env)
    assert result.returncode == 0, result.stderr
    assert "CREDENTIALS_WERE_EXECUTED" not in result.stdout
    command = next(
        line for line in result.stdout.splitlines() if line.startswith("command:")
    )
    assert shlex.split(command.removeprefix("command:"))[-1] == override


def test_normalized_filesystem_root_is_rejected(path_env):
    path_env["OPD_RUNTIME_ROOT"] = "/tmp/.."
    result = run_script("local_env.sh", [], path_env)
    assert result.returncode != 0
    assert "Runtime root cannot be /" in result.stderr


def test_retired_reference_probe_is_rejected_before_launch(path_env):
    result = run_script("run_probe.sh", ["probe_path_equiv"], path_env)
    assert result.returncode != 0
    assert "reference-inference methods were removed" in result.stderr
    assert "command:" not in result.stdout
    assert "run=" not in result.stdout
