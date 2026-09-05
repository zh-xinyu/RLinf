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
"""Offline safety tests for the MOPD setup wrapper; no package installs or CUDA."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "toolkits/mopd/setup_runtime.sh"


@pytest.fixture
def setup_env(tmp_path):
    """Isolate roots and make credential execution fail loudly."""
    env = dict(os.environ)
    for name in ("REPO", "OPD_DRY_RUN"):
        env.pop(name, None)
    credentials = tmp_path / "credentials.env"
    credentials.write_text("echo CREDENTIALS_EXECUTED; exit 37\n")
    env.update(
        OPD_LOCAL_ENV_FILE=str(tmp_path / "absent.env"),
        OPD_RUNTIME_ROOT=str(tmp_path / "runtime with spaces"),
        OPD_SHARED_ROOT=str(tmp_path / "shared with spaces"),
        WANDB_ENV_FILE=str(credentials),
    )
    return env


def run_setup(args, env):
    """Execute a setup guard or dry-run without starting an installer."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("stage", ["all", "install", "resources", "check"])
def test_setup_dry_run_is_nonmutating(stage, setup_env):
    result = run_setup([stage, "--dry-run"], setup_env)
    assert result.returncode == 0, result.stderr
    assert f"repos={setup_env['OPD_RUNTIME_ROOT']}/repos" in result.stdout
    assert "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96" in result.stdout
    assert "RoboCasa=921c9a5736a8d0ea5589657898aadcfa55a6a195" in result.stdout
    assert "Robosuite=5ce6643f3092639d08f7b0f90ed1c6a84f50552c" in result.stdout
    assert "CREDENTIALS_EXECUTED" not in result.stdout
    assert not Path(setup_env["OPD_RUNTIME_ROOT"]).exists()
    assert not Path(setup_env["OPD_SHARED_ROOT"]).exists()


@pytest.mark.parametrize("revision", ["main", "n1.5-release", "4af2b62"])
def test_checkout_rejects_non_commit_pins(revision, setup_env):
    # Load only this helper, never the provisioning entrypoint or external commands.
    function = SCRIPT.read_text().split("checkout_pinned() {", 1)[1].split("\n}", 1)[0]
    program = (
        'fail() { echo "$*" >&2; exit 1; };\n'
        "checkout_pinned() {" + function + "\n}\n"
        'checkout_pinned UNUSED /unused https://github.com/NVIDIA/Isaac-GR00T.git main "$1"'
    )
    result = subprocess.run(
        ["bash", "-c", program, "bash", revision],
        env=setup_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "Expected a full commit SHA" in result.stderr


@pytest.mark.parametrize("options", [[], ["--resume-install"]])
def test_existing_environment_is_protected(setup_env, options):
    runtime = Path(setup_env["OPD_RUNTIME_ROOT"])
    venv = runtime / "venv"
    venv.mkdir(parents=True)
    marker = venv / "user-file"
    marker.write_text("preserve this\n")
    result = run_setup(["install", *options], setup_env)
    assert result.returncode != 0
    assert "Existing venv is protected" in result.stderr
    assert marker.read_text() == "preserve this\n"
    assert not (runtime / "cache").exists()


def test_resources_require_prepared_shared_source(setup_env):
    result = run_setup(["resources"], setup_env)
    assert result.returncode != 0
    assert "Shared models/assets are not staged" in result.stderr
    assert not Path(setup_env["OPD_RUNTIME_ROOT"]).exists()


@pytest.mark.parametrize("root", ["/tmp", str(REPO), str(REPO.parent)])
def test_setup_rejects_broad_roots(setup_env, root):
    setup_env["OPD_RUNTIME_ROOT"] = root
    result = run_setup(["all", "--dry-run"], setup_env)
    assert result.returncode != 0


def test_setup_rejects_overlapping_storage(setup_env):
    setup_env["OPD_SHARED_ROOT"] = setup_env["OPD_RUNTIME_ROOT"] + "/shared"
    result = run_setup(["resources", "--dry-run"], setup_env)
    assert result.returncode != 0
    assert "Shared/runtime roots must be separate" in result.stderr


def test_common_installer_can_be_sourced_without_installing(setup_env):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; declare -F clone_or_reuse_repo',
            "bash",
            str(REPO / "requirements/install.sh"),
        ],
        env=setup_env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clone_or_reuse_repo"


def prepare_small_resources(setup_env, tmp_path):
    """Use a temporary main repo so setup logs never go to real project outputs."""
    fake_repo = tmp_path / "main repo"
    subprocess.run(["git", "init", "-q", str(fake_repo)], check=True)
    setup_env["REPO"] = str(fake_repo)
    shared = Path(setup_env["OPD_SHARED_ROOT"])
    relative = Path("models/robocasa365/gr00t_n1-5/foundation_model_learning")
    for checkpoint in (
        "pretraining/checkpoint-80000",
        "target_posttraining/atomic_seen/checkpoint-60000",
        "target_posttraining/composite_seen/checkpoint-60000",
        "target_posttraining/composite_unseen/checkpoint-60000",
    ):
        path = shared / relative / checkpoint
        path.mkdir(parents=True)
        (path / "config.json").write_text('{"fake": true}\n')
    assets = shared / "assets/robocasa365/fixtures"
    assets.mkdir(parents=True)
    (assets / "fixture.xml").write_text("<mujoco/>\n")
    return relative / "pretraining/checkpoint-80000/config.json"


def test_resource_copy_is_resumable_and_preserves_extras(setup_env, tmp_path):
    example = prepare_small_resources(setup_env, tmp_path)
    result = run_setup(["resources"], setup_env)
    assert result.returncode == 0, result.stderr + result.stdout
    runtime = Path(setup_env["OPD_RUNTIME_ROOT"])
    assert (runtime / example).read_bytes() == (
        Path(setup_env["OPD_SHARED_ROOT"]) / example
    ).read_bytes()
    extra = runtime / example.parent / "keep.txt"
    extra.write_text("unrelated data")
    result = run_setup(["resources"], setup_env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert extra.read_text() == "unrelated data"


def test_resource_conflict_fails_without_overwriting(setup_env, tmp_path):
    example = prepare_small_resources(setup_env, tmp_path)
    destination = Path(setup_env["OPD_RUNTIME_ROOT"]) / example
    destination.parent.mkdir(parents=True)
    destination.write_text("different existing content")
    result = run_setup(["resources"], setup_env)
    assert result.returncode != 0
    assert "Resource checksum mismatch" in result.stdout + result.stderr
    assert destination.read_text() == "different existing content"


def test_resource_copy_rejects_symlinked_parent(setup_env, tmp_path):
    prepare_small_resources(setup_env, tmp_path)
    runtime = Path(setup_env["OPD_RUNTIME_ROOT"])
    runtime.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime / "models").symlink_to(outside, target_is_directory=True)
    result = run_setup(["resources"], setup_env)
    assert result.returncode != 0
    assert "must stay under the runtime" in result.stdout + result.stderr
    assert not list(outside.iterdir())
