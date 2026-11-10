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
"""Restore and verify the versioned OPD environment without private node files."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "requirements/profiles/gr00t-n15-robocasa365-20261019"


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run an installation step, stopping at the first failure."""
    print("+ " + shlex.join(str(x) for x in command), flush=True)
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def resolve_profile(name: str | None = None) -> Path:
    """Resolve a bundled profile name, rejecting external paths and traversal."""
    if name is None:
        return PROFILE
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("--profile must be a bundled profile name, not a path")
    directory = PROFILE.parent / name
    if directory.is_symlink() or not (directory / "profile.json").is_file():
        raise ValueError(f"Unknown bundled profile: {name}")
    return directory


def load_profile(directory: Path | None = None) -> dict:
    """Read the selected profile, defaulting to the existing RoboCasa snapshot."""
    return json.loads(((directory or PROFILE) / "profile.json").read_text())


def package_pins(directory: Path | None = None) -> dict[str, str]:
    """Read exact PyPI versions, excluding comments and empty lines."""
    return dict(
        line.split("==", 1)
        for line in ((directory or PROFILE) / "requirements.txt")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    )


def check_platform(profile: dict) -> None:
    """Reject platforms for which this wheel snapshot is not defined."""
    if (platform.system(), platform.machine()) != (
        profile["platform"],
        profile["machine"],
    ):
        raise ValueError("This profile requires Linux x86_64 with NVIDIA CUDA.")


def libero_config(runtime: Path) -> dict[str, str]:
    """Return LIBERO paths owned by this runtime, never the user's global config."""
    benchmark = runtime / "repos/libero/libero/libero"
    return {
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "datasets": str(runtime / "datasets"),
        "assets": str(benchmark / "assets"),
    }


def profile_environment(runtime: Path, profile: dict) -> dict[str, str]:
    """Build process-local overrides for a profile without reading node config."""
    if profile.get("environment") != "libero":
        return {"PYTHONPATH": str(REPO)}
    return {
        "PYTHONPATH": os.pathsep.join((str(REPO), str(runtime / "repos/libero"))),
        "LIBERO_CONFIG_PATH": str(runtime / "libero-config"),
        "LIBERO_TYPE": "standard",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONNOUSERSITE": "1",
        "HF_HOME": str(runtime / "cache/huggingface"),
        "HF_MODULES_CACHE": str(runtime / "cache/huggingface/modules"),
        "HF_HUB_CACHE": str(runtime / "cache/huggingface/hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NUMBA_CACHE_DIR": str(runtime / "cache/numba"),
        "XDG_CACHE_HOME": str(runtime / "cache/xdg"),
        "TORCH_HOME": str(runtime / "cache/torch"),
        "TRITON_CACHE_DIR": str(runtime / "cache/triton"),
        "USE_TF": "0",
        "NO_ALBUMENTATIONS_UPDATE": "1",
    }


def prepare_profile_config(runtime: Path, profile: dict) -> None:
    """Create private config only during installation into a new runtime."""
    if profile.get("environment") == "libero":
        directory = runtime / "libero-config"
        directory.mkdir()
        # JSON is valid YAML; no bootstrap dependency on PyYAML is needed.
        (directory / "config.yaml").write_text(
            json.dumps(libero_config(runtime), indent=2) + "\n"
        )
        (runtime / "datasets").mkdir()


def install(runtime: Path, dry_run: bool, profile_name: str | None = None) -> None:
    """Install into a new directory, refusing to modify an existing runtime."""
    directory = resolve_profile(profile_name)
    profile = load_profile(directory)
    check_platform(profile)
    if runtime.exists():
        raise ValueError(f"Runtime already exists; choose a NEW directory: {runtime}")
    uv = shutil.which("uv")
    if not uv:
        raise ValueError(
            f"Install uv=={profile['uv']} first "
            f"(see {profile.get('setup_doc', 'OPD_ENVIRONMENT.md')})."
        )
    version = subprocess.check_output([uv, "--version"], text=True).split()[1]
    if version != profile["uv"]:
        raise ValueError(f"Expected uv=={profile['uv']}, got {version}")
    if dry_run:
        print(
            json.dumps(
                {
                    "runtime": str(runtime),
                    "profile": profile,
                    "pypi_packages": len(package_pins(directory)),
                },
                indent=2,
            )
        )
        return
    # No inherited node config, active venv, alternate package index or overlay.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("UV_", "PIP_"))
        and k
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "RLINF_PYTHON_EXECUTABLE",
            "BASH_ENV",
            "ENV",
            "GIT_LFS_SKIP_SMUDGE",
        }
    }
    runtime.mkdir(parents=True, exist_ok=False)
    env.update(
        UV_NO_CONFIG="1",
        UV_DEFAULT_INDEX="https://pypi.org/simple",
        UV_CACHE_DIR=str(runtime / "cache/uv"),
        UV_PYTHON_INSTALL_DIR=str(runtime / "python"),
        UV_PYTHON_PREFERENCE="only-managed",
        PATH=str(Path(uv).parent) + os.pathsep + env["PATH"],
        MUJOCO_GL="egl",
        PYTHONNOUSERSITE="1",
    )
    if profile.get("environment") == "libero":
        env["UV_LINK_MODE"] = "copy"
    # Reuse the common venv helper with its explicit snapshot mode. This avoids
    # resolving the broad RLinf extras again and changing the captured versions.
    run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1"; VENV_DIR="$2"; PYTHON_VERSION="$3"; '
            'create_and_sync_venv "$4"',
            "opd-profile",
            REPO / "requirements/install.sh",
            runtime / "venv",
            profile["python"],
            "create-only",
        ],
        cwd=runtime,
        env=env,
    )
    python = runtime / "venv/bin/python"
    pip = [
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--no-deps",
        "--build-constraint",
        str(directory / "build-constraints.txt"),
    ]
    run(pip + profile["bootstrap"], cwd=runtime, env=env)
    if profile.get("package_wheels"):
        run(pip + profile["package_wheels"], cwd=runtime, env=env)
    run(pip + ["-r", directory / "requirements.txt"], cwd=runtime, env=env)
    for source in profile["sources"]:
        checkout = runtime / "repos" / source["name"]
        # Skip only explicitly declared non-runtime LFS objects in this source.
        # Other repositories (especially simulator assets) retain normal smudging.
        source_env = dict(env)
        if source.get("skip_lfs_smudge", False):
            source_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        # Do not honor private GR00T_PATH / ROBOCASA_PATH overrides here.
        run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                'source "$1"; unset OPD_PROFILE_SOURCE; '
                'clone_or_reuse_repo OPD_PROFILE_SOURCE "$2" "$3" --no-checkout; '
                'git -C "$2" checkout --detach "$4"',
                "opd-profile",
                REPO / "requirements/install.sh",
                checkout,
                source["url"],
                source["commit"],
            ],
            cwd=runtime,
            env=source_env,
        )
        run(pip + ["--editable", checkout], cwd=runtime, env=env)
    # The snapshot requires this exact CUDA/Torch/ABI wheel: no version fallback.
    run(pip + [profile["flash_attention"]["url"]], cwd=runtime, env=env)
    prepare_profile_config(runtime, profile)
    env.update(profile_environment(runtime, profile))
    if profile.get("environment") == "libero":
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
        env.pop("OPD_LIBERO_AUTOMAP_EGL", None)
    run(
        [python, __file__, "verify", "--runtime", runtime, "--profile", directory.name],
        cwd=runtime,
        env=env,
    )
    activation = (
        "# Generated for this checkout; source before launching Ray.\n"
        f"source {shlex.quote(str(runtime / 'venv/bin/activate'))}\n"
        f"export REPO_PATH={shlex.quote(str(REPO))}\n"
        'export EMBODIED_PATH="$REPO_PATH/examples/embodiment"\n'
        'export PYTHONPATH="$REPO_PATH"\n'
        "export PYTHONNOUSERSITE=1\nexport MUJOCO_GL=egl\n"
        f"export OPD_ENV_PROFILE={shlex.quote(profile['id'])}\n"
    )
    if profile.get("environment") == "libero":
        activation += "unset MUJOCO_EGL_DEVICE_ID OPD_LIBERO_AUTOMAP_EGL\n"
        activation += "".join(
            f"export {key}={shlex.quote(value)}\n"
            for key, value in profile_environment(runtime, profile).items()
        )
    (runtime / "activate.sh").write_text(activation)
    shutil.copytree(directory, runtime / "profile")
    print(
        f"Software verified. Next: resources in "
        f"{profile.get('setup_doc', 'OPD_ENVIRONMENT.md')}.\n"
        f"source {shlex.quote(str(runtime / 'activate.sh'))}"
    )


def verify_source_install(source: dict, checkout: Path) -> None:
    """Require editable-install provenance from this interpreter's site-packages.

    A source checkout on PYTHONPATH may expose egg-info before the installed
    dist-info. That metadata has no direct_url.json and must not stand in for
    the installed distribution's provenance.
    """
    paths = sorted({sysconfig.get_paths()[key] for key in ("purelib", "platlib")})
    name = source["name"].lower().replace("-", "_")
    installed = [
        dist
        for dist in importlib.metadata.distributions(path=paths)
        if dist.metadata["Name"].lower().replace("-", "_") == name
    ]
    if len(installed) != 1:
        raise ValueError(f"Expected one installed distribution for {source['name']}")
    dist = installed[0]
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        raise ValueError(f"Missing installed direct_url.json for {source['name']}")
    metadata = json.loads(direct_url)
    if (
        dist.version != source["version"]
        or metadata.get("url") != checkout.as_uri()
        or metadata.get("dir_info", {}).get("editable") is not True
    ):
        raise ValueError(f"Incorrect installed source provenance: {source['name']}")


def verify(runtime: Path, gpu: bool = False, profile_name: str | None = None) -> dict:
    """Check all captured versions, source revisions, imports and optional CUDA."""
    directory = resolve_profile(profile_name)
    profile = load_profile(directory)
    check_platform(profile)
    if Path(sys.prefix).resolve() != runtime / "venv":
        raise ValueError("Run verify with the requested runtime/venv/bin/python.")
    if platform.python_version() != profile["python"]:
        raise ValueError(f"Expected Python {profile['python']}")
    # Import remote-code processors only from this runtime's fixed checkout.
    os.environ["HF_HOME"] = str(runtime / "cache/huggingface")
    os.environ["HF_MODULES_CACHE"] = str(runtime / "cache/huggingface/modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ.pop("TRANSFORMERS_CACHE", None)
    if profile.get("environment") == "libero":
        config = runtime / "libero-config/config.yaml"
        if not config.is_file() or json.loads(config.read_text()) != libero_config(
            runtime
        ):
            raise ValueError(
                "LIBERO private config is missing or references another runtime"
            )
        os.environ.update(profile_environment(runtime, profile))
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        os.environ.pop("OPD_LIBERO_AUTOMAP_EGL", None)
        # Setting PYTHONPATH alone does not change this already-running process.
        sys.path.insert(0, str(runtime / "repos/libero"))
    pins = package_pins(directory)
    pins.update(dict(pin.split("==") for pin in profile["bootstrap"]))
    for source in [*profile["sources"], profile["flash_attention"]]:
        pins[source["name"]] = source["version"]
    for name, expected in pins.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"{name}: expected {expected}, got {actual}")
    for source in profile["sources"]:
        checkout = runtime / "repos" / source["name"]
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != source["commit"]:
            raise ValueError(f"Wrong source commit: {checkout}: {actual}")
        # Assets can be replaced by the user's downloaded asset directory.
        excludes = [":!robocasa/models/assets"] if source["name"] == "robocasa" else []
        run(
            [
                "git",
                "-C",
                checkout,
                "diff",
                "--exit-code",
                "HEAD",
                "--",
                ".",
                *excludes,
            ],
            stdout=subprocess.DEVNULL,
        )
        verify_source_install(source, checkout)
    for module in profile.get(
        "verification_imports",
        (
            "torch",
            "ray",
            "mujoco",
            "robosuite",
            "robocasa",
            "gr00t",
            "flash_attn",
            "rlinf.config",
            "rlinf.envs.robocasa",
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model",
        ),
    ):
        importlib.import_module(module)
    for source in profile["sources"]:
        module = importlib.import_module(source["name"])
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(runtime / "repos" / source["name"]):
            raise ValueError(
                f"Import overlay detected for {source['name']}: {location}"
            )
    import torch
    from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
    from transformers import AutoProcessor

    if (
        torch.version.cuda != profile["torch_cuda"]
        or torch._C._GLIBCXX_USE_CXX11_ABI != profile["torch_cxx11_abi"]
    ):
        raise ValueError("Torch CUDA / C++ ABI does not match the FlashAttention wheel")
    AutoProcessor.from_pretrained(
        DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
    )
    if gpu:
        from flash_attn import flash_attn_func

        q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
        result = flash_attn_func(q, q, q)
        if not torch.isfinite(result).all():
            raise ValueError("FlashAttention returned non-finite values")
        torch.cuda.synchronize()
    report = {
        "profile": profile["id"],
        "verified_packages": len(pins),
        "python": platform.python_version(),
        "gpu_smoke": gpu,
        "resources_and_training_verified": False,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    """Expose installation and verification without any private launch helpers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "verify"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--profile", help=f"Bundled profile name (default: {PROFILE.name})"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not args.runtime.is_absolute():
        parser.error("--runtime must be an absolute path")
    if args.command == "install" and args.gpu:
        parser.error("--gpu is only available for verify")
    if args.command == "verify" and args.dry_run:
        parser.error("--dry-run is only available for install")
    runtime = args.runtime.resolve()
    if args.command == "install":
        install(runtime, args.dry_run, args.profile)
    else:
        verify(runtime, args.gpu, args.profile)


if __name__ == "__main__":
    main()
