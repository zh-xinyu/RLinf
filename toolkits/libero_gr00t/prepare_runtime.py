# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare process-local LIBERO launch files after a manual profile install.

This command installs no packages, imports no model, and starts no GPU or Ray
work. Existing launch files with different contents are never overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def profile_module():
    """Load the shared profile definition without importing robotics packages."""
    spec = importlib.util.spec_from_file_location(
        "opd_profile", REPO / "requirements/opd_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch_environment(runtime: Path, profile: dict) -> dict[str, str]:
    """Build the same scoped adapters used by the validated node wrappers."""
    module = profile_module()
    if profile.get("environment") != "libero":
        raise ValueError("This helper requires a LIBERO profile")
    env = module.profile_environment(runtime, profile)
    env.update(
        REPO_PATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        VIRTUAL_ENV=str(runtime / "venv"),
        OPD_ENV_PROFILE=profile["id"],
        OPD_LIBERO_AUTOMAP_EGL="1",
        OPD_BLACKWELL_SDPA_SAFE=("1" if profile["torch_cuda"] == "12.8" else "0"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        NCCL_P2P_DISABLE="1",
        NCCL_IB_DISABLE="1",
        USE_FLAX="0",
        JAX_PLATFORMS="cpu",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO / "toolkits/libero_gr00t/runtime_adapter"), env["PYTHONPATH"])
    )
    # CUDA_VISIBLE_DEVICES belongs to the allocation / individual Ray worker.
    return env


def prepare(runtime: Path, name: str) -> dict:
    """Write private simulator paths, a shell environment and a Python wrapper."""
    runtime = runtime.expanduser().resolve()
    module = profile_module()
    directory = module.resolve_profile(name)
    profile = module.load_profile(directory)
    env = launch_environment(runtime, profile)
    python = runtime / "venv/bin/python"
    if not python.is_file():
        raise ValueError(f"Install the runtime first: missing {python}")
    for source in profile["sources"]:
        if not (runtime / "repos" / source["name"]).is_dir():
            raise ValueError(f"Missing pinned source checkout: {source['name']}")
    shell = (
        "# Generated for this checkout and runtime. Source before starting Ray.\n"
        "unset PYTHONHOME PYTHONUSERBASE RLINF_PYTHON_EXECUTABLE BASH_ENV ENV\n"
        "unset MUJOCO_EGL_DEVICE_ID CONDA_PREFIX CONDA_DEFAULT_ENV\n"
        + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in env.items())
        + f'export PATH={shlex.quote(str(runtime / "venv/bin"))}:"$PATH"\n'
    )
    wrapper = (
        "#!/bin/bash\nset -euo pipefail\n"
        f"source {shlex.quote(str(runtime / 'libero_env.sh'))}\n"
        f'exec {shlex.quote(str(python))} "$@"\n'
    )
    files = {
        runtime / "libero-config/config.yaml": json.dumps(
            module.libero_config(runtime), indent=2
        )
        + "\n",
        runtime / "libero_env.sh": shell,
        runtime / "bin/python": wrapper,
    }
    # Validate all conflicts before writing any files; keep active jobs immutable.
    for path, content in files.items():
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"Choose a new runtime; refusing to overwrite {path}")
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x") as stream:
                stream.write(content)
    (runtime / "bin/python").chmod(0o755)
    (runtime / "datasets").mkdir(exist_ok=True)
    return {
        "runtime": str(runtime),
        "profile": name,
        "python": str(runtime / "bin/python"),
    }


def main() -> None:
    """Expose a manual-install configuration step with no experiment side effects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.runtime, args.profile), indent=2))


if __name__ == "__main__":
    main()
