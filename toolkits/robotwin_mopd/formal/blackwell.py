# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Install pinned Blackwell overlays and bind unchanged evaluation scenes."""

import argparse
import ctypes
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from toolkits.robotwin_mopd.formal.experiment import HERE, digest, read_machine

OIDN_URL = "https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz"
OIDN_SHA = "3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f"
NCCL_SHA = "dba12e429fe11268b895d0531ba96a7f679f35227d5b1ec77c5febbcd02281bd"
OIDN_FILES = {
    "libOpenImageDenoise.so.2.3.3": "9ac9dcd18318cae879efb01c14cff72c2c4275428a15e5ab6e94c8ff8a43c4dc",
    "libOpenImageDenoise_core.so.2.3.3": "eadb2437d6ac4a88e5746980495ba1091f7d14745e3000b82a9382093d22f351",
    "libOpenImageDenoise_device_cuda.so.2.3.3": "12b3ff87cfa4b6a90481b757f5a56ee69380a45e3dcaeadf8dd22cb73d4f714a",
}


def install(runtime: Path) -> None:
    """Patch an idle dedicated runtime; preserve the original venv packages."""
    runtime = runtime.resolve()
    env = runtime / "env.sh"
    if not env.is_file():
        raise ValueError("Install the dedicated runtime first")
    archive = runtime / "oidn-2.3.3.x86_64.linux.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(OIDN_URL, archive)
    if digest(archive) != OIDN_SHA:
        raise ValueError("OIDN archive hash mismatch")
    source = runtime / "vendor/oidn-2.3.3.x86_64.linux"
    if not source.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(runtime / "vendor", filter="data")
    vendor = runtime / "vendor/sapien-3.0.0b1-utf8-oidn233"
    if not vendor.exists():
        site = runtime / "venv/lib/python3.11/site-packages"
        for name in ("sapien", "sapien.libs"):
            shutil.copytree(site / name, vendor / name)
        loader = vendor / "sapien/_oidn_tricks.py"
        original = loader.read_text()
        if "2.0.1" not in original:
            raise ValueError("Unexpected original OIDN loader")
        loader.write_text(original.replace("2.0.1", "2.3.3"))
        for name in OIDN_FILES:
            shutil.copy2(source / "lib" / name, vendor / "sapien/oidn_library" / name)
        (vendor / "opd_sapien_manifest.json").write_text(
            json.dumps(
                {
                    "sapien": "3.0.0b1",
                    "oidn": "2.3.3",
                    "source_url": OIDN_URL,
                    "source_sha256": OIDN_SHA,
                    "files": OIDN_FILES,
                },
                indent=2,
            )
            + "\n"
        )
    for name, expected in OIDN_FILES.items():
        if digest(vendor / "sapien/oidn_library" / name) != expected:
            raise ValueError(f"OIDN library hash mismatch: {name}")
    nccl = runtime / "vendor/nccl-2.31.2"
    library = nccl / "nvidia/nccl/lib/libnccl.so.2"
    if not nccl.exists():
        subprocess.run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--python",
                str(runtime / "venv/bin/python"),
                "--target",
                str(nccl),
                "--no-deps",
                "nvidia-nccl-cu12==2.31.2",
            ],
            check=True,
        )
    if digest(library) != NCCL_SHA:
        raise ValueError("NCCL library hash mismatch")
    overlay = runtime / "blackwell.env"
    text = (
        f"export OPD_ROBOTWIN_SAPIEN_VENDOR={shlex.quote(str(vendor))}\n"
        'export PYTHONPATH="$OPD_ROBOTWIN_SAPIEN_VENDOR:${PYTHONPATH:-}"\n'
        f"export OPD_ROBOTWIN_NCCL_LIBRARY={shlex.quote(str(library))}\n"
        'case ":${LD_PRELOAD:-}:" in\n'
        '  *":$OPD_ROBOTWIN_NCCL_LIBRARY:"*) ;;\n'
        '  *) export LD_PRELOAD="$OPD_ROBOTWIN_NCCL_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}" ;;\n'
        "esac\n"
    )
    if overlay.exists() and overlay.read_text() != text:
        raise ValueError("Existing Blackwell activation differs; refuse overwrite")
    overlay.write_text(text)
    activation = f"source {shlex.quote(str(overlay))}"
    if activation not in env.read_text():
        backup = runtime / "env.before_formal_blackwell.sh"
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(env, backup)
        env.write_text(env.read_text() + "\n" + activation + "\n")
    print(
        f"Installed pinned overlays. Source {env} in a NEW shell before bind/preflight."
    )


def bind(machine_path: Path) -> None:
    """Accept only the exact verified simulator content; retain every scene."""
    import torch

    from rlinf.envs.robotwin.runtime_profile import validate_robotwin4_runtime
    from toolkits.robotwin_mopd.author_pool import load_pool, runtime_identity
    from toolkits.robotwin_mopd.feasible_pool import write_pool

    machine = read_machine(machine_path)
    if machine["profile"] != "blackwell-cu128-candidate":
        raise ValueError("This binding is for the Blackwell profile only")
    if torch.__version__.split("+")[0] != "2.7.1" or torch.version.cuda != "12.8":
        raise ValueError("Expected Torch2.7.1/cu128")
    runtime = validate_robotwin4_runtime()
    expected = json.loads((HERE / "blackwell_runtime_identity.json").read_text())
    if runtime_identity(runtime) != expected:
        raise ValueError("Runtime content differs from the verified action30 profile")
    lib = Path(os.environ["OPD_ROBOTWIN_NCCL_LIBRARY"])
    if digest(lib) != NCCL_SHA:
        raise ValueError("Wrong NCCL payload")
    version = ctypes.c_int()
    if (
        ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(version))
        or version.value != 23102
    ):
        raise ValueError("NCCL2.31.2 is not loaded; source runtime/env.sh first")
    directory = Path(machine["output_root"]) / "prepared/runtime_pools"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("formal100.json", "dev20.json"):
        pool = load_pool(HERE / name)
        pool.pop("pool_content_sha256")
        pool["protocol"]["runtime_profile"] = runtime
        target = directory / name
        if target.exists():
            previous = load_pool(target)
            previous.pop("pool_content_sha256")
            if previous != pool:
                raise ValueError("Existing runtime-bound pool differs")
        else:
            write_pool(target, pool)
    original = json.loads(machine_path.read_text())
    if original.get("runtime_pool_dir", str(directory)) != str(directory):
        raise ValueError("Existing machine runtime_pool_dir differs")
    original["runtime_pool_dir"] = str(directory)
    machine_path.write_text(json.dumps(original, indent=2) + "\n")
    print(f"Bound unchanged scenes to verified runtime: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("install").add_argument("--runtime", type=Path, required=True)
    subs.add_parser("bind").add_argument("--machine", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "install":
        install(args.runtime)
    else:
        bind(args.machine)


if __name__ == "__main__":
    main()
