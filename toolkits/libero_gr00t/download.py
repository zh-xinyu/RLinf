# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Download pinned inference-only GR00T checkpoints, verifying LFS SHA256."""

import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "libero_gr00t_20261104"
MODELS = {
    "student": (
        "twanghcmut/GR00T-N1.5-LIBERO-4suite-combined",
        "aee73136d034537f68c4e34d2619d3d357c7a2b1",
    ),
    "object": (
        "RLinf/RLinf-Gr00t-RL-Object-Step400",
        "5eb03ebbc74d9f6594475288ca46cec1b29beaca",
    ),
    "goal": (
        "RLinf/RLinf-Gr00t-RL-Goal-Step500",
        "f91ad61d5ff4b9f906af3849f2b097868d93e6de",
    ),
    "long": (
        "RLinf/RLinf-Gr00t-RL-Long-Step300",
        "2c3d290c05741051a2ff12c750de460503c077f1",
    ),
}


def download(item):
    name, (repo, revision) = item
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    meta = json.load(urllib.request.urlopen(url, timeout=60))
    rows = []
    for f in meta["siblings"]:
        filename = f["rfilename"]
        if not (
            filename.endswith(".safetensors")
            or filename
            in [
                "config.json",
                "experiment_cfg/metadata.json",
                "model.safetensors.index.json",
                "README.md",
            ]
        ):
            continue
        dest = ROOT / "models" / name / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = f.get("lfs", {}).get("sha256")
        expected_size = f.get("size")
        if not dest.exists():
            print("DOWNLOAD", name, filename, expected_size, flush=True)
            temporary = dest.with_suffix(dest.suffix + ".partial")
            offset = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(
                f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
                headers={"Range": f"bytes={offset}-"} if offset else {},
            )
            with urllib.request.urlopen(request, timeout=120) as src:
                append = (
                    offset > 0
                    and src.status == 206
                    and src.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    )
                )
                out = temporary.open("ab" if append else "wb")
                while chunk := src.read(8 * 1024 * 1024):
                    out.write(chunk)
                out.close()
            temporary.rename(dest)
        digest = hashlib.file_digest(dest.open("rb"), "sha256").hexdigest()
        if expected and digest != expected:
            raise ValueError(f"Hash mismatch: {dest}")
        if expected_size is not None and dest.stat().st_size != expected_size:
            raise ValueError(f"Size mismatch: {dest}")
        rows.append({"file": filename, "sha256": digest, "size": dest.stat().st_size})
        print("VERIFIED", name, filename, flush=True)
    return {"name": name, "repo": repo, "revision": revision, "files": rows}


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(download, MODELS.items()))
    (ROOT / "download_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
