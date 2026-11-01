# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Download pinned official four-task Aloha demonstrations without loading pickle."""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
FILES = {
    "adjust_bottle": (
        292557643,
        "74e8504ae0586467b0c03fe7b2423d01fc16ef32d2da1b279ea8a502cf748939",
    ),
    "handover_mic": (
        406382725,
        "c43714747294929689a542308235815205156b9e895c1de729288cf2e8feb787",
    ),
    "place_container_plate": (
        280925049,
        "55cbccb0d4851cec0018e0fac9719fb6041b50c8656bdc67401a39b610c472f9",
    ),
    "stack_bowls_two": (
        610959581,
        "afd9de1ac182f27e17d302f3f1624b5f303d404b089d7d32787d7ad848999095",
    ),
}


def download_one(root: Path, task: str) -> dict:
    """Verify a source ZIP and extract only HDF5 and human-readable metadata."""
    expected_size, expected_hash = FILES[task]
    url = f"https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{REVISION}/dataset/{task}/demo_clean.zip"
    archive = root / f"{task}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        with (
            urllib.request.urlopen(url, timeout=180) as response,
            partial.open("wb") as out,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
        partial.rename(archive)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    if archive.stat().st_size != expected_size or digest != expected_hash:
        raise ValueError(f"Source verification failed: {archive}")
    extracted = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if info.is_dir() or relative.suffix not in {".hdf5", ".json", ".txt"}:
                continue
            target = root / relative
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as dest:
                while chunk := source.read(8 * 1024 * 1024):
                    dest.write(chunk)
            extracted.append(str(relative))
    row = {
        "task": task,
        "url": url,
        "bytes": expected_size,
        "sha256": digest,
        "extracted": extracted,
    }
    (root / f"{task}.manifest.json").write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"VERIFIED {task}: {expected_size} bytes; {len(extracted)} safe members",
        flush=True,
    )
    return row


def main() -> None:
    """Fetch official revised clean demos into a new isolated directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda task: download_one(args.output, task), FILES))
    manifest = {
        "dataset": "TianxingChen/RoboTwin2.0",
        "revision": REVISION,
        "embodiment": "aloha_agilex",
        "selection": "Four tasks, 50 official clean demonstrations each",
        "note": "Official revised demo_clean archive (RGB JPEG/legal-joint corrections); not asserted identical to PACE training episodes. No pickle or video files extracted.",
        "files": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
