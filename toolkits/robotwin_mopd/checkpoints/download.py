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
"""Download pinned RoboTwin specialists and the official Pi0.5 base checkpoint."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests

TASKS = ("adjust_bottle", "handover_mic", "place_container_plate", "stack_bowls_two")
PACE_REPO = "niejunnan25/pi05-robotwin"
PACE_REVISION = "17ef545270cf26ed81169893b60f845f80ce6557"


def download_file(
    url: str,
    destination: Path,
    size: int,
    sha256: str | None = None,
    md5_base64: str | None = None,
    crc32c_base64: str | None = None,
) -> None:
    """Stream one artifact and verify its source-provided checksum and size."""
    import google_crc32c

    def new_digest():
        if sha256:
            return hashlib.sha256()
        return hashlib.md5() if md5_base64 else google_crc32c.Checksum()

    def checksum_matches(digest):
        actual = (
            digest.hexdigest() if sha256 else base64.b64encode(digest.digest()).decode()
        )
        return actual == (sha256 or md5_base64 or crc32c_base64)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        checksum = new_digest()
        with destination.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                checksum.update(chunk)
        if checksum_matches(checksum):
            return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(5):
        try:
            digest = new_digest()
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
            if temporary.stat().st_size != size:
                raise ValueError(f"Size mismatch: {destination}")
            if not checksum_matches(digest):
                raise ValueError(f"Source checksum mismatch: {destination}")
            temporary.replace(destination)
            print(
                json.dumps({"downloaded": str(destination), "bytes": size}), flush=True
            )
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=("pace", "base"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help="Replay exact GCS object generations from a saved manifest.",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.source == "pace":
        url = (
            f"https://huggingface.co/{PACE_REPO}/resolve/{PACE_REVISION}/manifest.json"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        manifest = response.json()
        (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        files = [
            f for m in manifest["models"] if m["task"] in TASKS for f in m["files"]
        ]
        files.append(
            {
                "path": manifest["tokenizer"]["path"],
                "bytes": manifest["tokenizer"]["bytes"],
                "sha256": manifest["tokenizer"]["sha256"],
            }
        )
        jobs = [
            (
                f"https://huggingface.co/{PACE_REPO}/resolve/{PACE_REVISION}/{f['path']}?download=true",
                args.root / f["path"],
                f["bytes"],
                f["sha256"],
            )
            for f in files
        ]
        (args.root / "source.json").write_text(
            json.dumps(
                {"repo": PACE_REPO, "revision": PACE_REVISION, "tasks": TASKS}, indent=2
            )
        )
    else:
        prefix = "checkpoints/pi05_base/"
        saved_manifest = args.base_manifest or args.root / "gcs_source_manifest.json"
        if saved_manifest.exists():
            manifest = json.loads(saved_manifest.read_text())
        elif args.base_manifest:
            raise FileNotFoundError(args.base_manifest)
        else:
            response = requests.get(
                "https://storage.googleapis.com/storage/v1/b/openpi-assets/o",
                params={"prefix": prefix},
                timeout=60,
            )
            response.raise_for_status()
            manifest = response.json()
        if manifest.get("nextPageToken"):
            raise ValueError("Unexpected paginated base manifest")
        (args.root / "gcs_source_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        jobs = [
            (
                f"https://storage.googleapis.com/openpi-assets/{quote(f['name'], safe='/')}?generation={f['generation']}",
                args.root / f["name"].removeprefix(prefix),
                int(f["size"]),
                None,
                f.get("md5Hash"),
                f["crc32c"],
            )
            for f in manifest["items"]
        ]
    print(
        json.dumps(
            {
                "source": args.source,
                "files": len(jobs),
                "bytes": sum(j[2] for j in jobs),
            }
        ),
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda job: download_file(*job), jobs))
    (args.root / "DOWNLOAD_COMPLETE.json").write_text(
        json.dumps(
            {"source": args.source, "files": len(jobs), "completed_unix": time.time()},
            indent=2,
        )
    )
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
