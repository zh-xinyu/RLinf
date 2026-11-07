# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fetch the immutable SFT500 release, verifying the local formal manifest."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from toolkits.robotwin_mopd.formal.experiment import HERE, digest

REPOSITORY = "theo-zhou/vla-mopd-checkpoints"
REVISION = "9e973cf5ff2cd606e962db6d94b93d2cf9b7bdde"
MANIFEST = "robotwin2/pi05/releases/sft500/resources.json"


def verify(path: Path, record: dict) -> None:
    """Verify content, including previously downloaded resources."""
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError(f"Resource identity mismatch: {path}")


def download(root: Path) -> None:
    """Reuse matching local resources without replacing existing files."""
    root = root.expanduser().absolute()
    local = json.loads((HERE / "resources.json").read_text())
    remote = json.loads(
        Path(hf_hub_download(REPOSITORY, MANIFEST, revision=REVISION)).read_text()
    )
    stripped = {
        name: {k: v for k, v in row.items() if k != "path_in_repo"}
        for name, row in remote["files"].items()
    }
    if stripped != local["files"]:
        raise ValueError("Published resources differ from the checked-out recipe")
    for rel, record in remote["files"].items():
        target = root / rel
        if target.exists() or target.is_symlink():
            verify(target, record)
        else:
            source = Path(
                hf_hub_download(REPOSITORY, record["path_in_repo"], revision=REVISION)
            ).resolve()
            verify(source, record)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        print(f"Verified {rel}", flush=True)
    (root / "hf_release.json").write_text(
        json.dumps(
            {"repository": REPOSITORY, "revision": REVISION, "manifest": MANIFEST},
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    args = parser.parse_args()
    download(args.resource_root)


if __name__ == "__main__":
    main()
