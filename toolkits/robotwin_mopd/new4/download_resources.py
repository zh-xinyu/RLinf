# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Download the immutable new4 SFT400 release and verify every file."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from toolkits.robotwin_mopd.new4.contracts import HERE, digest


def download(root: Path) -> None:
    """Materialize verified links from the pinned Hugging Face release."""
    release = json.loads((HERE / "hf_release.json").read_text())
    root = root.expanduser().absolute()
    remote_path = hf_hub_download(
        release["repository"], release["manifest"], revision=release["revision"]
    )
    remote = json.loads(Path(remote_path).read_text())
    local = json.loads((HERE / "resources.json").read_text())
    stripped = {
        name: {key: value for key, value in record.items() if key != "path_in_repo"}
        for name, record in remote["files"].items()
    }
    if stripped != local["files"]:
        raise ValueError("Published resources differ from the checked-out recipe")
    for name, record in remote["files"].items():
        target = root / name
        source = (
            target
            if target.exists() or target.is_symlink()
            else Path(
                hf_hub_download(
                    release["repository"],
                    record["path_in_repo"],
                    revision=release["revision"],
                )
            ).resolve()
        )
        if (
            not source.is_file()
            or source.stat().st_size != record["bytes"]
            or digest(source) != record["sha256"]
        ):
            raise ValueError(f"Resource identity mismatch: {name}")
        if source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        print(f"Verified {name}", flush=True)
    (root / "hf_release.json").write_text(json.dumps(release, indent=2) + "\n")


def main() -> None:
    """Download resources to an explicitly chosen target directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", required=True, type=Path)
    download(parser.parse_args().resource_root)


if __name__ == "__main__":
    main()
