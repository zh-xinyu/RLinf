# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Location-independent fingerprint of the Linux SAPIEN wheel payload."""

import hashlib
import json
from pathlib import Path

SCHEME = "sapien-installed-payload-sha256-v1"


def excluded(relative: Path) -> bool:
    """Exclude only generated Python bytecode, never source/binaries/resources."""
    return "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}


def digest_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def finalize(entries: list[dict]) -> dict:
    entries.sort(key=lambda entry: entry["path"])
    payload = {"scheme": SCHEME, "files": entries}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "scheme": SCHEME,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }


def fingerprint_sapien_package(package_root: Path) -> dict:
    """Hash actual package and sibling wheel libraries, independent of location."""
    package_root = package_root.resolve()
    roots = {"sapien": package_root, "sapien.libs": package_root.parent / "sapien.libs"}
    entries = []
    for logical_name, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"Missing required Linux wheel payload: {root}")
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if excluded(relative):
                continue
            if path.is_symlink():
                raise ValueError(f"Unsupported payload symlink: {path}")
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                content_sha256 = digest_stream(stream)
            entries.append(
                {
                    "path": f"{logical_name}/{relative.as_posix()}",
                    "size_bytes": path.stat().st_size,
                    "sha256": content_sha256,
                }
            )
    return finalize(entries)
