# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fingerprint relocation, generated-cache and substantive-change regression."""

import tempfile
from pathlib import Path

from rlinf.envs.robotwin.package_fingerprint import (
    fingerprint_sapien_package as fingerprint,
)


def test_content_identity_invariants():
    """Exercise relocation/cache invariance and meaningful content changes."""
    with tempfile.TemporaryDirectory(
        prefix="opd-sapien-fingerprint-test-"
    ) as directory:
        base = Path(directory)
        files = {
            "sapien/__init__.py": b"version='test'\n",
            "sapien/pysapien.cpython-311.so": b"native-binary",
            "sapien/vulkan_shader/rt/camera.rgen": b"raygen shader",
            "sapien.libs/libsvulkan2.so": b"renderer-binary",
        }
        for install in ("vendor", "normal/site-packages"):
            for relative, content in files.items():
                path = base / install / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        vendor = base / "vendor/sapien"
        normal = base / "normal/site-packages/sapien"
        original = fingerprint(vendor)["sha256"]
        assert fingerprint(normal)["sha256"] == original
        (vendor / "__pycache__").mkdir()
        (vendor / "__pycache__/__init__.cpython-311.pyc").write_bytes(b"path-specific")
        (vendor.parent / "opd_sapien_manifest.json").write_text("optional receipt")
        metadata = vendor.parent / "sapien-test.dist-info"
        metadata.mkdir()
        (metadata / "direct_url.json").write_text("absolute installation source")
        assert fingerprint(vendor)["sha256"] == original
        for relative, content in files.items():
            path = vendor.parent / relative
            path.write_bytes(content + b"changed")
            assert fingerprint(vendor)["sha256"] != original, relative
            path.write_bytes(content)
        missing = vendor / "vulkan_shader/rt/camera.rgen"
        missing.unlink()
        assert fingerprint(vendor)["sha256"] != original
        missing.write_bytes(files["sapien/vulkan_shader/rt/camera.rgen"])
        symlink = vendor / "outside.py"
        symlink.symlink_to(base / "untracked-file")
        try:
            fingerprint(vendor)
        except ValueError:
            pass
        else:
            raise AssertionError("Payload symlink was not rejected")
