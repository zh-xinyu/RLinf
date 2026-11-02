# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fail closed on the verified RoboTwin4 simulator profile.

This guard is opt-in for generic RoboTwin users. The dedicated four-task
configuration requires it even when a Ray worker loses launcher environment
variables. Importing SAPIEN here does not create a scene or renderer.
"""

import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

from rlinf.envs.robotwin.package_fingerprint import fingerprint_sapien_package

PROFILE = "pi05-four-task"
SAPIEN_VERSION = "3.0.0b1"
ROBOTWIN_REVISION = "dca9ec682688821c42944eacd7f5dd9bfb15396f"


@lru_cache(maxsize=4)
def _validate(robotwin_root: str, vendor: str) -> dict:
    """Validate once per process/profile, including the actual loaded module."""
    sapien = importlib.import_module("sapien")
    actual_version = getattr(sapien, "__version__", None)
    distribution_version = importlib.metadata.version("sapien")
    module_path = Path(sapien.__file__).resolve()
    if actual_version != SAPIEN_VERSION or distribution_version != SAPIEN_VERSION:
        raise RuntimeError(
            f"RoboTwin4 requires SAPIEN {SAPIEN_VERSION}; loaded module "
            f"{actual_version}, distribution {distribution_version}, {module_path}. "
            "SAPIEN 3.0.1 changes wrist-camera rendering. Activate the verified "
            "RoboTwin4 profile before starting a new owned Ray instance."
        )
    if vendor and Path(vendor).resolve() not in module_path.parents:
        raise RuntimeError(
            f"RoboTwin4 requested SAPIEN vendor {vendor}, but loaded {module_path}"
        )
    root = Path(robotwin_root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ROBOTWIN_REVISION:
        raise RuntimeError(
            f"RoboTwin4 requires source {ROBOTWIN_REVISION}; found {revision}"
        )
    source_files = (
        "envs/_base_task.py",
        "envs/robot/robot.py",
        "envs/camera/camera.py",
        "task_config/demo_clean.yml",
        "task_config/_camera_config.yml",
    )
    report = {
        "profile": PROFILE,
        "sapien_version": actual_version,
        "sapien_distribution_version": distribution_version,
        "sapien_module_path": str(module_path),
        "sapien_module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "robotwin_revision": revision,
        "robotwin_source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }
    content = fingerprint_sapien_package(module_path.parent)
    report["sapien_content_fingerprint"] = {
        key: value for key, value in content.items() if key != "files"
    }
    if vendor:
        receipt = Path(vendor) / "opd_sapien_manifest.json"
        if receipt.exists():
            report["sapien_vendor_manifest"] = json.loads(receipt.read_text())
    return report


def validate_robotwin4_runtime() -> dict:
    """Require the verified SAPIEN/source versions and return provenance."""
    root = os.environ.get("ROBOTWIN_PATH")
    if not root:
        raise RuntimeError("RoboTwin4 requires ROBOTWIN_PATH from its isolated profile")
    return _validate(root, os.environ.get("OPD_ROBOTWIN_SAPIEN_VENDOR", "")).copy()


def validate_requested_profile(required_profile: str | None = None) -> dict | None:
    """Check a static config requirement or an explicitly activated profile."""
    profile = required_profile or os.environ.get("OPD_ROBOTWIN_PROFILE")
    if not profile:
        return None
    if profile != PROFILE:
        raise ValueError(f"Unsupported RoboTwin runtime profile: {profile}")
    return validate_robotwin4_runtime()
