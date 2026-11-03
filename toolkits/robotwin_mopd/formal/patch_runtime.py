# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Apply the same OpenPI/SAPIEN/MPLib patches as the verified pilot finalizer."""

import importlib.util
import shutil
from pathlib import Path


def main() -> None:
    openpi = Path(importlib.util.find_spec("openpi").origin).parent
    transformers = Path(importlib.util.find_spec("transformers").origin).parent
    shutil.copytree(
        openpi / "models_pytorch/transformers_replace", transformers, dirs_exist_ok=True
    )
    loader = (
        Path(importlib.util.find_spec("sapien").origin).parent
        / "wrapper/urdf_loader.py"
    )
    text = loader.read_text()
    for filename in ("urdf_file", "srdf_file"):
        text = text.replace(
            f'with open({filename}, "r") as f:',
            f'with open({filename}, "r", encoding="utf-8") as f:',
        )
    loader.write_text(text)
    planner = Path(importlib.util.find_spec("mplib").origin).parent / "planner.py"
    text = planner.read_text().replace(
        "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
        "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
    )
    planner.write_text(text)


if __name__ == "__main__":
    main()
