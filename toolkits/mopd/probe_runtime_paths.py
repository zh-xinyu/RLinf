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
"""Validate local runtime paths and checkpoint headers without Ray or training."""

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def main() -> None:
    """Check the runtime, offline processor, and all four released checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--software-only", action="store_true", help="Skip assets and model files."
    )
    args = parser.parse_args()
    from safetensors import safe_open
    from transformers import AutoProcessor

    runtime = Path(os.environ["OPD_RUNTIME_ROOT"]).resolve()
    assert Path(sys.prefix).resolve() == runtime / "venv", sys.prefix
    assert Path(sys.base_prefix).resolve() == runtime / "python", sys.base_prefix
    report = {"runtime": str(runtime), "imports": {}, "checkpoints": {}}
    for name in ("torch", "ray", "gr00t", "robocasa", "robosuite", "flash_attn"):
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        assert location.is_relative_to(runtime), f"Nonlocal import: {location}"
        if name in ("gr00t", "robocasa"):
            expected = runtime / "repos" / name
            assert location.is_relative_to(expected), f"Wrong editable path: {location}"
            direct_url = json.loads(
                importlib.metadata.distribution(name).read_text("direct_url.json")
            )
            assert direct_url.get("dir_info", {}).get("editable"), direct_url
            assert direct_url["url"] == expected.as_uri(), direct_url
        report["imports"][name] = str(location)

    import robocasa
    from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH

    assets = (Path(robocasa.__file__).parent / "models/assets").resolve()
    assert assets == runtime / "assets/robocasa365", assets
    if not args.software_only:
        assert (assets / "fixtures").is_dir(), f"Missing fixture assets: {assets}"
    processor = AutoProcessor.from_pretrained(
        DEFAULT_EAGLE_PATH, trust_remote_code=True, local_files_only=True
    )
    report["processor"] = type(processor).__name__
    report["assets"] = str(assets)
    root = Path(os.environ["OPD_MODEL_ROOT"]).resolve()
    assert root.is_relative_to(runtime), root
    checkpoints = [root / "pretraining/checkpoint-80000"]
    checkpoints.extend(
        root / "target_posttraining" / suite / "checkpoint-60000"
        for suite in ("atomic_seen", "composite_seen", "composite_unseen")
    )
    for checkpoint in [] if args.software_only else checkpoints:
        assert (checkpoint / "config.json").is_file(), checkpoint
        assert (checkpoint / "experiment_cfg/metadata.json").is_file(), checkpoint
        weights = sorted(checkpoint.glob("*.safetensors"))
        assert weights, f"Missing weights: {checkpoint}"
        tensor_count = 0
        for weight in weights:
            with safe_open(weight, framework="pt", device="cpu") as handle:
                tensor_count += len(handle.keys())
        report["checkpoints"][str(checkpoint)] = tensor_count
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "ray",
            "gr00t",
            "robocasa",
            "transformers",
            "mujoco",
            "numpy",
        )
    }
    for name, expected in {
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "mujoco": "3.3.1",
        "numpy": "2.2.5",
    }.items():
        actual = report["versions"][name].split("+", 1)[0]
        assert actual == expected, f"Expected {name}=={expected}, got {actual}"
    report["software_only"] = args.software_only
    print(json.dumps(report, indent=2))
    print("RUNTIME_PATHS_OK")


if __name__ == "__main__":
    main()
