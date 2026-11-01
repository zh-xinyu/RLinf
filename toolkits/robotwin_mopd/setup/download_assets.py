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
"""Download the pinned official RoboTwin assets into an isolated checkout."""

import os
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(
    os.environ.get(
        "ROBOTWIN_RUNTIME_ROOT", "/localscratch2/liyifa11/XyFile/opd-robotwin4"
    )
)
REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
assets = ROOT / "repos/RoboTwin/assets"
archives = ("background_texture.zip", "embodiments.zip", "objects.zip")
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    repo_type="dataset",
    revision=REVISION,
    allow_patterns=list(archives),
    local_dir=assets,
    max_workers=3,
)
for name in archives:
    marker = assets / (name + ".extracted")
    if marker.exists():
        continue
    with zipfile.ZipFile(assets / name) as archive:
        for entry in archive.infolist():
            resolved = (assets / entry.filename).resolve()
            if not resolved.is_relative_to(assets.resolve()):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
        archive.extractall(assets)
    marker.write_text(REVISION + "\n")
print("Assets extracted", assets, flush=True)
