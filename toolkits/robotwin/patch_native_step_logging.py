# Copyright 2025 The RLinf Authors.
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
"""Suppress per-control-target terminal output in the pinned RoboTwin checkout."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robotwin_root", type=Path)
    args = parser.parse_args()
    path = args.robotwin_root / "envs/_base_task.py"
    text = path.read_text()
    original = '        print(f"step: \\033[92m{self.take_action_cnt} / {self.step_lim}\\033[0m", end="\\r")'
    updated = '        if getattr(self, "eval_verbose", False):\n    ' + original
    if updated in text:
        return
    if text.count(original) != 1:
        raise RuntimeError("Unexpected source version; refusing ambiguous patch")
    path.write_text(text.replace(original, updated))


if __name__ == "__main__":
    main()
