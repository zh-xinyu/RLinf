#!/usr/bin/env bash
# MPLib inference profile: no CUDA planner build or changes to shared runtimes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROBOTWIN_RUNTIME_ROOT="${ROBOTWIN_RUNTIME_ROOT:-/localscratch2/liyifa11/XyFile/opd-robotwin4}"
source "$ROBOTWIN_RUNTIME_ROOT/venv/bin/activate"
export UV_CACHE_DIR="$ROBOTWIN_RUNTIME_ROOT/cache/uv"
cd "$REPO_ROOT"
uv pip install -r toolkits/robotwin_mopd/setup/runtime-pins.txt
python - <<'PY'
from pathlib import Path
import importlib.util
import shutil
openpi = Path(importlib.util.find_spec('openpi').origin).parent
transformers = Path(importlib.util.find_spec('transformers').origin).parent
shutil.copytree(openpi / 'models_pytorch/transformers_replace', transformers, dirs_exist_ok=True)
sapien = Path(importlib.util.find_spec('sapien').origin).parent
loader = sapien / 'wrapper/urdf_loader.py'
s = loader.read_text()
s = s.replace('with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:')
s = s.replace('with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:')
loader.write_text(s)
mplib = Path(importlib.util.find_spec('mplib').origin).parent
planner = mplib / 'planner.py'
s = planner.read_text()
s = s.replace('if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:', 'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:')
planner.write_text(s)
PY
uv pip freeze > "$ROBOTWIN_RUNTIME_ROOT/runtime_freeze.txt"
bash "$REPO_ROOT/toolkits/robotwin_mopd/setup/write_environment.sh"
