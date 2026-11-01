#!/usr/bin/env bash
# Isolated pilot provisioning; use the repository installer helpers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROBOTWIN_RUNTIME_ROOT="${ROBOTWIN_RUNTIME_ROOT:-/localscratch2/liyifa11/XyFile/opd-robotwin4}"
cd "$REPO_ROOT"
source requirements/install.sh
VENV_DIR="$ROBOTWIN_RUNTIME_ROOT/venv"
PYTHON_VERSION=3.11.14
NO_ROOT=1
DISABLE_APEX=1
export UV_CACHE_DIR="$ROBOTWIN_RUNTIME_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROBOTWIN_RUNTIME_ROOT/python"
export RLINF_PYTHON_EXECUTABLE=/localscratch2/liyifa11/XyFile/opd/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
if [[ ! -x "$RLINF_PYTHON_EXECUTABLE" ]]; then
    unset RLINF_PYTHON_EXECUTABLE
fi
mkdir -p "$ROBOTWIN_RUNTIME_ROOT/repos" "$ROBOTWIN_RUNTIME_ROOT/cache"
ROBOTWIN_PATH=$(clone_or_reuse_repo ROBOTWIN_PATH "$ROBOTWIN_RUNTIME_ROOT/repos/RoboTwin" https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support --depth 1)
OPENPI_PATH=$(clone_or_reuse_repo OPENPI_PATH "$ROBOTWIN_RUNTIME_ROOT/repos/openpi" https://github.com/RLinf/openpi.git --depth 1)
pin_checkout() {
    local checkout="$1" revision="$2"
    if [[ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]]; then
        if [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
            echo "Refusing to replace modified checkout: $checkout" >&2
            return 1
        fi
        git -C "$checkout" fetch --depth 1 origin "$revision"
        git -C "$checkout" checkout --detach "$revision"
    fi
}
pin_checkout "$ROBOTWIN_PATH" dca9ec682688821c42944eacd7f5dd9bfb15396f
pin_checkout "$OPENPI_PATH" c5dc4b9296a1a4739bf52828f28a579f12dce763
printf '%s\n' "RoboTwin $(git -C "$ROBOTWIN_PATH" rev-parse HEAD)" "OpenPI $(git -C "$OPENPI_PATH" rev-parse HEAD)" > "$ROBOTWIN_RUNTIME_ROOT/source_revisions.txt"
create_and_sync_venv
install_common_embodied_deps
uv pip install "$OPENPI_PATH"
uv pip freeze > "$ROBOTWIN_RUNTIME_ROOT/base_freeze.txt"
