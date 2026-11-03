#!/usr/bin/env bash
# Dedicated research handoff profile; does not modify other model installers.
set -euo pipefail
if [[ $# -ne 2 || ( "$2" != action-cu124 && "$2" != blackwell-cu128-candidate ) ]]; then
    echo "Usage: bash install.sh /absolute/new/runtime {action-cu124|blackwell-cu128-candidate}" >&2
    exit 2
fi
opd_install_root="$1"
opd_install_profile="$2"
[[ "$opd_install_root" = /* && ! -e "$opd_install_root" ]] || {
    echo "Runtime must be a new absolute directory; existing environments are never replaced" >&2
    exit 2
}
opd_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$opd_repo"
source requirements/install.sh
export ROBOTWIN_RUNTIME_ROOT="$opd_install_root"
unset OPD_ROBOTWIN_SAPIEN_VENDOR PYTHONPATH
VENV_DIR="$ROBOTWIN_RUNTIME_ROOT/venv"
PYTHON_VERSION=3.11.14
export UV_CACHE_DIR="$ROBOTWIN_RUNTIME_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROBOTWIN_RUNTIME_ROOT/python"
mkdir -p "$ROBOTWIN_RUNTIME_ROOT/repos"
create_and_sync_venv create-only
ROBOTWIN_PATH=$(clone_or_reuse_repo ROBOTWIN_PATH "$ROBOTWIN_RUNTIME_ROOT/repos/RoboTwin" https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support --depth 1)
OPENPI_PATH=$(clone_or_reuse_repo OPENPI_PATH "$ROBOTWIN_RUNTIME_ROOT/repos/openpi" https://github.com/RLinf/openpi.git --depth 1)
for opd_pair in "$ROBOTWIN_PATH:dca9ec682688821c42944eacd7f5dd9bfb15396f" "$OPENPI_PATH:c5dc4b9296a1a4739bf52828f28a579f12dce763"; do
    opd_checkout="${opd_pair%:*}"
    opd_revision="${opd_pair##*:}"
    [[ -z "$(git -C "$opd_checkout" status --porcelain)" ]] || exit 2
    git -C "$opd_checkout" fetch --depth 1 origin "$opd_revision"
    git -C "$opd_checkout" checkout --detach "$opd_revision"
done
# Install frozen packages outside the repository's global uv torch overrides.
# No new core dependency override is applied to RLinf or other runtimes.
cd "$ROBOTWIN_RUNTIME_ROOT"
uv pip install --python "$VENV_DIR/bin/python" --no-deps -r "$opd_repo/toolkits/robotwin_mopd/formal/runtime-common.lock.txt"
if [[ "$opd_install_profile" == blackwell-cu128-candidate ]]; then
    uv pip install --python "$VENV_DIR/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 --constraint "$opd_repo/toolkits/robotwin_mopd/formal/cu128.constraints.txt"
    uv pip install --python "$VENV_DIR/bin/python" --no-deps torchcodec==0.3.0
else
    uv pip install --python "$VENV_DIR/bin/python" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python "$VENV_DIR/bin/python" --no-deps torchcodec==0.2.0
fi
uv pip install --python "$VENV_DIR/bin/python" --no-deps "$OPENPI_PATH" "$OPENPI_PATH/packages/openpi-client"
# Reuse verified simulator/transformers patches, without resolving packages again.
python "$opd_repo/toolkits/robotwin_mopd/formal/patch_runtime.py"
python "$opd_repo/toolkits/robotwin/patch_native_step_logging.py" "$ROBOTWIN_PATH"
bash "$opd_repo/toolkits/robotwin_mopd/setup/write_environment.sh"
uv pip freeze --python "$VENV_DIR/bin/python" > "$ROBOTWIN_RUNTIME_ROOT/runtime_freeze.txt"
printf '%s\n' "$opd_install_profile" > "$ROBOTWIN_RUNTIME_ROOT/formal_profile.txt"
echo "Installed candidate runtime. Assets, transferred models, preflight, rendering and training validation remain required."
