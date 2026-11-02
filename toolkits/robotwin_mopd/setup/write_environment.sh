#!/usr/bin/env bash
# Generate an explicit, machine-local activation file for the isolated pilot.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROBOTWIN_RUNTIME_ROOT="${ROBOTWIN_RUNTIME_ROOT:-/localscratch2/liyifa11/XyFile/opd-robotwin4}"
{
    printf 'export ROBOTWIN_RUNTIME_ROOT=%q\n' "$ROBOTWIN_RUNTIME_ROOT"
    printf 'export REPO_PATH=%q\n' "$REPO_ROOT"
    cat <<'ENV'
export ROBOTWIN_PATH="$ROBOTWIN_RUNTIME_ROOT/repos/RoboTwin"
export OPENPI_PATH="$ROBOTWIN_RUNTIME_ROOT/repos/openpi"
export VIRTUAL_ENV="$ROBOTWIN_RUNTIME_ROOT/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$REPO_PATH:$ROBOTWIN_PATH:${PYTHONPATH:-}"
export OPD_ROBOTWIN_PROFILE=pi05-four-task
# Optional temporary vendor for new processes while older jobs keep running.
# Prefer the canonical UTF-8 patched payload used by the normal finalizer.
# Fresh builds install the pinned version normally and need no vendor path.
if [[ -z "${OPD_ROBOTWIN_SAPIEN_VENDOR:-}" ]]; then
    for opd_sapien_candidate in \
        "$ROBOTWIN_RUNTIME_ROOT/vendor/sapien-3.0.0b1-utf8" \
        "$REPO_PATH/outputs/robotwin4/reference/sapien300b1_cp311_utf8_vendor" \
        "$REPO_PATH/../../robotwin4/reference/sapien300b1_cp311_utf8_vendor" \
        "$ROBOTWIN_RUNTIME_ROOT/vendor/sapien-3.0.0b1" \
        "$REPO_PATH/outputs/robotwin4/reference/sapien300b1_cp311_vendor" \
        "$REPO_PATH/../../robotwin4/reference/sapien300b1_cp311_vendor"; do
        if [[ -d "$opd_sapien_candidate/sapien" ]]; then
            export OPD_ROBOTWIN_SAPIEN_VENDOR="$(cd "$opd_sapien_candidate" && pwd)"
            break
        fi
    done
    unset opd_sapien_candidate
fi
if [[ -n "${OPD_ROBOTWIN_SAPIEN_VENDOR:-}" ]]; then
    if [[ ! -d "$OPD_ROBOTWIN_SAPIEN_VENDOR/sapien" ]]; then
        echo "Invalid RoboTwin SAPIEN vendor: $OPD_ROBOTWIN_SAPIEN_VENDOR" >&2
        return 1 2>/dev/null || exit 1
    fi
    export PYTHONPATH="$OPD_ROBOTWIN_SAPIEN_VENDOR:$PYTHONPATH"
fi
export PYTHONNOUSERSITE=1
export ASSETS_PATH="$ROBOTWIN_PATH"
export ROBOT_PLATFORM=ALOHA
export HF_HOME="$ROBOTWIN_RUNTIME_ROOT/cache/huggingface"
export UV_CACHE_DIR="$ROBOTWIN_RUNTIME_ROOT/cache/uv"
export OPENPI_DATA_HOME="$ROBOTWIN_RUNTIME_ROOT/cache/openpi"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
ENV
} > "$ROBOTWIN_RUNTIME_ROOT/env.sh"
