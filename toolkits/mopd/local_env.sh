#!/usr/bin/env bash
# Source this file before direct Python commands, or use the MOPD launchers.
# Loading paths never sources credentials, starts Ray, or creates output folders.

opd_load_paths() {
    local script_dir common_dir configured_runtime configured_shared configured_wandb
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 1
    REPO=$(realpath -e -- "${REPO:-$script_dir/../..}") || return 1
    common_dir=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir) || return 1
    OPD_MAIN_REPO=$(dirname -- "$common_dir")
    OPD_LOCAL_ENV_FILE=${OPD_LOCAL_ENV_FILE:-$OPD_MAIN_REPO/.opd.local.env}
    configured_runtime=${OPD_RUNTIME_ROOT:-}
    configured_shared=${OPD_SHARED_ROOT:-}
    configured_wandb=${WANDB_ENV_FILE:-}
    if [[ -f "$OPD_LOCAL_ENV_FILE" ]]; then
        source "$OPD_LOCAL_ENV_FILE" || return 1
    fi
    OPD_RUNTIME_ROOT=${configured_runtime:-${OPD_RUNTIME_ROOT:-}}
    OPD_SHARED_ROOT=${configured_shared:-${OPD_SHARED_ROOT:-}}
    WANDB_ENV_FILE=${configured_wandb:-${WANDB_ENV_FILE:-}}
    if [[ "$OPD_RUNTIME_ROOT" != /* || "$OPD_RUNTIME_ROOT" == / ]]; then
        echo "Set an absolute OPD_RUNTIME_ROOT in $OPD_LOCAL_ENV_FILE" >&2
        return 1
    fi
    OPD_RUNTIME_ROOT=$(realpath -m -- "$OPD_RUNTIME_ROOT") || return 1
    [[ "$OPD_RUNTIME_ROOT" != / ]] || { echo "Runtime root cannot be /" >&2; return 1; }
    if [[ -n "$OPD_SHARED_ROOT" ]]; then
        [[ "$OPD_SHARED_ROOT" == /* && "$OPD_SHARED_ROOT" != / ]] || return 1
        OPD_SHARED_ROOT=$(realpath -m -- "$OPD_SHARED_ROOT") || return 1
    fi
    if [[ -n "$WANDB_ENV_FILE" && "$WANDB_ENV_FILE" != /* ]]; then
        echo "WANDB_ENV_FILE must be absolute" >&2
        return 1
    fi
    VENV=$OPD_RUNTIME_ROOT/venv
    OPD_REPOS_ROOT=$OPD_RUNTIME_ROOT/repos
    OPD_OUTPUTS_ROOT=$OPD_MAIN_REPO/outputs
    OPD_MODEL_ROOT=$OPD_RUNTIME_ROOT/models/robocasa365/gr00t_n1-5/foundation_model_learning
    # Retained as a derived compatibility name for existing Python probes.
    ROBOCASA365_CHECKPOINT_ROOT=$OPD_MODEL_ROOT
    XDG_CACHE_HOME=$OPD_RUNTIME_ROOT/cache
    HF_HOME=$XDG_CACHE_HOME/huggingface
    HF_HUB_CACHE=$HF_HOME/hub
    HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
    HF_MODULES_CACHE=$HF_HOME/modules
    # HF_HOME/HF_HUB_CACHE replace the deprecated Transformers-specific override.
    unset TRANSFORMERS_CACHE
    UV_CACHE_DIR=$XDG_CACHE_HOME/uv
    TORCH_HOME=$XDG_CACHE_HOME/torch
    TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor
    TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
    CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
    NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
    TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/torch_extensions
    WANDB_CACHE_DIR=$XDG_CACHE_HOME/wandb
    export REPO OPD_MAIN_REPO OPD_LOCAL_ENV_FILE OPD_RUNTIME_ROOT OPD_SHARED_ROOT
    export VENV OPD_REPOS_ROOT OPD_OUTPUTS_ROOT OPD_MODEL_ROOT ROBOCASA365_CHECKPOINT_ROOT WANDB_ENV_FILE
    export XDG_CACHE_HOME HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_MODULES_CACHE
    export UV_CACHE_DIR TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH WANDB_CACHE_DIR
    export NUMBA_CACHE_DIR TORCH_EXTENSIONS_DIR
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export PATH="$VENV/bin:$OPD_RUNTIME_ROOT/python/bin:$PATH"
    export EMBODIED_PATH="$REPO/examples/embodiment" EVALUATIONS_PATH="$REPO/evaluations" REPO_PATH="$REPO"
}

opd_print_paths() {
    printf '%s\n' "code=$REPO" "outputs=$OPD_OUTPUTS_ROOT" \
        "runtime=$OPD_RUNTIME_ROOT" "venv=$VENV" "repos=$OPD_REPOS_ROOT" "models=$OPD_MODEL_ROOT" \
        "cache=$XDG_CACHE_HOME" "shared_source=$OPD_SHARED_ROOT" \
        "wandb_credentials_file=$WANDB_ENV_FILE"
}

opd_prepare_run() {
    local kind=$1 run_id=$2
    case "$kind" in train|eval|smoke) ;; *) echo "Invalid run kind: $kind" >&2; return 1 ;; esac
    if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
        echo "Run id must be a single safe directory name: $run_id" >&2
        return 1
    fi
    export OPD_RUN_DIR="$OPD_OUTPUTS_ROOT/$kind/$run_id"
    export RAY_START_LOG="$OPD_RUN_DIR/ray_start.log"
    opd_print_paths
    printf 'run=%s\n' "$OPD_RUN_DIR"
    [[ ${OPD_DRY_RUN:-0} == 1 ]] && return 0
    if [[ ! -x "$VENV/bin/python" || ! -f "$VENV/bin/activate" ]]; then
        echo "Runtime not installed: $VENV" >&2
        return 1
    fi
    mkdir -p -- "$OPD_OUTPUTS_ROOT/$kind" || return 1
    # Never overwrite an earlier run; use a new tag even for retries/resumes.
    mkdir -- "$OPD_RUN_DIR" || return 1
    mkdir -- "$OPD_RUN_DIR/artifacts" || return 1
    opd_print_paths > "$OPD_RUN_DIR/paths.txt"
}

opd_load_paths || { return 1 2>/dev/null || exit 1; }
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    opd_print_paths
fi
