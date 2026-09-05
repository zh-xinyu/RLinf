#!/usr/bin/env bash
# Provision the research GR00T N1.5 / RoboCasa365 runtime without starting Ray.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash toolkits/mopd/setup_runtime.sh <all|install|resources|check> [options]' \
        '  all             Install software, copy shared resources, run CPU checks.' \
        '  install         Fresh software install, including editable repos and CPU imports.' \
        '  resources       Copy the four released checkpoints and RoboCasa assets from shared storage.' \
        '  check           Offline CPU imports, processor, assets, and checkpoint-header checks.' \
        '  --dry-run       Print the plan; no downloads, writes, credentials, Ray, or CUDA.' \
        '  --resume-install  Retry only a partial install created by this script.' \
        '  --system-deps   Let requirements/sys_deps.sh install host packages (needs root).' \
        'Read .opd.local.env for OPD_RUNTIME_ROOT, OPD_SHARED_ROOT, WANDB_ENV_FILE.' \
        'Prerequisites: Linux x86_64, NVIDIA host driver, bash, git, uv, rsync, flock.'
}

fail() { echo "setup_runtime: $*" >&2; exit 1; }

stage=${1:---help}
case "$stage" in
    -h|--help) usage; exit 0 ;;
    all|install|resources|check) shift ;;
    *) usage >&2; fail "Unknown stage: $stage" ;;
esac
dry_run=${OPD_DRY_RUN:-0}
resume_install=0
system_deps=0
while (($#)); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --resume-install) resume_install=1 ;;
        --system-deps) system_deps=1 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
    shift
done

setup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$setup_dir/local_env.sh"

# Source pins from the working N1.5 research runtime, NOT the N1.6/N1.7 recipes.
profile_id=gr00t-n15-robocasa365-20260905
python_version=3.11.14
readonly gr00t_revision=4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
readonly robocasa_revision=921c9a5736a8d0ea5589657898aadcfa55a6a195
readonly robosuite_revision=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
export GR00T_PATH="$OPD_REPOS_ROOT/gr00t"
export ROBOCASA_PATH="$OPD_REPOS_ROOT/robocasa"
export ROBOCASA_ASSETS_PATH="$OPD_RUNTIME_ROOT/assets/robocasa365"
export ROBOCASA365_ROBOSUITE_REV="$robosuite_revision"
export RLINF_PYTHON_EXECUTABLE="$OPD_RUNTIME_ROOT/python/bin/python3.11"
export UV_PYTHON_INSTALL_DIR="$OPD_RUNTIME_ROOT/cache/python-bootstrap"
export UV_PYTHON_BIN_DIR="$OPD_RUNTIME_ROOT/cache/bin"
export MPLCONFIGDIR="$OPD_RUNTIME_ROOT/cache/matplotlib"
export NO_ALBUMENTATIONS_UPDATE=1
# Do not accidentally install into an activated conda/other-project environment.
unset VIRTUAL_ENV CONDA_PREFIX UV_PROJECT_ENVIRONMENT UV_PYTHON

opd_print_paths
printf 'stage=%s\nprofile=%s\nPython=%s\nGR00T=%s\nRoboCasa=%s\nRobosuite=%s\n' \
    "$stage" "$profile_id" "$python_version" "$gr00t_revision" "$robocasa_revision" "$robosuite_revision"

check_root() {
    local target=$1
    case "$target" in
        /|/tmp|/localscratch2|/egr|"$HOME"|"$REPO"|"$OPD_MAIN_REPO")
            fail "Choose a dedicated dependency directory, not $target" ;;
    esac
    [[ "$target" != "$REPO/"* && "$REPO" != "$target/"* ]] || fail "Runtime must be outside the repo"
}
check_root "$OPD_RUNTIME_ROOT"
if [[ -n "$OPD_SHARED_ROOT" ]]; then
    [[ "$OPD_SHARED_ROOT" != / && "$OPD_SHARED_ROOT" != "$OPD_RUNTIME_ROOT" \
        && "$OPD_RUNTIME_ROOT" != "$OPD_SHARED_ROOT/"* \
        && "$OPD_SHARED_ROOT" != "$OPD_RUNTIME_ROOT/"* ]] || fail "Shared/runtime roots must be separate"
fi

installer=(bash "$REPO/requirements/install.sh" embodied --model gr00t --env robocasa365
    --venv "$VENV" --python "$python_version")
((system_deps)) || installer+=(--no-root)
if [[ "$dry_run" == 1 ]]; then
    if [[ "$stage" == all || "$stage" == install ]]; then
        printf 'install: download local Python; clone pinned repos; '
        printf '%q ' "${installer[@]}"
        printf '\n'
    fi
    if [[ "$stage" == all || "$stage" == resources ]]; then
        printf 'resources: %s/{models,assets} -> %s/{models,assets} (no deletion or overwrite)\n' \
            "$OPD_SHARED_ROOT" "$OPD_RUNTIME_ROOT"
    fi
    if [[ "$stage" != resources ]]; then
        printf 'check: CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 probe_runtime_paths.py (no Ray)\n'
    fi
    exit 0
fi

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail "This profile is for Linux x86_64 / NVIDIA"
for command in git uv rsync flock; do
    command -v "$command" >/dev/null || fail "Missing prerequisite: $command"
done
if [[ "$stage" == all || "$stage" == resources ]]; then
    [[ -d "$OPD_SHARED_ROOT/models/robocasa365/gr00t_n1-5/foundation_model_learning" \
        && -d "$OPD_SHARED_ROOT/assets/robocasa365/fixtures" ]] || \
        fail "Shared models/assets are not staged; see toolkits/mopd/README.md"
fi

pending="$OPD_RUNTIME_ROOT/.setup-in-progress"
if [[ "$stage" == all || "$stage" == install ]]; then
    if [[ -e "$VENV" || -L "$VENV" ]]; then
        [[ "$resume_install" == 1 && -f "$pending" && $(<"$pending") == "$profile_id" ]] || \
            fail "Existing venv is protected: $VENV. Use check, or choose a new runtime root."
    fi
fi
mkdir -p -- "$OPD_RUNTIME_ROOT/cache"
exec 9>"$OPD_RUNTIME_ROOT/cache/setup.lock"
flock -n 9 || fail "Another setup process holds the runtime lock"
run_id="setup-${stage}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
setup_log_dir="$OPD_OUTPUTS_ROOT/smoke/$run_id"
mkdir -p -- "$setup_log_dir"
exec > >(tee "$setup_log_dir/setup.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$setup_log_dir/status"' EXIT
opd_print_paths > "$setup_log_dir/paths.txt"

checkout_pinned() {
    local var=$1 path=$2 url=$3 branch=$4 revision=$5
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || fail "Expected a full commit SHA, not a branch or tag: $revision"
    [[ $(realpath -m -- "$path") == "$path" ]] || fail "Source path must not redirect outside repos: $path"
    if [[ ! -e "$path" && ! -L "$path" ]]; then
        # Reuse the RLinf clone helper in a subshell; its globals stay isolated.
        (
            set +u
            source "$REPO/requirements/install.sh"
            clone_or_reuse_repo "$var" "$path" "$url" -b "$branch"
        )
        # The branch only bootstraps the clone; the installed tree is this exact commit.
        git -C "$path" checkout --detach "$revision"
    fi
    [[ $(git -C "$path" rev-parse HEAD) == "$revision" ]] || fail "Unexpected revision: $path"
    # External asset symlink replaces packaged seeds; macros_private is generated.
    local changes
    changes=$(git -C "$path" status --porcelain --untracked-files=normal -- . \
        ':(exclude)robocasa/models/assets' ':(exclude)robocasa/macros_private.py')
    [[ -z "$changes" ]] || fail "Source changes at $path; preserve/review them before reinstalling"
}

install_software() {
    mkdir -p -- "$OPD_REPOS_ROOT"
    [[ ! -L "$OPD_RUNTIME_ROOT/python" ]] || fail "The local Python prefix must not be a symlink"
    if [[ ! -e "$OPD_RUNTIME_ROOT/python" ]]; then
        uv python install --install-dir "$UV_PYTHON_INSTALL_DIR" --no-bin "$python_version"
        local downloaded_python downloaded_prefix
        downloaded_python=$(uv python find --managed-python --no-project "$python_version")
        downloaded_prefix=$("$downloaded_python" -c 'import sys; print(sys.base_prefix)')
        [[ "$downloaded_prefix" == "$UV_PYTHON_INSTALL_DIR/"* ]] || fail "Python downloaded outside runtime"
        # Relocate standalone CPython BEFORE building the venv, never afterwards.
        mv -T --no-clobber -- "$downloaded_prefix" "$OPD_RUNTIME_ROOT/python"
    fi
    [[ $("$RLINF_PYTHON_EXECUTABLE" -c 'import platform; print(platform.python_version())') == "$python_version" ]] || \
        fail "Unexpected local Python version"
    checkout_pinned GR00T_PATH "$GR00T_PATH" https://github.com/NVIDIA/Isaac-GR00T.git n1.5-release "$gr00t_revision"
    checkout_pinned ROBOCASA_PATH "$ROBOCASA_PATH" https://github.com/robocasa/robocasa.git main "$robocasa_revision"
    printf '%s\n' "$profile_id" > "$pending"
    (
        cd -- "$REPO"
        "${installer[@]}"
    )
    CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        "$VENV/bin/python" "$setup_dir/probe_runtime_paths.py" --software-only
    uv pip freeze --python "$VENV/bin/python" > "$setup_log_dir/requirements-installed.txt"
    mv -T -- "$pending" "$OPD_RUNTIME_ROOT/.setup-complete"
}

copy_resource() {
    local source=$1 destination=$2 differences
    [[ -d "$source" && ! -L "$destination" ]] || fail "Missing source or symlink destination: $source -> $destination"
    [[ $(realpath -m -- "$destination") == "$destination" ]] || fail "Resource destination must stay under the runtime: $destination"
    mkdir -p -- "$destination"
    rsync -a --ignore-existing --exclude=.cache/ "$source/" "$destination/"
    # Existing mismatched files are reported, never silently overwritten.
    differences=$(rsync -rlnc --exclude=.cache/ --out-format='%i %n%L' "$source/" "$destination/")
    [[ -z "$differences" ]] || fail "Resource checksum mismatch at $destination: $differences"
}

copy_resources() {
    local relative
    for relative in pretraining/checkpoint-80000 \
        target_posttraining/atomic_seen/checkpoint-60000 \
        target_posttraining/composite_seen/checkpoint-60000 \
        target_posttraining/composite_unseen/checkpoint-60000; do
        copy_resource "$OPD_SHARED_ROOT/models/robocasa365/gr00t_n1-5/foundation_model_learning/$relative" \
            "$OPD_MODEL_ROOT/$relative"
    done
    copy_resource "$OPD_SHARED_ROOT/assets/robocasa365" "$ROBOCASA_ASSETS_PATH"
}

check_runtime() {
    [[ -x "$VENV/bin/python" ]] || fail "Install software first"
    CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        "$VENV/bin/python" "$setup_dir/probe_runtime_paths.py"
}

case "$stage" in
    all) install_software; copy_resources; check_runtime ;;
    install) install_software ;;
    resources) copy_resources ;;
    check) check_runtime ;;
esac
printf 'SETUP_OK stage=%s logs=%s\n' "$stage" "$setup_log_dir"
