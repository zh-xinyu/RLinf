#!/usr/bin/env bash
# Launch one MOPD run with the two watchdogs this box has needed.
#
# Usage: bash toolkits/mopd/run_mopd.sh <config_name> [tag] [extra hydra overrides...]
#
# Deliberately NOT `set -u`: openpi-style venv activate scripts dereference an unset
# PYTHONPATH. And never put a long wait in the same shell command as the launch -- a
# command timeout would kill the job and the log would show a SIGTERM that looks external.
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/local_env.sh" || exit 1
CONFIG=${1:?config name required}
TAG=${2:-$(date +%Y%m%d-%H%M%S)}
shift 2 2>/dev/null || shift 1
EXTRA=("$@")

opd_prepare_run "${OPD_RUN_KIND:-train}" "${OPD_RUN_ID:-$CONFIG.$TAG}" || exit 1
OUT_DIR=$OPD_RUN_DIR
LOG=$OUT_DIR/train.log
STATUS=$OUT_DIR/status
COMMAND=("$VENV/bin/python" -u "$REPO/examples/embodiment/train_embodied_agent.py"
    --config-name "$CONFIG" "runner.logger.experiment_name=$TAG" "${EXTRA[@]}")
printf 'command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ ${OPD_DRY_RUN:-0} == 1 ]] && exit 0
printf '%q ' "${COMMAND[@]}" > "$OUT_DIR/command.txt"

# How long the run may take to print its first sign of real work. The startup deadlock
# this box produces emits *no* output at all, so a silence timer tuned for mid-run pacing
# catches it far too late -- one such hang burned 9h and blocked a whole queue.
STARTUP_MARKER=${STARTUP_MARKER:-"Generating Rollout Epochs"}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2700}   # 45 min: GR00T loads 4 checkpoints at startup
SILENCE_TIMEOUT=${SILENCE_TIMEOUT:-3600}   # 60 min without the log growing
GPU_FREE_MB=${GPU_FREE_MB:-30000}
GPU_WAIT_MAX=${GPU_WAIT_MAX:-7200}

# Activate the venv in THIS shell before any ray command. `ray start` run from an
# unactivated shell picks up miniconda's ray (2.44 / py3.9) while the driver inside the
# venv is ray 2.56 / py3.11, and the driver then dies on a version mismatch against a
# cluster we started ourselves.
source "$VENV/bin/activate" || exit 1

source "$SCRIPT_DIR/ray_utils.sh"

wait_for_gpus() {
    local waited=0
    while [ "$waited" -lt "$GPU_WAIT_MAX" ]; do
        local busy
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk -v lim=$((49140 - GPU_FREE_MB)) '$1 > lim {c++} END {print c+0}')
        if [ "$busy" -eq 0 ]; then
            [ "$waited" -gt 0 ] && log "  GPUs free after ${waited}s"
            return 0
        fi
        log "  waiting for GPUs ($busy busy), ${waited}s elapsed"
        sleep 120
        waited=$((waited + 120))
    done
    log "  ERROR: GPUs never freed"
    return 1
}

log "=== MOPD run: config=$CONFIG tag=$TAG ==="
echo "RUNNING" > "$STATUS"

wait_for_gpus || { echo "GPU_WAIT_TIMEOUT" > "$STATUS"; exit 1; }
restart_ray   || { echo "RAY_FAILED" > "$STATUS"; exit 1; }

cd "$REPO" || exit 1
setsid nohup bash -c '
    if [[ -n "$WANDB_ENV_FILE" && -f "$WANDB_ENV_FILE" ]]; then
        source "$WANDB_ENV_FILE" || exit 1
    else
        export WANDB_MODE=${WANDB_MODE:-offline}
    fi
    # A6000s here have no NVLink and the box has no InfiniBand.
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export MUJOCO_GL=egl
    # Validation OOMed at 44.7/47.5 GB needing one more 3.73 GiB block. Expandable
    # segments let the allocator grow a segment instead of demanding that much
    # contiguous free space, which is the shape of this failure.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    exec "$@"
' opd-driver "${COMMAND[@]}" > "$LOG" 2>&1 < /dev/null &
RUN_PID=$!
log "  launched pid $RUN_PID -> $LOG"

# --- watchdog 1: must reach real work -------------------------------------
waited=0
started=0
while [ "$waited" -lt "$STARTUP_TIMEOUT" ]; do
    if grep -q "$STARTUP_MARKER" "$LOG" 2>/dev/null; then
        started=1
        log "  startup marker seen after ${waited}s"
        break
    fi
    if ! kill -0 "$RUN_PID" 2>/dev/null; then
        log "  process exited during startup"
        break
    fi
    sleep 30
    waited=$((waited + 30))
done
if [ "$started" -eq 0 ] && kill -0 "$RUN_PID" 2>/dev/null; then
    log "  ERROR: no startup marker in ${STARTUP_TIMEOUT}s -- killing (startup deadlock)"
    pkill -9 -P "$RUN_PID" 2>/dev/null; kill -9 "$RUN_PID" 2>/dev/null
    purge_ray
    echo "STARTUP_TIMEOUT" > "$STATUS"
    exit 1
fi

# --- watchdog 2: log must keep growing ------------------------------------
last_size=0
stalled=0
while kill -0 "$RUN_PID" 2>/dev/null; do
    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$size" -eq "$last_size" ]; then
        stalled=$((stalled + 60))
        if [ "$stalled" -ge "$SILENCE_TIMEOUT" ]; then
            log "  ERROR: log silent for ${SILENCE_TIMEOUT}s -- killing"
            pkill -9 -P "$RUN_PID" 2>/dev/null; kill -9 "$RUN_PID" 2>/dev/null
            purge_ray
            echo "SILENCE_TIMEOUT" > "$STATUS"
            exit 1
        fi
    else
        stalled=0
        last_size=$size
    fi
    sleep 60
done

wait "$RUN_PID"; rc=$?
log "  finished rc=$rc"
if [ "$rc" -eq 0 ]; then echo "OK" > "$STATUS"; else echo "EXIT_$rc" > "$STATUS"; fi
exit "$rc"
