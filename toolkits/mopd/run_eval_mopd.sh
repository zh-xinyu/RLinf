#!/usr/bin/env bash
# Standalone evaluation of the *undistilled* student, under the same harness the training
# run validates with. Without this there is no baseline to compare the in-training
# validation numbers against.
#
# Usage: bash toolkits/mopd/run_eval_mopd.sh <tag> [wait_status_file] [hydra overrides...]
# Pass an empty wait_status_file ("") when supplying overrides without a wait.
# If wait_status_file is given, block until that run writes a terminal status.
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/local_env.sh" || exit 1
TAG=${1:?tag required}
WAIT_FILE=${2:-}
shift 2 2>/dev/null || shift 1
EXTRA=("$@")

opd_prepare_run eval "${OPD_RUN_ID:-eval.$TAG}" || exit 1
OUT_DIR=$OPD_RUN_DIR
LOG=$OUT_DIR/eval.log
STATUS=$OUT_DIR/status
COMMAND=("$VENV/bin/python" -u "$REPO/evaluations/eval_embodied_agent.py"
    --config-path "$REPO/evaluations/robocasa365"
    --config-name robocasa365_mopd_gr00t_eval
    "runner.logger.experiment_name=$TAG" "${EXTRA[@]}")
printf 'command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ ${OPD_DRY_RUN:-0} == 1 ]] && exit 0
printf '%q ' "${COMMAND[@]}" > "$OUT_DIR/command.txt"

source "$VENV/bin/activate" || exit 1
source "$SCRIPT_DIR/ray_utils.sh"

# Wait on a status FILE, never on `pgrep -f <name>`: that matches the shell that created
# any script containing the name, including this session's own harness shell.
if [ -n "$WAIT_FILE" ]; then
    waited=0
    while [ ! -f "$WAIT_FILE" ] || [ "$(cat "$WAIT_FILE")" = "RUNNING" ]; do
        sleep 60
        waited=$((waited + 60))
        if [ "$waited" -ge 43200 ]; then
            log "ERROR: $WAIT_FILE still RUNNING after 12h; giving up"
            echo "WAIT_TIMEOUT" > "$STATUS"
            exit 1
        fi
    done
    log "upstream run finished with status $(cat "$WAIT_FILE"); starting eval"
    sleep 60   # let GPUs and Ray settle
fi

echo "RUNNING" > "$STATUS"
restart_ray || { echo "RAY_FAILED" > "$STATUS"; exit 1; }

cd "$REPO" || exit 1
setsid nohup bash -c '
    if [[ -n "$WANDB_ENV_FILE" && -f "$WANDB_ENV_FILE" ]]; then
        source "$WANDB_ENV_FILE" || exit 1
    else
        export WANDB_MODE=${WANDB_MODE:-offline}
    fi
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export MUJOCO_GL=egl
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}
    export HYDRA_FULL_ERROR=1
    exec "$@"
' opd-eval "${COMMAND[@]}" > "$LOG" 2>&1 < /dev/null &
RUN_PID=$!
log "  eval launched pid $RUN_PID -> $LOG"

# silence watchdog only: an eval prints steadily once it starts
last_size=0; stalled=0
while kill -0 "$RUN_PID" 2>/dev/null; do
    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$size" -eq "$last_size" ]; then
        stalled=$((stalled + 60))
        if [ "$stalled" -ge 3600 ]; then
            log "  ERROR: eval log silent for 1h -- killing"
            pkill -9 -P "$RUN_PID" 2>/dev/null; kill -9 "$RUN_PID" 2>/dev/null
            purge_ray; echo "SILENCE_TIMEOUT" > "$STATUS"; exit 1
        fi
    else
        stalled=0; last_size=$size
    fi
    sleep 60
done
wait "$RUN_PID"; rc=$?
log "  eval finished rc=$rc"
if [ "$rc" -eq 0 ]; then echo "OK" > "$STATUS"; else echo "EXIT_$rc" > "$STATUS"; fi
