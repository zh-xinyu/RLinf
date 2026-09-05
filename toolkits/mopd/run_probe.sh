#!/usr/bin/env bash
# Usage: bash toolkits/mopd/run_probe.sh probe_precision_init [probe arguments...]
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/local_env.sh"
PROBE=${1:?probe name required}
shift
[[ "$PROBE" =~ ^probe_[a-z0-9_]+$ && -f "$SCRIPT_DIR/$PROBE.py" ]] || {
    echo "Unknown probe: $PROBE" >&2; exit 1;
}
if [[ "$PROBE" == probe_path_equiv ]]; then
    echo "Historical probe: reference-inference methods were removed. Use probe_flow_ode for the current rollout path." >&2
    exit 1
fi
opd_prepare_run smoke "${OPD_RUN_ID:-$PROBE.$(date +%Y%m%d-%H%M%S)}"
COMMAND=("$VENV/bin/python" -u "$SCRIPT_DIR/$PROBE.py" "$@")
printf 'command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ ${OPD_DRY_RUN:-0} == 1 ]] && exit 0
printf '%q ' "${COMMAND[@]}" > "$OPD_RUN_DIR/command.txt"
export MUJOCO_GL=${MUJOCO_GL:-egl}
cd "$OPD_RUN_DIR"
"${COMMAND[@]}" 2>&1 | tee "$OPD_RUN_DIR/probe.log"
