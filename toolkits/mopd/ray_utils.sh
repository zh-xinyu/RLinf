#!/usr/bin/env bash
# Ray lifecycle helpers for the MOPD runs on this box.
#
# Two hazards this box has actually produced (see .project_results/mopd/RAY_ON_THIS_BOX.md):
#  1. `ray stop --force` does not reliably kill gcs_server. A survivor keeps port 6379 and
#     the next driver either fails with a session-name assert or, worse, connects and then
#     blocks forever in CoreWorker init because no raylet answers.
#  2. With the default --num-cpus on a many-core box the raylet prestarts one worker per
#     CPU; they all miss the registration timeout and the raylet never answers the driver.
# So: purge in a verify-and-repeat loop, start with a modest --num-cpus, and prove the
# cluster works by round-tripping a real remote task -- `ray status` only proves the GCS
# answers, which is exactly the state that hangs.

RAY_TMP=${RAY_TMP:-/dev/shm/ray_mopd}
# The raylet prestarts exactly this many Python workers
# (services.py: --num_prestart_python_workers={num_cpus}) and must still register with the
# GCS inside a hardcoded 30s (node.py: raylet_start_wait_time_s = 30, no env var). Each
# worker imports ray, which costs ~19s cold in this venv, so a high value makes startup a
# coin flip on how warm the page cache is. 32 still covers RLinf's ~24 actors.
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}

# --- Ray startup timeouts, at file scope so every script that sources this gets them ---
# Total dashboard budget (ray_constants.py, default 60s).
# Ray overwrites CUDA_VISIBLE_DEVICES inside each worker with its own GPU ids, which
# silently discards an outer restriction: pinning the driver to physical GPU 7 still lands
# the actor on physical GPU 0 (Ray id 0). On a shared box that means running straight into
# someone else's job. This tells Ray to leave the variable alone.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_DASHBOARD_STARTUP_TIMEOUT_S="${RAY_DASHBOARD_STARTUP_TIMEOUT_S:-300}"
# Per-subprocess-module readiness budget (dashboard/consts.py, default 30s). The dashboard
# runs nine such modules, each its own process; a cold `import ray` is ~19s here and
# MetricsHead does more on top, so it misses 30s and takes the whole `ray start` down with
# "Module MetricsHead failed to start. Timeout after 30.0 seconds." Raising the *overall*
# timeout above does not help -- this is the one that actually fires.
export RAY_DASHBOARD_SUBPROCESS_MODULE_WAIT_READY_TIMEOUT="${RAY_DASHBOARD_SUBPROCESS_MODULE_WAIT_READY_TIMEOUT:-180}"

log() { echo "[$(date '+%F %T')] $*"; }

purge_ray() {
    ray stop --force > /dev/null 2>&1
    for pass in 1 2 3 4 5; do
        # `grep -v grep` matters: without it the counting grep matches its own command line
        # and the purge loops forever against a phantom survivor.
        local pids
        pids=$(ps -eo pid,cmd --no-headers 2>/dev/null \
               | grep -iE "gcs_server|raylet|ray/core|ray::" | grep -v grep | awk '{print $1}')
        local port_pids
        port_pids=$( (lsof -ti :6379; lsof -ti :8265) 2>/dev/null | sort -u )
        if [ -z "$pids" ] && [ -z "$port_pids" ]; then
            rm -rf "$RAY_TMP" /tmp/ray 2>/dev/null
            log "  purge_ray: clear after pass $pass"
            return 0
        fi
        [ -n "$pids" ] && kill -9 $pids 2>/dev/null
        [ -n "$port_pids" ] && kill -9 $port_pids 2>/dev/null
        sleep 5
    done
    log "  WARN: purge_ray could not fully clear Ray"
    rm -rf "$RAY_TMP" /tmp/ray 2>/dev/null
    return 1
}

verify_ray() {
    # A real remote task, not `ray status`.
    python - <<'PY'
import sys
import ray
try:
    ray.init(address="auto", log_to_driver=False)
    @ray.remote
    def ping():
        return "pong"
    assert ray.get(ping.remote(), timeout=60) == "pong"
    print("ray verify: pong")
    # RLinf also needs the dashboard/state API; a cluster without it passes the task
    # round-trip and then kills the driver later.
    import urllib.request
    urllib.request.urlopen("http://127.0.0.1:8265/api/v0/actors", timeout=30).read(1)
    print("ray verify: dashboard api ok")
except Exception as exc:  # noqa: BLE001
    print(f"ray verify FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1)
PY
}

# Pull ray's modules into the page cache before the raylet forks its prestart workers.
# A cold `import ray` is ~19s here; warm it once and the concurrent worker imports stop
# competing for the raylet's 30s registration budget.
warm_ray_import() {
    local t0=$SECONDS
    # Warm ray itself AND the dashboard agent / metrics modules. The raylet waits for the
    # dashboard agent to write session/metrics_agent_port_* and *fatally crashes* when it
    # does not appear in time ("Check failed: status.has_value() Timed out waiting for
    # file ... metrics_agent_port_...", node_manager.cc:3449). That agent is a separate
    # Python process, so its cold import is on the same ~19s shared-filesystem budget.
    python - <<'PYWARM' > /dev/null 2>&1
for mod in (
    "ray",
    "ray.dashboard.agent",
    "ray.dashboard.head",
    "ray.dashboard.modules.metrics.metrics_head",
    "ray.dashboard.modules.reporter.reporter_agent",
):
    try:
        __import__(mod)
    except Exception:
        pass
PYWARM
    log "  warmed ray + dashboard-agent imports in $((SECONDS - t0))s"
}

restart_ray() {
    purge_ray
    warm_ray_import
    for try in 1 2 3; do
        # The dashboard must stay ON: RLinf's runner queries the Ray state API on 8265,
        # and with the dashboard disabled it dies with ServerUnavailable after the actors
        # have already initialised. But its default 60s startup budget is not enough on
        # this box -- it imports its modules off a shared filesystem -- and when it misses
        # the deadline `ray start` fails outright with no cluster left behind.
        # Never discard this output: three silent failures cost an hour of guessing.
        # Ray's resource ledger, not CUDA_VISIBLE_DEVICES, decides how many workers get
        # created. Without this, restricting visibility to one GPU still yields eight
        # rollout workers all mapped onto it -- 8 x ~20GB and an immediate OOM.
        local gpu_arg=""
        if [ -n "${RAY_NUM_GPUS:-}" ]; then
            gpu_arg="--num-gpus=$RAY_NUM_GPUS"
        fi
        ray start --head --port=6379 --num-cpus="$RAY_NUM_CPUS" $gpu_arg \
            --temp-dir="$RAY_TMP" --include-dashboard=true \
            --dashboard-host=127.0.0.1 --dashboard-port=8265 \
            >> "${RAY_START_LOG:-/tmp/ray_start.log}" 2>&1
        sleep 20
        if verify_ray; then
            log "  ray up and verified (try $try, --num-cpus=$RAY_NUM_CPUS)"
            return 0
        fi
        log "  WARN: ray did not verify (try $try/3)"
        purge_ray
        sleep 10
    done
    log "  ERROR: ray would not start"
    return 1
}
