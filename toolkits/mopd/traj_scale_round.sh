#!/usr/bin/env bash
# Single-teacher OPD with the rollout scaled to ManiSkill's regime.
#
# What the previous rounds were missing was not samples but TRAJECTORIES. ManiSkill's
# working M1 collected 320 per round against our 96, and its batch was the whole pool
# (5120 = 320 x 16) -- it never subsampled either. Our episodes are ~4x longer, so we had
# comparable sample counts drawn from a third as many independent rollouts, and the
# transitions inside one episode are highly correlated.
#
#   total 320 of 327 valid   <- previous rounds: the batch WAS the pool, 98%
#   total 320 of 799 valid   <- with rollout_epoch 2: actually sampling, 40%
#
# rollout_epoch is the knob that buys trajectories without more parallel envs: it costs
# wall-clock, not CPU width. It needed one fix -- envs resample tasks between epochs, and
# task descriptions tokenise to different lengths, so stacking the stored prompts failed
# ("got [24, 825] at entry 0 and [24, 826] at entry 66"). Ragged prompt fields are now
# padded to the longest step, which is the same operation the eagle processor already
# performs within a batch, one token wider -- not the fixed-length padding that once cost
# this pipeline all of its success rate.
#
# micro_batch_size 32 matches ManiSkill's. Measured peak 26.8 GB of 49; at 8 it was 14.9,
# and training time per step fell to 61 s, so it is no longer the bottleneck.
#
# global_batch_size is the whole pool (~25k transitions from 768 trajectories), not a
# subsample. With one optimizer step per round -- required to stay on-policy, since the OPD
# loss has no clipping and its advantage uses the l_S recorded at rollout time -- the
# discarded half would have been exactly as on-policy as the half kept, so dropping it only
# widens the gradient variance. ManiSkill's working setup was likewise batch == pool.
#
# Sampling one denoise step per transition is a different matter and is kept: that is an
# unbiased Monte Carlo estimate of the expectation over tau, which both setups make.
#
# ENV COUNT IS CAPPED BY EGL CONTEXTS PER GPU, not by RAM or by GPU memory. Placement
# gives each of the 8 env workers its own card (visible_accelerators=['<rank>'],
# isolate_accelerator=True), and robosuite picks the EGL device from CUDA_VISIBLE_DEVICES,
# so total_num_envs/8 env subprocesses render on one card. Two measurements bound it:
#
#   57 concurrent envs come up on an idle card, the 58th dies with
#     mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd
#   40 concurrent envs come up with 27 GB already held on the card (the training peak at
#     micro_batch_size 32) -- the ceiling was not reached, so 40 is a verified-safe floor.
#
# Each env holds TWO contexts during reset churn, not one. RoboCasa envs report
# renderer == 'mjviewer', and robosuite's reset() only calls _destroy_sim() when
# renderer == 'mujoco', so a hard reset drops the old sim as a reference cycle and builds
# a new context beside it. The cyclic GC does reclaim it -- live contexts go 1 -> 2 and
# stay at 2 over 12 resets -- but the steady state is 2 per env process.
#
# That is what killed the 192-env round: 24 envs/GPU x 2 = 48 contexts, above the 40 that
# is verified safe under load, and it failed at exactly step 8 on every retry. 128 envs is
# 16/GPU x 2 = 32 contexts, under the verified floor, and rollout_epoch 6 holds the
# trajectory count at 768. The epoch knob itself is safe: 20 close()/create() cycles leak
# nothing, because close() -> sim.free() calls gc.collect() explicitly.
#
# 10 steps, deliberately short: this round asks whether scaling the rollout changes the
# TREND, not whether it converges. Every previous round showed success falling as the
# objective fell, and that was clearly visible inside 10 steps. Matching ManiSkill's 25
# updates is the follow-up if the trend reverses. Budget: rollout 1420 s + teacher 412 s +
# training ~120 s per step, about 32 min, so ~5.5 h per run.
set -o pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/local_env.sh" || exit 1
opd_prepare_run smoke "traj_scale_round.$(date +%Y%m%d-%H%M%S)" || exit 1
[[ ${OPD_DRY_RUN:-0} == 1 ]] && exit 0
OUT=$OPD_OUTPUTS_ROOT/train
LOG=$OPD_RUN_DIR/batch.log
STATE=$OPD_RUN_DIR/status
PROJECT=robocasa_opd_verify
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
: > "$LOG"

progress_watchdog() {
    local out=$1 limit=$2 last=-1 stalled=0
    while :; do
        sleep 300
        [ -f "$out" ] || continue
        local now; now=$(grep -c "Routed teacher velocities over" "$out" 2>/dev/null || echo 0)
        if [ "$now" -eq "$last" ]; then
            stalled=$((stalled + 300))
            if [ "$stalled" -ge "$limit" ]; then
                say "WATCHDOG: no new step in ${limit}s (stuck at $now); killing"
                pkill -9 -f "train_embodied_agent.py --config-name robocasa365_mopd" 2>/dev/null
                return 1
            fi
        else stalled=0; last=$now; fi
    done
}

run_one() {   # tag cfg lr
    local tag=$1 cfg=$2 lr=${3:-2.0e-5}
    local st out run_id
    for a in 1 2 3; do
        run_id=$cfg.$tag.$(date +%Y%m%d-%H%M%S).attempt$a
        st=$OUT/$run_id/status
        out=$OUT/$run_id/train.log
        say "RUN $tag (128 envs x 6 rollout epochs = 768 trajectories) attempt $a"
        progress_watchdog "$out" 7200 & local wd=$!
        OPD_RUN_ID=$run_id bash "$SCRIPT_DIR/run_mopd.sh" "$cfg" "$tag" \
            "runner.max_steps=10" "runner.val_check_interval=5" \
            "runner.save_interval=1000" \
            "runner.logger.project_name=$PROJECT" \
            "runner.logger.experiment_name=$tag" \
            "env.train.task_soup=[atomic_seen]" "env.train.task_filter=null" \
            "env.train.total_num_envs=128" "env.train.rollout_epoch=6" \
            "env.train.max_episode_steps=1056" \
            "env.train.max_steps_per_rollout_epoch=1056" \
            "env.eval.task_soup=[atomic_seen]" "env.eval.task_filter=null" \
            "env.eval.episode_horizon_source=task_horizon" \
            "env.eval.max_episode_steps=1056" \
            "env.eval.max_steps_per_rollout_epoch=1056" \
            "env.eval.total_num_envs=48" \
            "actor.global_batch_size=20480" "actor.micro_batch_size=32" \
            "actor.fsdp_config.gradient_checkpointing=True" \
            "actor.optim.lr=$lr" "algorithm.update_epoch=1" >> "$LOG" 2>&1
        kill -9 "$wd" 2>/dev/null
        local s; s=$(cat "$st" 2>/dev/null)
        say "RUN $tag -> $s"
        if [ "$s" = "OK" ]; then
            say "  eval success_once : $(grep -oE 'success_at_end=[0-9.]+ *│success_once=[0-9.]+' "$out" 2>/dev/null | grep -oE 'success_once=[0-9.]+' | sed 's/success_once=//' | tr '\n' ' ')"
            say "  velocity_cosine   : $(grep -oE 'velocity_cosine=[0-9.]+' "$out" 2>/dev/null | sed 's/velocity_cosine=//' | tr '\n' ' ')"
            say "  objective         : $(grep -oE '(velocity_mse|opd_reward)=[-0-9.e+]+' "$out" 2>/dev/null | tr '\n' ' ')"
            say "  train succ_once   : $(grep -oE 'success_once=[0-9.]+' "$out" 2>/dev/null | sed 's/success_once=//' | tr '\n' ' ')"
            say "  pool              : $(grep -oE 'total [0-9]+ of [0-9]+ valid' "$out" 2>/dev/null | head -1)"
            return 0
        fi
        # Each of the 8 ranks logs this line once per step, so divide.
        local lines; lines=$(grep -c 'Routed teacher velocities over' "$out" 2>/dev/null || echo 0)
        say "  retained attempt log: $out"
        say "  steps done: $((lines / 8))  err: $(grep -E 'Error|Traceback' "$out" 2>/dev/null | tail -1 | cut -c1-150)"
        sleep 120
    done
    return 1
}

# M4 first: M1 at 2e-5 is already measured (two 5-step replicates, below), so the
# open question is whether supervising the sampled transition behaves differently.
echo "M4" > "$STATE"; run_one t768_m4_atomic robocasa365_mopd_m4_gr00t 2.0e-5
# M1 at 5x lr is a falsification test, not an attempt to make M1 work. The standing
# objection to the negative result is that cosine only moved 0.952 -> 0.970, so nothing
# was really asked of the student. If success still falls monotonically at 1e-4, that
# objection is closed; if it reverses, the relation is not monotone after all. Note that
# ManiSkill's working M1 STARTED at cosine 0.994, so "reach 0.99" is not a target this
# evidence supports -- that setup never crossed this region.
echo "M1_LR1E4" > "$STATE"; run_one t768_m1_lr1e4_atomic robocasa365_mopd_gr00t 1.0e-4
echo "DONE" > "$STATE"
say "baseline for comparison: atomic_seen student 0.5079, teacher 0.854"
