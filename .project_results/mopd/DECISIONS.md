# Routed MOPD — decision log

Autonomous stretch starting 2026-08-18 (user resting). Each entry: the decision, why, and
what would change if it turns out wrong. Newest at the bottom.

**Kept because the *why* behind the config is hard to reconstruct from the code.** Entries
overtaken by later work are marked in place rather than removed — D8 was reversed outright.

Standing user decisions (given before the stretch):
- GR00T N1.5 + RoboCasa365, on this box.
- Method 1 supervision (velocity MSE) + direct task-suite -> teacher routing.
- Teachers: group the rollout batch by suite, one teacher on GPU at a time.
- Pilot scale: all three suites, a few tasks each.
- Shared GPU: progress over cross-arm comparability; shrink to fit is allowed.
- Autonomy: push through to real success-rate numbers, with watchdogs and resume.
- Box hazards named by the user: stale Ray head must be purged and *verified* repeatedly;
  A6000s have no NVLink, so NCCL_P2P_DISABLE=1 (and NCCL_IB_DISABLE=1, no InfiniBand).

---

## D1. Branch off `opd-mechanism-study`, not off the upstream base

`opd-mopd-routed` starts at d04795ec. The only correct Method-1 implementation is
post-`4ab7fafe` (flow-timestep pairing fix); `opd-velocity-distill` predates it and carries
the bug. Local `main` additionally carries the torch 2.11 migration (#1410), which would
disturb the working venv for no benefit. M3/M4 code rides along but is inert unless its
`loss_type` is selected.

## D2. GR00T is forced — the pi0 shortcut does not exist

`robocasa/robocasa365_checkpoints` has per-suite specialists **only** under
`gr00t_n1-5/foundation_model_learning/target_posttraining/`. Its `pi0` and `pi05` entries are
multitask_learning only. So reusing the already-working openpi venv and its existing
`nft_forward` was not an option.

## D3. Skip optimizer states in the checkpoint download

`optimizer.pt` is 8.55GB of each 16.14GB checkpoint and is only needed to resume that
training run. We load weights for init and for frozen teachers. 64.5GB -> 25GB downloaded.

## D4. Expose GR00T's velocity by splitting `sample_mean_var_val`, not by duplicating it

`FlowMatchingActionHeadForRLActionPrediction.sample_mean_var_val` computed `v_t` internally
and returned only mean/std. Rather than write a parallel velocity path (which could drift
from the sampler), the velocity block was extracted into `get_velocity()` and the sampler now
calls it. Teacher and student are therefore scored by construction on the same computation.
`ForwardType.NFT` -> `nft_forward` mirrors the OpenPI entry point the M1 actor code expects.

*If wrong:* nothing silently — a divergence would show up as the teacher==student smoke
failing to give velocity_mse ~ 0.

## D5. Give each RoboCasa365 task its originating soup label

`rlinf/envs/robocasa365/utils.py` flattened `get_ds_soup()` results across soups and, when
more than one soup was configured, labelled every task with the joined string
`"atomic_seen+composite_seen+composite_unseen"`. Routing needs a per-task label, so the
registry now remembers which soup produced each task (`task_soup_of`), used as the fallback
before the joined string. Single-soup behaviour is unchanged. A task listed in two soups
keeps the first, matching the existing dedupe.

## D6. Teachers stay resident in host RAM; only the GPU copy is exclusive

The user's instruction was one teacher on the GPU at a time. Taken literally to "one
teacher in memory at a time" it would mean re-reading ~8GB of weights from the shared
filesystem three times per training step, which would dominate step time. The box has
944GB of RAM free and each teacher is ~7.6GB, so three cached teachers cost ~23GB per
actor rank. Teachers are therefore constructed once, cached on CPU, and moved to the GPU
one at a time -- peak GPU memory matches the single-teacher case, which is what the
instruction was protecting.

*If wrong:* host RAM pressure with many ranks. Falls back to evicting the cache after each
group, at the cost of reload time.

## D7. `split: target`, not `pretrain`

`composite_unseen` has no `pretrain` datasets -- `get_ds_soup(..., split="pretrain")`
returns 0 tasks -- and RLinf's env builds its task list from that call, so a `pretrain`
config would silently train on two suites while claiming three. All three suites resolve
under `split: target` (18/16/16 = the 50-task leaderboard set). The MOPD setting doc has
been corrected.

## D8. ~~GR00T's rollout uses one stochastic denoise step (`joint_logprob: False`)~~ **REVERSED 2026-08-24**

The original decision reasoned that GR00T N1.5's training path had no deterministic mode —
`get_rl_action` picks `mode="train"` for the steps named by `denoise_inds`, and every
available `noise_method` (flow_sde / flow_cps / reinflow) adds noise — so `joint_logprob:
False` was chosen as the minimum-noise rollout available.

**That described the code correctly and drew the wrong conclusion.** It gave M1 M3-Hybrid's
stochastic rollout. GR00T's `mode="eval"` branch *is* OpenPI's `flow_ode` (both reduce to
`x + v·delta`); it was simply unreachable during training. A `flow_ode` branch was added, and
the config is now `flow_ode` + `joint_logprob: True` — a fully deterministic ODE rollout,
matching the mechanism study's M1.

Note `joint_logprob: True` means the opposite of what D8 assumed under a stochastic
`noise_method`: it makes every step take the `mode="train"` branch, which under `flow_ode`
is deterministic. Detail: [METHOD_SUPPORT_GR00T.md](METHOD_SUPPORT_GR00T.md).

## D9. Pilot tasks are the shortest two per suite

Task horizons span 450 (atomic) to 4350 (composite) steps, and RoboCasa is a CPU
simulator. The pilot uses TurnOnMicrowave / CloseToasterOvenDoor (450 each),
ScrubCuttingBoard / RinseSinkBasin (1200 / 1350) and CuttingToolSelection /
CategorizeCondiments (1200 / 1650). Training truncates episodes at 300 steps -- velocity
distillation needs on-policy *states*, not completions -- while evaluation uses each task's
own horizon, since a composite task truncated at 300 can never succeed.

*Updated 2026-08-24:* training truncates at 320, not 300 (a multiple of the 16-step action
chunk). In-training validation now uses `episode_horizon_source: max_episode_steps` at 1664 —
long enough for every pilot task and weighting the six equally; the standalone eval config
uses `task_horizon`, which weights short tasks more heavily. The two are not comparable.
See [RUNBOOK.md](RUNBOOK.md) §3.

*Known bias:* truncated training rollouts only cover early-episode states, and the
success signal in this pilot will come almost entirely from the two atomic tasks.

## D10. Full-horizon training rollouts, with a suite-balanced sample for the batch

Training truncated every episode at 320 steps, which covers 71% of an atomic trajectory but
only 19-27% of a composite one — so the student was distilled on the first quarter of what it
is evaluated on. D9 justified the truncation as "velocity distillation needs on-policy states,
not completions", which is true and does not imply a quarter of them is enough.

Episodes now run their own horizon (`episode_horizon_source: task_horizon`) and the training
batch is a fixed-size sample drawn from what the rollout produced. At a fixed env-step budget
the sample count is unchanged — samples = env-steps / 16 whatever the distribution — so what
is traded is episode *diversity* for state coverage, not compute for coverage.

*If wrong:* fewer distinct episodes per epoch means samples inside one episode are temporally
correlated and each epoch sees fewer initial states. Watch the gradient noise. Falls back by
setting `episode_horizon_source: max_episode_steps` and turning the sampler off.

## D11. Balance the suites by quota, not by loss weight — for now

Routing sends each sample to its suite's teacher, so the sampling distribution sets each
teacher's share of the gradient. Sampling naturally weights suites by horizon: at the full
50-task set, atomic would carry 13% of the gradient rather than 33%.

Two mechanisms reach `P(suite) = 1/3` and both issue the same number of teacher forwards.
Importance weighting covers ~22% more distinct states for that compute; the per-suite quota is
simpler and its correctness holds by construction, since the drawn counts *are* the balance and
the existing routing log shows them.

Chose the quota. The weighting variant's failure mode is a weight tensor desynchronised from
its samples through the micro-batch split — a silently mis-weighted gradient that looks normal
in every metric, which is the failure mode this work has repeatedly hit.

*If wrong:* the cost is distinct-state coverage, not correctness. Importance weighting is
recorded as an upgrade to make once there is a baseline to verify it against — and it must not
change the result.

## D12. `task_sampling_strategy: ordered` for training

`random` draws each env's task i.i.d. over the union of the suites, so one epoch is
multinomial: the per-suite mix swings ~18% at pilot scale and, at the full 50-task set, 27% of
tasks get no env at all in a given epoch. `ordered` is a deterministic round-robin with a
cursor that persists across epochs and is offset per worker.

This stratifies rather than removes randomness: which env runs which task becomes
deterministic, while *which instance* of that task — scene, object placement, initial state —
stays random, since training keeps `use_fixed_reset_state_ids: False`.

It also carries a constraint that is easy to miss: the suite-balanced quota is drawn per rank,
and `ordered` gives each env worker `total_num_envs / num_env_workers` consecutive tasks. That
count must be a multiple of the task-filter length, or some ranks hold no samples for some
suite. See D10's config: 48 envs, not 24.

## D13. Full-task multi-teacher M4 runs with natural weighting, not suite balance

The 2026-08-28 full-50-task multi-teacher M4 run sets `suite_balanced_sampling: False`. That
is ablation A1 in the sampling spec, not the main setting, and it must be stated whenever the
run is reported: the suites are weighted by horizon, roughly 13% / 41% / 46%.

The reason is a wall-clock limit, not a preference. Suite balance is only *correct* when every
rank holds every suite, because DDP averages the ranks; with the 50-task list, block-ordered
by soup, and `ordered` sampling, that needs 35 consecutive tasks per worker — 280 envs, ~810
GB, 573k env-steps per step. `random` sampling does not fix it either. See
[WEIGHTING.md](WEIGHTING.md) §9.

*If wrong:* the atomic teacher contributes an eighth of the gradient rather than a third, so a
null result on atomic is weaker evidence than it looks. Rerun with the §9 fix once the shared
repo is free.

## D14. Evaluation may not truncate; training may

`validate_robocasa365_eval_horizons` refuses an eval rollout shorter than the longest resolved
horizon, and it is right to: truncating evaluation drops the long tasks' failures and inflates
`success_once`. The first attempt at the full-task run set both to 2048 and was stopped by that
check.

Training keeps 2048 while evaluation uses 4352. Training truncation costs state coverage on the
22 of 50 tasks longer than 2048 — a real cost, taken because a 4352-step training epoch is
twice the wall clock and, with `auto_reset: False`, spends most of it on envs that already
finished.

*If wrong:* the long tasks are under-trained relative to how they are scored. Raise the
training epoch when there is time for it.
