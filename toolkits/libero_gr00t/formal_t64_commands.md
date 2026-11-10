# Historical T64 LIBERO experiments

This page records the older Offline-Distilled-50 runs and node-specific commands.
For the pending teacher-rollout SFT initialization and portable installation,
start at [the experiment contract](../../LIBERO_EXPERIMENTS.md).

T64 is the selected primary setting. Use `experiment/libero-gr00t-rollout-batching`
on action10 GPUs0–3. The stopped action00 run remains historical, with its source,
31completed rounds and round25 checkpoint preserved separately.

## Matrix and comparison

| Arm | Rollout allocation / suite loss | Refresh R | Rounds | Updates | Save at updates |
| --- | --- | ---: | ---: | ---: | --- |
| M0 | Uniform / equal | 1 | 100 | 100 | 25/50/75/100 |
| M1 | Uniform / equal | 2 | 100 | 200 | 50/100/150/200 |
| M2 | Adaptive / synchronized | 2 | 100 | 200 | 50/100/150/200 |
| M3 | Adaptive / synchronized | 1 | 100 | 100 | 25/50/75/100 |

M0→M1 isolates rollout reuse; M1→M2 measures adaptive allocation with synchronized
suite weighting; M0→M3 tests that contribution without reuse; M3→M2 measures reuse
under adaptive allocation. Allocation and synchronized loss weighting form one
contribution here. Their individual effects require a separate mechanism ablation.
Cached-R is not part of this main matrix.

Every arm collects6400student trajectories. B and LR stay fixed across R;
optimizer compute deliberately differs. Use rollout rounds as the primary x-axis
(each round64trajectories), and also retain trajectories, real simulator calls,
optimizer updates and wall time. Equal trajectory counts do not imply equal
simulation steps because suite horizons and policy episode lengths differ.

## Fixed recipe

- Accepted Offline-Distilled-50 initialization; its61%screen is not the final
  round0score. All arms start fresh from this same checkpoint and metadata.
- RLinf Spatial-Step400, Object-Step400, Goal-Step500 and Long-Step300 teachers.
  Keep their own normalization/preprocessing and BF16 rotary/head contract.
- Four actor/rollout/env ranks,16parallel environments/rank, one epoch/round.
  Uniform quotas16/16/16/16; rotate through all ten tasks of each suite.
- B512, microbatch2/rank, LR3e-5, four optimizer-update warmup steps, constant
  scheduler, AdamW(.9,.95), decay0, clip1. Seed0first; seeds1/2are planned repeats.
- Original M3, four ODE steps, full16×32latent; execute5/5/5/10actions with
  native physical horizons240/240/320/480 and10reset settling steps.
- Native B1 rollout inference for every live slot. All arms enable
  `--skip-finished-rollout`, replaying the discarded noise draws for exact
  live-trajectory/RNG equivalence. Experimental BF16 model batching stays off.
- Adaptive: rho0.5, epsilon0.01, warmup/window20rounds, interval10rounds,
  minimum5episodes/task, quota-change cap6/suite/decision. Use the verified
 800-episode calibration, rebound to this execution signature without new physics.
  Suite loss coefficients are4times realized quota fractions; update batches
  remain suite balanced. Learning-progress signals remain diagnostic only.

Action10 has shared GPUs. All arms use the same12GiB actor cache floor and20GiB
retention bound, rather than the old action00floor26/bound28. This changes no
numerical hyperparameter or RNG draw. It reserves reusable allocator space, not
exclusive access. Preflight requires30GiB free/card. Report shared-GPU wall time
as observed runtime, not a dedicated-hardware throughput claim.

## Launch

```bash
OPD_ROOT=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try
OPD_REPO="$OPD_ROOT/outputs/worktrees/libero_gr00t_rollout_batching"
OPD_PYTHON="$OPD_REPO/outputs/runtime_libero_profile_action10_20261105_r2/bin/python"
export OPD_LIBERO_TEACHER_REFERENCE="$OPD_ROOT/outputs/libero_gr00t_formal_T64_20261108/official_teacher_reference.json"
export OPD_WANDB_GROUP=T64_offline50_handoff_seed0
export OPD_WANDB_RUN_NAME=M1_Uniform-R2_T64_B512_seed0
export WANDB_MODE=online
cd "$OPD_REPO"
"$OPD_PYTHON" -m toolkits.libero_gr00t.launch_formal \
  --arm m1 --trajectory-budget 64 --seed 0 --python "$OPD_PYTHON" \
  --models "$OPD_ROOT/outputs/libero_gr00t_20261104/models" \
  --student "$OPD_ROOT/outputs/libero_gr00t_20261104/offline_distillation_20261107/training/checkpoint-000050/hf" \
  --spatial-teacher "$OPD_ROOT/outputs/libero_feasibility_20261104/models/groot_spatial_rl" \
  --gpus 0 1 2 3 --cpus 32 --timeout-seconds 172800 \
  --actor-cache-retention-gib 20 --actor-cache-preallocate-gib 12 \
  --skip-finished-rollout --log-cuda-memory \
  --output "$OPD_ROOT/outputs/libero_gr00t_formal_T64_20261108/M1_Uniform-R2_seed0" --run
```

W&B project: `FORMAL_libero_gr00t`. Use distinct persistent run IDs. The default
is a dry run without `--run`. Do not duplicate a queued launch.

Execution order: one-round/two-update integration acceptance (offline W&B,
excluded from formal results), M1, M2, shared initial/final M1/final M2 evaluation,
M0, M3, then remaining saved-checkpoint evaluations. Any failure stops the queue;
there is no automatic hyperparameter change or result-dependent early stopping.

## Independent evaluation and retained results

Evaluate round0once and every arm at rounds25/50/75/100:17checkpoints total,
2000episodes/checkpoint (all40tasks ×50official states),34000evaluation episodes.
These evaluation interactions are recorded separately from student training and
shared initialization/calibration costs.

Use the [shared rollout-worker evaluator](parallel_evaluation.md),5parallel
states/task/GPU, episode-private seed1234protocol, same K/horizons and strict
full-weight loading. Raw2000-pair coverage must pass before publishing a score.
The first paired8-episode engineering check matched264actions bitwise; it does
not replace full evaluation.

Primary plot: independent suite-macro success vs rollout rounds, with per-suite
curves and seed uncertainty after repetition. Main table: each suite and macro
success at round100, plus training trajectories/simulator steps, optimizer
updates and observed wall time. Preserve raw episode results, source/runtime
identities, resolved configs, training JSONL and W&B URLs. The result table starts
as a template; fill only audited results, never training rollout success.

Artifacts and queue status: `outputs/libero_gr00t_formal_T64_20261108/` in the
shared project root. Prior T32/T64 runs remain separate historical experiments.

After a second-round rollout-reload OOM in the first attempt, unused actor cache
is explicitly released to its12GiBfloor before the synchronization handle
returns. Numerical parameters remain frozen. The failed two-update M1 attempt
is preserved; the replacement M1 uses a new artifact directory/run ID.
