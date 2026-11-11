# LIBERO: squared SR gap with synchronized rollout and query allocation

Implementation branch: `experiment/libero-gr00t-adaptive-query`, based on the
candidate snapshot `e5b21cc6`. This aligns the method with RoboTwin while keeping
LIBERO's tested numerical and interaction settings. It does not copy RoboTwin's
T32, B256 or50-round budget. The controller uses the agreed single10-round
timescale described below.

## Method

Compute each suite's student SR as the equal average of its ten task SRs over
the window of completed physical rounds. The fixed teacher calibration uses the
same task-macro aggregation. Define:

```text
gap[s] = max(teacher_sr[s] - student_window_sr[s], 0)
probability[s] = gap[s]**2 / sum(gap**2)
```

All-zero gaps use a uniform probability target. Project this target to integer
rollout quotas q using the existing minimum-coverage and maximum-change bounds.
Use the **actual projected q**, not the probability target, to allocate queries.

For the fixed T64/B512/four-rank setting, local batch size is128 and local suite
query counts are exactly `2*q[s]`. Global counts are `8*q[s]`, totaling512.
For example q=`4/28/20/12` gives local queries=`8/56/40/24` and global queries=
`32/224/160/96`. Generic budgets use the same largest-remainder rule as RoboTwin
and reject a per-rank allocation that would leave a suite without a query.

No extra suite loss coefficients are applied. The objective is the mean of all
real queries, retaining the original Method3 time weights and per-query executed-
prefix normalization. Native text buckets retain their padding masks and scale
loss by genuine rows/local batch. Real replacement draws remain genuine queries.

The runner freezes q before the first teacher query. First sampling and every
flow refresh use that same mapping. The actor rejects a missing mapping or a
mapping changed after initial supervision. Refresh draws new observations from
the full current rollout pool, regenerates current-student flow with new noise,
and queries the matching teacher at the resulting flow state.

## LIBERO training and controller settings

| Setting | Value |
| --- | --- |
| Model/checkpoint interfaces | Existing student grouped11D state and own metadata; four fixed PPO teachers and their own transforms |
| Execution prefixes / physical horizons | 5/5/5/10;240/240/320/480 |
| Flow | Full16×32 latent, four ODE steps, existing student-coordinate Method3/noise0.3 |
| Deployment | One node/four actor/env/rollout ranks,16env slots/rank |
| T / B / microbatch | 64 /512 /2per rank |
| Optimizer | LR3e-5, AdamW(.9,.95), decay0, clip1 |
| LR warmup / scheduler | Four optimizer updates, then constant |
| Allocator warmup/window/interval | 10/10/10physical rounds |
| Coverage / change cap | One episode/task validity check; six trajectories/suite/decision |
| Main budget / saves | 100physical rounds; save every10rounds |
| W&B | `FORMAL_libero_gr00t`; default group `T64_gap_query_W10_I10_seed0` |

Use one physical-round timescale: warmup=window=interval=10. First adaptive
decision is for **round11**, then21/31/41/...; each window uses ten new completed
rounds. This intentionally replaces the initially inherited20/20/10controller.
M3 time weighting and the four-optimizer-update LR warmup are unchanged.

With four workers and one rollout epoch, minimum quota is4/suite. A ten-round
window therefore contains at least40episodes/suite, or four per task under
continuous round-robin scheduling. Set min_task_episodes=1as a validity check:
every task must have an SR denominator. It does not truncate the records used,
reduce the actual number of episodes, or add a tunable statistical confidence gate.
Unexpected missing-task coverage still holds the quota and records a diagnostic.

Retain max_quota_change=6as the fixed bound used by the RoboTwin runs. It is not
mathematically required by the gap law, but prevents abrupt jumps in physical and
query resource shares. It remains a disclosed controller setting rather than a
separate sweep. No new EMA, mixing coefficient or demand proxy is introduced.

## Matrix and entrypoint

| Arm | Rollout / query allocation | R | Updates | Save at updates |
| --- | --- | ---: | ---: | --- |
| M0 | Uniform / uniform | 1 | 100 | 10/20/.../100 |
| M1 | Uniform / uniform | 2 | 200 | 20/40/.../200 |
| M2 | Adaptive / synchronized | 2 | 200 | 20/40/.../200 |
| M3 | Adaptive / synchronized | 1 | 100 | 10/20/.../100 |

New YAMLs are `examples/embodiment/config/libero_mopd_adaptive_query_m0.yaml`
through `m3.yaml`, sharing `libero_mopd_adaptive_query_common.yaml`.
The historical formal YAMLs and `launch_formal.py` retain their old recipes.
The new entrypoint defaults to T64 and rejects T32 or learning-parameter CLI
overrides. Cache/GPU resource flags continue through the existing owned launcher.

After [runtime preparation](../../LIBERO_ENVIRONMENT.md), set model/artifact
paths and create a plan (this command does not start training):

```bash
export OPD_WANDB_GROUP=T64_gap_query_W10_I10_seed0
export OPD_WANDB_RUN_NAME=M2_Adaptive-R2_T64_B512_seed0
export OPD_LIBERO_TEACHER_REFERENCE=/absolute/path/to/verified_teacher_reference.json
export WANDB_MODE=offline
"$LIBERO_PYTHON" -m toolkits.libero_gr00t.launch_adaptive_query \
  --arm m2 --seed 0 --python "$LIBERO_PYTHON" \
  --models "$LIBERO_MODELS" --student "$LIBERO_STUDENT" \
  --spatial-teacher "$LIBERO_SPATIAL" --gpus 0 1 2 3 --cpus 32 \
  --actor-cache-retention-gib 20 --actor-cache-preallocate-gib 12 \
  --skip-finished-rollout --log-cuda-memory \
  --output "$LIBERO_RESULTS/M2_gap_query_plan"
```

Use `m0/m1/m2/m3` to select the arm, with separate output paths/run names.
All arms require an explicitly supplied, identical accepted student. This change
does not select a new initializer or accept the pending teacher-rollout SFT100
candidate. Initialization acceptance and content-verified reference relocation
remain separate gates listed in [the experiment contract](../../LIBERO_EXPERIMENTS.md).
Changing the allocator law does not require recollecting teacher calibration
when teacher weights and the observation/action/execution signature are unchanged.

GPU acceptance must precede real `--run` launches. No GPU experiment is started
by this development change. Independent evaluation still uses the shared rollout-
worker evaluator, fixed40tasks×50states and the same K/horizons. The10-round save
and evaluation grid remains unchanged. Checkpoints from the old weighting method
must not be resumed as the new method; use the same base initialization and a
fresh optimizer. New-method continuation retains strict same-recipe validation.

## Diagnostics and acceptance

- `train/actor/query_draws/<suite>`: actual global query count per update.
- `train/actor/query_distinct_source_rows/<suite>`: distinct local source indices,
  summed across ranks; this does not deduplicate semantically identical images.
- Existing pool sizes, replacement fractions and per-suite raw velocity MSE.
- `train/actor/macro_velocity_mse` and `macro_analytic_kl`: suite-equal metrics,
  alongside the query-composition-weighted aggregate. Missing suites withhold macro.
- `train/new_rollout_macro_velocity_mse`: first update only; retain existing
  per-suite first-update metrics and per-inner-update records.
- Existing physical-round, trajectory, executed-step and allocator-state records.

Counts use the existing rank-AVG metric collective followed by multiplication
by world size to recover global sums; no extra sampling-time collective is added.
CPU regressions exercise the actual first-teacher/refresh path with unequal
query shares, native bucket padding and refreshed student weights/noise; compare
the prefix-normalized gradient to an independent per-query objective; replay
40tasks at the preserved controller settings; and compare all four resolved
recipes against the historical T64 controls. GPU/FSDP/model acceptance remains
pending on an allocated node.

## Experiment planning revision on2026-11-11

The agreed save/independent-evaluation grid is now every10physical rounds through
round100, plus a shared round0 evaluation. Keep100rounds and the existing2000-
episode checkpoint protocol. This is41checkpoint evaluations for the first seed
(82,000episodes); three training seeds with one shared round0 need242,000episodes.
Evaluation is scheduled separately and does not run inside the training loop.
The initial development receipts below preserve the earlier25-round grid; current
YAML/launcher tests cover the new10-round grid.

Plan seed0 for all four arms first, then repeat with seeds1/2, using the same
initialization and evaluation state/episode-RNG protocol. Set `--seed` and the
W&B group/run name explicitly for each repeat. Three training seeds
allow reporting seed-level mean/standard deviation; a single seed is exploratory.

The following are recommendations, **not active configuration changes**:

- Try microbatch4 in a bounded GPU timing/memory/gradient acceptance check while
  keeping B512. Select one microbatch for all four arms before formal comparison.
  Native token-length buckets and floating-point accumulation mean mathematical
  equivalence does not imply bitwise-identical training or a guaranteed speedup.
- Keep Method3 noise_level0.3. It defines the reference local-SDE KL weights;
  ODE rollout and refresh remain deterministic conditional on the sampled initial
  noise. Existing T64 run configurations also used0.3.

## Development validation on2026-11-11

411CPU regressions passed, including21new adaptive-query tests. Changed Python
files passed Ruff lint/format; local documentation links and shell syntax passed.
A separate100-round/6400-record synthetic replay produced identical rollout plans
and local query quotas to the RoboTwin donor implementation on the same inputs.
The replay used LIBERO's preserved T64/20-round-window settings in both allocators;
it did not run physics or load a model. The donor source hashes and four resolved
configuration-only recipes are retained in the shared-root artifact directory
`outputs/libero_adaptive_query_development_20261111/`.

The saved `*_CONFIG_ONLY.yaml` files intentionally contain placeholder model,
reference and runtime paths. They are inspection artifacts, not executable runs.
No GPU smoke, policy evaluation or training was launched. Target-machine and
initializer acceptance remain pending.

The measurement-grid revision passed27targeted CPU regressions (the21new-method
checks and six historical T64 checks), plus Ruff lint/format and doc/shell checks.
Loss, microbatch and controller timing are unchanged. No GPU job was launched.

## Controller simplification

The user selected window10and interval10to reduce independent tuning choices.
The active YAML now also ties allocator warmup to10, sets the task-coverage
validity check to1, and keeps the donor's max-change6bound. Historical formal
YAMLs and their20/20/10/gate5settings remain untouched. Synthetic checks establish
quota/coverage/timing correctness; they do not establish stable policy gains.
Microbatch remains2pending the separate GPU speed/memory check.

The single-timescale revision passed94targeted CPU regressions. Coverage gates
1/2/3/4produced identical plans on the fixed40-task layout, confirming that the
active gate1is a validity check rather than an extra allocation preference.
Uniform and Adaptive plans match over the first ten physical rounds; the first
adaptive decision is for round11. No GPU job or model experiment was launched.
