# MOPD development guide

`opd-project-main` contains Base MOPD, full-pool flow refresh, adaptive rollout
allocation, synchronized suite loss weighting, per-suite diagnostics, and GR00T
memory optimizations. The [experiment guide](docs/opd/collaboration/README.md)
defines the current formal matrix. The pilot recipes below document development
history and must not be substituted for that matrix.

Project-facing documentation is maintained in English. Personal node profiles,
email task assignments, and local research notes are not runtime dependencies.

## Integration history

- The 2026-10-19 integration included nine commits through `a3761fd5` from
  `dev/mopd-groot-robocasa365`: FP32 action-head master weights, checkpoint-specific
  normalization, balanced suite sampling, completed-episode freezing, LR
  schedules, and runtime path handling.
- Evaluation commit `628a5ae3` is patch-equivalent to event-weighting commit
  `4f0a5017`. See the [evaluation protocol](evaluations/robocasa365/FORMAL_EVAL.md).
- Refresh was consolidated from `dev/groot-robocasa365-decoupled-refresh` and
  checked against the diagnostic worktree's full-pool refresh/teacher-query path.
  Timing metrics and W&B grouping were retained.
- The 2026-10-31 integration added adaptive allocation, rollout-boundary resume,
  suite metrics/weights, and memory improvements from
  `dev/mopd-adaptive-rollout-allocation`.
- Event weighting, Cached, fixed-observation probes, and SOPD-specific recipes
  remain in historical branches/worktrees. They are not silently mapped to
  Refresh: `flow_reuse_mode=cached`, `flow_observation_sampling=fixed_batch`, and
  enabled `flow_probe` are rejected on the mainline.

## Historical controlled recipes

| Config under `examples/embodiment/config/` | T | B | R | Updates | Save interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| `robocasa365_mopd_base_t96_controlled_gr00t_full` | 96 | 2048 | 1 | 200 | 40 |
| `robocasa365_mopd_refresh_r2_t96_gr00t_full` | 96 | 2048 | 2 | 200 | 40 |

These differ only in `algorithm.flow_refresh_updates`. Both use an explicitly
provided C10, three specialists, LR3e-5, constant scheduling, warmup10 student
updates, micro32, eight actor ranks, four-step ODE, noise_level0.5, and native
horizons. T96 is48 environments x2 epochs, with32 trajectory slots per suite.
For exactly768 samples per suite, set B2304 in both arms. Keep T/B/LR aligned.
Total updates and enabled save/evaluation intervals must be divisible by R.
The historical controlled comparison fixed update counts; the formal matrix
instead fixes physical rounds and scales total updates with R.

Other retained recipes:

- `robocasa365_mopd_gr00t_full`: original T96/B8192/warmup3, 100 rounds, save25.
- `robocasa365_mopd_refresh_r2_gr00t_full` and
  `robocasa365_mopd_base_controlled_gr00t_full`: T48/B2048/warmup10, 200 updates,
  save20; project `Formal_Robocasa_Groot_MOPD`.
- `robocasa365_mopd_refresh_gr00t_full`: early R4/T96/B8192/warmup3;
  50 physical rounds =200 updates, saved every10 physical rounds. It retains
  historical step semantics; prefer optimizer-update recipes for new work.

## Refresh semantics

1. Collect physical trajectories with the current rollout policy and retain
   every valid observation in the round's pool.
2. The first update uses rollout-generated flow states and teacher targets.
3. Each of the remaining R-1 updates samples a new suite-balanced batch from
   the full pool, generates ODE flows with the updated student and fresh noise,
   samples flow time steps, and queries the corresponding frozen teacher.
4. Each inner update performs one optimizer step. After R updates, synchronize
   rollout weights and collect the next physical round.

Repeated observations remain distinct samples with independent noise/flow draws.
Masks exclude padding and completed-episode chunks. Refresh runs without
computed gradients through the regular FSDP `__call__` path. Controlled refresh
isolates flow RNG consumption from the training RNG.

## Counters, checkpoints, and continuation

| Field or file | Meaning |
| --- | --- |
| `runner.step_unit=optimizer_update` | Loss logging, warmup, max_steps, and save intervals count student updates |
| `optimizer_updates.jsonl` | One record per optimizer update |
| `train/new_rollout_velocity_mse` | Loss on the first update of each physical round |
| `rollout_count`, `rollout_success/*` | Physical-round success diagnostics |
| `time/step`, `time/rollout_round` | Physical rollout plus R updates |
| `time/amortized_update` | Round time divided by R |
| `runner_state.json` | Saved update, rollout, and environment-step counters |

Optimizer-update recipes default to fresh runs. Explicit continuation can set
`runner.resume_dir=<global_step_N>` and
`+runner.resume_mode=adaptive_round_boundary`. A complete round boundary restores
model, Adam, scheduler, sampling counters, allocation history, and task cursors.
R, controller settings, reference, and worker layout must match; verify the
original training configuration as well. This requires the synchronous runner.
Proposal RNG restarts, so continuation is not bitwise replay.

`max_steps` is the cumulative update target. Record discarded interactions under
`+runner.recovery_interactions` (`rollout_rounds`, `trajectories`,
`simulation_steps`); they count toward total cost, not the trained policy's
interaction lineage. Local checkpoints are needed for evaluation/continuation;
the collaborator's result delivery excludes weights and optimizer states.

## Run on a configured node

Install the [fixed runtime](OPD_ENVIRONMENT.md) first. The formal matrix has a
portable launcher in the [experiment guide](docs/opd/collaboration/README.md).
For direct historical-recipe use, prepare the environment before starting Ray:

```bash
export REPO_PATH="$PWD"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export OPD_MODEL_ROOT=/absolute/path/to/foundation_model_learning
export OPD_STUDENT_CHECKPOINT=/absolute/path/to/C10/checkpoint-60000
export OPD_RUN_DIR=/absolute/path/to/new-run-directory
export MUJOCO_GL=egl
export RLINF_NODE_RANK=0

python examples/embodiment/train_embodied_agent.py \
  --config-name robocasa365_mopd_refresh_r2_t96_gr00t_full \
  'hydra.run.dir=${oc.env:OPD_RUN_DIR}' \
  runner.logger.experiment_name=refresh-r2-pilot \
  +runner.logger.wandb_group=mopd-refresh
```

Use the corresponding Base config for R1. The historical default project is
`Formal_Robocasa_Groot_MOPD`; `runner.logger.project_name=...` changes it when
using this direct Hydra entrypoint. Authenticate locally; do not store tokens
in configs. Allocate only the intended node/GPUs to Ray. Private launchers under
`toolkits/mopd/` are not present in a fresh clone.

## Adaptive allocation

The initial pilot `robocasa365_mopd_adaptive_r2_gr00t_full` used T96/B2304/R2,
200 updates =100 rounds =9,600 trajectories, LR3e-5, constant scheduling,
warmup10 updates, save40, and W&B project `RoboCasa_MultiTeacher_Diagnostics`.
T denotes episode slots per round, not a fixed simulation-step budget.

After20 uniform rounds, make a decision every10 rounds using the previous20
rounds of **task-macro** student success. Require at least5 episodes per task;
otherwise retain the previous quotas. For suite s:

```text
gap_s = max(teacher_success_s - student_success_s, 0)
p_s = (1-rho)/3 + rho*(gap_s+epsilon)/sum_j(gap_j+epsilon)
rho = 0.5; epsilon = 0.01
```

Project desired shares to integer quotas summing to96, with per-decision
changes of at most8 per suite. Eight workers x2 epochs require at least16
trajectories per suite to keep every suite in every local pool. Within a suite,
tasks rotate round-robin. Task occurrence determines reset seeds, aligning
shared prefixes of each task's initial-state sequence across allocations.
Action-noise identity across different worker layouts is not guaranteed.

The original pilot changed physical experience only: B768/768/768, the M3 loss,
hard routing, and flow refresh stayed fixed. Signed learning progress was
logged but did not control allocation. Synchronous allocation disables bootstrap
overlap and uses one rollout stage, group_size1, and native horizons. Uniform
controls use the same collector with `runner.rollout_allocation.mode=uniform`.

### Shared teacher reference

Formal runs use the verified portable reference in the
[resource guide](docs/opd/collaboration/RESOURCES.md). For future recalibration,
the development command is:

```bash
python evaluations/robocasa365/calibrate_teachers.py \
  --output /absolute/path/to/new-calibration-directory \
  --gpus 0,1,2,3,4,5,6,7 --episodes 20 --parallel-envs 5
```

Calibration and training run sequentially. The reference must be COMPLETE and
match task membership, horizons, checkpoint paths, and effective inference
semantics. Its reset-seed sequence is independent of training. Report calibration
cost separately and include it in total method cost. These episodes do not train
the student or replace independent evaluation; the reference has sampling error.

### Interaction accounting

| Field or file | Meaning |
| --- | --- |
| `update_step` | Student optimizer update |
| `interaction/rollout_rounds` | Completed physical rounds |
| `interaction/rollout_trajectories` | Completed episodes;96 per round in this recipe |
| `interaction/simulation_steps`, `interaction/control_steps`, `interaction/env_steps` | Executed control steps, excluding terminal padding, reset settling, and internal MuJoCo substeps |
| `interaction/total_simulation_steps`, `interaction/total_trajectories` | Training plus applicable calibration and recovery costs |
| `calibration/*`, `teacher_reference.json` | Calibration budget, wall time, and reference snapshot |
| `rollout_episodes.jsonl` | Task, suite, round, worker/slot, seed, policy version, success, executed steps |
| `rollout_allocation.jsonl` | Round quotas, decision evidence, and budgets |
| `rollout_allocation_state.json` | Checkpointed history, task cursors, and counters |

Executed control steps must match the actor's valid-control-mask total each
round. Flow refresh adds no physical interactions. Equal T and rounds do not
imply equal simulation steps because allocations change episode lengths.
Preserve rounds, steps, updates, and time for comparison. Training-window success
feeds the controller and cannot replace independent evaluation.

## Suite grouping and losses

Full MOPD recipes enable:

```yaml
algorithm:
  suite_batch_order: grouped
  log_suite_velocity_metrics: true
```

Stable sorting by teacher ID reorders already sampled and labeled rows without
resampling or extra teacher queries. It retains duplicate rows, within-suite
order, microbatch size, accumulation scaling, final gradient clipping, and one
optimizer step. With B2304/eight GPUs/micro32, each rank has96 rows per suite and
3+3+3 forward/backward microbatches. This implements the same equal-weight
objective as mixed ordering; dropout assignment and floating-point ordering can
still differ. Nondivisible older batches keep boundary-spanning microbatches
and actual sample-count weights; B2048 retains688/680/680 samples globally.

Grouping requires FSDP M1/M3, routed experts, balanced sampling, chunk-level
masks, update_epoch1, and disabled reward filtering. Selected training chunks
must be fully valid. Unsupported settings are rejected instead of changing mask
denominators while claiming objective equivalence.

For each suite, GPU statistics accumulate squared-error sums, M3 time-weighted
error sums, and valid velocity-element counts. One small SUM collective combines
ranks before division. Do not average microbatch/rank means. Statistics add no
model forwards and do not participate in autograd.

- `train/actor/velocity_mse/<suite>`: raw velocity MSE.
- `train/actor/analytic_kl/<suite>`: M3 time-weighted loss before accumulation
  scaling; absent for M1.
- `train/actor/valid_velocity_elements/<suite>`: element count, including duplicates.
- `train/new_rollout_velocity_mse/<suite>`: first update of each physical round.

Statistics reset at every optimizer update, not once per R updates. Refresh
markers identify pool update index and refreshed samples. Missing suites record
zero counts without inventing zero losses. `suite_batch_order=mixed` retains
mixed ordering with diagnostics; `log_suite_velocity_metrics=false` disables
suite statistics. Other tasks without these fields retain their old defaults.

### Static and synchronized suite weights

`algorithm.suite_loss_weights` maps all teachers to positive coefficients,
normalized to mean one. With equal per-rank suite counts and valid chunks, the
objective is `sum_s(weight_s * M3_s) / K`. Weights multiply the numerator before
accumulation; normalizing each homogeneous microbatch by its weight sum would
incorrectly cancel the weighting. Raw global/per-suite MSE and analytic-KL
metrics remain unweighted across suites. `actor/optimization_loss` records the
weighted objective before accumulation scaling; `total_loss` retains its
existing scaled convention. Unequal per-rank suite counts are rejected.

`algorithm.sync_suite_loss_weights=true` instead uses `alpha_s=3*q_s/T` from the
actual integer rollout quotas, fixed across R updates. It requires synchronous
allocation and is mutually exclusive with static suite weights.

## Diagnostic probes and memory

- `robocasa365_mopd_allocation_probe_gr00t_full` forks a complete uniform U40
  warmup checkpoint and runs to U80 with T96/B2304/micro32/R2.
  `OPD_PROBE_CHECKPOINT` supplies the checkpoint. `resume_mode=allocation_probe`
  permits allocation mode/static-quota changes only at this boundary, retaining
  task cursors, Adam/scheduler, and counters. Proposal RNG restarts. These forks
  are diagnostics, not new independent C10 seeds.
- `robocasa365_mopd_allocation_probe_memory_gr00t_full` extends to U120, save20,
  with feature-only GR00T. `actor.model.skip_backbone_logits=true` bypasses the
  unused Qwen3 vocabulary projection while retaining hidden-state selection,
  parameters, tied embeddings, and checkpoint keys. It propagates to routed
  teachers and training rollout copies. LM labels/logit requests are unsupported.
  Pinned third-party sources are not modified.
- `algorithm.log_cuda_memory=true` logs rank-local allocated/reserved phase peaks
  for teacher passes, student updates, and flow refresh. These exclude EGL and
  other processes/users; do not sum phase peaks. Process/device monitoring is
  needed for whole-GPU memory. Finished input microbatches are released promptly.
- `actor.cuda_cache_retention_gib` optionally reduces allocator cache release in
  actor processes. It is not a GPU reservation or a memory lock. Formal dedicated
  GPU recipes set it to0; shared-node pilots used18.
- `robocasa365_mopd_suite_weight_probe_gr00t_full` used a common U40 fork to U120,
  T96/B2304/R2/micro32/LR3e-5, weights0.5/1.25/1.25, and either Uniform32/32/32 or
  Static24/36/36 allocation. These were sensitivity tests, not a validated law.

## Validation history

The 2026-10-19 Base/Refresh integration passed70 CPU regressions plus Ruff checks.
Tests covered full-pool sampling, duplicate observations with independent flow
noise/targets, current student parameters, RNG isolation, update/round/save
counts, logging, config rejection, GR00T precision, environment lifecycle, and
the evaluation protocol. That integration did not rerun full GPU training.

Suite tests compare real actor-loop gradients, Adam parameters/moments, logs,
and forward/backward counts across grouped/mixed batches; they include B2304,
B2048, irregular statistics masks, and two-process Gloo reductions. Weighting
tests cover explicit gradients, all-one equivalence, raw metrics, and rejected
invalid inputs. Feature-only tests cover outputs/gradients; historical GPU
checks used synthetic normalized observations and were not whole-system memory
benchmarks. The release's clean-snapshot166-test result and remaining receiving-
machine checks are documented in [VALIDATION.md](docs/opd/collaboration/VALIDATION.md).
