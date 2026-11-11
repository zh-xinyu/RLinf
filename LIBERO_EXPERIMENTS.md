# LIBERO experiment contract and freeze status

## Scope and readiness

This branch contains four-suite GR00T N1.5 MOPD, decoupled flow refresh, adaptive
rollout allocation with synchronized query counts, teacher-rollout SFT,
and independent evaluation through the training rollout implementation.

**Status: new-method development, not a launch-ready final experiment release.**
The current branch is `experiment/libero-gr00t-adaptive-query`, based on the
candidate snapshot `e5b21cc6`. The source-freeze candidate remains available as
`release/libero-gr00t-candidate`. Use the [current method/commands](toolkits/libero_gr00t/adaptive_query.md)
for squared-gap rollout/query allocation. Historical formal YAMLs/launchers are
retained separately. Share a committed snapshot with its explicit commit ID.
The root RoboCasa workspace is separate.

| Component | Evidence / remaining gate |
| --- | --- |
| Training environment and teacher/student interfaces | Existing successful mixed-K MOPD runs; checkpoint-specific metadata and affine flow-coordinate adapters retained |
| Independent evaluation | Shared environment/rollout implementation;8paired episodes across4suites,264bitwise-identical predictions vs the reference path; strict899-tensor trained checkpoint reload |
| Teacher-rollout SFT trainer | Native flow matching, finite100-update training, strict resumed optimizer state and exported head/metadata identity |
| New SFT100 initializer | Screening incomplete:76/80episodes; confirmation and SFT100→MOPD transition probe pending |
| Portable runtime entrypoints | Previously used node hooks packaged in source; CPU checks pass; fresh target-image EGL/NCCL/model acceptance pending |
| Adaptive calibration portability | Raw800-episode calibration exists; content-verified relocation to collaborator paths remains to be packaged |

Do not replace the missing acceptance evidence with a partial aggregate, an
older initializer, or installation success. No new training/evaluation is
started by this source/documentation audit.

## Fixed model and execution contract

All40tasks participate:10each from Spatial, Object, Goal and Long (`libero_10`).
Keep these PPO specialist teachers:

| Suite | Public teacher | Reproduced SR /20states per task |
| --- | --- | ---: |
| Spatial | `RLinf/RLinf-Gr00t-RL-Spatial-Step400` | 185/200=92.5% |
| Object | `RLinf/RLinf-Gr00t-RL-Object-Step400` | 193/200=96.5% |
| Goal | `RLinf/RLinf-Gr00t-RL-Goal-Step500` | 165/200=82.5% |
| Long | `RLinf/RLinf-Gr00t-RL-Long-Step300` | 180/200=90.0% |

Equal-suite teacher mean is90.375%. These are local calibration results using
official initial states0–19, not RLinf's published scores. Long K5 was86.0%; the
frozen protocol uses K10. Retain the original raw counts and checkpoint hashes.

- Generate a16×32flow latent with four deterministic ODE steps; execute
  K=5/5/5/10physical actions. Returned storage width10does not change latent16.
- Physical horizons240/240/320/480, ten reset-settling steps, freeze finished
  environments. Count executed physics separately from padded calls/settling.
- Native B1 inference and unpadded student prompts. Experimental model batching
  remains off. `--skip-finished-rollout` skips dead slots while preserving the
  reference live-slot RNG draws.
- Student keeps its grouped11D state, original normalization, FP32 trainable
  action head, BF16 frozen VLM and BF16 compute. Teachers keep their scalar8D
  state, own normalization, fixed570-token prompts and BF16 head/rotary policy.
- Affine state/velocity transport aligns the checkpoints' action coordinates.
  MOPD uses Method3 (`analytic_local_kl`, student-coordinate shared-covariance
  SDE weighting, noise0.3); generated flow trajectories remain ODE trajectories.
  This weighting convention is explicit, not an assertion of the teacher's
  native SDE pushforward or exact equivalence of coarse Euler samplers.

## Initialization under acceptance

The intended replacement is ordinary teacher-rollout SFT, distinct from the
historical **Offline-Distilled-50** initializer that used an M3 velocity target.
Historical MOPD curves and the existing `formal_*` YAML comments refer to that
older initializer. They must not be relabeled as new SFT-initialized results.

The new SFT starts from `RLobot-jun/gr00t-n1.5-libero-fewshot-combined-1k`, revision
`9f0ac4146e3bbd54daa1efad285b334a48d1aa79`, with a fresh optimizer. It uses a fixed
pool of196successful teacher trajectories /2762decision observations; original
collection cost includes277attempts and40,047policy simulation steps plus2770
settling steps. Action labels concatenate actually executed commands within each
episode, mask terminal padding and invert the simulator's gripper convention.
No new physics, teacher forward or student ODE rollout occurs during SFT updates.

Native GR00T conditional flow matching uses the upstream Beta(1.5,1.0) time law,
FP32 sampling, original metadata, frozen VLM, no image augmentation and suite/task
balanced observation sampling. B128, micro2, two ranks, LR1e-4, warmup10updates
then constant, AdamW(.95,.999), decay1e-5, clip1, seed42. Checkpoint100is exported;
its native SFT loss averaged0.10522over updates91–100. SFT loss is not MOPD MSE.

Current screen: Spatial9/20, Object12/20, Goal8/20, Long7/16;36/76observed
successes, with four missing Long episodes. This is incomplete coverage, so no
four-suite headline score is assigned. The screen stopped after a Ray
infrastructure failure; no completed training was lost. There is no proven
policy-inference failure or established OOM cause for the latest Raylet failure.

Next acceptance gates, when GPU time is authorized: complete the screen; confirm
the chosen initializer on the predeclared200episodes; run a short MOPD transition
probe. Freeze and distribute exactly one accepted HF export plus content hashes.
All arms must start from that same export with a fresh MOPD optimizer. The
public upload location for this new artifact is still pending.

## Main matrix once initialization is accepted

| Arm | Rollout allocation / query allocation | R | Rounds | Optimizer updates | Save at updates |
| --- | --- | ---: | ---: | ---: | --- |
| M0 | Uniform / equal | 1 | 100 | 100 | 25/50/75/100 |
| M1 | Uniform / equal | 2 | 100 | 200 | 50/100/150/200 |
| M2 | Adaptive / synchronized | 2 | 100 | 200 | 50/100/150/200 |
| M3 | Adaptive / synchronized | 1 | 100 | 100 | 25/50/75/100 |

Shared: one node/four ranks; T64trajectories/round,16environment slots/rank,
B512, micro2/rank, LR3e-5, four **optimizer-update** warmup steps then constant,
AdamW(.9,.95), decay0, clip1. Each update samples the full rollout observation
pool with128examples/suite in Uniform arms; Adaptive uses global counts8*q_s.
Refresh regenerates flow noise/trajectories and
teacher targets for the current student. B/LR are not divided by R. Seeds0/1/2
are repeats; start with0. Existing YAMLs and launch validation encode this recipe.

Adaptive uses the verified teacher calibration, rho1, epsilon0 and squared
positive suite SR gaps. Retain the20-round warmup/window,10-round decision
interval, minimum5episodes/task and quota-change cap6/suite/decision. Uniform
quotas are16/16/16/16. Query counts follow actual projected quotas; extra suite
loss coefficients are disabled. Learning progress is diagnostic only. First
adaptive decision is round21. These controller timing settings remain LIBERO's
tested values rather than adopting RoboTwin's10/10/5-round settings.

All arms collect6400student trajectories; optimizer compute differs by design.
Use rollout rounds as the main x-axis, and retain trajectories, actual simulation
steps, optimizer updates and wall time. Equal trajectory budgets do not imply
equal physics budgets when suite allocation and episode lengths change.
Use the same operational allocator/cache settings across compared arms and
record GPU sharing. Saving every25rounds and independent evaluation are separate
jobs: saving does not automatically launch evaluation.

## Portable command contract

First follow [LIBERO_ENVIRONMENT.md](LIBERO_ENVIRONMENT.md). Arrange separate
artifact directories: `LIBERO_MODELS/object`, `goal`, `long`, a Spatial checkpoint
directory and the accepted student export. Include `config.json`, all required
weight shards/index and `experiment_cfg/metadata.json`; retain immutable upstream
revisions and hashes. The historical `download.py` has a private output layout
and an older strong student; it is not the current portable acquisition entry.

Example **plan only** for M1, using paths populated after initialization acceptance:

```bash
export OPD_WANDB_GROUP=T64_gap_query_seed0
export OPD_WANDB_RUN_NAME=M1_Uniform-R2_T64_B512_seed0
export WANDB_MODE=offline
"$LIBERO_PYTHON" -m toolkits.libero_gr00t.launch_adaptive_query \
  --arm m1 --seed 0 --python "$LIBERO_PYTHON" \
  --models "$LIBERO_MODELS" --student "$LIBERO_STUDENT" \
  --spatial-teacher "$LIBERO_SPATIAL" --gpus 0 1 2 3 --cpus 32 \
  --actor-cache-retention-gib 20 --actor-cache-preallocate-gib 12 \
  --skip-finished-rollout --log-cuda-memory \
  --output "$LIBERO_RESULTS/M1_plan"
```

Without `--run`, the launcher writes a plan. Inspect the resolved config and
model identities. Run acceptance and actual training only under an allocation,
using a new output directory and `--run`. W&B project is `FORMAL_libero_gr00t`;
offline/local records remain sufficient. M0/M3 use100updates; M1/M2 use200.
For M2/M3, `OPD_LIBERO_TEACHER_REFERENCE` must identify a validated reference
bound to the relocated, hash-identical teacher files and execution signature.
The old reference embeds local paths: do not edit those paths by hand or silently
recalibrate. Preparing that portable bundle is a remaining handoff task.

## Independent evaluation and required delivery

Use `toolkits.libero_gr00t.launch_parallel_evaluation`, with the resolved training
config. Its workers subclass the training `EnvWorker` and `MultiStepRolloutWorker`,
use `LiberoEnv`, `prepare_actions` and `EmbodiedEvalRunner`. Exact evaluation
packets and private episode RNG replace training quota sampling; the physics,
observation conversion, action execution and policy interfaces are shared.
The historical standalone K8 YAML/launcher is not the current protocol.

Evaluate round0once and each arm at25/50/75/100rounds:17checkpoints, each40tasks
×50official initial states0–49=2000episodes. Fixed seed1234, private episode
noise seed `1234 + local_task_id*100 + trial_id`; five parallel states/task/GPU.
Require full pair coverage before computing a reportable score.

```bash
"$LIBERO_PYTHON" -m toolkits.libero_gr00t.launch_parallel_evaluation \
  --python "$LIBERO_PYTHON" --config "$LIBERO_RESOLVED_CONFIG" \
  --baseline --gpus 0 1 2 3 --envs-per-task 5 --cpus 8 \
  --output "$LIBERO_RESULTS/initial_evaluation_plan"
```

This also defaults to a plan. For trained checkpoints replace `--baseline` with
`--weights /absolute/path/to/full_weights.pt`; retain the same base student.
`--cpus 8` is per active GPU/task. A future engineering parity check may select
tasks0/10/20/30and states40/41 with `--envs-per-task 2 --trace`; it is not a
formal score. See [evaluation details](toolkits/libero_gr00t/parallel_evaluation.md).

Deliver raw episode JSON/CSV, audited summaries, resolved train/eval configs,
model/source/runtime identities, training JSONL/TensorBoard, failures and resume
records. W&B URLs are optional. Keep checkpoints locally until all evaluations
and audits finish. Report a summary CSV with columns:

```text
arm,seed,round,optimizer_update,suite,task_id,successes,episodes,success_rate,training_trajectories,training_simulation_steps,training_wall_seconds,checkpoint_identity,raw_evaluation_path
```

Keep task rows and separate suite/macro aggregates; never substitute rollout
`success_once` windows for independent evaluation. The primary figure is suite-
macro SR vs rollout rounds; the main table reports round100per-suite/macro SR
and interaction/compute costs. Initialization and calibration budgets are shared
preparation costs reported separately from online training and evaluation.

## Source audit on2026-11-10

468CPU regression tests passed across model-independent M3/refresh math, native
microbatch planning, mixed-K execution, metadata/teacher reference contracts,
allocation/resume state, launchers, strict evaluation coverage, SFT labels and
portable runtime setup. CUDA devices were hidden; no Ray cluster, simulator run,
training or GPU evaluation was launched. Changed Python files also passed Ruff
lint/format checks. The vendored Robosuite adapters retain upstream formatting
and license notices; their selection logic has an isolated CPU test.

This is code/regression acceptance, not a new installation or new hardware
acceptance. Historical GPU evidence uses the old initializer; SFT100→MOPD and
the collaborator image remain pending as listed above. Current runs are stopped.

## Adaptive-query development on2026-11-11

The new method retains the tested T64/B512/micro2/LR3e-5,100-round recipe and
20/20/10allocator timing. It changes the priority to squared positive SR gaps
and replaces synchronized suite loss coefficients with synchronized query counts.
411CPU regressions passed;21specifically cover the new method. A100-round
synthetic donor comparison matched every plan/query allocation. GPU acceptance
and initialization confirmation remain separate. See the
[current implementation guide](toolkits/libero_gr00t/adaptive_query.md).
