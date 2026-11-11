# LIBERO experiment handoff: start with Uniform

## Current scope

Prepare the approved platform image first, then run Uniform-R1/R2/R4, each
with training seeds 0, 1 and 2 (nine runs total). Internal pip configuration and file transfer are supplied
by the cluster team. No custom container import is required. Hardware/image
acceptance and the new student acceptance are still pending; this is the work
queue, not a claim that the target cluster has already been validated.

Use branch `release/libero-gr00t` and record its agreed commit.
Do not use the RoboCasa release or the historical LIBERO formal launcher.

| Branch | Purpose |
| --- | --- |
| `release/libero-gr00t` | Current collaborator source: Uniform-R1/R2/R4 x three seeds; retained Adaptive implementation |
| `release/libero-gr00t-candidate` at `e5b21cc6` | Older locally retained source/environment snapshot, before the query-method port and Uniform-R4 release; not this handoff |

The current release descends from that candidate; it does not overwrite it. Only
`release/libero-gr00t` is the current published collaborator branch. Pin the
commit in `source_receipt.json`, rather than selecting a branch by a similar name.

```bash
git clone --branch release/libero-gr00t \
  https://github.com/zh-xinyu/RLinf.git libero-mopd
cd libero-mopd
git rev-parse HEAD
```

Before launching, check out the exact commit recorded in the supplied handoff
receipt. Future branch updates must not change a running experiment's source.

## 1. Installation and artifact queue

Follow [the platform-image guide](LIBERO_CLUSTER_SETUP.md) for numbered manual
commands. It preserves the internal package source. The public-network installer
is a reference, not the internal-cluster entrypoint.

1. Select Linux x86_64, Python 3.11.14, four GPUs on one node, 32 CPUs and
   preferably 256 GB host RAM. Record the actual GPU/image/driver before choosing
   the Ampere/Ada or Blackwell profile. EGL graphics access is required.
2. Create the small bootstrap venv; install `uv==0.10.4` through internal pip.
3. Create one isolated runtime and install its exact pinned packages, GR00T and
   LIBERO source/assets. Training and evaluation both use this same runtime.
4. Run `prepare_runtime`, then validate packages, source identities, EGL mapping,
   collectives and checkpoint loading inside an allocated GPU job.
5. Transfer the four teachers and the accepted student, retaining original
   metadata, all weight shards and file hashes. Model transfer is separate from
   Git clone. No credentials or personal `.opd.action*.env` files are needed.

| Artifact | Source / handling |
| --- | --- |
| Spatial teacher | `RLinf/RLinf-Gr00t-RL-Spatial-Step400` |
| Object teacher | `RLinf/RLinf-Gr00t-RL-Object-Step400` |
| Goal teacher | `RLinf/RLinf-Gr00t-RL-Goal-Step500` |
| Long teacher | `RLinf/RLinf-Gr00t-RL-Long-Step300` |
| Shared student | Teacher-rollout SFT100 HF export; supplied separately after acceptance, not the historical download script's student |
| Runtime sources | Pinned GR00T and RLinf-LIBERO revisions from the selected profile, including real LIBERO assets/initial states |
| Runtime wheels | Selected profile's exact pins plus matching FlashAttention; mirror/import any unavailable internal packages |
| Teacher SR reference | Not required by this Uniform queue; prepare separately for later Adaptive work |

The new student HF export is approximately 7.59 GB (decimal). Include
`config.json`, both safetensors shards, the index, `experiment_cfg/metadata.json`
and `offline_export.json`. Its upload/transfer location is not assigned yet.
Full teacher/source/wheel delivery identities must be recorded before admission.
Do not download a changing `main` and assume it matches the calibrated artifact.

Uniform still needs all four teachers for supervision, but **does not need their
SR reference**. Teacher calibration can therefore run separately from Uniform.
The teacher trajectory dataset is needed to reproduce initialization/offline
baselines, but is not required to train from the supplied frozen student.

## 2. Ordered experiment list

| Priority | Experiment | Training seeds | R | Rounds/run | Updates/run | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | Node and student admission | Disposable | 2 | 2 | 4 | Runtime and artifacts staged |
| 1 | Shared initial evaluation | Fixed evaluation seed 1234 | — | 0 | 0 | Inference admission |
| 2 | Uniform-R1 | 0, 1, 2 | 1 | 100 | 100 | Accepted shared student and training admission |
| 3 | Uniform-R2 | 0, 1, 2 | 2 | 100 | 200 | Same student and recipe |
| 4 | Uniform-R4 | 0, 1, 2 | 4 | 100 | 400 | Same student and recipe |

All nine runs are requested. Suggested execution order: R1/R2/R4 with seed 0,
then R1/R2/R4 with seed 1, then seed 2. Parallel execution is fine on separate
admitted allocations. This order gives an early complete reuse comparison;
seeds 1/2 are not an optional or indefinitely deferred extension.

**Adaptive is not assigned to the collaborator in this batch.** Its source and
the common runtime stay available. The project owner will tune it locally and
provide a later pinned configuration/handoff. Do not run the retained M2/M3 YAMLs
or treat the present controller values as a finalized collaborator recipe.
Teacher50 calibration and portable reference binding remain separate preparation.

The short admission run is disposable, uses a fresh output directory and is not
a main-result run. Its bounded configuration must be reviewed jointly before
execution; the formal launcher deliberately rejects arbitrary budget overrides.
There is no separate automated admission command promised by this handoff.

The SFT100 screen currently covers 76/80 episodes. Before formal training, audit
the missing four episodes and the predeclared confirmation states 42–46 against
the shared round-0 evaluation, which includes states 0–49. These records can
complete acceptance without duplicate evaluation. Confirm the intended 30–70%
initial SR range and a working SFT100-to-MOPD transition. Do not substitute the
older Offline-Distilled-50 checkpoint silently. If acceptance fails, pause the
formal queue and agree on one replacement initialization for every arm.

## 3. Fixed settings and fairness

| Setting | All main runs |
| --- | --- |
| Suites / tasks | Spatial, Object, Goal, Long; 10 tasks each |
| Execution length / horizon | K=5/5/5/10; horizons 240/240/320/480 |
| Flow / loss | Four ODE steps, full 16x32 latent; M3 `analytic_local_kl`, noise level 0.3 |
| Layout / collection | One node, four ranks, 16 env slots/rank; T64 trajectories/round |
| Training batch | Global B512, microbatch 4 per rank by default; test 8 at admission |
| Optimizer | LR 3e-5, AdamW (.9,.95), decay 0, clip 1 |
| LR schedule | Warmup 4 optimizer updates, then constant |
| Save / independent evaluation | Every 10 physical rollout rounds through 100 |
| Training trajectories | 6400 per run; actual physics steps recorded separately |

Keep B, LR, optimizer, initialization, execution, evaluation and runtime identical.
R1/R2/R4 perform one/two/four student updates per physical rollout; do not divide
B or LR by R. Warmup remains four optimizer updates for every run.
The formal config files are tracked under
`examples/embodiment/config/libero_mopd_uniform_r{1,2,4}.yaml`.
No learning-parameter sweep is part of the first queue. Keep native B1 inference;
microbatch selection is an admission step and must be consistent across all nine
runs. The common Adaptive runtime supports it, but Adaptive numerical configs
remain internal until the owner supplies the later recipe.

## 4. Manual command entrypoint with automatic follow-up evaluation

Use [LIBERO_COMMANDS.md](LIBERO_COMMANDS.md) for the complete manual workflow,
including all nine training commands and checkpoint evaluation by physical round.
Copy `toolkits/libero_gr00t/configs/uniform_machine.example.json` to
`outputs/local/libero_machine.json`, edit its absolute model/runtime/results paths
and admitted resources once, then select the experiment using R and seed:

```bash
# Plan only; no training job starts.
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 4 --seed 1
# Actual experiment after admission, using a separate automatically named path.
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 4 --seed 1 --run
```

With `--run`, the wrapper completes the full 100-round training, waits for
owned-Ray cleanup, and automatically evaluates the shared initial student and
saved rounds 10 through 100 on the same GPUs. Add `--train-only` only when a
separate training allocation is intended. `--evaluate-all --run` evaluates an
already completed run without training again.

The wrapper calls the configured runtime without requiring activation or repeated
exports. R/seed determine the tracked numerical preset, result directory and W&B
labels together. Machine JSON fields cannot override T/B/LR/rounds. Existing
outputs are rejected; use `--output` for a separately agreed new path.

Default microbatch is 4/card; test 8 through first teacher supervision, every R4
refresh/update, backward and save/resume on all GPUs before selecting it. The
nominal accumulation is 32 slots/rank at micro4 and 16 at micro8; native text
buckets can add masked slots. Choose one admitted value and cache settings for
all nine runs, based on actual peaks and update wall time. Keep B512/LR fixed.
Rollout inference stays native B1 and independent evaluation stays unchanged.
The mathematical objective is preserved; BF16 accumulation is not bitwise exact.

W&B project is `FORMAL_libero_gr00t`. Machine JSON defaults to offline; online is
optional using the platform's normal login. Runtime/image/model and initial
student acceptance remain prerequisites, not actions performed by this wrapper.

## 5. Independent evaluations

Evaluation uses the training environment/rollout implementation in the same
runtime. The manual entrypoint schedules it automatically after the entire training run
finishes. Checkpoint saving during training does not start concurrent evaluation.
Every checkpoint uses 40 tasks x 50 official states 0–49, seed 1234, five parallel
states/task/GPU and eight CPUs per active GPU. Require complete episode coverage.

Normal training commands automatically queue every saved round. The following
single-checkpoint commands are optional: `--eval-round 0` measures the shared
initial student; a nonzero value selects a saved physical round, not extra training:

```bash
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 1 --seed 0 --eval-round 0 --run
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 4 --seed 1 --eval-round 40 --run
```

Omit `--run` to inspect each evaluation plan first. The latter command selects
update160 automatically. Full instructions and relocated-config handling are in
[the command guide](LIBERO_COMMANDS.md). Checkpoints retain the layout:

```text
<run>/libero_uniform/checkpoints/global_step_<update>/actor/model_state_dict/full_weights.pt
```

| Reuse | Physical rounds | Corresponding checkpoint update indices |
| --- | --- | --- |
| R1 | 10/20/…/100 | 10/20/…/100 |
| R2 | 10/20/…/100 | 20/40/…/200 |
| R4 | 10/20/…/100 | 40/80/…/400 |

Do not pass the training seed to evaluation: the evaluation seed remains 1234.
Reuse the shared initial result across all nine runs. There are 90 trained
checkpoint evaluations plus one shared initial evaluation: **182,000 episodes**.
The first seed's three controls plus the shared initial evaluation need 62,000.
Record evaluation costs separately from training interaction budgets.

## 6. Required results

Return the following, with paths referenced from the summary CSV:

- Raw independent `episodes.json`/`episodes.csv`, task summaries, coverage audit
  and failures for each checkpoint; do not replace them with rollout windows.
- One task-level results CSV: arm, training seed, round, update, suite, global
  task ID, successes, episodes, SR, training trajectories/physics/wall time,
  checkpoint identity and raw evaluation path. Missing values stay empty.
- A separate suite/macro summary table. Task rows have 50 episodes; suite rows
  have 500; macro averages the four suite SRs equally.
- Resolved train/eval configs, launch/provenance/runtime receipts, training
  logs/metrics, allocator records and any failed/resumed-run records. Preserve
  per-suite raw MSE, suite-macro MSE, physical rounds, trajectories, simulation
  steps, update index and wall time. W&B links are optional.

Retain checkpoints on the cluster until all evaluations and audits finish; no
checkpoint transfer is requested as a result deliverable. The primary figure is
independent suite-macro SR vs rollout rounds. The final table reports per-suite
and macro SR at round 100 (mean and standard deviation over three training
seeds), plus training interaction/compute costs. Also retain individual-seed
results; do not combine counts as if training seeds were extra evaluation states. Model
initialization and teacher calibration are shared preparation costs.

## Automatic evaluation receipts

Reserve job time for training plus its sequential post-training evaluation queue.
Each run writes `post_training_evaluation.json`, with per-round output paths and
failure/attempt history. A successful training-only status does not imply all
measurements finished; require this receipt's COMPLETE status and audited raw
results. Shared initial and matching completed checkpoint results are reused only
with model/protocol/source and coverage checks. Failed or stale folders are never
silently replaced. See the command guide for `--evaluate-all --run` recovery and
`--config` handling for a relocated completed run.
