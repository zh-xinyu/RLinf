# GR00T N1.5 / LIBERO MOPD exploration

This branch explores four-suite capability consolidation with public checkpoints,
real LIBERO rollouts, and the production Method-3 loss. The standalone pilot and
production Ray/FSDP configuration are separate paths; neither is a frozen paper recipe.

## Checkpoints and screening

| Candidate | Source | Screening successes |
| --- | --- | --- |
| Original strong student | `twanghcmut/GR00T-N1.5-LIBERO-4suite-combined` | 39/40 |
| Current few-shot student, native text padding | `RLobot-jun/gr00t-n1.5-libero-fewshot-combined-1k` | 9/40 |
| Routed RL specialists | RLinf Spatial-Step400, Object-Step400, Goal-Step500, Long-Step300 | 33/40 |

Each screen covers ten tasks per suite and one official initial state per task.
These small counts motivate the weaker candidate; they are not formal scores or
proof that distillation improves the student. The earlier strong pairing had no
observed teacher headroom. An earlier B=4, two-round joint smoke verified finite
updates and synchronized replicas, not improved task success.

The current student is pinned to revision
`9f0ac4146e3bbd54daa1efad285b334a48d1aa79`; the original student used
`aee73136d034537f68c4e34d2619d3d357c7a2b1`. Download manifests record teacher
revisions and file hashes. The few-shot release does not identify its exact
training-code revision, so compatibility does not establish reproduction of its
training procedure.

## Model and metadata contract

- Generate 16 actions, execute eight, and use four ODE steps. Main/wrist images
  are rotated according to the LIBERO adapter; image tiling is disabled.
  Episode horizons are 240/320/320/520, with ten settling steps after reset.
- The few-shot student now supports training through the production
  `LiberoGroupedDataConfig` and `libero_grouped` observation adapter. Its grouped
  state converts axis-angle rotation to 6D rotation (11 valid state dimensions);
  controller actions remain 7D. Teachers retain their scalar LIBERO schema.
- Student tag `new_embodiment` and teacher tag `libero_franka` both explicitly
  select projector 31. Each checkpoint keeps its own normalization metadata.
- Student inference uses native, unpadded **B=1** prompts. Teachers use their own
  fixed-570 token contract. Production native rollout splits batches into B=1
  calls and pads token fields only for storage; student training/refresh removes
  storage padding. Keep training microbatch one for mixed prompt lengths.
- The pilot audits the lightweight teacher state adapter against the full
  transform, the projector ID, actual image tensors, and unpadded token contents.
  Teacher preprocessing is built independently. A separate endpoint audit checks
  that transported actions decode to the same physical commands using the actual
  checkpoint transforms. Policy keys/shapes are checked; extra critic keys are allowed.

Cross-metadata M3 uses the existing endpoint map and ODE transport:

```text
a_student = scale * a_teacher + shift
A(t) = 1 + t * (scale - 1)
x_student(t) = A(t) * x_teacher(t) + t * shift
v_teacher_in_student = A(t) * v_teacher + (scale - 1) * x_teacher + shift
```

`gr00t_cross_metadata_m3_mode: student_coordinate_sde` defines both local Gaussian
kernels in student coordinates with the same student noise schedule, retaining
M3's weighted velocity MSE. This is not the pushforward of the teacher's native
SDE, nor a claim that coarse four-step Euler samplers are exactly equivalent.

## Standalone full-episode pilot

The executable recipe is [configs/pilot_uniform_r2.yaml](configs/pilot_uniform_r2.yaml):

| Item | Setting |
| --- | --- |
| Collection | Four GPU ranks, one suite and one complete episode per rank per round (T=4) |
| Updates | Refresh-R2, 20 rounds / 40 optimizer updates, fresh pool each round |
| Batch | Global B=64, microbatch=1; 16 samples per suite per update |
| Optimizer | LR=3e-5, four-update warmup, AdamW betas=(0.9, 0.95), weight decay=0 |
| Precision / M3 | FP32 student action head, BF16 autocast, noise level=0.3 |
| Training coverage | Cycle all ten tasks per suite; initial states 0 then 2 |
| Evaluation | All ten tasks per suite at initial state 1, at rounds 0/10/20 |

NCCL averages gradients after equal local accumulation, giving every suite equal
weight. Replica fingerprints and finite gradients are checked after updates.
Evaluation restores training RNG state. One held-out initial state per task is
still a screening protocol, not the final multi-episode evaluation.

From the worktree root, use the configured local runtime and downloaded assets:

```bash
PROBE_PYTHON="$PWD/outputs/runtime/bin/python"
PROBE_MODELS="$PWD/../../libero_gr00t_20261104/models"
PROBE_SPATIAL="$PWD/../../libero_feasibility_20261104/models/groot_spatial_rl"

"$PROBE_PYTHON" toolkits/libero_gr00t/launch_pilot.py \
  --python "$PROBE_PYTHON" \
  --config toolkits/libero_gr00t/configs/pilot_uniform_r2.yaml \
  --models "$PROBE_MODELS" --student "$PROBE_MODELS/student_fewshot" \
  --spatial-teacher "$PROBE_SPATIAL" --output outputs/libero_mopd_pilot
```

The output directory must be new. Optionally pass `--gpus ID ID ID ID`; otherwise
the launcher selects the four GPUs with most free memory and requires at least
30 GiB free on each. The 24-GiB PyTorch allocator cap does not reserve memory or
include every CUDA/EGL allocation. Existing jobs are not stopped. Timings on
shared GPUs are not dedicated-machine throughput measurements.

Outputs include `recipe.yaml`, `launch.json`, per-suite logs/JSONL, separate
training/evaluation interaction counts, and action-head checkpoints at updates
20 and 40. Saved heads reference the unchanged base checkpoint and are reloaded
for a consistency check. They contain no optimizer state; resume is unsupported.
The T4 pilot completed: initial9/40, update20=2/40, update40=4/40.
The matched LR3e-6 control completed9→7→6/40. Both failed the learning gate;
falling velocity MSE alone is insufficient to freeze this setting.

The local `outputs/runtime/bin/python` wrapper reuses a read-only base environment
and a LIBERO overlay. Inspect `outputs/runtime/env.sh` and `provenance.json` for
versions and asset paths. This is not a portable installation profile and does
not modify shared Ray state. The older `launch.py` entrypoint remains available
for paired screening and short correctness probes.

## Production configuration draft

[libero_four_suite_mopd_refresh_r2_gr00t_pilot.yaml](../../examples/embodiment/config/libero_four_suite_mopd_refresh_r2_gr00t_pilot.yaml)
uses the production runner with four collocated actor/env/rollout ranks. Each
rank receives all four suites: **T=16**, B=64, microbatch=1, R2, 20 rounds. Routing
keys are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
Checkpoint paths come from the `OPD_*` environment variables listed in the YAML.

Hydra composition, algorithm contracts, environment sampling, metadata adapters,
and logging have CPU tests. Full production GPU training remains unverified.
Unlike the standalone recipe, this draft samples all official initial states
and disables in-training evaluation; it does not implement the [0, 2]/[1] split.
The two paths therefore are not matched experimental controls.

For an integration-only production smoke, prepare a plan first:

```bash
"$PROBE_PYTHON" toolkits/libero_gr00t/launch_production.py \
  --python "$PROBE_PYTHON" --models "$PROBE_MODELS" \
  --student "$PROBE_MODELS/student_fewshot" --spatial-teacher "$PROBE_SPATIAL" \
  --gpus 0 1 4 5 --output outputs/production_smoke_plan --dry-run
```

GPU IDs are physical and must be chosen for the current machine. The default
only writes a resolved config and launch manifest. To execute, use a new output
directory and explicit `--run`; it requires 30 GiB free on each selected GPU.
The launcher uses private Ray ports/address files and owner-scoped cleanup,
without joining or stopping existing clusters. Defaults are two updates,
checkpoint at update two, and **32-step incomplete episodes** with capped suite
horizons; this measures integration, not success rate. `--rollout-steps 520`
retains full native suite horizons. No production smoke completion is claimed.
To test the T32 quota path, add `--config-name libero_mopd_candidate_m1
--rollout-steps 520 --updates 2` (on one command line) and use a new output
directory. The smoke launcher always overrides the long-run budget and final
save interval with `--updates`; it does not silently launch150 rounds.

Opt-in `runner.log_physics_step_counts` records actual policy, settling, total
simulator calls, and resets by summing all ranks. The metrics end in `_session`
because they restart after process recreation. Existing mask-based
`interaction/env_steps` remains distinct and also appears as
`interaction/valid_policy_steps`. The four-suite recipe enables
`freeze_finished_episodes`: each slot stops physics at its first success or
native timeout, retaining its terminal observation and outcome through the
remaining chunk/round. Policy physics counts then match the valid control-step
mask. The legacy single-suite default remains unchanged.

The production environment also supports `task_sampling_strategy: suite_quota`
with `runner.rollout_allocation.enabled: true`. The existing uniform, static,
and adaptive controllers share exact slot plans, suite/local-task identities,
task-occurrence-based official initial states, and per-slot reset seeds. Use
this same collector for all formal comparison arms. Allocation requires full
native horizons and at least one slot from every suite on each worker. On four
workers, T16 leaves no adaptive freedom; T32 with eight slots per worker permits
nonuniform integer global quotas, including quotas not divisible by four.
Adaptive mode still requires a matching per-task teacher calibration artifact;
the tiny screening results are not that artifact. This integration has CPU
coverage; a complete production GPU training run remains to be verified.

Candidate matrix configs are available under
[`examples/embodiment/config`](../../examples/embodiment/config/), sharing
[`libero_mopd_candidate_common.yaml`](../../examples/embodiment/config/libero_mopd_candidate_common.yaml).
They are **not frozen formal recipes**: learning acceptance is still pending.

| Config suffix | Allocation / loss weighting | Refresh | Updates / save interval |
| --- | --- | --- | --- |
| `libero_mopd_candidate_m0` | Uniform / equal | R1 | 150 / 25 |
| `libero_mopd_candidate_m1` | Uniform / equal | R2 | 300 / 50 |
| `libero_mopd_candidate_m2` | Adaptive / synchronized | R2 | 300 / 50 |
| `libero_mopd_candidate_m3` | Adaptive / synchronized | R1 | 150 / 25 |

All use T32, B64, microbatch 1, LR 3e-6, four warmup updates, and 150 physical
rounds. Evaluate the initial policy and every saved 25 rounds separately with
[`libero_four_suite_gr00t_mopd_eval.yaml`](../../evaluations/libero/libero_four_suite_gr00t_mopd_eval.yaml).
M2/M3 require `OPD_LIBERO_TEACHER_REFERENCE` pointing to a validated calibration
JSON. M0/M1 do not require it. Config composition and validation are CPU-tested;
these files do not launch experiments automatically.


## Independent evaluation result audit

After one checkpoint evaluation, pass its raw per-worker logs to:

```bash
"$PROBE_PYTHON" toolkits/libero_gr00t/summarize_evaluation.py \
  --logs /path/to/one-evaluation/worker-0.log /path/to/one-evaluation/worker-1.log \
         /path/to/one-evaluation/worker-2.log /path/to/one-evaluation/worker-3.log \
  --output outputs/one_checkpoint_evaluation_summary
```

Use actual worker filenames; do not combine checkpoints or runs. The default
requires all40 tasks x50 trials, writes `episodes.csv` and `summary.json`, and
withholds SR with a nonzero exit if coverage is incomplete. Identical duplicated
log lines count once; conflicting duplicates and out-of-range IDs fail. Preserve
raw logs, their recorded hashes, resolved evaluation configuration and checkpoint
identity. Log coverage alone cannot verify correct horizons or checkpoint loading.

## Production checkpoint reload smoke

Once production training writes a complete `full_weights.pt`, validate it in a
new single-GPU process. This does not join or start Ray. Select an available GPU
with at least20 GiB free (the default preflight); the PyTorch cap is16 GiB and is
not a memory reservation. From the worktree root:

```bash
CUDA_VISIBLE_DEVICES=6 "$PROBE_PYTHON" toolkits/libero_gr00t/evaluate_full_checkpoint.py \
  --weights /path/to/run/checkpoints/global_step_4/actor/model_state_dict/full_weights.pt \
  --config /path/to/run/resolved_config.yaml \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --tasks 0 --trials 1 --output outputs/full_checkpoint_reload_smoke
```

Use the actual saved checkpoint path and a new output directory. `--student`
optionally relocates the original base checkpoint with its config and metadata;
full weights alone do not replace those assets. The tool reads the exact actor
model config, checks the grouped-state/native-text/FP32-head contract, then loads
all keys strictly on CPU before moving the policy to GPU. Partial head files,
nested training states, mismatched shapes/dtypes and nonfinite weights fail.
In particular, it does not silently cast a BF16 saved head to FP32. Before
loading, it also requires every `backbone.*` tensor to exactly match the original
base policy. The protocol records changed/unchanged action-head tensor counts,
unchanged backbone counts, and separately lists changes to other state such as
version buffers. Unchanged individual head tensors are permitted; frozen VLM
drift is rejected rather than silently accepted.

Defaults evaluate one task and one official initial state per suite at native
horizons with first-success stopping and ten settling steps. These are complete
episodes but only a reload/inference smoke, not formal performance evidence.
`protocol.json` records weight/config/metadata hashes, the model contract and
checkpoint update/round; `episodes.jsonl` records individual outcomes and timing;
`summary.json` is written only after all requested episodes complete. This
standalone check does not validate the distributed evaluator's episode scheduler.
The exact full-weight GPU reload remains unverified until a real produced
checkpoint passes this smoke.

## Shared-GPU smoke synchronization

The LIBERO recipes use the existing bucket synchronizer with original dtypes
(`bucket_dtype: null`) and immediate loading. Actor and rollout start from the
same frozen base checkpoint, so unchanged frozen parameters need no repeated
transfer. Dense FP32 action-head updates made the earlier COO patch sync allocate
large temporary index/value tensors and caused an OOM between rounds, after two
successful optimizer updates. Bucket sync avoids that patch assembly without
changing the optimizer, objective, or weight precision.

The32 MiB bucket setting is a target: a tensor is never split, and the largest
current head tensor is576 MiB. Actual GPU peak still requires measurement. The
bounded production launcher now aborts reported CUDA OOMs even when they occur
in background worker threads; `--timeout-seconds` bounds the driver runtime
(default3600). Cleanup remains limited to this launch's owned Ray processes.

## Candidate precision and noise seeds

Before learning validation, the LIBERO actor uses AMP-only mixed precision:
FSDP parameter, reduction and buffer dtype overrides are `null`, input casting is
false, and BF16 autocast is enabled. The loaded frozen VLM stays BF16 while the
head and explicitly preserved rotary buffers stay FP32. The earlier production
smokes used FSDP BF16 casting; those are integration records, not matched learning
controls for the standalone pilot. Real FSDP parity/memory checks are required
before accepting the updated recipe.

Candidates explicitly set `rollout.seed: 0` in addition to actor, environment
and controller seeds. Rollout workers are separate processes: actor/environment
seeds alone did not determine their flow noise. The opt-in seed is applied after
model initialization/warmup as `rollout.seed + worker_rank`. Changing training
seed must change all four fields together. Recipes omitting `rollout.seed` keep
the previous behavior. This is start-of-run determinism, not exact RNG restoration
on checkpoint resume or invariance to worker count. Independent distributed
evaluation uses its own fixed rollout seed 1234.

Production warmup retains its existing convention: with four warmup updates,
LR multipliers start at `[0, .25, .5, .75, 1]`; the standalone pilot used
`[.25, .5, .75, 1]`. The first production call updates Adam moments but not weights.
Compare `velocity_mse` or `analytic_kl`: logged production `total_loss` is divided
by the gradient-accumulation count. No change to the underlying M3 objective or
gradient normalization is introduced here.

## FSDP versus unwrapped AMP parity

Before accepting the AMP-only production recipe, run the following bounded check
on one explicitly selected GPU with at least32 GiB free:

```bash
CUDA_VISIBLE_DEVICES=6 "$PROBE_PYTHON" toolkits/libero_gr00t/fsdp_parity.py \
  --config /path/to/amp-only-run/resolved_config.yaml \
  --output outputs/fsdp_amp_parity
```

The tool requires all three FSDP mixed-precision dtypes to be null and BF16
`amp_autocast`. It uses the actual `FSDPStrategy.wrap_model` factory and configured
wrap policy/`use_orig_params`, with a private world-one NCCL file process group.
A real Spatial task0 initial-state0 observation, fixed full action latent and
flow index2 are reused before/after wrapping the same model. The VLM remains
BF16, the head FP32, and all rotary `inv_freq` tensors must remain FP32.

It compares every NFT output value and seven selected action-head parameter
gradients across encoders, decoder, conditioning attention, DiT and time embedding.
The scalar loss is velocity squared mean for numerical diagnosis, not a teacher
performance objective. The reference gradients are moved to CPU before wrapping;
the PyTorch memory cap defaults to28 GiB. `fixture.pt` records the exact input,
`result.json` records per-tensor errors/dtypes/memory and acceptance, and `status`
records completion. This verifies neither four-rank gradient scaling nor an
optimizer step, distributed communication, or task success. CPU tests verify
flat-gradient name/padding handling; real model parity requires the GPU run.
