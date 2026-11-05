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
