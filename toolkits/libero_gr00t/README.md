# GR00T N1.5 / LIBERO feasibility probe

This experimental branch starts at `opd-project-main` commit `251ac605`. It
uses existing public checkpoints, real LIBERO simulation, and the production
Method-3 loss. It does not replace the RoboCasa recipe or implement the full
Ray/FSDP multi-suite LIBERO runner.

## Checkpoints and input contract

- Student: `twanghcmut/GR00T-N1.5-LIBERO-4suite-combined`, revision
  `aee73136d034537f68c4e34d2619d3d357c7a2b1`.
- Teachers: RLinf GR00T RL Spatial-Step400, Object-Step400, Goal-Step500,
  Long-Step300. The download manifest pins revisions and checks LFS hashes.
- Generate 16 actions, execute 8, four ODE steps; standard rotated main/wrist
  images and 8-dimensional proprioception. The probe explicitly disables
  image tiling. Episode horizons are 240/320/320/520.
- Both selected checkpoint tags use projector **31** in N1.5. The names differ:
  student `new_embodiment`, teachers `libero_franka`. A process-global patched
  table is insufficient: Pydantic captures defaults at import time. The explicit
  `embodiment_projector_id` pins each transform instance independently.
- `data_config_class` separates the LIBERO schema from the projector tag.
- Each model retains its original metadata. Before teacher supervision, the
  probe compares the lightweight state adapter and projector ID with the full
  teacher transform, and checks exact equality of reused image/token inputs.
- Checkpoint policy keys and tensor shapes are checked; extra critic keys are
  allowed, missing policy tensors are not.

## Cross-metadata M3

Reuse RoboCasa's endpoint map, `a_student = scale * a_teacher + shift`, and
its time-dependent ODE coordinate transport:

```
A(t) = 1 + t * (scale - 1)
x_student(t) = A(t) * x_teacher(t) + t * shift
v_teacher_in_student = A(t) * v_teacher + (scale - 1) * x_teacher + shift
```

This preserves the common Gaussian prior and continuous ODE endpoint semantics.
It does not assert exact equality of coarse four-step Euler samplers.

The explicit experimental option
`algorithm.gr00t_cross_metadata_m3_mode: student_coordinate_sde` constructs
**both local Gaussian transition kernels in student coordinates**, with the
same student noise schedule. The existing M3 weighted velocity MSE is then
unchanged. This is **not the pushforward of the teacher's native SDE**. The
native-SDE interpretation would require a transported covariance and Jacobian.
Default cross-metadata M3 rejection and the M4 rejection remain intact.

## Local execution

The verified action01 runtime is an isolated wrapper at
`outputs/runtime/bin/python` inside this worktree. It reuses the read-only
`/localscratch2/liyifa11/XyFile/opd/venv` environment and an environment-only
LIBERO overlay. See `outputs/runtime/provenance.json` and `env.sh` for exact
versions and asset paths. This is a local feasibility setup, not a portable
installation profile. No shared environment or Ray cluster is modified.

From the worktree root, set paths to the existing artifact directories:

```bash
PROBE_PYTHON="$PWD/outputs/runtime/bin/python"
PROBE_MODELS="$PWD/../../libero_gr00t_20261104/models"
PROBE_SPATIAL="$PWD/../../libero_feasibility_20261104/models/groot_spatial_rl"

# Independent paired screening: all 10 tasks, one official init state each.
python3 toolkits/libero_gr00t/launch.py --mode eval \
  --python "$PROBE_PYTHON" --models "$PROBE_MODELS" \
  --spatial-teacher "$PROBE_SPATIAL" --output outputs/paired_screen

# One shared student: four suite ranks average gradients before every update.
python3 toolkits/libero_gr00t/launch.py --mode mopd \
  --python "$PROBE_PYTHON" --models "$PROBE_MODELS" \
  --spatial-teacher "$PROBE_SPATIAL" --output outputs/joint_smoke \
  --student-head-precision fp32 --rounds 2 --reuse 2 --rollout-chunks 4
```

The launcher selects the GPUs with most free memory and requires 30 GiB free
on each at launch. `--gpus` can explicitly select them. Each child uses a
24-GiB PyTorch allocator limit, low CPU priority and one CPU thread. This limit
does not reserve memory or guarantee exclusivity; CUDA contexts/EGL allocations
are outside the allocator limit. Existing jobs are never stopped. Shared GPU
compute makes these timings unsuitable as dedicated-machine throughput results.

MOPD uses one observation per suite per update (global B=4), LR=3e-5,
AdamW, equal suite gradients and two fresh-flow updates per physical segment.
Every round replaces the observation pool with current-student observations.
Each rank owns a different frozen teacher; Gloo averages CPU copies of gradients.
Post-update parameter fingerprints must agree across all ranks. The first round
and second round therefore exercise rollout both before and after joint updates.
This intentionally simple communication path is for correctness, not speed.

The default BF16 action head matches the initial inference/single-teacher smoke.
The joint probe can restore the student's original FP32 head and use autocast;
precision is logged. Short training segments are explicitly marked incomplete
and do not estimate success rates. No trained checkpoint is saved by this probe.
Ten episodes per suite are screening evidence, not formal evaluation.

## Completed four-card joint smoke

The FP32 student action-head probe completed two physical rollout rounds and
four joint optimizer updates (Refresh-R2), global B=4, LR=3e-5. Four suite ranks
averaged gradients before each update. Sampled parameter fingerprints agreed
across all ranks after every update, and fresh noise hashes differed at every
update. Teacher-input audits and finite-gradient checks passed.

Total interaction was 256 policy steps plus 80 settling steps. Maximum PyTorch
reserved memory was 23.08 GiB per process, allocated memory 20.48 GiB. These
short incomplete segments do not measure task SR. No trained checkpoint was
saved; this is a correctness probe, not evidence of performance improvement.
The full production Ray/FSDP LIBERO multi-suite runner remains future work.

Raw evidence: `outputs/libero_gr00t_20261104/joint_mopd_smoke/*.jsonl` and
`joint_mopd_summary.json` in that artifact directory. All 22 targeted unit tests
and Ruff checks passed. No branch has been pushed or merged.

Paired screening (10 tasks × one init state per suite): student 9/10, 10/10,
10/10, 10/10; specialists 8/10, 8/10, 8/10, 9/10 for Spatial, Object, Goal,
Long respectively. This pairing has no demonstrated teacher headroom, so no
long training run was started. These are screening counts, not formal scores.

## Few-shot candidate screening

`launch.py --student PATH` overrides only the student checkpoint. The optional
`fewshot.py` adapter supports the grouped schema published in
`RLobot-jun/gr00t-n1.5-libero-fewshot-combined-1k` (revision
`9f0ac4146e3bbd54daa1efad285b334a48d1aa79`). It preserves the checkpoint's
normalization and uses the publisher's public `LiberoDataConfig` convention:
axis-angle state becomes 6D rotation (11 valid state dimensions), while controller
actions remain 7D. It is **evaluation-only**; MOPD rejects this schema until
state and action-coordinate transport are audited. The exact training-code
revision is absent from the checkpoint release, so this is a provisional
compatibility assessment, not a reproduction claim. The shared screening
protocol retains RLinf's text padding and BF16 inference.
