# Candidate four-arm LIBERO experiment entry

**Status: candidate, not frozen.** These commands describe the planned native
policy matrix after learning, calibration and adaptive integration acceptance.
They do not authorize automatic formal launches and do not alter the current
native continuation or its queues. Do not use the canonical fixed570 profile or
`--allocation-smoke` for this matrix.

All arms use the same initial student and four routed teachers, T32 trajectories
per physical rollout round, native suite horizons, B64, microbatch1, LR3e-6,
four optimizer-update warmup calls, and a constant scheduler. The launcher uses
four explicitly selected physical GPUs and its private Ray instance.

| Arm | Config name | Allocation / loss weights | Reuse | Physical rounds | `--updates` | `--save-every-updates` |
| --- | --- | --- | --- | ---: | ---: | ---: |
| M0 | `libero_mopd_candidate_m0` | Uniform / equal | R1 | 150 | 150 | 25 |
| M1 | `libero_mopd_candidate_m1` | Uniform / equal | Refresh-R2 | 150 | 300 | 50 |
| M2 | `libero_mopd_candidate_m2` | Adaptive / synchronized | Refresh-R2 | 150 | 300 | 50 |
| M3 | `libero_mopd_candidate_m3` | Adaptive / synchronized | R1 | 150 | 150 | 25 |

Every arm requires `--rollout-steps 520`; the shorter native horizons for Spatial,
Object and Goal remain240/320/320. Thus every saved checkpoint corresponds to
25 additional physical rounds. Independent evaluation is separate, at rounds
0/25/50/75/100/125/150; saving does not automatically run evaluation.

For example, M1 at seed0:

```bash
outputs/runtime/bin/python toolkits/libero_gr00t/launch_production.py \
  --python outputs/runtime/bin/python \
  --models /absolute/path/to/models \
  --student /absolute/path/to/student_fewshot \
  --spatial-teacher /absolute/path/to/groot_spatial_rl \
  --output /absolute/path/to/new_m1_seed0_run \
  --config-name libero_mopd_candidate_m1 \
  --seed 0 --student-padding-profile native \
  --updates 300 --save-every-updates 50 --rollout-steps 520 \
  --timeout-seconds 172800 \
  --gpus <four_available_physical_GPU_IDs> --dry-run
```

Replace the config/update/save values according to the table and use a new output
folder for every arm/seed. `--models` must contain `object`, `goal`, and `long`
checkpoint directories. The Spatial checkpoint is provided separately. Inspect
the dry-run manifest and resolved config first. The launcher refuses existing
output folders; an eventual authorized `--run` invocation therefore needs a new
run folder. The48-hour bound in the example is an explicit timeout, not an
estimated runtime. Current artifacts remain labelled candidate/smoke-only until
the experiment recipe is actually frozen; changing the label is not validation.

Use `--seed 0`, `--seed 1`, or `--seed 2` for fresh independent repeats. This single
option sets **actor.seed, rollout.seed, env.train.seed and
runner.rollout_allocation.seed** together and records all four effective values
in `launch.json`. Omission preserves the YAML values exactly. Valid overrides are
integers0–2147483647; the upper bound leaves room for worker offsets within the
32-bit RNG seed range. This controls RNG streams, not bitwise cross-hardware
reproducibility. Do not reset optimizer state midway through a seed's run.
A resume must retain the source seed: changing it is rejected by strict recipe
comparison. Use a new run from the original student for a new seed.

M2 and M3 additionally require a completed, protocol-matched reference:

```bash
export OPD_LIBERO_TEACHER_REFERENCE=/absolute/path/to/teacher_reference.json
```

Use the same validated reference for M2/M3 and all training seeds. This reference
comes from the real20-episode/task calibration described in
[teacher_calibration.md](teacher_calibration.md), not the earlier one-episode
screen. Training seeds do not change the fixed calibration protocol. M0/M1 do not
require this environment variable. Calibration cost is recorded separately.

For standalone matched screening and the distributed50-episode/task evaluation,
see the [toolkit evaluation instructions](README.md#independent-evaluation-result-audit).
Keep the native policy profile and chosen evaluator/RNG protocol fixed across
initialization and trained checkpoints. The full evaluation must validate all
2000 unique task/state pairs before its result is accepted.
