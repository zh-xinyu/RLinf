# LIBERO specialist calibration for Adaptive-Sync

M2 and M3 require one shared, complete calibration reference. The earlier
one-episode/task screening is insufficient. The collector below uses the existing
native teacher probe, preserving each specialist's scalar state metadata, fixed
text padding, BF16 head, four ODE steps, and eight executed actions. Calibration
measures the native specialist; it does not evaluate a transported teacher SDE.

Prepare from a resolved candidate training configuration. This CPU command hashes
all four teachers' inference files and captures the relevant local sampler/model
source before any episodes run. Use a new output directory and the same isolated
runtime as the pilot:

```bash
outputs/runtime/bin/python toolkits/libero_gr00t/teacher_reference.py prepare \
  --config /absolute/path/to/resolved_config.yaml \
  --python outputs/runtime/bin/python \
  --output /absolute/path/to/teacher_calibration \
  --episodes-per-task 20
```

Run each of `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` once.
Select an available physical GPU explicitly for each process. Each invocation
requires at least 30 GiB free before startup and limits its child probe's PyTorch
allocator to 24 GiB; that cap is not a reservation. The four runs may execute
sequentially or on four separately selected GPUs. Do not overlap another job on
the same card merely because this command has a memory cap.

```bash
CUDA_VISIBLE_DEVICES=<available_GPU> outputs/runtime/bin/python \
  toolkits/libero_gr00t/teacher_reference.py run-suite \
  --capture /absolute/path/to/teacher_calibration --suite libero_spatial
```

The default timeout is four hours per suite. The wrapper only terminates its own
child process group. A failed or interrupted suite leaves its status and raw logs;
use a new capture for a clean rerun rather than appending or silently replacing
those records. Source or teacher content changes after preparation are rejected.

After all four succeed:

```bash
outputs/runtime/bin/python toolkits/libero_gr00t/teacher_reference.py finalize \
  --capture /absolute/path/to/teacher_calibration
```

`teacher_reference.json` is written only after validating all 800 unique episodes
(40 tasks × 20 official initial states), exact suite routing, native horizons,
first-success/timeout completion, the recorded invocation, unchanged teacher and
source hashes, and the shared training-signature validator. It contains task-level
success counts and policy simulator calls, separate settling calls, and elapsed
wall time spanning the suite runs. `simulation_steps` follows the runner's existing
calibration convention of policy calls; settling calls remain separately recorded.
Set `OPD_LIBERO_TEACHER_REFERENCE` to this file when composing M2/M3.

The sampler uses official initial-state indices 0–19 and per-episode action noise
seed `1234 + local_task_id * 100 + trial`. The environment seed is fixed at1234.
Training uses task-occurrence initial-state cycling and its own reset/noise seeds;
calibration estimates teacher capability under the same observation/action/horizon
contract, not paired training trajectories. These states are not held out from
training and this calibration is not the formal 50-episode/task evaluation.

Keep the entire capture directory, including `manifest.json`, the resolved
training config, per-suite raw JSONL/status/console logs, and final reference.
Checkpoint identity is content-based, not merely a directory name. The current
reference also binds local absolute checkpoint paths: relocation needs explicit
validation and regeneration of the path-bound reference, never hand editing.

The wrapper and exporter have CPU regression tests. A completed GPU calibration
and an actual M2/M3 production smoke remain required before declaring adaptive
LIBERO experiments ready. Existing runtime dependencies are reused; the manifest
captures local policy source and checkpoint content, not a portable environment
lockfile or remote execution attestation.

## Bounded adaptive integration smoke

After the actual calibration is complete, use `--allocation-smoke` with M2 or M3
to exercise allocation and synchronized loss weighting without waiting20 rounds.
This flag leaves formal YAMLs unchanged and applies only these controller values:
`warmup_rounds=2`, `window_rounds=2`, `interval_rounds=1`, `min_task_episodes=1`,
and `rho=1.0`. Epsilon0.01 and the quota-change bound3 remain unchanged.

The first two T32 rounds provide16 episodes/suite and cover all10 tasks. The first
eligible decision is therefore before round3. At least three complete rounds and
full native horizons are required; the launcher rejects uniform arms, truncated
rollouts, and shorter jobs. Four rounds are recommended to check consecutive
rounds. For M2 use eight updates; for M3 use four. For example, after setting the
real `OPD_LIBERO_TEACHER_REFERENCE` file:

```bash
outputs/runtime/bin/python toolkits/libero_gr00t/launch_production.py \
  --python outputs/runtime/bin/python \
  --models /absolute/path/to/models \
  --student /absolute/path/to/student_fewshot \
  --spatial-teacher /absolute/path/to/groot_spatial_rl \
  --output /absolute/path/to/adaptive_integration_smoke \
  --config-name libero_mopd_candidate_m2 --allocation-smoke \
  --rollout-steps 520 --updates 8 --timeout-seconds 7200 \
  --gpus <four_available_physical_GPU_IDs> --run
```

The launch manifest explicitly records the overrides and marks the run as smoke
only, with `performance_evidence=false`. Do not reuse this shortened success
window as the formal method's tuning result.

Acceptance requires a real `decision=true, reason=gap` in round3 or later and at
least one nonuniform quota vector. Quotas must sum32, remain at least4/suite,
change by at most3/suite/decision, and match actual completed episode counts.
Each optimizer record's `train/actor/suite_loss_weight/<suite>` must equal that
round's quota divided by8 (mean-one weights); both M2 inner updates must use the
same weights. Also check finite losses/gradients, the expected update/interaction
counts, final checkpoint reload, and preserved per-worker suite coverage.

Unequal measured gaps are not guaranteed. If all quotas remain8 because actual
gaps are equal or integer rounding preserves uniformity, report the nonuniform
path as unexercised and extend with real episodes; never fabricate a reference
or silently force quotas to declare the adaptive smoke successful.
