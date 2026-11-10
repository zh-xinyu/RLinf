# Independent LIBERO evaluation through rollout workers

Use `launch_parallel_evaluation.py` on the rollout-efficiency branch. It reuses
`LiberoEnv`, `EnvWorker`, `MultiStepRolloutWorker`, `prepare_actions` and
`EmbodiedEvalRunner` from training. There is no learner, optimizer or teacher
query during student evaluation. Small worker adapters supply exact evaluation
packets and independent episode RNG streams; physics and policy inference stay
on the training implementation.

The original standalone evaluator remains a reference implementation.
`launch_evaluation.py` and its historical K8 YAML are **legacy**; do not use them
for current mixed-prefix formal experiments.

## Fixed protocol

- All 40 tasks; 50 official initial states per task, indices 0–49.
- Spatial/Object/Goal/Long: execute 5/5/5/10 actions per decision; terminate at
  first success or 240/240/320/480 physical steps.
- Full 16×32 flow latent, four ODE steps, native B1 prompts, unchanged checkpoint
  metadata/normalization and FP32 student head with BF16 inference autocast.
- Policy noise: `1234 + local_task_id * 100 + trial_id`, with a private persistent
  Torch RNG stream per episode. This matches the previous standalone evaluator
  and makes episode noise independent of GPU, slot order and concurrency.
- Reset seed 1234, ten settling steps with the gripper open. Settling steps are
  recorded separately from policy-controlled simulation steps.
- Five parallel states of one task per GPU by default, ten packets per task.
  GPUs receive disjoint task shards. Change `--envs-per-task` only to a divisor
  of the selected state count (1, 2, 5, 10, 25, 50 for the full protocol).
- Finished slots freeze physics and skip inference. Live inference stays B1;
  experimental BF16 model batching is disabled.

Parallelism changes evaluation scheduling, not the requested task/state pairs.
Every episode appears once. Missing, duplicated or incomplete outcomes cause an
error; an incomplete aggregate does not receive a success rate.

## Launch

Use a node-specific runtime wrapper from this worktree. The example below is for
**action10**; action00 is reserved for the separate RoboTwin experiment.

```bash
OPD_ROOT=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try
cd "$OPD_ROOT/outputs/worktrees/libero_gr00t_rollout_batching"
OPD_PYTHON="$PWD/outputs/runtime_libero_profile_action10_20261105_r2/bin/python"
OPD_TRAIN_CONFIG="$OPD_ROOT/outputs/libero_gr00t_formal_T64_20261107/M1_Uniform-R2_seed0/resolved_config.yaml"

"$OPD_PYTHON" -m toolkits.libero_gr00t.launch_parallel_evaluation \
  --python "$OPD_PYTHON" --config "$OPD_TRAIN_CONFIG" \
  --baseline --gpus 0 1 2 3 --envs-per-task 5 --cpus 8 \
  --output "$OPD_ROOT/outputs/libero_evaluation/initial_student"
```

This creates a **dry run**. Inspect `manifest.json` and each task's
`resolved_config.yaml`; add `--run` and choose a new output directory to execute.
GPU IDs are physical IDs. Each GPU owns a private Ray cluster for its current
single task, with owner-scoped cleanup; existing clusters/jobs are untouched.
The default GPU preflight requires 20 GiB free, and the model allocator is capped
at 16 GiB. This is an availability check and allocator cap, not a GPU reservation.
The `--cpus 8` value is **per active GPU/task**, so four tasks request 32 CPUs.

For trained weights replace `--baseline` with:

```bash
--weights "$OPD_ROOT/outputs/libero_gr00t_formal_T64_20261107/M1_Uniform-R2_seed0/libero_formal/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
```

The base checkpoint is read from the resolved training recipe. `--student` is
only for relocating that same base. Full weights are checked for exact keys,
shapes, dtypes, finite values and frozen-backbone identity before loading. A
head-only checkpoint is rejected. The update must end a complete rollout round.

For an engineering check, add `--tasks 0 --trials 40 41 --envs-per-task 2 --trace`.
A two-episode check is not a formal success-rate estimate. Keep `--trace` off for
formal evaluation: it hashes every returned action and is only a parity diagnostic.
`benchmark_evaluation_reference.py` records the same selected pairs through the
previous single-environment path for comparison.

## Outputs

- `manifest.json`: checkpoint/config hashes, source identity, seed, protocol and
  requested task/state/GPU selection.
- `task_*/resolved_config.yaml`, `launch.json`, `driver.log`, `ray.log`, worker
  logs and `status`: exact recipes, owned processes and failure diagnostics.
- `task_*/episodes.json`: one record per completed state, including success,
  suite, task/state identity, native horizon and actual simulation steps.
- `task_*/reload.json`: strict full-weight validation for trained checkpoints.
- `episodes.csv`, `summary.json`: audited raw outcomes and equal-suite aggregate.

These are the independent evaluation artifacts to retain alongside training
records. The launcher writes raw results locally and does not require W&B.
Saving a training checkpoint still does not automatically start evaluation.

## Validation on action10

The accepted Offline-Distilled-50 student was checked at task0 in each suite,
initial states40/41. All8episodes and264live B1 predictions matched the previous
standalone path **bitwise**, including both successful and timeout episodes.
Success flags and actual terminal simulation steps matched exactly. This checks
both K5 and K10; it is an engineering parity check, not an SR estimate.

The saved T64 M1 checkpoint at update50 also passed full899-tensor validation:
585unchanged backbone tensors and314changed action-head tensors. A two-packet
single-slot probe checks exact trial advancement across resets. CPU regressions
cover private episode RNG, reset IDs, completion/coverage audits, the legacy
evaluator, formal recipes and resume behavior (97tests passed).

Local evidence: `outputs/libero_parallel_eval_20261108/` in the shared project
root, including `baseline_parity.json` and the raw worker/reference traces.
A full2000-episode evaluation is still a separate formal experiment.
