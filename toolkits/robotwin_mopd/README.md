# Four-task RoboTwin MOPD pilot

This isolated pilot trains one base π0.5 student from four PACE single-task
teachers. The initial experiments have not demonstrated task-success gains. It uses the existing M3 objective, task-balanced supervision,
native control, and current-student flow refresh. See
`docs/opd/robotwin2/EXPERIMENTS_20261101.md` for the experiment plan and
`docs/opd/robotwin2/PILOT_RESULTS_20261101.md` for measured results and exact runs.

- Runtime and simulator: [setup/README.md](setup/README.md).
- Pinned checkpoints, conversion and JAX/PyTorch checks:
  [checkpoints/README.md](checkpoints/README.md).
- Actor precision: `docs/opd/robotwin2/ACTOR_PRECISION.md`.

Activate the isolated runtime. For online W&B, first load the credentials from
`WANDB_ENV_FILE` specified by the selected machine profile; the runtime does not
copy credentials or log in automatically. Keep credentials out of this repository.

```bash
source /path/to/robotwin-runtime/env.sh
source "$WANDB_ENV_FILE"
cd "$REPO_PATH"
export ROBOTWIN_OUTPUT_ROOT=/absolute/workspace/outputs/robotwin4
export USE_TF=0
python -m toolkits.robotwin_mopd.run --run-id four_task_smoke \
  --output-root "$ROBOTWIN_OUTPUT_ROOT/train" --smoke
python -m toolkits.robotwin_mopd.run --run-id four_task_uniform_r2_pilot \
  --output-root "$ROBOTWIN_OUTPUT_ROOT/train" runner.max_steps=40
```

Smoke runs one complete physical round and two student updates, then saves a
checkpoint. The initial pilot override is 20 rounds / 40 updates; the YAML's
80-update horizon is available for extension. Saving occurs every 20 updates.
The launcher defaults to eight available GPUs and refuses existing Ray clusters;
cleanup targets only processes started by that invocation. Each run stores its
resolved configuration, source snapshot, diff, commands and status. An optional
`numactl --interleave=all` prefix applies only to this launcher and descendants;
set `ROBOTWIN_NUMA_POLICY=interleave=all` to record it in launch metadata.

Standalone evaluation does not launch Ray:

```bash
CUDA_VISIBLE_DEVICES=0 python -m toolkits.robotwin_mopd.evaluate \
  --task adjust_bottle --checkpoint teacher --episodes 5 --seed 200000 \
  --output "$ROBOTWIN_OUTPUT_ROOT/eval/teacher_pilot_adjust_bottle.json"
python -m toolkits.robotwin_mopd.calibrate \
  --eval-dir "$ROBOTWIN_OUTPUT_ROOT/eval" \
  --output "$ROBOTWIN_OUTPUT_ROOT/eval/teacher_reference_pilot.json"
python -m toolkits.robotwin_mopd.summarize "$ROBOTWIN_OUTPUT_ROOT/train/RUN_ID"
```

The offline summarizer selects the first/last `--window-rounds` observed
`rollout_count` values in `optimizer_updates.jsonl`, so R2 and R4 use the same
physical-round window. Window details include exact rollout IDs, update steps,
and finite/missing/invalid metric counts; a partially logged round includes only
its recorded updates. Missing MSE values do not shift the window, and its mean
uses the available finite values. Invalid/nonfinite values make that window's
mean `null` and `loss_all_finite=false`; missing metrics without invalid values
make `loss_all_finite=null`. Updates without a valid rollout count are reported
separately and excluded from MSE windows. SR remains training-rollout SR.

Evaluate all four specialists before building the calibration. For adaptive
runs set `OPD_TEACHER_REFERENCE` to that JSON and pass
`runner.rollout_allocation.mode=adaptive algorithm.sync_suite_loss_weights=true`.
Small pilot calibration counts are not formal benchmark results.

The `simulation_steps` metric counts executed control targets, excluding frozen
terminal padding; it is not the number of SAPIEN integration substeps. Completed
episodes and optimizer updates are also available in JSONL for offline analysis.
Native resets retry only physically unstable scenes and record actual seeds;
there is no planner-success filtering.

## Shared-host GPU subset

For a four-GPU pilot, explicitly select physical devices and pass `--num-gpus 4`.
The launcher replaces `all` placement with those physical GPU IDs because RLinf
can discover GPUs outside the CUDA mask. Other placement overrides are rejected
in subset mode. Existing jobs are not stopped. For example:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m toolkits.robotwin_mopd.run \
  --num-gpus 4 --run-id specialist_init_pilot --output-root "$ROBOTWIN_OUTPUT_ROOT/train" \
  env.train.total_num_envs=16 actor.global_batch_size=128 actor.micro_batch_size=4 \
  runner.max_steps=20 runner.save_interval=20
```

This command still uses the configured base initialization unless
`actor.model.model_path` is explicitly overridden. See
`docs/opd/robotwin2/TRAIN_PATH_DEBUG_20261101.md` for the current debugging plan.

The diagnostic module `toolkits.robotwin_mopd.teacher_train_rollout` tests frozen
routed teachers through the production training rollout/env workers, without
creating an actor or optimizer. It requires a separately owner-scoped Ray
instance with matching GPU visibility. `--student-checkpoint PATH` instead tests
a single shared policy through the original mixed-batch prediction method.
