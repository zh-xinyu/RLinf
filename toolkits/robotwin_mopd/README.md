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
The launcher requires eight available GPUs and refuses existing Ray clusters;
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

Evaluate all four specialists before building the calibration. For adaptive
runs set `OPD_TEACHER_REFERENCE` to that JSON and pass
`runner.rollout_allocation.mode=adaptive algorithm.sync_suite_loss_weights=true`.
Small pilot calibration counts are not formal benchmark results.

The `simulation_steps` metric counts executed control targets, excluding frozen
terminal padding; it is not the number of SAPIEN integration substeps. Completed
episodes and optimizer updates are also available in JSONL for offline analysis.
Native resets retry only physically unstable scenes and record actual seeds;
there is no planner-success filtering.
