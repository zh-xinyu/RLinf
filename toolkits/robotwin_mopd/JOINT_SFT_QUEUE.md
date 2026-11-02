# Joint SFT → MOPD followup queue

`joint_sft_followup.py` evaluates SFT100/200/500, uses the **predeclared SFT500** checkpoint for a fresh MOPD run, and evaluates MOPD40/80. The existing gates and training parameters remain unchanged. Run this from the RoboTwin worktree with its isolated runtime activated.

## Recover after an upstream evaluation failure

A new queue can wait for an **evaluation-only recovery supervisor**. It does not rerun the previous joint OPD training. Preserve the original failed launch/status JSON and logs.

For the 2026-11-01 recovery:

```bash
source /localscratch2/liyifa11/XyFile/opd-robotwin4/env.sh
cd "$REPO_PATH"
export ROBOTWIN_OUTPUT=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try/outputs/robotwin4
python -m toolkits.robotwin_mopd.joint_sft_followup \
  --base "$ROBOTWIN_OUTPUT" \
  --queue-tag recovery_20261101 \
  --upstream-launch "$ROBOTWIN_OUTPUT/setup/longer_opd_eval_recovery_20261101.launch.json" \
  --upstream-status "$ROBOTWIN_OUTPUT/setup/longer_opd_eval_recovery_20261101.status.json" \
  --gpu-drain-grace-seconds 120
```

The command creates new files:

- `setup/joint_sft_followup_launch_recovery_20261101.json`
- `setup/joint_sft_followup_queue_recovery_20261101.json`

The upstream launch file must contain its exact `pid` and `create_time`. The status must contain the matching `pid` and reach `COMPLETE`; that exact process must also exit. Failed/review states prevent continuation. Both upstream options must be supplied together.

Without new options, the original `20261101` queue filenames and original longer-OPD dependency remain unchanged. A queue tag only changes queue identity/status files; it does not overwrite or rename training/evaluation artifacts. Existing queue identity/status files prevent duplicate launches. Partial prior evaluations or an existing MOPD output require manual review rather than automatic overwrite.

## GPU drain grace

`--gpu-drain-grace-seconds` defaults to **0**, retaining immediate refusal when GPUs are occupied. An explicit value from 0 to 600 allows bounded polling for release after an owned job finishes. The same strict memory and compute-process checks apply every time. It does not terminate processes, reset GPUs, reduce the free-memory requirement, or move jobs to other GPUs.

During the grace period, status is `WAITING_GPU_DRAIN`. If GPUs remain occupied, the queue exits with `WAITING_GPU_REVIEW`; it does not retry indefinitely. The whole queue retains its ten-hour deadline, including owned subprocess execution. Frozen evaluations use physical GPUs 4–7; MOPD starts only after all eight GPUs pass the idle checks.

## Result interpretation

The viability gate counts eight **episodes** per task, not necessarily eight independent scenes: native reset retries can duplicate actual seeds. Gate and comparison reports include episode counts and unique actual scene-seed counts. Comparisons report whether actual seed sequences match; action noise is not fixed. SFT100/200 are descriptive and never select the initialization checkpoint.

CPU validation:

```bash
CUDA_VISIBLE_DEVICES='' USE_TF=0 python -m pytest -q \
  tests/unit_tests/test_robotwin_joint_sft_followup.py
```
