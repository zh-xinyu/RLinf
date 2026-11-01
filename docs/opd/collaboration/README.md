# RoboCasa365 experiment guide

This is the versioned technical guide for collaborators and coding agents.
Machine assignments, input download access, and delivery locations belong in the
separately distributed email task sheet, which is not tracked in Git.

Training code is fixed at tag **`robocasa-formal-150r-20261031`**, commit
`42d22dd446582466c5bf0096e57d5050ceff6b91`. Subsequent documentation updates do
not move that tag. The current assignment evaluates **every checkpoint at
50 episodes per task**, superseding the tagged guide's smaller intermediate
budget. No training or evaluation code change is needed for this choice.

## 1. Clone and configure the environment

```bash
git clone --branch robocasa-formal-150r-20261031 https://github.com/zh-xinyu/RLinf.git
cd RLinf
git rev-parse HEAD
```

Install `gr00t-n15-robocasa365-20261019` using the
[environment guide](../../../OPD_ENVIRONMENT.md). Skip its clone commands if
you already checked out the release. Complete software verification and the
`--gpu` check. Personal node files, `shs/`, and `toolkits/mopd/` are not required.
If installation fails, preserve logs rather than upgrading pinned dependencies.

Follow the [resource guide](RESOURCES.md) to download teachers, receive C10,
prepare simulator assets, verify model hashes, and bind the teacher reference.
Use the existing calibration at 20 episodes per task; do not recalibrate
independently on each machine.

Create a trusted machine-local shell file outside the repository:

```bash
export OPD_RUNTIME_ROOT=/absolute/path/opd-runtime
source "$OPD_RUNTIME_ROOT/activate.sh"
export OPD_STUDENT_CHECKPOINT=/absolute/path/C10/checkpoint-60000
export OPD_MODEL_ROOT=/absolute/path/teachers/gr00t_n1-5/foundation_model_learning
export OPD_TEACHER_REFERENCE=/absolute/path/inputs/teacher_reference.json
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
# Set WANDB_ENTITY to the agreed team slug; authenticate with your own account.
```

Source this file before launching, then enter the fixed checkout. If the
checkout moves, correct its path in the runtime's `activate.sh`. Use
`wandb login` for authentication; do not commit credentials.

The recipe assumes a dedicated single node with eight 48-GB GPUs, sufficient
CPU/RAM, and storage for local checkpoints. Outputs default to `outputs/`;
this directory can be linked to persistent storage before starting runs.

## 2. Experiment matrix

| ID | Rollout allocation | Suite loss weights | R | Updates U | Save interval U |
| --- | --- | --- | ---: | ---: | ---: |
| M0 | Uniform | Equal | 1 | 150 | 25 |
| M1 | Uniform | Equal | 2, Refresh | 300 | 50 |
| M2 | Adaptive | Synchronized with allocation | 2, Refresh | 300 | 50 |
| M3 | Adaptive | Synchronized with allocation | 1 | 150 | 25 |

Shared controls:

- Fresh C10 weights and optimizer; training seed 0; all 50 tasks with native horizons.
- **150 physical rollout rounds**, T96 = 48 environments x 2 epochs per round;
  14,400 training trajectories per run.
- B2304, 768 samples per suite; eight ranks, microbatch32 per GPU, nine
  accumulation microbatches per optimizer update.
- LR3e-5, constant scheduler, warmup10 **student updates**; M3
  `analytic_local_kl`, four-step ODE, chunk16, noise_level0.5. The loss name M3
  and experiment ID M3 refer to different things.
- R2 refresh uses the full observation pool, new noise, the updated student's
  flow trajectories, and newly queried teacher targets.
- Every arm uses the same `suite_quota` collector. M0/M1 retain 32/32/32 quotas.
- Adaptive settings: rho0.5, epsilon0.01, 20 uniform warmup **rounds**, a
  20-round window, decisions every10 rounds, at least5 episodes per task,
  minimum16 trajectories per suite, and maximum quota change8 per decision.
  Loss weights are `alpha_s = 3*q_s/96`, fixed across the round's R updates.
- All arms use feature-only GR00T and `cuda_cache_retention_gib=0`.
  Do not change microbatch or other controls in only one arm.

W&B project: `RoboCasa_MultiTeacher_Diagnostics`; group: `robocasa_formal_150r`.
Set `WANDB_ENTITY` to the shared team supplied by the project owner. The same
project name under different entities does not identify a shared project.
The code explicitly supplies its project name, so `WANDB_PROJECT` alone does
not override it. Agree any logging configuration change before running.
The first assignment covers four runs at seed0; additional training seeds are
scheduled separately.

## 3. Verify and train

Check resolved configuration without starting Ray or using GPUs:

```bash
python -m toolkits.robocasa_mopd.run train m2 --run-id check-m2 --dry-run
```

On eight available GPUs, complete two physical training rounds and a
single-task evaluation smoke. Training smoke retains the full task collector
and native horizons, limits training to two rounds, and disables W&B.

```bash
python -m toolkits.robocasa_mopd.run train m2 --run-id setup-m2-smoke --smoke
python -m toolkits.robocasa_mopd.run eval --run-id setup-c10-eval-smoke --smoke
```

The launcher starts and cleans up only its own Ray processes. It refuses an
existing cluster or busy GPUs; never stop another user's cluster. Run one
launcher per node. For serial training in tmux:

```bash
set -e
python -m toolkits.robocasa_mopd.run train m0 --run-id robocasa-M0-uniform-r1-seed0-formal
python -m toolkits.robocasa_mopd.run train m1 --run-id robocasa-M1-uniform-r2-seed0-formal
python -m toolkits.robocasa_mopd.run train m2 --run-id robocasa-M2-adaptive-sync-r2-seed0-formal
python -m toolkits.robocasa_mopd.run train m3 --run-id robocasa-M3-adaptive-sync-r1-seed0-formal
```

Equivalent nodes may run different arms concurrently. Keep the parameters
fixed. After interruption or OOM, preserve logs and contact the project owner;
the launcher does not silently resume, reduce the batch, or restart training.
Resume requires checking the complete rollout boundary and interaction costs.

## 4. Independent evaluation

Evaluate saved rounds **25/50/75/100/125/150** and a shared C10 round0 baseline.
Every evaluation uses all50 tasks, **50 episodes per task**, seed7, and native
horizons. Omit `--intermediate`; successful runs have `mode=formal` and `OK`.
There are 25 evaluations and 62,500 episodes across the four arms plus C10,
excluding smoke and calibration. Report this cost separately from training.

Training saves checkpoints without inline evaluation. Evaluate afterward or on
another available node. Keep local checkpoints until their evaluations finish;
**trained checkpoint files are not required deliverables**.

```bash
python -m toolkits.robocasa_mopd.run eval --run-id robocasa-C10-round0

# Example: M2 round25 corresponds to U50; M0/M3 round25 corresponds to U25.
python -m toolkits.robocasa_mopd.run eval \
  --run-id robocasa-M2-seed0-round25 \
  --weights "$PWD/outputs/train/robocasa-M2-adaptive-sync-r2-seed0-formal/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"

python -m toolkits.robocasa_mopd.run eval \
  --run-id robocasa-M2-seed0-round150 \
  --weights "$PWD/outputs/train/robocasa-M2-adaptive-sync-r2-seed0-formal/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
```

Use `U=R*round` for every other checkpoint. Evaluate all scheduled checkpoints,
not only those with high training success. Independent evaluation writes local
JSON/CSV files; it does not automatically populate the training W&B run.

## 5. Deliverables

Deliver the following three parts. Do not upload model weights or optimizer
states, and do not include `checkpoints/` in the training-material archive.
No remote access to trained checkpoints is required.

### Summary table

One CSV row per evaluated checkpoint: the shared C10 baseline and six rows per
arm, for25 rows total. Include experiment/seed, round, update count, three suite
success rates, suite-macro SR, task-macro SR, cumulative training trajectories,
simulation steps, training time, evaluation identifier, and W&B URL.
Use rates in [0,1] in CSV; display percentages in presentation tables.

For suite rates a, b, c, suite-macro SR is `(a+b+c)/3` and task-macro SR is
`(18*a+16*b+16*c)/50`. The top-level `summary.json` `success_once` is task-macro
SR. Fifty evaluation episodes are not fifty independent training seeds.

### Raw evaluation results: evidence for checkpoint performance

Deliver `outputs/eval/<eval-id>/` for all25 evaluations, including:

- `per_task_results.csv`: task/suite, episode count, success count, and SR.
- `summary.json`: suite and overall aggregates.
- `plan.json`: evaluated weight hash, task/protocol settings, seed, and source provenance.
- Per-task metrics, resolved configs, diagnostic logs, and completion status.

Do not drop failed tasks or return only rounded percentages/screenshots.
Videos are not required. Checkpoint identity and weight hashes remain in the
metadata even though model files are not transferred.

### Training materials: evidence for the learning process

Deliver non-checkpoint files from `outputs/train/<run-id>/` and
`outputs/launches/<run-id>/`, including:

- Resolved config, code/profile identifiers, machine details, driver/Ray logs,
  and interruption/restart notes.
- `optimizer_updates.jsonl`: update-level losses and diagnostics.
- `rollout_episodes.jsonl`: task/suite outcomes and actual interaction counts.
- `rollout_allocation.jsonl`: quotas, controller decisions, and budgets.
- W&B run links and available local W&B/TensorBoard metric files.

A shared W&B workspace is useful for monitoring loss, per-suite success,
allocation/weights, LR, time, and interaction counts. It does not replace the
raw files. Large collections of manually exported screenshots are unnecessary.

Main performance plots use **rollout rounds**; retain actual simulation steps
and wall time for secondary comparisons. W&B training steps count optimizer
updates, so equal update counts are not equal rollout budgets across R1/R2.
`time/step` is one rollout plus R updates; `time/amortized_update` divides by R.
Training success is a controller/diagnostic signal, not independent evaluation.

## Validation boundary

See [VALIDATION.md](VALIDATION.md). Receiving-machine GPU smoke checks are
required before long runs. RoboTwin development uses a separate branch/profile
and does not modify the frozen training release or upgrade this runtime.
