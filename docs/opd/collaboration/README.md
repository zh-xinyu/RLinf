# RoboCasa365 reproduction workflow

Use tag **`robocasa-formal-150r-handoff-20261101`** for this assignment. This revision makes
W&B optional and documents the public C10 download. It retains the unified
machine configuration, isolated run logs, and result collection from the
2026-10-31 handoff. Training algorithms, the four experiment YAMLs, the runtime
profile, and input model contents are unchanged. Old tags are retained, not moved.

The private email task sheet supplies collaborator assignments and access
information. This public guide explains how to execute them. All commands below
run from the cloned repository root. No manual exports or activation are needed.

## A. Prepare this node once

Follow [OPD_ENVIRONMENT.md](../../../OPD_ENVIRONMENT.md), in order:

1. Clone the assigned tag and check system prerequisites.
2. Copy [machine.example.json](../../../toolkits/robocasa_mopd/machine.example.json)
   to `$HOME/opd-config/robocasa.json`; fill in the hostname, four paths, and W&B
   preference (disabled by default). It contains machine settings only, not learning hyperparameters.
3. Run `install`, `verify`, and `verify --gpu`.
4. Run `download-teachers` and `assets`; download the pinned public C10 from [RESOURCES.md](RESOURCES.md#c10-input).
5. Run `prepare`, `login`, and `preflight`.
6. Run `smoke` on eight available GPUs and check both jobs succeeded.

The common command prefix is:

```text
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json"
```

Append one subcommand below. `--dry-run` before the subcommand prints the
commands without writes/downloads/GPU use. `preflight` actually resolves all
four Hydra configs but does not start training. Results are stored in the
configured `output_root`, accessed through the checkout's `outputs/` link.

## B. Four ready-to-run configurations

**The YAML files are already written. Collaborators do not create or edit them.**
Machine paths come from the JSON; fixed hyperparameters come from the YAMLs.
Each arm inherits [formal_common](../../../examples/embodiment/config/robocasa365_mopd_formal_common.yaml)
and changes only R, allocation/synchronization, and update/save counts.

| ID / command | YAML | Allocation / loss weights | R | Rounds | Updates | Save every U |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `m0` | [formal_m0](../../../examples/embodiment/config/robocasa365_mopd_formal_m0.yaml) | Uniform / equal | 1 | 150 | 150 | 25 |
| `m1` | [formal_m1](../../../examples/embodiment/config/robocasa365_mopd_formal_m1.yaml) | Uniform / equal | 2 | 150 | 300 | 50 |
| `m2` | [formal_m2](../../../examples/embodiment/config/robocasa365_mopd_formal_m2.yaml) | Adaptive / synchronized | 2 | 150 | 300 | 50 |
| `m3` | [formal_m3](../../../examples/embodiment/config/robocasa365_mopd_formal_m3.yaml) | Adaptive / synchronized | 1 | 150 | 150 | 25 |

All configs have the full filename prefix `robocasa365_mopd_` and suffix `.yaml`
in `examples/embodiment/config/`. Their resolved copies are saved with each run.

Shared controls: C10 with a fresh optimizer; training seed0; all50 tasks with
native horizons; T96=48 environments x2 epochs; B2304 (768/suite); eight GPUs,
micro32/GPU, nine accumulation microbatches/update; LR3e-5, constant scheduler,
warmup10 **updates**; M3 `analytic_local_kl`, four-step ODE, chunk16,
noise_level0.5. Loss M3 and experiment ID M3 are separate labels.

R2 refresh samples the full observation pool, draws new noise, and regenerates
student flows and teacher targets. All arms use the same suite-quota collector.
Uniform allocation is32/32/32. Adaptive settings are rho0.5, epsilon0.01,
20 uniform warmup **rounds**, window20, interval10, minimum5 episodes/task,
minimum16 trajectories/suite, maximum quota change8. Synchronized loss weights
are `alpha_s=3*q_s/96` using actual integer quotas, fixed for all R updates.
Feature-only GR00T is enabled; CUDA cache retention is0 on dedicated GPUs.

The first batch has one training seed, not a multi-seed aggregate. Do not
change hyperparameters or initialize from an old pilot optimizer.

## C. Train the four arms

Run these in tmux, serially on one node, or assign separate equivalent nodes:

```bash
set -e
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" train m0
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" train m1
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" train m2
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" train m3
```

Each command chooses its fixed YAML/run ID automatically. The launcher owns its
Ray processes and refuses existing clusters, busy GPUs, dirty tracked files,
and existing output directories. It does not stop peer jobs or silently resume.
After an interruption, preserve logs and contact the owner; do not change batch
size or rerun into the same directory.

Run IDs are `robocasa-M0-uniform-r1-seed0-formal`,
`robocasa-M1-uniform-r2-seed0-formal`,
`robocasa-M2-adaptive-sync-r2-seed0-formal`, and
`robocasa-M3-adaptive-sync-r1-seed0-formal`.

## D. Evaluate C10 and all six saved checkpoints per arm

Every checkpoint uses all50 tasks, **50 episodes/task**, seed7, native horizons,
and the same task-native protocol. Evaluate rounds25/50/75/100/125/150.
C10 round0 is evaluated once and shared across all arms. Teacher calibration
stays at the existing20 episodes/task and is not rerun.

```bash
set -e
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" eval c10
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" eval m0
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" eval m1
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" eval m2
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" eval m3
```

`eval mN` executes six evaluations in order and maps `U=R*round` to checkpoint
paths automatically. `eval c10` executes one. No `--intermediate` option is used.
There are25 evaluations and62,500 episodes in total. Training and evaluation
costs are recorded separately. Training itself does not run inline evaluation.

To select one remaining checkpoint after reviewing a failed series:

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" eval m2 --round 75
```

Existing evaluation directories are not overwritten. Completed points are not
automatically skipped or recomputed. Preserve any failed directory for diagnosis.
Run the remaining points explicitly after agreeing how to handle the failure.

Each evaluation creates `outputs/eval/<eval-id>/` with `plan.json`,
`per_task_results.csv`, `summary.json`, task metrics/logs/configs, and `status`.
A complete result has status `OK`. Independent evaluation does not automatically
populate the training W&B run.

## E. Produce the summary table

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" summarize
```

Output: **`outputs/deliverables/checkpoint_results.csv`**, with25 rows. The
`checkpoint_results.csv` sent with the email is a **blank template**, not an
experimental result. The collector can be run during progress: unavailable
measurements stay blank and statuses identify pending/incomplete results.
It recomputes scores from the exact task counts, checks the evaluation budget,
checkpoint identity, and consistency with `summary.json`. Invalid results get
`INVALID` and a reason; the command returns an error.

| Column group | Source / interpretation |
| --- | --- |
| Experiment, seed, round, U, checkpoint ID | Fixed matrix and schedule |
| Three suite SRs | Validated `per_task_results.csv`, rates in [0,1] |
| Suite-macro SR | Equal mean of the three suite SRs |
| Task-macro SR | Mean across all50 tasks;18/16/16 suite task counts |
| Expected trajectories | Nominal budget `96*round`, not a measured result |
| Training trajectories / simulation steps | Matching update in `optimizer_updates.jsonl` |
| Training round seconds | Sum of TensorBoard `time/rollout_round` through that point; excludes startup and independent evaluation |
| Weight hashes, eval directory | Evaluation plan and local result files |
| W&B URL | Optional; extracted when available, otherwise leave blank |
| Status / notes | Missing or invalid data, never fabricated zero scores |

Do not confuse `summary.json`'s task-macro SR with suite-macro SR. The latter
is `(a+b+c)/3`; task-macro is `(18*a+16*b+16*c)/50`. Fifty evaluation episodes
are not fifty training seeds. Full job wall time can be reported separately
from accumulated round time.

## F. Deliver results without model checkpoints

1. **The filled25-row CSV.** Check status and notes before sending.
2. **Raw evaluation results:** all25 `outputs/eval/<eval-id>/` directories.
   Include per-task success counts/denominators, summary, plan, status, and logs.
   Do not omit failed tasks or provide only rounded percentages. Videos are optional.
3. **Training materials:** non-checkpoint files in each `outputs/train/<run-id>/`
   plus `outputs/launches/<run-id>/`: resolved config, JSONL metrics, TensorBoard
   files, driver/Ray logs, node/software provenance, and failure/restart notes.
   Include local W&B files or accessible run URLs only if W&B was used.

**Do not deliver weights, optimizer states, or remote checkpoint access.**
Exclude directories named `checkpoints` when packaging training materials.
Keep them locally long enough to complete the required evaluation.
The runner nests them at
`outputs/train/<run-id>/<run-id>/checkpoints/global_step_<U>/`; the unified
entrypoint handles this path, so collaborators do not calculate it manually.
Each run's JSONL/TensorBoard/W&B metrics are isolated in `outputs/train/<run-id>/`.

The paper's main performance axis is **rollout rounds**. Retain simulation
steps and time for secondary comparisons. Training W&B steps are optimizer
updates; equal U is not equal rollout budget across R1/R2. `time/step` means one
rollout plus R updates, while `time/amortized_update` divides that time by R.
Training success is a controller signal, not independent evaluation performance.

## W&B and validation

W&B is optional. The template uses `wandb_mode: "disabled"` and
`wandb_entity: ""`; training then uses only TensorBoard and JSONL, without a
W&B login or backend. Offline mode keeps local W&B records without logging in.
For online mode, choose your own username or any team you can write to and log
in with your own account. Joining the project owner's team is not required.
Online project: `RoboCasa_MultiTeacher_Diagnostics`; group: `robocasa_formal_150r`.
`WANDB_PROJECT` alone does not override the explicit YAML project.

The same CSV schema and raw-file requirements apply to all modes. A missing
`wandb_url` does not prevent a result from being marked `COMPLETE`.
See [VALIDATION.md](VALIDATION.md) for checks already performed. A receiving-machine eight-GPU smoke is required. RoboTwin
uses a separate branch/profile and does not change this reproduction bundle.
