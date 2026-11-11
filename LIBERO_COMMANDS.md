# LIBERO Uniform experiments: manual commands

Use `release/libero-gr00t` at the commit recorded in the handoff receipt.
Complete [installation](LIBERO_CLUSTER_SETUP.md), stage the four teachers and
accepted student, and pass node/student admission before actual experiments.
These commands do not install packages or automatically queue nine jobs.

## 1. Set machine paths once

Run from the repository root. Copy the tracked example to a private ignored file:

```bash
mkdir -p outputs/local
cp toolkits/libero_gr00t/configs/uniform_machine.example.json \
  outputs/local/libero_machine.json
```

Edit `outputs/local/libero_machine.json` using a text editor. Replace its six
absolute paths: `python`, `models`, `spatial_teacher`, `student`, `results`, and
`csv_dir`. The last two are configured independently: `results` holds training
checkpoints/logs and raw evaluations; `csv_dir` is the explicit NAS destination
for the shared CSVs and their identity/lock files. There is no implicit CSV fallback.
`python` must point to the prepared **runtime wrapper** `libero-runtime/bin/python`,
not `venv/bin/python` or the image interpreter. The other keys are resource/logging
settings, not learning hyperparameters:

| Machine setting | Default / meaning |
| --- | --- |
| `csv_dir` | Required absolute NAS path; prepare it before actual launches |
| `gpus` | `[0,1,2,3]`, four visible GPU IDs in the scheduled job |
| `cpus` | 32, total training CPUs |
| `microbatch_size` | 4/card; select 8 only after full node admission, consistently for all nine runs |
| `wandb_mode` | `offline`; `online` or `disabled` also allowed |
| `timeout_seconds` | 604800, seven-day driver timeout; the scheduler allocation must also be long enough |
| `actor_cache_retention_gib` / `actor_cache_preallocate_gib` | 20 / 12, subject to node admission |

The models directory contains `object/`, `goal/`, `long/`. Spatial is the separate
`spatial_teacher` path. Preserve metadata and all weight shards. The same accepted
student must be used in every run. No tokens belong in this JSON. For online W&B,
use the platform's existing login/account/team configuration. Project is
`FORMAL_libero_gr00t`; run/group labels are derived automatically from R/seed.

For example, the machine JSON may contain:

```json
{
  "results": "/local/work/libero_runs",
  "csv_dir": "/nas/project/libero_mopd_results"
}
```

These are two entries within the full template, not a complete machine file.
After confirming the NAS is mounted, create the configured CSV directory once:

```bash
mkdir -p /nas/project/libero_mopd_results
```

Replace this illustrative path with the cluster's actual NAS path. Every actual
training-plus-evaluation or evaluation command checks that `csv_dir` already
exists and permits a temporary write/fsync before launching GPU work. The probe
is removed. A missing/read-only directory stops the command; the exporter does
not create an alternative local destination. Plan-only commands need no NAS write.
The configured NAS must remain accessible when results are exported after training.

The short entrypoint runs with the image's ordinary `python3`, uses only its
standard library, and invokes the exact runtime wrapper from the JSON. Activating
a venv and setting repeated shell exports are unnecessary. The private JSON stays
under ignored `outputs/`; the example and formal YAMLs are tracked.

## 2. Inspect a resolved experiment before launching

```bash
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 4 --seed 0
```

Without `--run`, it creates a plan under
`<results>/Uniform-R4_seed0_plan/`. Inspect `resolved_config.yaml`, `launch.json`
and the source/model identities. No training or Ray job starts. A plan requires
an installed runtime and staged artifacts. `--help` needs only ordinary Python.

The preset recipe is:

| Parameter | Uniform-R1 | Uniform-R2 | Uniform-R4 |
| --- | ---: | ---: | ---: |
| Rollout rounds | 100 | 100 | 100 |
| T / trajectories per round | 64 | 64 | 64 |
| Global B | 512 | 512 | 512 |
| LR | 3e-5 | 3e-5 | 3e-5 |
| Warmup optimizer updates / scheduler | 4 / constant | 4 / constant | 4 / constant |
| Optimizer updates | 100 | 200 | 400 |
| Save every optimizer updates | 10 | 20 | 40 |
| Save / evaluate every physical rounds | 10 | 10 | 10 |

Microbatch comes from the admitted machine JSON, shared across all runs. K stays
5/5/5/10; flow/loss, metadata adapters and evaluation are unchanged. Learning
settings are in `examples/embodiment/config/libero_mopd_uniform_r{1,2,4}.yaml`;
do not edit them per run or add LR/B fields to the machine JSON.

## 3. Evaluate the shared initialization once

First make an R1/seed0 training plan, then its evaluation plan:

```bash
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 1 --seed 0
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 1 --seed 0 --eval-round 0
```

After reviewing the evaluation plan and completing inference admission:

```bash
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 1 --seed 0 --eval-round 0 --run
```

This evaluates the base HF student, not a trained checkpoint. It uses the training
plan's resolved recipe if the actual run does not exist yet. Results go to
`<results>/evaluations/initial_student/`; share this single result across all nine
runs. Complete initialization acceptance before starting the formal queue.

## 4. Launch training followed by automatic evaluation

Each `--run` command trains all 100 physical rounds, waits for training's owned
Ray processes to exit, then automatically evaluates the shared initial student
and saved rounds 10/20/30/40/50/60/70/80/90/100 on the same allocated GPUs. It does
not evaluate after every optimizer update or pause training every ten rounds.
The checkpoint scores are still measured at every ten-round save boundary.

A completed shared initial evaluation is reused only after checking model/source
identities, inference protocol and full raw episode coverage. Completed matching
checkpoint evaluations can also be reused. Incomplete/stale outputs are not
silently overwritten. Retain the allocation for **training plus evaluation**;
the first run can require 22,000 evaluation episodes, later runs 20,000 each.
The command returns successfully only after all scheduled evaluations finish.

Each line below is a complete independent command. Run only the line wanted;
do not paste all nine into a shared four-GPU allocation as background jobs.

```bash
# Seed 0
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 1 --seed 0 --run
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 2 --seed 0 --run
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 4 --seed 0 --run
# Seed 1
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 1 --seed 1 --run
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 2 --seed 1 --run
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 4 --seed 1 --run
# Seed 2
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 1 --seed 2 --run
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 2 --seed 2 --run
python3 -m toolkits.libero_gr00t.run_uniform --machine outputs/local/libero_machine.json --r 4 --seed 2 --run
```

For R4/seed1, output is `<results>/Uniform-R4_seed1/`, the W&B group is
`T64_uniform_seed1`, and the name is `Uniform-R4_T64_B512_seed1`. These names are
derived from the same arguments used by training; stale shell run names are
replaced. The launcher prints the full underlying command before execution.

Existing output directories are rejected. For a repeated plan or a separately
agreed fresh retry, choose a new `--output /absolute/new/path`. This starts a new
run; it is not a resume option. Strict resume uses the lower-level launcher and
the same recipe, as described in its existing checkpoint/resume contract.

## 5. Optional separate training or checkpoint evaluation

The normal `--run` command already includes all scheduled evaluations. For a
training-only job, add `--train-only`. To finish/recheck the evaluation queue for
an already completed training run without retraining, use:

```bash
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 4 --seed 1 --evaluate-all --run
```

This reads `post_training_evaluation.json` from the training directory. Complete
matching results are audited and reused. A stage previously recorded as failed
or interrupted receives a new `_attemptN` output directory on this explicit retry;
its original outputs are retained, and the receipt identifies the current result.
If training failed or any scheduled checkpoint is missing, evaluation does not
start. This command resumes evaluation, not optimizer training.

A single-checkpoint evaluation remains available for inspection or manual work.
For example, round 40 of Uniform-R4/seed1:

```bash
# Review a plan first.
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 4 --seed 1 --eval-round 40
# Execute on an admitted allocation after that checkpoint exists.
python3 -m toolkits.libero_gr00t.run_uniform \
  --machine outputs/local/libero_machine.json --r 4 --seed 1 --eval-round 40 --run
```

The wrapper chooses update **160**, without manual checkpoint arithmetic. Results
go to `<results>/evaluations/Uniform-R4_seed1_round040/`. Repeat for rounds
10/20/30/40/50/60/70/80/90/100 of every run. Independent evaluations run as sequential child jobs after training. The normal
entrypoint queues them automatically; these single-checkpoint commands are optional.

All evaluations use 40 tasks x 50 official states, evaluation seed **1234** and
five parallel states/task/GPU. `--seed 1` identifies the training run; it does not
change evaluation RNG. The evaluator shares the training environment/rollout
implementation. With 32 machine CPUs it requests eight CPUs per active GPU.
Keep coverage audits and raw episode records before filling the CSV templates.

If a run was moved or uses a custom output path, add
`--config /absolute/relocated/run/resolved_config.yaml` to the evaluation command.
Its checkpoints must retain the `libero_uniform/checkpoints/global_step_<R*round>`
layout under that config's directory; R/seed must identify that same experiment.

See [the handoff](LIBERO_HANDOFF.md) for initialization gates, files to return,
the nine-run evaluation budget and the deferred Adaptive scope.

## 6. Automatically merged CSV results

After each complete checkpoint evaluation, the entrypoint updates two files in
the machine JSON's explicit **`csv_dir` directory**, shared across all nine runs:

| File | Rows / contents |
| --- | --- |
| `checkpoint_summary.csv` | One row per checkpoint: R/seed/round/update, Spatial/Object/Goal/Long SR and suite-macro SR, episode count, interaction counters and raw output path |
| `checkpoint_results.csv` | One row per task/checkpoint: successes, episodes and SR, with the same run/checkpoint identity and budgets; matches the task-level handoff template |

The complete nine-run matrix has 91 summary rows (90 trained checkpoints plus
one shared initial result) and 3640 task rows. SR values are fractions from 0 to 1.
Filter the summary to round100 to obtain the main-result rows for the three
training seeds. You do not need to fill these CSVs manually or copy templates into
the NAS CSV directory. The final means/std over training seeds can be computed
from these per-seed rows; training seeds are not pooled as extra evaluation states.

Writes use a shared file lock and atomic replacement. Re-exporting the same
checkpoint updates its key instead of appending a duplicate. Incomplete outcomes,
changed checkpoint identity or a different base/protocol cannot silently replace
existing scores. Source/model/protocol identities are also saved in
`results_identity.json`. Keep the same `csv_dir` for the intended nine runs;
different CSV destinations produce separate tables. Training/raw outputs retain
their independently configured `results` location.

Actual simulation counters are read from `libero_uniform/rollout_allocation.jsonl`
when available. Training wall time stays empty when not independently available;
missing cost fields are not fabricated or confused with evaluation time. Retain
the original training logs and per-checkpoint evaluation folders.

CSV export also runs after optional single-checkpoint evaluation. If raw
evaluation succeeded but export failed, `post_training_evaluation.json` records
EVALUATED rather than COMPLETE. `--evaluate-all --run` can audit/reuse that result
and retry the export. A CSV-only recovery command, using ordinary Python and no
GPU job, is also available:

```bash
python3 -m toolkits.libero_gr00t.collect_results \
  --csv-dir /absolute/nas/path/libero-mopd-results \
  --evaluation /absolute/path/results/evaluations/Uniform-R4_seed1_round040 \
  --r 4 --seed 1 --round 40 \
  --training-run /absolute/path/results/Uniform-R4_seed1
```
