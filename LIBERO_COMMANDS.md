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

Edit `outputs/local/libero_machine.json` using a text editor. Replace its five
absolute paths: `python`, `models`, `spatial_teacher`, `student`, and `results`.
`python` must point to the prepared **runtime wrapper** `libero-runtime/bin/python`,
not `venv/bin/python` or the image interpreter. The other keys are resource/logging
settings, not learning hyperparameters:

| Machine setting | Default / meaning |
| --- | --- |
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

## 4. Launch one of the nine training runs

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

## 5. Evaluate saved checkpoints by physical round

For example, evaluate round 40 of Uniform-R4/seed1:

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
10/20/30/40/50/60/70/80/90/100 of every run. Independent evaluation is a separate
job; training does not launch it automatically.

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
