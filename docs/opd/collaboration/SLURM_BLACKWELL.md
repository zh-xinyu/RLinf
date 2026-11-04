# RoboCasa on Slurm: four or eight GPUs

This branch separates **hardware settings** from the frozen scientific recipe.
Use `release/robocasa-formal`, then check out the exact commit supplied with
an experiment assignment. The earlier `robocasa-mopd-joint-sft-full-v1` tag does
not contain this adaptation. Keep a clean, fixed checkout throughout all jobs.

The initial target is one node with four RTX PRO 6000 Blackwell GPUs, 32 allocated
CPUs, 384 GiB host RAM, and driver 580.178.04. The supplied partition name was
ambiguous: confirm the site's partition and account before submission. This
guide never assumes a partition name or modifies the system driver.

## 1. Choose hardware and software independently

| Setting | Existing A6000 machines | Blackwell Slurm target |
| --- | --- | --- |
| `num_gpus` | 8 (4 also supported) | 4 (8 also supported) |
| `runtime_profile` | `gr00t-n15-robocasa365-20261019` | `gr00t-n15-robocasa365-blackwell-cu128` |
| Torch / bundled CUDA | 2.6.0 / 12.4 | 2.7.1+cu128 / 12.8 |
| FlashAttention | 2.7.4.post1, old C++ ABI | 2.8.3.post2, C++11 ABI |
| `scheduler` | `local` | `slurm` |
| `resource_mode` | `shared` | `dedicated` |
| Global batch / microbatch per GPU | 2304 / 32 | 2304 / 32 |
| Accumulation microbatches per update | 9 on eight GPUs | 18 on four GPUs |

Four GPUs do **not** halve global batch, rollout trajectories, or learning rate.
The same four tracked YAMLs are used. Only component placement, accumulation
implied by world size, and cache reservation change. Dedicated mode disables
proactive actor cache reservation; Slurm supplies resource isolation. The model,
loss, and environment code are shared. Numerical trajectories need not be bitwise
identical across GPU counts or Torch versions. Compare all four arms on the same
hardware/profile, and keep their provenance with the results.

The new profile has passed a fresh CPU installation, version/import checks, and
CPU regression tests. **Blackwell GPU kernels and end-to-end training still need
the receiving machine's smoke test.** See [validation](VALIDATION.md) and the
[profile notes](../../../requirements/profiles/gr00t-n15-robocasa365-blackwell-cu128/README.md).

## 2. Clone and fill one machine JSON

```bash
git clone --branch release/robocasa-formal https://github.com/zh-xinyu/RLinf.git
cd RLinf
# Use the exact commit from the task sheet before installation.
git checkout <ASSIGNED_COMMIT>
mkdir -p "$HOME/opd-config"
cp toolkits/robocasa_mopd/machine.slurm.example.json "$HOME/opd-config/robocasa-slurm.json"
```

Edit the JSON, replacing every `/absolute/shared/path/...` placeholder. Keep
`hostname: "slurm"`, `scheduler: "slurm"`, `num_gpus: 4`, `ray_cpus: 32`, the
Blackwell profile, and `resource_mode: "dedicated"`. Use absolute paths visible
at the **same locations on login and compute nodes** for this checkout, machine
JSON, runtime, resources, and outputs. A compute-node-local temporary disk is
not suitable for a serial multi-job handoff unless explicitly staged each time.

- `runtime_root`: a **new** directory for the runtime and upstream sources.
- `resource_root`: teachers, student downloads, and the bound reference.
- `student_checkpoint`: the complete Joint-SFT-Full checkpoint directory; the
  template places it below `resource_root/project-checkpoints/`.
- `output_root`: a new results directory. The launcher binds `outputs/` to it.
- `wandb_mode`: `disabled`, `offline`, or `online`; W&B is optional.
- `wandb_entity`: empty when disabled; otherwise your actual writable entity/team
  slug. It may differ from your login username. Never put tokens in the JSON.

The JSON contains machine/resource settings; learning parameters remain in
`examples/embodiment/config/robocasa365_mopd_formal_{common,m0,m1,m2,m3}.yaml`.

## 3. Install, download, and check inputs

Prerequisites: Linux x86_64, glibc >=2.28, host Python >=3.9 with `venv`, Git,
build tools, EGL/OpenGL libraries, and access to Python/PyPI/GitHub/HF/RoboCasa
asset servers. Driver 580.178.04 can run this CUDA12.8 runtime; a separate local
CUDA toolkit is not needed for the prebuilt Torch/FlashAttention wheels. Ask the
cluster administrator about missing system libraries. Do not upgrade pinned
packages to resolve an installation problem.

Run preparation where cluster policy permits installation and downloads. It
does not require GPUs. The runtime installer refuses an existing target path.

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" paths
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" install
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" verify
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" download-teachers
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" download-student
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" assets
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" prepare
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" login
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" check-logging
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" preflight
```

`assets` is interactive: accept the pinned RoboCasa downloader's prompts. Assets
live under `<runtime_root>/repos/robocasa/robocasa/models/assets/`; the software
profile does not hash-lock the full asset tree. `prepare` verifies all inference
file hashes and writes a machine-local teacher reference. Do not rerun it over
an existing reference; use `preflight` for subsequent checks. Student and teacher
downloads use the fixed HF revisions in [checkpoints.json](checkpoints.json).
No separate private checkpoint transfer is required.

`login` and `check-logging` are no-ops outside online mode. In online mode the
latter writes and reads a small setup run before expensive GPU work. Its
`exclude-from-paper` tag and `launch_checks` group distinguish it from results.
The experiment project is `FORMAL_robocasa_mopd`; joining a particular team is
not required. All modes retain local TensorBoard and JSONL metrics.

## 4. Submit the mandatory smoke

From the repository root, replace `CONFIRMED_PARTITION` and add
`--account=YOUR_ACCOUNT` when required by the site:

```bash
sbatch --parsable --partition=CONFIRMED_PARTITION \
  toolkits/robocasa_mopd/slurm.sbatch \
  "$HOME/opd-config/robocasa-slurm.json" smoke
```

The template requests one node, one task, four GPUs, 32 CPUs, 384G RAM and a
14-day limit. Override `--time` if the site's limit is shorter. `srun` passes the
allocation to Ray; do not manually set `CUDA_VISIBLE_DEVICES`, start Ray, or call
`ray stop`. The launcher restricts component placement to assigned GPUs and
checks the requested GPU/CPU counts. Whole GPUs are required; MIG is unsupported.
It refuses pre-existing Ray processes/ports rather than stopping other jobs.

Smoke performs, in order:

1. Runtime verification and BF16 FlashAttention forward/backward on every GPU.
2. An optional online logging write/read check.
3. M2 training for two physical rounds / four student updates, saving step4.
4. Independent evaluation of that **saved trained checkpoint** on OpenDrawer,
   one episode, exercising checkpoint loading, EGL and simulator execution.

Inspect `slurm-robocasa-mopd-<JOB_ID>.out`, Slurm completion/exit status, and:

- `outputs/launches/setup-m2-smoke/status`: `OK`.
- `outputs/launches/setup-joint-sft-full-eval-smoke/status`: `OK`.
- `outputs/smoke/setup-joint-sft-full-eval-smoke/status`: `OK_SMOKE`.

Success in the one-episode task is not an acceptance threshold; completed
execution is. Long Slurm runs refuse missing/failed smoke records or records
from a different commit, profile, GPU count, input manifest or resource mode.
Keep the checkout fixed. After a failure, preserve logs and diagnose it before
retrying; existing output directories are never silently overwritten.

## 5. Run the full experiment and independent evaluations

The [experiment matrix](README.md#b-four-ready-to-run-configurations) is unchanged:
M0 uniform R1; M1 uniform refresh R2; M2 adaptive-sync refresh R2; M3 adaptive-sync
R1. Each starts from Joint-SFT-Full with a fresh optimizer, runs 150 physical
rounds, and saves at rounds25/50/75/100/125/150. Independent evaluation is50
episodes per task, all50 tasks, seed7, native horizons. Evaluate the initial
model once and every saved checkpoint: **25 evaluations / 62,500 episodes**.

After inspecting the smoke, preview the serial chain with its numeric job ID:

```bash
python3 toolkits/robocasa_mopd/submit_slurm.py \
  --machine "$HOME/opd-config/robocasa-slurm.json" \
  --partition CONFIRMED_PARTITION --smoke-job-id 123456 --dry-run
```

Replace `123456` with the actual successful smoke job ID, then remove `--dry-run`
to submit. Add `--account YOUR_ACCOUNT` if required, and use `--time-limit` for
a site-specific limit. This submits **29 jobs**: four training jobs, then25
independent evaluations. Each depends on the previous job's success; no training
and evaluation run concurrently. Accepted job IDs are recorded incrementally
under `outputs/submissions/`. An interrupted submission may already have queued
jobs: inspect that record before submitting anything again. If the smoke ID is
no longer retained by Slurm, use the individual commands below after checking
the retained smoke records; do not invent a predecessor ID.

Individual stages can also be submitted explicitly:

```bash
sbatch --partition=CONFIRMED_PARTITION toolkits/robocasa_mopd/slurm.sbatch "$HOME/opd-config/robocasa-slurm.json" train m0
sbatch --partition=CONFIRMED_PARTITION toolkits/robocasa_mopd/slurm.sbatch "$HOME/opd-config/robocasa-slurm.json" eval joint-sft-full
sbatch --partition=CONFIRMED_PARTITION toolkits/robocasa_mopd/slurm.sbatch "$HOME/opd-config/robocasa-slurm.json" eval m0 --round 25
```

These examples are not an additional queue to run alongside the chain. Submit
only missing stages, serially, after checking dependencies and existing outputs.
The launcher does not automatically resume a timed-out training run. Preserve
its logs and agree on recovery before rerunning.

For eight GPUs, change only `num_gpus` to8 and use `sbatch --gpus=8` for smoke
and manually submitted jobs. The chain reads the count from the machine JSON.
Use a separate output directory and a matching eight-GPU smoke. Select software
by GPU architecture; four GPUs do not inherently require the Blackwell profile.

## 6. Collect and deliver

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa-slurm.json" summarize
```

Deliver `outputs/deliverables/checkpoint_results.csv` (25 rows), all raw
`outputs/eval/<id>/` directories, and non-checkpoint files from the four
`outputs/train/<id>/` and `outputs/launches/<id>/` directories. Also retain smoke
logs, Slurm stdout/job IDs, and a credential-free machine JSON. See the
[detailed delivery contract](README.md#f-deliver-results-without-model-checkpoints).
The CSV records training/evaluation GPU counts, software profiles and commits
when present in the source records, in addition to scores and measured budgets.
Do not enter assumed values into missing measurements.

Checkpoints are **not** a deliverable; keep them locally until evaluation is
complete. W&B URLs are optional. The paper's primary x-axis is physical rollout
rounds, not optimizer updates. Cross-hardware wall-clock comparisons must label
the hardware and runtime; do not attribute those differences to the method.
