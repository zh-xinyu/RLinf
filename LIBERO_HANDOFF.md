# LIBERO experiment handoff

The current assignment is **six Offline-Distilled-50 experiments, seed0,
80rounds**, with automatic tiered evaluation. Use the existing installed runtime
and machine JSON; local GPU/CPU/microbatch/logging choices remain local.

Start with **[LIBERO_OFFLINE50.md](LIBERO_OFFLINE50.md)** for checkpoint download,
three server queues, six standalone commands, exact settings, automatic measurement, CSV paths,
recovery and timing estimates. The two Adaptive variants retain the squared-gap
law and differ only in rho0.5/1; interval5 and warmup/window10 are shared.

The legacy `run_experiment` / `run_uniform` commands retain their historical
100-round behavior. Use the new `queue_offline50` / `run_offline50` commands for
this assignment; no environment rebuild is needed. Existing teacher checkpoints
are unchanged. The restored student is linked in the current command guide.

For a fresh installation only, use [LIBERO_CLUSTER_SETUP.md](LIBERO_CLUSTER_SETUP.md).
Earlier experimental contracts and commands remain historical references rather
than the current six-job assignment. Preserve local adaptations when updating the
tracking branch `release/libero-gr00t`.

---

## Historical handoff (preserved; use the revision above)

# LIBERO collaborator handoff

Start here. The current assignment is **Uniform-R1/R2/R4, seeds 0/1/2: nine runs**.
Adaptive will be assigned after owner tuning. Environment and student acceptance
are completed jointly before long runs; model files are supplied separately.

Student candidate: [teacher-rollout SFT100 HF export](https://huggingface.co/theo-zhou/vla-mopd-checkpoints/tree/939b3f95b2f269af9242794497020bc4abd43027/libero/groot-n15/teacher-rollout-sft/checkpoint-100),
approximately7.59GB. Download the complete folder and retain its metadata/index.
The included manifest records file hashes and candidate acceptance status.

## 1. Get the code and install once

```bash
git clone --branch release/libero-gr00t https://github.com/zh-xinyu/RLinf.git libero-mopd
cd libero-mopd
git rev-parse HEAD
```

Compare the commit with the handoff message. Use `release/libero-gr00t`; the older
local `release/libero-gr00t-candidate` is not this handoff.

Follow **[LIBERO_CLUSTER_SETUP.md](LIBERO_CLUSTER_SETUP.md)** one stage at a time.
The platform provides an approved Linux image and internal pip source; no custom
image import is needed. Request four GPUs on one node, 32 CPUs and preferably 256 GB
RAM. Choose the Ampere/Ada or Blackwell profile with the owner. One small bootstrap
venv installs tools; one runtime runs both training and evaluation. Complete the
node smoke checks and common student acceptance together before the main queue.

## 2. Fill one machine file

```bash
mkdir -p outputs/local
cp toolkits/libero_gr00t/configs/uniform_machine.example.json outputs/local/libero_machine.json
```

Edit its six absolute paths: runtime wrapper `python`, `models`, `spatial_teacher`,
`student`, `results` and **`csv_dir`**. The models directory contains `object/`,
`goal/`, `long/`; Spatial and the shared student are separately specified.
`results` stores checkpoints/logs/raw evaluation; `csv_dir` is the explicitly
prepared writable NAS directory shared by all nine runs. Set GPU IDs to the
allocated job's visible IDs. Microbatch defaults to 4/card; use 8 only after joint
acceptance and keep the selected value across all runs. W&B is optional/offline.
No per-run LR/B edits, repeated shell exports or hand-filled result CSVs are needed.

## 3. Run the assigned experiments

| Reuse | Seeds | Physical rounds/run | Student updates/run |
| --- | --- | ---: | ---: |
| Uniform-R1 | 0,1,2 | 100 | 100 |
| Uniform-R2 | 0,1,2 | 100 | 200 |
| Uniform-R4 | 0,1,2 | 100 | 400 |

T64, global B512, LR3e-5 and four-update warmup/constant schedule are preset.
Start with R1/R2/R4 seed 0, then repeat with seeds 1 and 2. Run each command in an
allocated job; give simultaneous runs separate GPU allocations.

```bash
# Review a plan first: no training starts.
python3 -m toolkits.libero_gr00t.run_experiment --machine outputs/local/libero_machine.json --r 1 --seed 0
# Actual run: train, then automatically evaluate all saved checkpoints.
python3 -m toolkits.libero_gr00t.run_experiment --machine outputs/local/libero_machine.json --r 1 --seed 0 --run
```

For the other eight runs, change only `--r` to 1/2/4 and `--seed` to 0/1/2.
Names and paths are automatic. Saving is every 10 physical rounds. After all 100
rounds, the command releases training processes and evaluates the shared initial
student plus rounds 10/20/.../100, each 40 tasks x 50 official states, fixed eval
seed 1234. Shared initial results are audited/reused. Reserve time for training
**and** evaluation: the full nine-run matrix has 182,000 evaluation episodes.

If training finished but measurement was interrupted, recover without retraining:

```bash
python3 -m toolkits.libero_gr00t.run_experiment --machine outputs/local/libero_machine.json --r 1 --seed 0 --evaluate-all --run
```

## 4. Return generated results

Return the NAS **`checkpoint_summary.csv`**, **`checkpoint_results.csv`** and
`results_identity.json`. They update automatically after every complete checkpoint;
all nine runs produce 91 summary rows and 3640 task rows including one shared start.
SRs are fractions 0–1. Also retain/share raw train/evaluation logs, episode results,
resolved configs and each run's `post_training_evaluation.json`. W&B links are
optional. Keep checkpoints until evaluation/audit finish; checkpoint transfer is
not requested. Send failures/logs for joint debugging rather than changing recipes.

## 5. Later Adaptive update

Wait for the owner's release notice. After the active job using this checkout
finishes, update the tracking branch and use the same machine file/runtime:

```bash
git switch release/libero-gr00t
git pull --ff-only origin release/libero-gr00t
python3 -m toolkits.libero_gr00t.run_experiment --machine outputs/local/libero_machine.json --method adaptive --r 2 --seed 0 --run
```

The owner publishes parameters and teacher reference together; local binding,
evaluation and CSV export are automatic. No environment rebuild/path edits are
needed for those parameter-only updates. The eventual Adaptive matrix is provided
separately. Detailed command/retry options are in [LIBERO_COMMANDS.md](LIBERO_COMMANDS.md);
owner tuning/publication notes are in [LIBERO_ADAPTIVE.md](LIBERO_ADAPTIVE.md).
