# LIBERO: six Offline50 experiments, 80 rounds

Use branch `release/libero-gr00t`. This update adds new commands and static YAMLs;
it requires **no package installation, runtime rebuild, or changes to existing
machine JSONs**. The existing GPU/CPU, microbatch, runtime, model-path and logging
settings remain local. W&B is optional; disabled/offline configurations remain valid.
The legacy `run_uniform` / `run_experiment` commands keep their old behavior.

## 1. Update the existing checkout

After jobs using this checkout finish:

```bash
git status --short
git fetch origin
git pull --ff-only origin release/libero-gr00t
git rev-parse HEAD
```

Keep local adaptations. If Git refuses because of local changes, retain and merge
those changes; do not reset/clean the checkout or replace the working environment.
The update does not change dependency profiles or the existing machine template.

## 2. Stage the restored student once

All runs switch from teacher-rollout SFT100 to **Offline-Distilled-50**. Download
[the complete checkpoint](https://huggingface.co/theo-zhou/vla-mopd-checkpoints/tree/6cfe84dee38d843fff2340873da8cb1272eef474/libero/groot-n15/offline-velocity-distillation/checkpoint-50)
at revision `6cfe84dee38d843fff2340873da8cb1272eef474`. All shards, index, config and
`experiment_cfg/metadata.json` are required. Content identity is
`61263131573a9d9980dfa05b57c325fdd025929314d7b3c54e88e9544b10dda3`.

On an internet-connected machine with Hugging Face tools:

```bash
hf download theo-zhou/vla-mopd-checkpoints \
  --revision 6cfe84dee38d843fff2340873da8cb1272eef474 \
  --include 'libero/groot-n15/offline-velocity-distillation/checkpoint-50/*' \
  --local-dir /absolute/model-store/vla-mopd-checkpoints
```

Replace the illustrative destination. If the cluster cannot access HF, import
these exact files through its approved transfer route. The local student path is
`/absolute/model-store/vla-mopd-checkpoints/libero/groot-n15/offline-velocity-distillation/checkpoint-50`.
Existing official teacher files remain unchanged. The published adaptive reference
reuses the previously audited **20 episodes/task** calibration; it is not a new
50-episode calibration. Calibration/initializer collection costs are shared and
recorded separately from student-training interactions.

In the commands below replace `/absolute/offline50/hf` with this complete local
checkpoint directory, and use the path of the **existing** machine JSON. The
`--student` override is held in memory; the original JSON is never edited. If its
student path already points to these exact bytes, `--student` can be omitted.

```bash
python3 -m toolkits.libero_gr00t.run_offline50 \
  --machine outputs/local/libero_machine.json \
  --student /absolute/offline50/hf --r 1 --verify-only
```

This checks all student file hashes without GPU work. Supplying SFT100 or only an
action head fails before training. Six numerical presets are tracked under
`examples/embodiment/config/libero_mopd_*offline50*.yaml`.

## 3. Assign one queue to each four-GPU server

All runs: seed0, T64, B512, LR3e-5, four optimizer-update LR warmup steps then
constant, original M3 and fresh optimizer. K5/5/5/10, native B1 inference, metadata
adapters and four ODE steps are unchanged. Every run collects5120trajectories.

| Run | Allocator warmup / window | Interval | R | Rounds | Updates | Server |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Uniform-R1 | — | — | 1 | 80 | 80 | 1 |
| Uniform-R2 | — | — | 2 | 80 | 160 | 2 |
| Adaptive-I10-R1 | 10 / 10 rounds | 10 | 1 | 80 | 80 | 3 |
| Adaptive-I10-R2 | 10 / 10 rounds | 10 | 2 | 80 | 160 | 1 |
| Adaptive-I5-R1 | 10 / 10 rounds | 5 | 1 | 80 | 80 | 2 |
| Adaptive-I5-R2 | 10 / 10 rounds | 5 | 2 | 80 | 160 | 3 |

Both Adaptive settings use squared positive teacher/student SR gaps, rho1,
epsilon0, minimum suite quota4 and maximum quota change8 per decision. Realized
rollout quotas determine query counts; there is no additional suite loss weight.
Both start adapting at round11. I10 makes7decisions through80rounds; I5 makes14.
Their ten-round windows overlap for I5. Keep both results; neither setting is a
cross-seed replication. Each server performs160rounds/240updates in total.

First inspect a plan on each server (change `--server` to its assigned1/2/3):

```bash
python3 -m toolkits.libero_gr00t.queue_offline50 \
  --machine outputs/local/libero_machine.json \
  --student /absolute/offline50/hf --server 1
```

Then run the matching command in each server's four-GPU allocation:

```bash
# Server 1
python3 -m toolkits.libero_gr00t.queue_offline50 --machine outputs/local/libero_machine.json --student /absolute/offline50/hf --server 1 --run
# Server 2
python3 -m toolkits.libero_gr00t.queue_offline50 --machine outputs/local/libero_machine.json --student /absolute/offline50/hf --server 2 --run
# Server 3
python3 -m toolkits.libero_gr00t.queue_offline50 --machine outputs/local/libero_machine.json --student /absolute/offline50/hf --server 3 --run
```

Each queue trains its two jobs first, then automatically evaluates both. Run one
queue per four-GPU allocation. No separate evaluation command is needed normally.
For one job, use `run_offline50` instead; select `--r 1|2`, and for Adaptive add
`--method adaptive --interval 10|5`. Its `--run` trains and evaluates that job.

## 4. Automatic evaluation and result delivery

Save every10rounds. Measure rounds10/20/30/40/60 with40tasks ×20official states
0–19; measure the shared initial and round80 with40tasks ×50states0–49. Seed1234,
episode-private noise, five states/task/GPU, and the training rollout/environment
implementation remain unchanged. The initial and final **curve points use the
first twenty raw trials of their full fifty-trial evaluation**, without new physics.

Evaluation prioritizes shared initial and final results, then intermediate curves.
The initial result is locked/audited/reused through the shared NAS CSV namespace.
All three machine files must point `csv_dir` at the same existing prepared NAS
root to reuse it across servers. Each job adds6000evaluation episodes; the shared
initial adds2000once: **38,000 episodes for all six jobs**. Different/unshared CSV
roots produce separate initial measurements instead.

Existing destinations are preserved. New outputs go under:

- `<results>/offline50_80r/Offline50_80r_<method>-R<R>_seed0/`
- `<csv_dir>/offline50_80r/checkpoint_summary.csv` and `checkpoint_results.csv`:
  **7summary rows /280task rows**,50episodes/task, including one shared initial.
- `<csv_dir>/offline50_80r/curve20_checkpoint_summary.csv` and
  `curve20_checkpoint_results.csv`: **37summary rows /1480task rows**,20episodes/task.
- `results_identity.json` and `curve20_results_identity.json`, each run's `offline50_experiment.json`,
  `post_training_evaluation_n20.json`, raw logs/episodes/configs and evaluation timing.

The configured parent `csv_dir` must already exist and permit write/fsync. The
new child namespace is created automatically; no local fallback or JSON edit is
used. Adaptive-I5/I10 have distinct run labels/CSV arms. W&B, when enabled through
existing local settings, uses distinct Offline50 labels in `FORMAL_libero_gr00t`;
no login/account/team change is required.

Return the four CSVs, two protocol identities, and share raw training/evaluation
artifacts. Retain checkpoints until audits finish; transferring them is not required.

### Recovery without retraining

The evaluation-only10-to-20upgrade accepts unchanged completed80-round training
from the preceding release. Full50-state endpoint results can be reused after
identity audit. New20-state curve CSVs and the `_n20` evaluation receipt are
separate; earlier ten-state results/receipts are retained, not overwritten.
Existing ten-state intermediate measurements are not combined automatically:
the new queue produces separate twenty-state intermediate measurements. The
38,000-episode total describes a fresh campaign, excluding earlier evaluations.

Rerun the same server queue command: completed training is retained, complete
measurements are audited/reused, and an interrupted measurement retries only
unfinished tasks. Failed task directories are archived, not deleted. Existing
incomplete training stops for explicit recovery instead of being overwritten.

For evaluation-only recovery of one run:

```bash
python3 -m toolkits.libero_gr00t.run_offline50 \
  --machine outputs/local/libero_machine.json --student /absolute/offline50/hf \
  --method adaptive --interval 10 --r 2 --seed 0 --evaluate-all --run
```

Use that run's exact method/interval/R. `--config /relocated/run/resolved_config.yaml`
can identify a relocated completed run. Source/model/protocol mismatches are
rejected; do not mix code updates into an active training/evaluation session.

## Timing estimate and validation boundary

On our A6000 nodes, a complete200-episode/eight-GPU measurement took8.5minutes;
completed tasks in a separate2000-episode measurement averaged8.7minutes/task.
The older protocol suggested approximately **20–35minutes per400-episode/four-GPU
measurement** and **1–2hours per2000-episode endpoint**, not a cloud-hardware
guarantee. The current intermediate protocol uses800episodes, so those points require more
work. A rough planning allowance is **4–8hours/run** for five intermediate points
and one final endpoint; shared initial adds1–2hours once. This is an extrapolation,
not an800-episode cloud benchmark. Use the first cloud `evaluation_timing.json`
and per-task
`timing.json` to replace this estimate. Host/CPU/GPU contention and policy SR matter.

This update preserves policy numerics and dependencies. Release validation covers
CPU tests and actual plan generation; the new cloud six-run queue still needs its
first real invocation. Existing rollout/evaluation GPU evidence comes from the
same underlying implementation, not a claim that all six new runs completed.
