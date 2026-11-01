# GR00T N1.5 RoboCasa365 formal evaluation

This directory defines the frozen, sole protocol for future GR00T N1.5
RoboCasa365 target-task evaluations: `rlinf-robocasa365-target50-native-v1`.
Run it from this checkout, not from a cold-start repository. Keep each score
with its plan and protocol metadata; earlier v1/v1-compat exploratory scores
are not interchangeable with this task-native protocol. Portable setup and
launch commands are in the [collaborator guide](../../docs/opd/collaboration/README.md).

## Protocol

- The checked-in [manifest](target50_manifest.json) fixes the 18 atomic-seen,
  16 composite-seen, and 16 composite-unseen tasks and their RoboCasa-native
  horizons. Each task is evaluated separately with 50 episodes: five parallel
  environments for ten reset epochs (2,500 episodes total).
- Use one GR00T N1.5 policy, BF16 inference, four denoising steps, 16-action
  chunks, and the RoboCasa PandaOmron observation/action adapters in the
  [YAML](robocasa365_mopd_gr00t_formal.yaml). The environment stops advancing
  physics at its native task horizon. A final partial chunk is padded only for
  RLinf’s fixed-size rollout tensor. Success does not end an episode early;
  `success_once` records whether success occurred at any point during it.
- Seed 7 fixes task assignment and RoboCasa's seeded reset sequence. The
  formal-only rollout worker also reseeds Python, NumPy, and PyTorch after
  loading weights, fixing GR00T's flow-noise stream. This controls RNG inputs,
  but does not promise bitwise determinism across hardware. The result is the
  mean `success_once` over exactly 50 episodes per task; suite and overall
  scores weight each task equally. Report `success_at_end` separately.
- The plan records the RLinf commit and relevant file hashes, official GR00T
  and RoboCasa commits, selected Python package versions, model metadata,
  all base-model shards, and the RLinf weight override hash. A run refuses
  source or runtime drift.
  Keep the plan and result directory together when sharing a score.

The runtime must contain the official GR00T and RoboCasa checkouts at the
commits enforced in `formal_eval.py`. The entrypoint reads the node-local
`OPD_RUNTIME_ROOT`; it does not install or download models. The base checkpoint
supplies GR00T architecture and normalization metadata. Use `--weights` for
an RLinf `full_weights.pt` policy; omit it only when evaluating the base HF
checkpoint itself.

## Intermediate checkpoints

`plan --intermediate` keeps the same 50 tasks, native horizons, seed, and model
semantics, but uses 20 episodes per task (five environments, four epochs). It
records `mode=intermediate` and `OK_INTERMEDIATE`; the default final protocol
remains 50 episodes with `mode=formal`. Do not pool these budgets without
labeling them. Both modes validate their exact task and episode counts.

## Run

Set up the OPD runtime, assets, and a *private* single-node Ray cluster first.
Ray must advertise the selected GPU count, and
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` must be set before starting it.
The entrypoint will not start or stop Ray. On a shared machine, use an
owner-scoped local launcher, if available, to avoid affecting other users’ clusters.

```bash
export OPD_RUNTIME_ROOT=/path/to/local/opd
source "$OPD_RUNTIME_ROOT/venv/bin/activate"
export RAY_ADDRESS=127.0.0.1:6379
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl

python evaluations/robocasa365/formal_eval.py plan \
  --run-id HOST-DATE-CHECKPOINT \
  --base-checkpoint /path/to/base/checkpoint \
  --weights /path/to/full_weights.pt \
  --gpus 0
python evaluations/robocasa365/formal_eval.py run \
  --plan outputs/eval/HOST-DATE-CHECKPOINT/plan.json
```

Use `--gpus 0,1,...` only for devices that are genuinely free and advertised
by the private Ray cluster. The dispatcher assigns whole tasks to GPUs; it
does not change the five-env/50-episode budget. It first completes one task,
then distributes the rest. Use a unique run ID: an existing result directory
is never overwritten. Outputs live in Git-ignored `outputs/eval/<run-id>/`:
`plan.json`, per-task resolved configs and logs, `metrics.json`,
`per_task_results.csv`, and `summary.json`. The overall status becomes `OK`
only after all 50 tasks pass episode-count, native-horizon, and metric checks.

For a short integration check, append `--smoke-task OpenDrawer
--smoke-episodes 1` to `plan`. This writes under `outputs/smoke/` and finishes
with `OK_SMOKE`; it is never a formal score. A change to tasks, budget,
environment semantics, model sampling, or metrics requires an explicitly named
mode/protocol and a fresh evaluation of every checkpoint being compared. Only
the documented intermediate mode may use the smaller 20-episode budget.
