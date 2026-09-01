# Runbook — GR00T N1.5 on RoboCasa365 in RLinf

The working setup, end to end: what the pipeline is, how to build it on a fresh node, and
the exact commands to evaluate and to train. Written 2026-08-24, after the inference path
was verified and the two gr00t packages were consolidated into one.

Everything below has been run. Where a number is quoted it was measured, not expected.

---

## 1. What the path actually is

RLinf does **not** vendor GR00T. It subclasses the installed `gr00t` package, and both
subclasses were already upstream at the merge base — we did not invent them:

```
gr00t (NVIDIA Isaac-GR00T, n1.5-release, editable install in the venv)
  │
  ├── GR00T_N1_5                          ──subclassed by──▶  GR00T_N1_5_ForRLActionPrediction
  └── FlowmatchingActionHead              ──subclassed by──▶  FlowMatchingActionHeadForRLActionPrediction
                                                                 (rlinf/models/embodiment/gr00t/gr00t_n1d5/)
```

One request flows like this:

```
Robocasa365Env  (rlinf/envs/robocasa365/)                     robosuite / MuJoCo, CPU
   │  obs: 3 cameras @256x256, 16-d state, task_description, task_groups
   ▼
convert_robocasa_obs_to_gr00t_format   OBS_CONVERSION["robocasa365"]
   │  (rlinf/models/embodiment/gr00t/simulation_io.py)
   ▼
RoboCasaPandaOmronDataConfig  →  modality transforms  →  Eagle VL backbone  →  vl_embs
   │  (our local port; 16/16 fields verified against the fork's panda_omron before it was removed)
   ▼
FlowMatchingActionHeadForRLActionPrediction.get_rl_action
   │  4 Euler steps, horizon 16, action_dim 32 (12 valid), embodiment_id 31
   ▼
convert_to_robocasa_action  →  prepare_actions  →  env.step
      gripper and base_mode thresholded at 0.5 to -1 / +1
```

**Training and evaluation share this path.** The reference delegation that once bypassed it
for evaluation is deleted, so an evaluation number says something about the training path.

### The one model-side branch that differs by mode

`sample_mean_var_val` picks a transition per denoise step:

| `noise_method` | `joint_logprob` | rollout | used by |
|---|---|---|---|
| `flow_ode` | `True` | fully deterministic Euler | **M1, M3** |
| `flow_sde` | `False` | one stochastic step (Hybrid SDE) | M3-Hybrid, M4 |

`joint_logprob: True` means *every* step takes the `mode="train"` branch — which under
`flow_ode` is deterministic. It is not a synonym for "more noise". `flow_ode` +
`joint_logprob: False` raises, deliberately.

---

## 2. Environment setup on a fresh node

Three steps. `install.sh` covers only the first. Everything goes on node-local NVMe:
NFS cold reads here run at 11-26 MB/s and Ray's startup budgets are tens of seconds, so a
venv on NFS makes `ray start` fail outright.

```bash
REPO=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try
VENV=/localscratch2/$USER/venvs/gr00t_robocasa365
CKPT=/localscratch2/$USER/models/robocasa365
cd "$REPO"

# 1. venv  (~31 min; installs gr00t + robocasa + robosuite + flash-attn)
bash requirements/install.sh embodied --venv "$VENV" \
     --model gr00t --env robocasa365 --no-root

# 2. RoboCasa kitchen assets  (~13 GB; install.sh does NOT do this and the env cannot reset without them)
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
yes | python -m robocasa.scripts.download_kitchen_assets

# 3. checkpoints  (~29 GB; optimizer.pt is skipped -- it is never loaded)
rsync -a --exclude 'optimizer.pt' \
      /egr/research-actionlab/liyifa11/models/robocasa365/ "$CKPT/"
```

**The repo is not installed into the venv.** `install.sh` defaults to
`--no-install-project`, so `rlinf` is imported from the working directory via `PYTHONPATH`.
One repo checkout therefore serves any number of venvs, and switching venvs changes no source
file — which is also why the venv can live on NVMe while the repo stays on the shared mount.

There is **no gr00t swap step.** `install.sh` installs NVIDIA's official `n1.5-release` and
that is the only gr00t involved. The RoboCasa benchmark fork was removed on 2026-08-23; the
loader uses our local `RoboCasaPandaOmronDataConfig` instead of its `panda_omron`.

### Verify before running anything

```bash
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python - <<'PY'
import gr00t, robocasa, robosuite
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING as T
from gr00t.experiment.data_config import DATA_CONFIG_MAP as M
print('gr00t            :', gr00t.__file__)          # .../venvs/.../gr00t/gr00t/__init__.py
print('panda_omron      :', 'panda_omron' in M)      # False -> official package, local port in use
print('new_embodiment id:', T['new_embodiment'])     # MUST be 31
PY
```

`new_embodiment` mapping to anything but **31** means the wrong per-embodiment projectors
will be used — silently, with every weight loading fine. That was the root cause of the
original zero-success investigation.

### Point the configs at the new paths

- `examples/embodiment/config/robocasa365_mopd_gr00t.yaml` — `rollout.model.model_path`,
  `actor.model.model_path`, and the three `rollout.expert_models.*.model_path`
- `evaluations/robocasa365/robocasa365_mopd_gr00t_eval.yaml` — `rollout.model.model_path`
- `toolkits/mopd/run_mopd.sh`, `toolkits/mopd/run_eval_mopd.sh` — the `VENV=` line

---

## 3. Evaluation

```bash
cd "$REPO"
CKPT=/localscratch2/$USER/models/robocasa365/gr00t_n1-5/foundation_model_learning

bash toolkits/mopd/run_eval_mopd.sh <tag> "" \
  "rollout.model.model_path=$CKPT/target_posttraining/atomic_seen/checkpoint-60000" \
  "env.eval.task_filter=[TurnOnMicrowave,CloseToasterOvenDoor]" \
  "env.eval.max_episode_steps=464" \
  "env.eval.max_steps_per_rollout_epoch=464" \
  "env.eval.total_num_envs=16" \
  "runner.logger.experiment_name=robocasa365_<tag>"
```

The script purges and *verifies* Ray (a real remote task, not `ray status`), launches the
driver detached, and runs a silence watchdog. Results land in
`outputs/mopd/eval.<tag>.out`, with a terminal status in `outputs/mopd/eval.<tag>.status`.

**Horizons must be multiples of `num_action_chunks` (16)** — `validate_embodied_cfg` asserts
it. Values that cover each suite's own tasks:

| suite | tasks | horizon |
|---|---|---|
| `atomic_seen` | TurnOnMicrowave, CloseToasterOvenDoor | 464 |
| `composite_seen` | ScrubCuttingBoard, RinseSinkBasin | 1360 |
| `composite_unseen` | CuttingToolSelection, CategorizeCondiments | 1664 |

### `episode_horizon_source` changes what the number means

- `task_horizon` (standalone eval default) — each task runs its own native horizon, so short
  tasks finish early and `auto_reset` starts another episode. 24 envs produced **40**
  trajectories. `success_once` averages over episodes, so this **over-weights short tasks**.
- `max_episode_steps` (in-training validation) — one episode per env, every task weighted
  equally. 48 envs produced **48** trajectories.

The same undistilled student scores **0.35** under the first and **0.2708** under the second.
Never compare across the two. Per-suite runs are unaffected (all had envs = trajectories).

---

## 4. Training

```bash
cd "$REPO"
bash toolkits/mopd/run_mopd.sh robocasa365_mopd_gr00t <tag> [hydra overrides...]
```

A short smoke run:

```bash
bash toolkits/mopd/run_mopd.sh robocasa365_mopd_gr00t smoke \
  "runner.max_steps=2" "runner.val_check_interval=1000" "runner.save_interval=1000"
```

### Why `total_num_envs: 64`

RoboCasa365 is a CPU/MuJoCo simulator, one process per env. Measured with random actions
(env only, no policy) on the six pilot tasks, all in **one** pool:

| envs in one pool | reset | env-steps/s | RSS/env |
|---|---|---|---|
| 8 | 14 s | 101 | ~2.9 GB |
| 16 | 17 s | 148 | ~2.9 GB |
| 32 | 31 s | 258 | ~2.9 GB |
| 64 | **> 9 min, never finished resetting** | — | — |

Scaling within one pool is poor and breaks down entirely by 64, and **RAM is the binding
constraint, not GPU** — ~2.9 GB per env process took host RAM from 63 GB to 246 GB at 64 envs.

This matters less than it looks because RLinf splits `total_num_envs` across 8 env workers,
each with its own pool, so the operating point is the *per-worker* count. Eight pools of 8
aggregate far better than one pool of 64. Hence 64 total (8 per worker, ~190 GB of env RAM),
which leaves room for the student and the three cached teachers (~23 GB per actor rank).

**Do not override `env.train.total_num_envs` alone.** `global_batch_size: 1280` is sized for
64 envs: 64 envs x 20 policy steps / 8 ranks = 160 = 1280/8. Changing one without the other
trips `rollout_size % batch_size_per_rank == 0`.

Progress lands in `outputs/mopd/robocasa365_mopd_gr00t.<tag>.out`; watch for

```
Routed teacher velocities over {'atomic_seen': N, 'composite_seen': N, 'composite_unseen': N}
```

The three shares differing across ranks is what confirms routing is per sample, not a fixed
split.

---

## 5. Settings that are easy to get wrong

| setting | value | why |
|---|---|---|
| prompt length | longest sequence in the current RoboCasa365 batch | fixed-length padding contaminated unmasked tokens and is disabled on this path; `padding_value: 1024` remains in the config but is inert here |
| `embodiment_tag` | `new_embodiment` → id **31** | RLinf's tag-table patch maps it to 10 (the N1.6/N1.7 convention); these checkpoints need 31, and the wrong id silently selects another robot's projectors |
| `num_action_chunks` | `16` | the official eval executes the full 16-step horizon; emitting fewer changes the control regime |
| `env.*.split` | `target` | `pretrain` resolves to **zero** `composite_unseen` tasks |
| `noise_method` / `joint_logprob` | `flow_ode` / `True` for M1 | see §1; `flow_sde` + `False` is M3-Hybrid's rollout, not M1's |
| FSDP `transformer_layer_cls_to_wrap` | named classes, no `ValueHead` | no critic is built when `add_value_head: False` |
| `MUJOCO_GL` | `egl` | robosuite offscreen rendering |
| NCCL | `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` | A6000s, no NVLink, no InfiniBand |
| wandb | `source /egr/research-actionlab/liyifa11/XyCode/.wandb_env` inside the launch command | the key is file-scoped; never `wandb login` |

### Integration gaps that cost a smoke run each

Every one of these produced a hard failure while bringing the pipeline up, and each would
cost the same time again on a rebuild:

| symptom | cause and fix |
|---|---|
| `Version mismatch: cluster Ray 2.44/py3.9 vs process 2.56/py3.11` | `ray start` ran without the venv active and picked up miniconda's ray. Activate the venv **before any ray command**. |
| `ServerUnavailable: 127.0.0.1:8265/api/v0/actors`, *after* actors initialised | the dashboard was disabled. RLinf's runner queries the Ray state API, so `--include-dashboard=false` kills the driver late. Keep the dashboard and raise its timeout instead. |
| `Could not find the transformer layer class to wrap` | FSDP builds its wrap policy from GR00T's `_no_split_modules`, which lists `ValueHead` — never constructed when `add_value_head: False`. Name the classes that do exist via `wrap_policy.transformer_layer_cls_to_wrap`. |
| `routed OPD needs a per-sample task label; env obs has keys [...]` | `EnvOutput.prepare_observations` normalises observations to a fixed key set, so neither `task_metadata` nor an ad-hoc list survives. `task_groups` was added to that contract. |
| `velocity distillation requires rollout.expert_model` | the single-teacher assert did not know about `expert_models`. |

Also: the model casts to bf16 **after** `from_pretrained`, which would quantise the RoPE
`inv_freq` buffers that `from_pretrained` deliberately keeps in fp32. The loader snapshots
them before the cast and restores them after. Casting bf16 back to fp32 does **not** recover
the precision, so the order matters.

---

## 6. Sharing a busy node

Three GPU-restriction mechanisms are independent and **all** must be set, or workers land on
someone else's card:

```bash
CUDA_VISIBLE_DEVICES=7 \        # visibility
RAY_NUM_GPUS=1 \                # Ray's resource ledger
MOPD_PLACEMENT=0 \              # RLinf's component_placement ("all" ignores the other two)
bash toolkits/mopd/run_eval_mopd.sh ...
```

Plus `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, already defaulted in `ray_utils.sh`:
Ray otherwise overwrites the variable inside each worker with its own GPU ids and discards
the outer restriction.

---

## 7. Where the rest is written down

| document | what it holds |
|---|---|
| [INSTALL.md](INSTALL.md) | how the gr00t package is wired in, and why the fork was dropped |
| [RESULTS.md](RESULTS.md) | every measured number, including withdrawn ones |
| [CHANGES.md](CHANGES.md) | the defects found and the fix for each |
| [METHOD_SUPPORT_GR00T.md](METHOD_SUPPORT_GR00T.md) | M1/M3/M3-Hybrid/M4 status on GR00T |
| [DECISIONS.md](DECISIONS.md) | why the config is the way it is |
