# Routed multi-teacher OPD — GR00T N1.5 on RoboCasa365

The implementation is maintained on the curated `opd-project-main` branch. One GR00T N1.5
student is distilled from three RoboCasa task-suite specialists, with each rollout sample
supervised by the teacher post-trained on its suite.

Start with this overview, then use [RESULTS.md](RESULTS.md) for measured results and
[DECISIONS.md](DECISIONS.md) for configuration rationale. The chronological results include
withdrawn interpretations, which are explicitly marked where later fixes invalidated them.

| document | what it holds |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | **the working setup, all commands, and the settings that are easy to get wrong** |
| [RESULTS.md](RESULTS.md) | every measured number, including withdrawn ones |
| [CHANGES.md](CHANGES.md) | the defects found and the fix for each |
| [METHOD_SUPPORT_GR00T.md](METHOD_SUPPORT_GR00T.md) | M1 / M3 / M3-Hybrid / M4 status on GR00T — read before running M3 or M4 |
| [Sampling Strategy...](Sampling%20Strategy%20for%20Flow-Based%20VLA%20Multi-Teacher%20OPD.md) | **method spec**: the sampling hierarchy, the main setting, and the A0-A4 ablation register |
| [M4_ENABLEMENT.md](M4_ENABLEMENT.md) | **M4 on GR00T**: three gaps that had never executed, and what changed |
| [WEIGHTING.md](WEIGHTING.md) | **how the optimisation budget is allocated across suites, tasks and trajectories** — the adopted main setting and why the alternatives were not taken |
| [PLAN_state_coverage.md](PLAN_state_coverage.md) | implementation plan for full-horizon rollouts with sampled batches, plus review notes on the spec |
| [DECISIONS.md](DECISIONS.md) | why the config is the way it is; entries overtaken by later work are marked in place |
| [INSTALL.md](INSTALL.md) | how the gr00t package is wired in, and why the fork was dropped |

## Status

The M1 and M4 paths have both completed end to end, but neither has beaten the undistilled
student in the recorded runs. This does not establish that single-transition flow
objectives are intrinsically uninformative. A later audit found that the old per-suite M4
reward diagnostic grouped unscored, zero-filled rollout rows; its mechanistic conclusion
is withdrawn. The implementation now logs only sampled rows, including replacement
duplicates, but existing logs cannot validate the corrected diagnostic. Host-memory growth
in the 768-trajectory setting also remains to be localized before longer runs.

## Method

Rollout: the student controls the simulator. Supervision is Method 1 (on-policy velocity
distillation), the mechanism the pi0.5 study selected:

```
L = mean || v_S(A_t^tau, o_t, tau) - stopgrad( v_T^{k(task)}(A_t^tau, o_t, tau) ) ||^2
```

`k(task)` is a deterministic router: a task's RoboCasa suite (`atomic_seen`,
`composite_seen`, `composite_unseen`) selects its teacher. No reward, no advantage, no
critic, no PPO.

Teachers are frozen, cached in host RAM, and moved to the GPU **one at a time**: after
rollout the batch is partitioned by suite and each teacher scores only its own partition,
so peak GPU memory is that of a single teacher no matter how many are configured.

## The working recipe

```bash
bash toolkits/mopd/run_mopd.sh robocasa365_mopd_gr00t <tag> [hydra overrides...]
```

That script is the whole recipe: it waits for free GPUs, purges and *verifies* Ray, launches
the driver detached, and runs a startup watchdog plus a silence watchdog. What it sets, and
why, is in [RUNBOOK.md](RUNBOOK.md) §5; the parts that are easy to get wrong:

| ingredient | value | why |
|---|---|---|
| venv | `gr00t_robocasa365` on **node-local NVMe** (`install.sh --model gr00t --env robocasa365`) | needed a new branch in `install_gr00t_model`; the combination had no Docker stage or CI. A venv on the NFS mount cannot meet Ray's startup deadlines — see [RUNBOOK.md](RUNBOOK.md) §2 |
| ray | activate the venv **first**, `--num-cpus=32`, dashboard **on** | otherwise miniconda's ray 2.44 starts the cluster, or RLinf's state-API query kills the driver |
| NCCL | `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` | A6000s, no NVLink, no InfiniBand |
| render | `MUJOCO_GL=egl` | robosuite offscreen rendering |
| wandb | `source /egr/research-actionlab/liyifa11/XyCode/.wandb_env` in the launch command | key is file-scoped; never `wandb login` |
| split | `target` | `pretrain` resolves to **zero** `composite_unseen` tasks |
| prompt | no fixed-length padding on the RoboCasa365 path | padding contaminated unmasked tokens; the config's `padding_value: 1024` is inert on this path |
| FSDP | `wrap_policy.transformer_layer_cls_to_wrap` without `ValueHead` | no critic is built when `add_value_head: False` |

## Checkpoints

`/egr/research-actionlab/liyifa11/models/robocasa365/gr00t_n1-5/foundation_model_learning/`

- student init: `pretraining/checkpoint-80000`
- teachers: `target_posttraining/{atomic_seen,composite_seen,composite_unseen}/checkpoint-60000`

Optimizer states were deliberately not downloaded (half the bytes, never loaded).
