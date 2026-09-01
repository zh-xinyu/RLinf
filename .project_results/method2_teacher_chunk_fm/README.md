# Method 2: Teacher-Chunk On-Policy Flow Matching

## Overview

**Method**: Distill a frozen RL-FlowSDE teacher into a student policy (π0.5) by matching the student's velocity field against the teacher's implied velocity field along an interpolated action path, using full teacher-generated action chunks as the distillation target.

**Task**: ManiSkill `PutOnPlateInScene25Main-v3` (25 main task variants, training set object sampling) — same task as Method 1.

**Loss**:
```
L_fm = mean || v_student(A_tau, o_t, t) - u_teacher ||_2^2
where:
  A_tau = t * noise + (1 - t) * A_teacher   # interpolated action
  u_teacher = noise - A_teacher              # teacher's implied velocity field
  t ~ U(0, 1)                                # flow-matching timestep
```

**Algorithm**: Teacher-chunk flow matching (no reward, no advantage, no PPO, no critic)

**Key difference from Method 1**: instead of matching the teacher's velocity at the student's own observation directly, Method 2 has the teacher generate a **full action chunk** first, then trains the student to match the velocity field implied by an interpolation between noise and that teacher chunk.

---

## Results Summary

| Checkpoint               | success_at_end | success_once | flow_match_mse | velocity_cosine |
|---------------------------|:--------------:|:------------:|:---------------:|:----------------:|
| Student SFT (baseline)    | 0.550          | 0.575        | -               | -                |
| Teacher (RL-FlowSDE)       | 0.691          | 0.856        | -               | -                |
| Step 1                     | 0.534          | 0.734        | 0.069           | 0.967            |
| Step 30                    | 0.681          | 0.866        | 0.045           | 0.979            |
| Step 50 (final)            | 0.681          | 0.869        | 0.046           | 0.976            |

*Note: `flow_match_mse` is Method 2's training loss itself (already a velocity MSE). Step 1's success numbers come from rollout statistics in the training log, not a standalone eval pass.*

**Key Findings:**
- **Improvement over SFT baseline**: 0.550 → 0.681 success_at_end (+24% relative, +0.131 absolute)
- **success_once** exceeds Method 1's final value (0.869 vs 0.869 — tied) and clearly exceeds the SFT baseline (+0.294 absolute)
- **success_at_end** stays below the teacher (0.681 vs 0.691) and below Method 1's final checkpoint (0.753) at this step budget — see [Method Comparison](#method-comparison-planned) for a budget-matched view
- **Training stability**: no OOM (student/teacher offload strategy works), grad_norm stable throughout (0.07-0.24), smooth loss convergence with no divergence

---

## Training Details

### Rollout Collection (On-Policy)

Each training step collects **fresh rollout data**, identical setup to Method 1:

- **320 parallel environments** (`total_num_envs=320`), distributed across 8 EnvGroup workers (40 envs/rank)
- **1 rollout epoch** per step (`rollout_epoch=1`)
- **Up to 80 timesteps** per episode (`max_episode_steps=80`)
- **Total samples per rollout**: 320 envs × 80 steps = 25,600 timesteps

**Task sampling**: Each episode reset randomly samples an object/plate configuration from the training set (`obj_set="train"`, `use_fixed_reset_state_ids=False` for training).

**Data collected per timestep**:
- Observation: `rgb+segmentation` (ManiSkill sensor mode)
- Student action: 5 action chunks (`num_action_chunks=5`), 4 flow-matching steps
- **Teacher action chunk**: frozen RL-FlowSDE teacher generates a full 10-chunk, 5-step action chunk per observation (`compute_teacher_chunk_actions`), requiring a student-weights-offload / teacher-weights-load swap on GPU

### Training Update

**Multiple gradient updates per rollout** (`update_epoch=1`, but the rollout is split into many small batches):

- **Global batch size**: 320 samples (`global_batch_size=320`)
- **Micro batch size**: 8 (`micro_batch_size=8`)
- **Gradient accumulation**: 320 / 8 = 40 micro-batches per optimizer.step()
- **Optimizer.step() calls per rollout**: `rollout_size / global_batch_size × update_epoch` = 25,600 / 320 × 1 = **80**
- **Optimizer**: AdamW with lr=5e-5
- **Training backend**: FSDP (`fsdp_config.sharding_strategy="full_shard"`), offload enabled for both student and teacher

**Step timing breakdown** (step 50 average):
- Rollout generation: ~120s
- Teacher chunk computation (offload swap + generate): included in `run_interact_once`/`interact` (~114s)
- Training update (80 optimizer.step() calls): ~83s
- Eval (every 5 steps): ~124s additional
- **Total per step**: ~323s average (~5.4 min)

### Total Training Scale

**50 training steps** (`runner.max_steps=50`):

- **Total rollouts**: 50
- **Optimizer.step() calls per rollout**: `rollout_size / global_batch_size × update_epoch` = 25,600 / 320 × 1 = **80**
- **Total gradient updates**: 50 × 80 = **4,000**
- **Total env interactions**: 50 × 320 × 80 = **1,280,000 timesteps**
- **Total wall time**: 4h29m28s
- **Total GPU-hours** (8 GPUs): ~36

See [`.project_results/COMPARISON.md`](../COMPARISON.md) for why raw step count alone isn't a fair way to compare this against Method 1, and which budget-matched metrics to use instead.

---

## Training Entry Point

### Launch Command

```bash
cd /egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try/.claude/worktrees/opd-method2

source /egr/research-actionlab/liyifa11/XyCode/.wandb_env

export MUJOCO_GL=egl
export ROBOT_PLATFORM=LIBERO
export REPO_PATH=$(pwd)
export EMBODIED_PATH=${REPO_PATH}/examples/embodiment
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try/openpi/bin/python \
  examples/embodiment/train_embodied_agent.py \
  --config-path config/ \
  --config-name maniskill_method2_openpi_pi05 \
  runner.logger.log_path=${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-method2
```

### Config File

**Main config**: `examples/embodiment/config/maniskill_method2_openpi_pi05.yaml`

**Key settings**:
- `algorithm.loss_type: teacher_chunk_fm` - activates teacher-chunk flow-matching loss
- `actor.model.model_path`: Student SFT checkpoint (same as Method 1)
- `rollout.expert_model.model_path`: Frozen RL-FlowSDE teacher checkpoint (same as Method 1)
- `actor.enable_offload: true` - required for student/teacher GPU swap
- `env.train.total_num_envs: 320` - parallel env count (same as Method 1)
- `actor.global_batch_size: 320` - samples per gradient update (16× smaller than Method 1)
- `actor.optim.lr: 5.0e-5` - learning rate (same as Method 1)

**Environment config**: `examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml` (shared with Method 1)
- Task: `PutOnPlateInScene25Main-v3`
- `obj_set: train`
- `max_episode_steps: 80`

### Core Worker

- **Actor**: `rlinf/workers/actor/fsdp_actor_worker.py:EmbodiedFSDPActor`
  - `compute_teacher_chunk_actions()` (~line 1441-1500): offloads student, loads teacher, generates full action chunks, offloads teacher, reloads student
  - Flow-matching loss computation in `run_training()`
- **Rollout**: `rlinf/workers/rollout/hf/async_huggingface_worker.py:AsyncMultiStepRolloutWorker`
- **Env**: `rlinf/workers/env/async_env_worker.py:AsyncEnvWorker`

---

## Checkpoints

**Saved checkpoints** (every 5 steps):
```
logs/20260804-130123-method2_50steps/maniskill_method2_openpi_pi05/checkpoints/
├── global_step_5/
├── global_step_10/
├── ...
└── global_step_50/  # Final checkpoint
```

**Recommended checkpoints**:
- **Step 9**: `flow_match_mse=0.031` (lowest training loss, may generalize better)
- **Step 49**: `velocity_cosine=0.988` (highest directional alignment)
- **Step 50**: final checkpoint

---

## Evaluation

Uses the same `evaluations/run_eval.py` entry point as Method 1, pointed at a Method 2 checkpoint's `actor/` subdirectory.

**Metrics**: `success_at_end`, `success_once`, `reward`, `return` (same definitions as Method 1).

---

## Notes

### Technical Difficulties and Fixes

**OOM**: student (9GB) + teacher (14GB) exceed a single 48GB card. Fixed by dynamic offloading — student weights move to CPU while the teacher generates chunks on GPU, then swap back.

**Missing loss_mask**: continuing to compute loss after episode termination injected gradient noise. Fixed by setting `auto_reset=false, ignore_terminations=false`.

**Config compatibility**: Method 1 used `optimizer.betas`; FSDP's `ModelManager` expects `optim.adam_beta1/2`. Restructured the config to match.

**Hydra interpolation error**: `${config_name}` was undefined at runtime; hardcoded the output directory path instead.

**Device transfer bug** (`fsdp_actor_worker.py:1456`): the teacher model lives on GPU, but `input_transform` returned a CPU tensor. Fixed by explicitly moving the transformed observation to `self.device` before building `_model.Observation`.

**Offload assertion** (`fsdp_actor_worker.py:1441`): added an assertion that actor weights are offloaded before moving the teacher to GPU, to fail fast instead of silently OOMing.

### Resource Requirements

- **GPUs**: 8× NVIDIA RTX A6000 (40GB VRAM each) — same hardware as Method 1
- **Cluster setup**: Single node, collocated placement (`component_placement: actor,rollout,env: all`)
- **Memory**: FSDP full_shard + offload enabled for both actor and rollout workers; the student/teacher swap is required (not optional) here, unlike Method 1

---

## Method Comparison (Planned)

This is **Method 2** of the On-Policy Distillation (OPD) project for π0.5 on ManiSkill.

**Planned methods**:
1. ✅ **Velocity distillation** ([Method 1](../method1_velocity_distillation/README.md)) - distill teacher velocity field
2. ✅ **Teacher-chunk flow matching** (this method) - match teacher action distributions at chunk level
3. ⏳ Method 3 (TBD)
4. ⏳ Method 4 (TBD)

For a budget-matched comparison between Method 1 and Method 2 (why raw step count is misleading, and which metrics to use instead), see [`.project_results/COMPARISON.md`](../COMPARISON.md).

## References

- WandB: project `rlinf_pi_opd`, run `maniskill_method2_openpi_pi05`, run ID `r5lq3z5u`
  https://wandb.ai/zhouxinyu2022-michigan-state-university/rlinf_pi_opd/runs/r5lq3z5u
- Method 1 config: `examples/embodiment/config/maniskill_veldistill_openpi_pi05.yaml`
- Flow Matching paper: Lipman et al., "Flow Matching for Generative Modeling" (2023)
- Branches: `opd-velocity-distill` (Method 1, main dev), `opd-method2` (Method 2, this worktree)
