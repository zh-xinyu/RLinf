# Method 1: On-Policy Velocity Distillation

## Overview

**Method**: Distill a frozen RL-FlowSDE teacher's velocity field into a student policy (π0.5) via MSE loss.

**Task**: ManiSkill `PutOnPlateInScene25Main-v3` (25 main task variants, training set object sampling)

**Loss**: `L_vel = mean || v_student(A_t^tau, o_t, tau) - stop_grad(v_teacher(...)) ||_2^2`

**Algorithm**: Pure velocity distillation (no reward, no advantage, no PPO, no critic)

---

## Results Summary

| Checkpoint               | success_at_end | success_once | velocity_mse | velocity_cosine |
|---------------------------|:--------------:|:------------:|:-------------:|:----------------:|
| Student SFT (baseline)    | 0.550          | 0.575        | -             | -                |
| Teacher (RL-FlowSDE)      | 0.691          | 0.856        | -             | -                |
| Step 25                   | 0.741          | 0.856        | -             | -                |
| Step 50                   | 0.744          | 0.869        | -             | -                |
| Step 200 (final)          | 0.753          | 0.869        | 3.06e-04      | 1.000            |

**Key Findings:**
- **Improvement over SFT baseline**: 0.550 → 0.753 success_at_end (+37% relative, +0.203 absolute)
- **Approach to teacher**: Student reaches 0.753 vs teacher's 0.691 (surpasses teacher on this metric)
- **Velocity alignment**: velocity_cosine=1.000 indicates perfect directional alignment; velocity_mse=3.06e-04 shows tight magnitude matching
- **Training stability**: Smooth convergence from step 25→200, no crashes after Vulkan ICD path fix

---

## Training Details

### Rollout Collection (On-Policy)

Each training step collects **fresh rollout data**:

- **320 parallel environments** (`total_num_envs=320`), distributed across 8 EnvGroup workers (40 envs/rank)
- **1 rollout epoch** per step (`rollout_epoch=1`)
- **Up to 80 timesteps** per episode (`max_episode_steps=80`, `max_steps_per_rollout_epoch=80`)
- **Total samples per rollout**: 320 envs × 80 steps = 25,600 timesteps

**Task sampling**: Each episode reset randomly samples an object/plate configuration from the training set (`obj_set="train"`, `use_fixed_reset_state_ids=False` for training).

**Data collected per timestep**:
- Observation: `rgb+segmentation` (ManiSkill sensor mode)
- Student action: 8-step action horizon, 5 action chunks (`num_action_chunks=5`, `action_horizon=8`)
- Flow states/chains: ODE rollout intermediate states from student's flow model
- **Teacher velocity**: Frozen RL-FlowSDE teacher computes target velocity field for each state (~140s per step for teacher inference)

### Training Update

**One gradient update per rollout** (`update_epoch=1`):

- **Global batch size**: 5,120 samples (`global_batch_size=5120`)
- **Micro batch size**: 32 (`micro_batch_size=32`)
- **Gradient accumulation**: 5120 / 32 = 160 micro-batches per update
- **Optimizer**: AdamW with lr=5e-5, grad_clip=1.0
- **Training backend**: FSDP (`fsdp_config.sharding_strategy="full_shard"`)

**Step timing breakdown** (step 200 average):
- Rollout generation: ~110s
- Teacher velocity computation: ~140s (compute_teacher_velocities)
- Training update: ~205s
- Eval (every 25 steps): ~225s additional
- **Total per step**: ~412s average (~7 min)

### Total Training Scale

**200 training steps** (`runner.max_steps=200`):

- **Total rollouts**: 200
- **Optimizer.step() calls per rollout**: `rollout_size / global_batch_size × update_epoch` = 25,600 / 5,120 × 1 = **5**
- **Total gradient updates**: 200 × 5 = **1,000**
- **Total env interactions**: 200 × 320 × 80 = **5,120,000 timesteps**
- **Total wall time**: ~22h59m (step 0→25: 2h56m28s + step 25→200: 20h02m06s)
- **Total GPU-hours** (8 GPUs): ~184

See [`.project_results/COMPARISON.md`](../COMPARISON.md) for why raw step count alone isn't a fair way to compare this against Method 2, and which budget-matched metrics to use instead.

---

## Training Entry Point

### Launch Command

```bash
cd /egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try

# 1. Source wandb credentials
source /egr/research-actionlab/liyifa11/XyCode/.wandb_env

# 2. Set environment variables
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export EMBODIED_PATH=${PWD}/examples/embodiment
export PYTHONPATH=${PWD}

# 3. Activate venv (contains fixed Vulkan ICD path)
source openpi/bin/activate

# 4. Launch training
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name maniskill_veldistill_openpi_pi05 \
  runner.logger.log_path="outputs/my_run_name" \
  runner.max_steps=200 \
  runner.save_interval=25 \
  runner.val_check_interval=25
```

**Resume from checkpoint:**
```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name maniskill_veldistill_openpi_pi05 \
  runner.resume_dir="outputs/.../checkpoints/global_step_N" \
  runner.max_steps=200
```

### Config File

**Main config**: `examples/embodiment/config/maniskill_veldistill_openpi_pi05.yaml`

**Key settings**:
- `algorithm.loss_type: velocity_distill` - activates velocity MSE loss
- `actor.model.model_path`: Student SFT checkpoint
- `rollout.expert_model.model_path`: Frozen RL-FlowSDE teacher checkpoint
- `actor.model.add_value_head: False` - no critic (pure distillation)
- `env.train.total_num_envs: 320` - parallel env count
- `actor.global_batch_size: 5120` - samples per gradient update
- `actor.optim.lr: 5.0e-5` - learning rate

**Environment config**: `examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml`
- Task: `PutOnPlateInScene25Main-v3`
- `obj_set: train` - training object set
- `max_episode_steps: 80` - episode truncation

---

## Checkpoints

**Retained checkpoint**:

```
outputs/veldistill-200step-resume-20260803-204545/
└── maniskill_veldistill_openpi_pi05/checkpoints/global_step_200/  # Final checkpoint
```

Intermediate checkpoints were removed after their evaluation metrics had been recorded.

Each checkpoint directory contains:
- `actor/` - FSDP-sharded student policy weights
- `optim_state/` - Optimizer state for resume
- `scheduler_state/` - LR scheduler state
- `config.yaml` - Full Hydra config snapshot

---

## Evaluation

**Standalone evaluation** using a saved checkpoint:

```bash
cd /egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try

source /egr/research-actionlab/liyifa11/XyCode/.wandb_env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export EMBODIED_PATH=${PWD}/examples/embodiment
export PYTHONPATH=${PWD}
source openpi/bin/activate

python evaluations/run_eval.py \
  --config-path evaluations/maniskill/ \
  --config-name maniskill_veldistill_openpi_pi05_eval \
  ckpt_path="outputs/.../checkpoints/global_step_200/actor" \
  logger.log_path="outputs/eval-step200-$(date +%H%M%S)"
```

**Eval config**: `evaluations/maniskill/maniskill_veldistill_openpi_pi05_eval.yaml`

**Metrics**:
- `success_at_end`: Task success rate at episode end (primary metric)
- `success_once`: Task success at any point during episode (optimistic metric)
- `reward`: Per-step reward average
- `return`: Cumulative episode return

---

## Notes

### Fixed Issues

**Vulkan ICD path bug**: `openpi/bin/activate` originally hardcoded `/etc/vulkan/icd.d/nvidia_icd.json` (nonexistent path on this machine). Fixed by editing the activate script to point at `/usr/share/vulkan/icd.d/nvidia_icd.json`. This fix is **machine-specific** - verify the correct path on different nodes.

**Ray/rlinf import path**: Always `cd` into the main repo root before launching training (not a worktree). Python's cwd-prepended `sys.path` can pick up the wrong `rlinf` copy if launched from a worktree.

### Resource Requirements

- **GPUs**: 8× NVIDIA RTX A6000 (40GB VRAM each)
- **Cluster setup**: Single node, collocated placement (`component_placement: "actor,env,rollout: all"`)
- **Memory**: FSDP full_shard + offload enabled for both actor and rollout workers
- **Storage**: ~20GB per checkpoint (FSDP-sharded weights + optimizer state)

---

## Method Comparison (Planned)

This is **Method 1** of the On-Policy Distillation (OPD) project for π0.5 on ManiSkill.

**Planned methods**:
1. ✅ **Velocity distillation** (this method) - distill teacher velocity field
2. ✅ **Teacher-chunk flow matching** ([Method 2](../method2_teacher_chunk_fm/README.md)) - match teacher action distributions at chunk level
3. ⏳ Method 3 (TBD)
4. ⏳ Method 4 (TBD)

Each method gets its own git branch off `opd` and results directory under `.project_results/`.

For a budget-matched comparison between Method 1 and Method 2 (why raw step count is misleading, and which metrics to use instead), see [`.project_results/COMPARISON.md`](../COMPARISON.md).
