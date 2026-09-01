# On-Policy Distillation Method Comparison

## Why Raw "Steps" Isn't a Fair Comparison

Method 1 (velocity distillation) and Method 2 (teacher-chunk flow matching) use very different **batch sizes**, which means that one "global_step" in Method 1 performs a different amount of training work than one "global_step" in Method 2.

### Key Difference: `global_batch_size`

| Method | `global_batch_size` | `micro_batch_size` | Optimizer steps per rollout |
|--------|--------------------:|-------------------:|----------------------------:|
| Method 1 (velocity distill) | 5,120 | 32 | ~5 |
| Method 2 (teacher-chunk FM) | 320 | 8 | ~80 |

Both methods collect **~25,600 samples** per rollout (320 envs × 80 steps × 1 epoch), but they split those samples differently:

- **Method 1**: splits into ~5 large batches of 5,120 samples each → **5 gradient updates per rollout**
- **Method 2**: splits into ~80 small batches of 320 samples each → **80 gradient updates per rollout**

This means Method 2's "50 steps" contains **16× more gradient updates** than you'd expect from the step count alone.

### Training Scale Breakdown

#### Method 1 (200 steps)
- **Global steps**: 200
- **Gradient updates**: 200 × 5 = **1,000**
- **Env interactions**: 200 × 320 × 80 = **5,120,000 timesteps**
- **Wall time**: ~23h (step 0→25: 2h57m + step 25→200: 20h02m)
- **GPU-hours** (8 GPUs): ~184
- **Time per gradient update**: 82.8s

#### Method 2 (50 steps)
- **Global steps**: 50
- **Gradient updates**: 50 × 80 = **4,000**
- **Env interactions**: 50 × 320 × 80 = **1,280,000 timesteps**
- **Wall time**: ~4h29m (4:29:28)
- **GPU-hours** (8 GPUs): ~36
- **Time per gradient update**: 4.0s

## Fair Comparison Axes

To fairly compare the two methods, use **budget-matched** metrics instead of raw step count:

### 1. **Gradient-Update Budget** (recommended primary metric)

Compare checkpoints with similar total gradient updates:

| Method | Checkpoint | Global Steps | Gradient Updates | success_at_end | success_once |
|--------|------------|-------------:|------------------:|---------------:|-------------:|
| Method 1 | Step 25 | 25 | 125 | 0.741 | 0.856 |
| Method 2 | Step 5 | 5 | 400 | - | - |
| Method 1 | Step 50 | 50 | 250 | 0.744 | 0.869 |
| Method 2 | Step 10 | 10 | 800 | - | - |
| Method 1 | Step 200 | 200 | **1,000** | 0.753 | 0.869 |
| Method 2 | Step 50 | 50 | **4,000** | 0.681 | 0.869 |

**Key insight**: Method 2's 50-step run performed **4× more gradient updates** than Method 1's full 200-step run.

### 2. **Wall-Clock Budget** (useful for resource planning)

Compare checkpoints with similar total GPU time:

| Method | Checkpoint | Wall Time | GPU-hours | success_at_end | success_once |
|--------|------------|-----------|----------:|---------------:|-------------:|
| Method 1 | Step 25 | ~2h57m | ~24 | 0.741 | 0.856 |
| Method 2 | Step 10 | ~54m | ~7 | - | - |
| Method 1 | Step 50 | ~5h48m | ~46 | 0.744 | 0.869 |
| Method 2 | Step 20 | ~1h48m | ~14 | - | - |
| Method 1 | Step 200 | ~23h | ~184 | 0.753 | 0.869 |
| Method 2 | Step 50 | ~4h29m | ~36 | 0.681 | 0.869 |

**Key insight**: Method 2 reached 50 steps in **~20% of Method 1's wall time** for 200 steps.

### 3. **Env-Interaction Budget** (data efficiency)

Compare checkpoints with similar total environment samples:

| Method | Checkpoint | Total Samples | success_at_end | success_once |
|--------|------------|---------------:|---------------:|-------------:|
| Method 1 | Step 50 | 1,280,000 | 0.744 | 0.869 |
| Method 2 | Step 50 | 1,280,000 | 0.681 | 0.869 |
| Method 1 | Step 200 | 5,120,000 | 0.753 | 0.869 |

**Key insight**: At the same 1.28M sample budget, Method 1 (step 50) outperforms Method 2 (step 50) on `success_at_end` (74.4% vs 68.1%), but both achieve identical `success_once` (86.9%).

## Recommendations

### For Method Comparison Papers/Reports

**Primary metric**: Use **gradient-update budget** or **wall-clock budget**. These directly reflect compute cost and training progress.

**Secondary metric**: Report **data efficiency** (env-interaction budget) to show sample complexity.

**Don't use**: Raw "steps" or "epochs" alone without clarifying what they mean in terms of gradient updates.

### For Future Experiments

When comparing methods with different batch sizes:

1. **Plan checkpoint saves** at matched gradient-update or wall-clock budgets (e.g., save Method 2 checkpoints at steps 1, 2, 5, 10, 13, 25, 38, 50 to cover the same gradient-update range as Method 1's 25/50/100/200).

2. **Report training curves** using gradient updates on the x-axis, not raw steps.

3. **Include a conversion table** showing the mapping between steps and gradient updates for each method (like the table above).

## Technical Details

The number of gradient updates per "global_step" is determined by:

```python
# From rlinf/workers/actor/fsdp_actor_worker.py:1584-1690
rollout_size = total_num_envs * max_episode_steps * rollout_epoch  # ~25,600
batch_size_per_rank = global_batch_size / world_size
num_optimizer_steps_per_rollout = (rollout_size / batch_size_per_rank) * update_epoch
```

For both methods:
- `total_num_envs = 320`
- `max_episode_steps = 80`
- `rollout_epoch = 1`
- `update_epoch = 1`
- `world_size = 8` (8 GPUs, FSDP sharding)

The only difference is `global_batch_size`, which determines how many optimizer steps are needed to consume one rollout.
