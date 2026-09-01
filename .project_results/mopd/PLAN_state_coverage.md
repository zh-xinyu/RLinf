# Plan — full-horizon training rollouts with sampled batches

Status: **plan only, nothing implemented.** Written 2026-08-25.

The sampling *method* — the hierarchy, the main setting and the ablation matrix — is specified
in [`Sampling Strategy for Flow-Based VLA Multi-Teacher OPD.md`](Sampling%20Strategy%20for%20Flow-Based%20VLA%20Multi-Teacher%20OPD.md).
This page is the implementation plan for it, plus the review notes in §"Review of the method
spec" below. The two were written independently and agree on the main setting: balance at the
suite level, sample naturally within a suite.

## The problem

Training truncates every episode at 320 steps while evaluation runs each task's native
horizon. Coverage of the trajectory the student is actually evaluated on:

| task | horizon | covered by 320 |
|---|---|---|
| TurnOnMicrowave | 450 | 71% |
| CloseToasterOvenDoor | 450 | 71% |
| ScrubCuttingBoard | 1200 | **27%** |
| RinseSinkBasin | 1350 | **24%** |
| CuttingToolSelection | 1200 | **27%** |
| CategorizeCondiments | 1650 | **19%** |

On the composite suites the student is distilled on the first quarter of the trajectory and
evaluated on all of it. That removes most of what on-policy distillation is for: the teacher's
guidance never reaches the states the student actually reaches at test time.

`DECISIONS.md` D9 justified 320 as "velocity distillation needs on-policy *states*, not
completions". True, and it does not follow that a quarter of the states is enough.

## The change

Run training episodes to `episode_horizon_source: task_horizon` and **sample a fixed number
of transitions at collection time** to build the training batch.

### Why the simulator cost does not change

Training samples = total env-steps / 16. At a fixed env-step budget the sample count is
invariant to how those steps are distributed. The six pilot tasks average 1050 steps:

```
now:  64 envs x  320 steps = 20,480 env-steps -> 1,280 samples   (first 19-27% of each task)
plan: ~20 envs x 1050 steps = 20,480 env-steps -> 1,280 samples   (100% of each task)
```

**Same simulator budget, same batch size.** What is traded is episode *diversity*: ~20 distinct
episodes per epoch instead of 64, with samples inside one episode temporally correlated. For
on-policy distillation that looks like the right trade, but it is a trade, not a free lunch.

### Why sampling at collection, not after materialising `[T, B]`

Memory is **not** the reason — the observation buffer would grow to roughly 750 MB per rank
on CPU, which this box does not notice. The reasons are:

1. **Fixed shape by construction.** `[T=104, B]` is ~70% padding for atomic tasks (450 steps
   = 28 policy steps of 104). Sampling at collection means every stored transition is valid
   and `loss_mask` degenerates to all-ones on this path.
2. **No wasted data movement.** That buffer is merged, flattened and moved between CPU and
   GPU. Carrying 70% padding through all of it is pure wall-clock.
3. **Batch size becomes an independent knob.** `global_batch_size: 1280` is currently coupled
   to the env count (64 x 20 / 8 = 160 per rank); getting that wrong raises
   `rollout_size % batch_size_per_rank`. "Draw exactly N per epoch" removes the coupling.

### The enabler, verified

Every loss on this branch is **per transition** — no temporal recursion, so dropping the time
axis costs nothing:

| method | loss | structure |
|---|---|---|
| M1 | `compute_velocity_distill_loss` | per-transition velocity MSE |
| M3 / M3-Hybrid | `compute_analytic_local_kl_loss` | same, weighted by `w(tau)` |
| M4 | `compute_opd_advantages` | `teacher_logprobs - prev_logprobs`, elementwise |

None of them is GAE-like. This is what makes a flat sampled batch admissible at all.

### Where it goes

`huggingface_worker.generate` already returns one `RolloutResult` **per policy step**; the
stacking into `[T, B]` happens downstream in `embodied_io_struct` (`merge_rollout_results`,
`torch.stack`). So the sampler is a filter at the accumulation point — contained, and not
inside the model call.

**It must be opt-in.** That accumulation path is shared with the PPO/GRPO configs.

Episode length is not known in advance (training runs with `ignore_terminations: False`, so a
success ends the episode early). **Reservoir sampling** handles this: each env keeps a fixed
number of slots and fills them uniformly over whatever stream it sees.

## Review of the method spec

The main setting — `s ~ U(S)`, `x ~ U(B_s)`, `tau ~ p(tau)`, with equal rollout trajectories
per task — is sound, and it is what the pool arithmetic below independently recommends. Five
things to add, two of which change what gets built.

### 1. Realise `P(s) = 1/3` by importance weighting, not resampling *(affects the build)*

Drawing a per-suite quota with replacement fills the quota but duplicates states. At full
scale (see the table below) atomic would be duplicated **2.54x**: no new information, 2.54x
the teacher forward.

Sample uniformly from the union and weight instead:

```
w(x) = |B| / (S * |B_{s(x)}|)
```

This is an unbiased estimator of the *same* objective, with lower variance (no repeated draws)
and no wasted compute. It is an implementation detail — the method is still "suite-balanced".

### 2. Long episodes are disproportionately failures *(affects interpretation)*

Training runs with `ignore_terminations: False`, so **a successful episode ends early**. Long
episodes are therefore disproportionately failures and timeouts, and natural within-suite
sampling weights by state count — so **the student is distilled mostly on states from episodes
it failed**.

That may well be what distillation should do: teacher guidance is most useful where the student
fails. But it is a strong implicit bias that the spec does not name, and it should be stated
rather than assumed.

It also raises **A4's priority**: episode-balanced sampling is the only entry in the matrix
that tests this, and the spec's own motivation for A4 says as much. A4 belongs ahead of A3,
second only to A1.

### 3. `M_{s,k} = M_s` is only exactly achievable when `num_envs % num_tasks == 0`

At full scale that is 50 tasks, so an epoch needs at least 50 envs for exact per-task balance.
`task_sampling_strategy: ordered` keeps a cursor across epochs, so balance holds **across**
epochs but not within one. Consequence: the per-suite weight must be computed from the
**realised** counts each step, never assumed.

### 4. The matrix is missing the baseline that motivates the change

The reason for this redesign is that 320-step truncation covers 19-27% of a composite
trajectory. None of M / A1-A4 tests that: it is a change to *rollout collection*, not to
sampling, so it sits outside the matrix. It still needs recording as the comparison that
justifies the rebuild — call it **A0: truncated-horizon collection**.

### 5. `p(tau)` in this codebase

`p(tau)` is uniform over the four denoise steps: with `joint_logprob: True`,
`compute_teacher_velocities` draws `step_idx` uniformly per sample. M1 is uniform `p(tau)`
with no weight; M3 is uniform `p(tau)` plus `w(tau)` in the loss. Worth stating so the spec
maps onto the code.

### Full-scale suite weights, Main against A1

Equal trajectories per task, `split: target`:

| suite | K | sum of horizons | A1 natural weight | Main weight | Main / A1 |
|---|---|---|---|---|---|
| atomic_seen | 18 | 11,850 | 13.1% | 33.3% | **2.54x** |
| composite_seen | 16 | 37,050 | 41.0% | 33.3% | 0.81x |
| composite_unseen | 16 | 41,550 | 45.9% | 33.3% | 0.73x |

So A1 vs Main is a 2.5x swing on the atomic suite — a substantive ablation, not a detail.

Cost at full scale, one trajectory per task: 90,450 env-steps per epoch, about **113 s of pure
simulation** at the measured ~800 env-steps/s, yielding ~5,650 samples. Affordable.

Note `K_s` is 18/16/16, not equal, so equal trajectories per task still gives the suites
different trajectory *counts*. That does not affect the training weight (fixed at 1/3) but it
does affect pool sizes.

## Balancing the teachers — the pool arithmetic

Routing sends each sample to its suite's teacher, so **the sampling distribution sets each
teacher's share of the gradient**. Sampling uniformly over the union weights suites by
horizon, which is ablation A1, not the main setting.

With equal envs per suite (7/7/7) at the current pilot budget the pools come out as:

| suite | mean horizon | pool | 1/3 of a 1280 batch |  |
|---|---|---|---|---|
| atomic_seen | 450 | **197** | 427 | **short by 2.2x** |
| composite_seen | 1275 | 558 | 427 | 0.76 |
| composite_unseen | 1425 | 623 | 427 | 0.68 |

**A shortfall on the short-horizon suite is the normal case**, at pilot scale (2.2x) and at
full scale (2.54x). This is what decides the mechanism: three ways to reach `P(s) = 1/3`,
which are substitutes rather than complements.

| mechanism | effect | cost |
|---|---|---|
| per-suite quota, drawn with replacement | equal counts | duplicate teacher forwards; effective sample size unchanged |
| env allocation ∝ 1/horizon | pools balanced without resampling | few distinct composite episodes per epoch |
| **per-suite importance weight** | equal gradient share | none; weight from realised counts |

### Chosen for the first build: per-suite quota *(reliability over efficiency)*

Both mechanisms issue exactly **N teacher forwards** — an earlier draft of this page claimed
the quota costs 2.54x more, which was wrong. The quota's waste is that some of those N
recompute the same state, not that there are more of them. Measured at pilot scale
(pools 197/558/623, N=1280):

| | atomic | composite_seen | composite_unseen | distinct states |
|---|---|---|---|---|
| quota 427 each | 197 (+230 duplicates) | 427 of 558 | 427 of 623 | **1051** |
| uniform + weight | 183 | 518 | 579 | **1280** |

So importance weighting buys **~229 more distinct states per epoch (+22%)** at equal compute.
Real, but modest.

Against that, the two differ sharply in how they fail:

| | per-suite quota | uniform + importance weight |
|---|---|---|
| sampler | group by suite, draw per suite | uniform draw (simpler) |
| loss change | **none** | needs an explicit `sample_weight` |
| batch plumbing | none | a per-sample weight must survive `forward_inputs` → micro-batch split → loss |
| correctness | **by construction**: the suites are N/3 each, and step 1's log is then an assertion | needs the weight verified against its samples |
| failure mode | an unfillable quota is visible | **weight/sample desync silently misweights the gradient, and the loss curve looks normal** |

The second row of that last column is the failure mode this project has been burned by
repeatedly. **Build the quota first.** Importance weighting is recorded as an upgrade, to be
done once there is a working baseline to verify it against — at which point the upgrade must
not change the result.

Refinement that is nearly free: draw **without** replacement when the pool covers the quota and
only fall back to replacement for the shortfall. Then the composite suites carry no duplicates
at all; atomic still does, because its pool is genuinely that small.

`compute_velocity_distill_loss` already computes `(sq_err * mask).sum() / mask.sum()`, i.e. a
weighted mean, so the upgrade's loss-side change is small. The difficulty is entirely in the
plumbing.

Env allocation ∝ 1/horizon stays available as a third knob. Upweighting a noisy estimate
amplifies its variance, and atomic will have the fewest distinct states under any of these.

### Why duplicates are acceptable

Uniform random sampling is an **unbiased** estimator of the objective whatever the sample
count, so drawing fewer distinct states raises variance, not bias — and the pool is
regenerated every epoch from fresh rollouts, so coverage accumulates across epochs. What the
duplicates do cost is *effective* sample size: atomic occupies 427 slots but its gradient is
built from ~197 distinct states, so that suite's gradient is noisier than its slot count
suggests. That is inherent — the pool only has 197 — not a consequence of the mechanism.

### No second stratification level inside a suite

Within a suite the two pilot tasks have different horizons, so uniform-over-transitions gives
the longer one more weight — the same horizon effect one level down. But **both tasks in a
suite route to the same teacher**, so it does not touch teacher weighting at all; it only
changes which states that teacher supervises. Uniform over transitions is the unbiased
estimator of the student's on-policy state occupancy, which is what the loss is defined
against.

This is the spec's main setting, and making it a configurable knob would be machinery with
nothing to control — the same speculative generality as the `teacher_router` knob just
removed. Task-level balancing is ablation A2; it does not belong in the default path.

## What was built

| step | outcome |
|---|---|
| log realised per-suite counts | **already existed** — `Routed teacher velocities over {...}` in `_teacher_velocity_pass` is exactly that count. One planned step deleted. |
| `episode_horizon_source: task_horizon`, `task_sampling_strategy: ordered` | done, together, in the training config |
| suite-balanced sampler | `EmbodiedFSDPActor._suite_balanced_sample`, gated on `algorithm.suite_balanced_sampling` |
| cap the teacher pass | `compute_teacher_velocities` scores only the drawn transitions and scatters them back into the `[T, B]` layout |
| cap the student pass | `run_training` reuses the same indices instead of `randperm`, so the student never runs on a transition the loss ignores |
| decouple the batch size | `actor.global_batch_size` is now the number of samples drawn per update, independent of the env count |

The sampler was reachable in a smaller change than expected: `process_nested_dict_for_train`
already flattens `[T, B]` and indexes with a permutation tensor, so suite-balanced sampling is
"pass a different index tensor". The correctness hinge is that
`flatten_nested_tensor_time_batch` and `process_nested_dict_for_train` both flatten with
`value.reshape(-1, *value.shape[2:])`, so a flat index means the same `(t, b)` at both sites —
checked before writing any of it.

### Two things found while building, both would have been silent

**The draw must advance every step.** The first version seeded on `seed + rank` alone. It also
referenced `self.global_step`, which does not exist on the actor — but had it existed and been
constant, every update would have trained on the *same* transitions, and no metric would show
it. Replaced with a counter the worker owns.

**Every rank's envs must cover every suite.** The quota is drawn per rank, and `ordered` gives
each env worker `total_num_envs / num_env_workers` *consecutive* tasks from the filter. At the
24 envs first written into the config that is three consecutive tasks, so half the workers
would hold no `atomic_seen` and half no `composite_unseen`:

```
24 envs: worker 0 -> tasks [0,1,2] -> {atomic, composite_seen}      no composite_unseen
         worker 1 -> tasks [3,4,5] -> {composite_seen, composite_unseen}   no atomic
48 envs: every worker -> tasks [0..5] -> all three suites, two envs each
```

Fixed by 48 envs, and the rule is now in the assert's message rather than only in a comment:
`total_num_envs / num_env_workers` must be a multiple of the task-filter length.

At 48 envs the per-rank pools are 56 / 159 / 178 against a quota of ~53, so the first build
runs with **no duplication at all** — the 2.2x figure below applies to a smaller env count.

### A latent bug the change surfaced

`_teacher_velocity_pass` computed its micro-batch chunk count from the **whole** flat batch and
then used it to split each **per-suite partition**:

```python
num_chunks = ceil(flat_batch_size / micro_batch_size)   # from the whole batch
...
_run(teacher_model, subset)                             # applied to one suite's rows
```

`split_dict_to_chunk` returns fewer chunks than asked for when the size does not divide, and
then indexes past the end — `IndexError: list index out of range`. This has been latent since
routed multi-teacher OPD was added: the partitions happened to be 80/40/40 against 20 chunks,
all exact. Suite-balanced sampling makes them 54/53/53, which is not, so the first smoke run
died on it.

Fixed at the root: the chunk count now comes from the size of the call's own input. Verified
across 41 / 47 / 53 / 54 / 160 / 225 — chunk counts correct and no elements lost.

### What the `distinct` column revealed

The sampler logs both drawn and distinct counts, and the first run showed
`drawn {atomic 54, ...}` against `distinct {atomic 41, ...}` — so the atomic pool per rank is
**41-47**, not the ~56 predicted from its 450-step horizon.

The gap is the failure-length coupling in review note §2, showing up in the data: training runs
with `ignore_terminations: False`, atomic is the suite the student succeeds at most often, and a
success **ends the episode early**. Fewer valid transitions follow.

Worth keeping the `distinct` column for exactly this reason — without it the pool would look
like whatever the horizon predicts, and the duplication would be invisible.

### Remaining

Smoke: three suites near equal in the sample log, `velocity_cosine` around 0.98, and
`Suite-balanced sample: ...` showing the realised mix.

**Not now:** none of the ablations, A0 included.

## How the rollout is composed

Worth stating because it is easy to assume otherwise: **all three suites are rolled out
together**. The task list is the union of the three soups, and every env draws one task from
that union, so each batch contains all three suites mixed. That is exactly why
`_teacher_velocity_pass` partitions by suite and moves one teacher to the GPU at a time.

### Why `ordered` matters more than it looks

`task_sampling_strategy: random` draws each env's task i.i.d. uniformly over the union, so a
single epoch is multinomial, not balanced:

| scale | envs per task | relative spread | tasks getting zero envs |
|---|---|---|---|
| pilot, 6 tasks / 64 envs | 10.7 | 28% | 0% |
| pilot, at *suite* level | 21.3 | **18%** | — |
| full, 50 tasks / 64 envs | 1.28 | **87%** | **27%** |

At full scale **27% of tasks get no env at all in a given epoch**. At pilot scale the suite
mix swings 18%, which moves the atomic duplication factor between 2.2x and 3.8x epoch to
epoch — the quota still delivers N/3, but from a pool of varying quality.

`ordered` is a deterministic round-robin over the task list with a cursor that persists across
epochs, offset per worker by `seed_offset * num_group` so the workers tile disjoint slices
rather than all starting at 0. Simulated for this config (8 workers x 8 envs, 6 tasks):

```
epoch 0: per task [11, 11, 11, 11, 10, 10]   per suite [22, 22, 20]
epoch 1: per task [11, 11, 10, 10, 11, 11]   per suite [22, 20, 22]
epoch 2: per task [10, 10, 11, 11, 11, 11]   per suite [20, 22, 22]
```

Exact equality is impossible when the env count is not a multiple of the task count, but the
remainder rotates between epochs and cancels.

**This does not make the run deterministic.** It stratifies:

| | `random` | `ordered` |
|---|---|---|
| which env runs which task | i.i.d. draw | **deterministic round-robin** |
| the scene, object placement and initial state within that task | random | **still random** |

Training has `use_fixed_reset_state_ids: False`, so each rollout of a task is a different
instance of it. The randomness moves from "which task" — bookkeeping — to "which instance of
that task", which is the diversity that actually matters.

## Open questions

- Episode diversity at ~20 envs: samples within one episode are temporally correlated, and
  each epoch sees far fewer distinct initial states. Worth watching the gradient noise.
- Late-timestep transitions are a biased subset: an episode that succeeds ends early, so
  surviving late states come disproportionately from *failing* episodes. That is arguably what
  distillation should target, but it is a selection effect and should be stated, not assumed.
