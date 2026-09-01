# How MOPD's optimisation budget is allocated — decisions and the adopted setting

Consolidates the weighting discussion. The method spec is
[`Sampling Strategy for Flow-Based VLA Multi-Teacher OPD.md`](Sampling%20Strategy%20for%20Flow-Based%20VLA%20Multi-Teacher%20OPD.md);
the implementation plan and review notes are in
[PLAN_state_coverage.md](PLAN_state_coverage.md); the short-form entries are D10–D12 in
[DECISIONS.md](DECISIONS.md). This page is the reasoning in one place.

---

## 1. The problem: nobody chooses the weights, the horizons do

Routing sends every sampled transition to the teacher of its task's suite. So **the sampling
distribution *is* the per-teacher loss weight** — there is no separate weighting knob, and
leaving it implicit does not leave it neutral.

Sampling uniformly over collected transitions weights each suite by how many transitions it
produced, which is set by episode length:

| suite | K (tasks) | Σ horizons | natural weight | equal weight | ratio |
|---|---|---|---|---|---|
| atomic_seen | 18 | 11,850 | 13.1% | 33.3% | **2.54x** |
| composite_seen | 16 | 37,050 | 41.0% | 33.3% | 0.81x |
| composite_unseen | 16 | 41,550 | 45.9% | 33.3% | 0.73x |

*(full 50-task set, equal rollout trajectories per task)*

So doing nothing gives the atomic teacher **13% of the gradient instead of 33%** — an
allocation decided by RoboCasa's horizon lengths, not by us.

This is the same class of trap as `episode_horizon_source`, which changes what `success_once`
*means* (the same checkpoint scores 0.35 or 0.2708 depending on the setting). A knob that
looks like bookkeeping and is actually a reweighting.

---

## 2. The hierarchy, and where a weight can be applied

```
Suite → Task → Trajectory → State / action chunk → Flow timestep τ
```

A weighting decision exists at every level. What was decided at each:

| level | decision | rationale |
|---|---|---|
| **Suite** | **balanced, P(s) = 1/3** | each teacher gets an equal share of the gradient; this is the thing MOPD is testing |
| **Task** (within suite) | equal rollout trajectories, `M_{s,k} = M_s` | equal *exposure*; achieved by `task_sampling_strategy: ordered` |
| **Trajectory / state** | **natural**, `x ~ U(B_s)` | preserves the student's on-policy state occupancy inside a suite |
| **Flow timestep** | uniform over the 4 denoise steps | M1 uses it unweighted; M3 applies `w(τ)` in the loss, not in the sampling |

### Why nothing is balanced *inside* a suite

Within a suite, tasks have different horizons (composite_unseen: 1200 vs 1650), so
uniform-over-transitions gives the longer task ~1.4x the weight — the same horizon effect one
level down.

**But both tasks in a suite route to the same teacher.** It does not touch per-teacher
weighting at all, only which states that teacher supervises. Uniform over transitions is the
unbiased estimator of the student's on-policy state occupancy, which is what the loss is
defined against.

Adding a per-task quota inside a suite would be machinery controlling nothing — the same
speculative generality as the `teacher_router` knob that was removed in `c9ff2e90`. Task-level
balancing is ablation **A2**; it does not belong in the default path.

---

## 3. The adopted main setting

```
s ~ U(S)          suites uniform            — equal optimisation budget per teacher
x ~ U(B_s)        states uniform in suite   — natural on-policy occupancy
τ ~ p(τ)          uniform over denoise steps
with M_{s,k} = M_s                          — equal rollout trajectories per task
```

> **Equal optimisation budget across suites, equal rollout exposure across tasks, natural
> on-policy state occupancy within each suite.**

Implemented as `algorithm.suite_balanced_sampling: True` plus
`env.train.task_sampling_strategy: ordered`.

---

## 4. Three ways to reach P(s) = 1/3 — they are substitutes, not complements

The pool arithmetic decides which is practical. Short-horizon suites produce fewer
transitions, so their pool is structurally smaller and a shortfall against an equal quota is
**the normal case, not an edge case** (2.2x at pilot scale, 2.54x at full scale).

| mechanism | effect | cost |
|---|---|---|
| **per-suite quota**, drawn with replacement for the shortfall | equal counts by construction | some of the N teacher forwards recompute the same state |
| **per-suite importance weight** on a uniform draw | equal gradient share | none — but a per-sample weight must survive the micro-batch split |
| **env allocation ∝ 1/horizon** | pools naturally balanced | few distinct composite episodes per epoch |

### Correcting an earlier claim

An earlier draft said the quota costs "2.54x the teacher forward". **That was wrong.** Both
mechanisms issue exactly **N** teacher forwards; the quota's waste is that some of those N
recompute an already-scored state. Measured at pilot scale (pools 197/558/623, N=1280):

| | atomic | composite_seen | composite_unseen | distinct states |
|---|---|---|---|---|
| quota 427 each | 197 (+230 duplicates) | 427 of 558 | 427 of 623 | **1051** |
| uniform + weight | 183 | 518 | 579 | **1280** |

Importance weighting therefore buys **~229 more distinct states per epoch (+22%)** at equal
compute. Real, but modest.

### Decision: quota first

Chosen for the first build on reliability grounds, at the user's direction ("先按可靠方法来"):

- the quota needs **no loss change and no batch plumbing**; the weighting variant needs a
  per-sample tensor to survive `forward_inputs` → micro-batch split → loss
- the quota's correctness holds **by construction** — the drawn counts *are* the balance, and
  the existing routing log displays them, so step 1 of the plan became an assertion rather
  than a diagnostic
- the failure modes are not comparable: an unfillable quota is visible, whereas a weight
  desynchronised from its samples silently mis-weights the gradient and looks normal in every
  metric — the failure mode this work has repeatedly hit

Refinement kept: draw **without** replacement where the pool covers the quota, with
replacement only for the shortfall. Composite suites then carry no duplicates at all.

**Recorded as an upgrade:** importance weighting, to be done once there is a baseline to
verify it against — and it must not change the result.
`compute_velocity_distill_loss` already computes `(sq_err * mask).sum() / mask.sum()`, a
weighted mean, so its loss-side change is small; the difficulty is entirely the plumbing.

### Why duplicates are acceptable

Uniform random sampling is an **unbiased** estimator whatever the sample count, so fewer
distinct states raises variance, not bias — and the pool is regenerated from fresh rollouts
every step, so coverage accumulates. What duplicates do cost is *effective* sample size:
atomic occupies its full quota but its gradient is built from however many distinct states its
pool holds. That is inherent to the pool being small, not a property of the mechanism.

---

## 5. A constraint that is easy to miss

**The quota is drawn per rank.** With `ordered` sampling each env worker takes
`total_num_envs / num_env_workers` *consecutive* tasks from the filter, so that count must be
a multiple of the task-filter length or some ranks hold no samples for some suite:

```
24 envs, 6 tasks: worker 0 → tasks [0,1,2] → {atomic, comp_seen}    no composite_unseen
                  worker 1 → tasks [3,4,5] → {comp_seen, comp_uns}  no atomic
48 envs, 6 tasks: every worker → tasks [0..5] → all three suites
```

This is in the assert's message, not only in a comment.

---

## 6. What `ordered` does to randomness

It stratifies rather than removes it:

| | `random` | `ordered` |
|---|---|---|
| which env runs which task | i.i.d. draw over the union | deterministic round-robin, cursor persists across epochs |
| the scene, objects and initial state within that task | random | **still random** |

Training keeps `use_fixed_reset_state_ids: False`. The randomness moves from "which task" —
bookkeeping — to "which instance of that task", which is the diversity that matters.

`random` leaves the per-suite mix swinging ~18% epoch to epoch at pilot scale, and at the full
50-task set **27% of tasks get no env at all** in a given epoch.

---

## 7. Verified in the first run

```
Suite-balanced sample: drawn {atomic 54, composite_seen 53, composite_unseen 53},
                       distinct {42, 53, 53}
Routed teacher velocities over {atomic 54, composite_seen 53, composite_unseen 53}
```

The routing counts match the sample counts exactly, which is what shows the teacher pass runs
only on the drawn subset rather than the whole `[T, B]`.

The `distinct` column earned its place immediately: it showed the atomic pool at **41–47**
rather than the ~56 its 450-step horizon predicts. The gap is the failure-length coupling —
training runs `ignore_terminations: False`, atomic is the suite the student succeeds at most
often, and a success ends the episode early, leaving fewer valid transitions. Without that
column the pool would look like whatever the horizon predicts and the duplication would be
invisible.

---

## 8. Open

- **Importance weighting** as the upgrade path (§4).
- **Per-suite weighting is not yet implicated in the training collapse.** The first M1 run
  degraded the student, but the balance itself was verified correct every step, and the
  causes measured out so far are elsewhere. See [RESULTS.md](RESULTS.md).
- **The ablations A0–A4** in the method spec — none run. A0 (truncated-horizon collection) is
  the baseline that justifies the whole rebuild and was added to the register during review.

---

## 9. A silent defect in the suite quota: per-rank balance is not global balance

Found 2026-08-28 while checking whether full-task multi-teacher training could keep suite
balancing. **The quota does not do what §4 claims when the ranks hold different suite mixes,
and nothing reports it.**

### What the code actually does

```python
suites = sorted(set(teacher_ids.tolist()))     # <- the suites PRESENT on this rank
for i, suite in enumerate(suites):
    pool  = nonzero((teacher_ids == suite) & valid)
    quota = target // len(suites) + (1 if i < target % len(suites) else 0)
    assert pool.numel() > 0, "...every rank's envs must cover every suite..."
```

`suites` is derived from the teacher ids that happen to be on this rank, not from the
configured teachers. A rank holding only `atomic_seen` therefore gets `suites = [atomic]`,
`quota = target`, and a non-empty pool — **the assert cannot fire for the case it names.**

### Why that matters

Gradients are averaged across ranks, so the global suite weighting is the *mean of the
per-rank weightings*. Balance holds only when every rank carries every suite.

With the full 50-task list, `ordered` task sampling and 24 envs over 8 workers:

```
task list:  0 ─── 17 │ 18 ─── 33 │ 34 ─── 49
            atomic(18)  comp_seen(16) comp_unseen(16)

worker 0-5 -> tasks 0-17    all atomic_seen
worker 6-7 -> tasks 18-23   all composite_seen
composite_unseen            NO RANK HOLDS IT
```

Each rank balances perfectly within what it has, and the global mix comes out
**6/8 atomic, 2/8 composite_seen, 0 composite_unseen** — one teacher receives no gradient at
all. The `Routed teacher velocities over {...}` log prints each rank's own distribution, so
every line looks healthy.

This is worse than a crash: a crash is visible.

### The env count that actually makes it correct

Every rank needs a consecutive span of the task list touching all three blocks. The worst
starting position decides the length:

| span starts at | must cover | length |
|---|---|---|
| 0 (atomic start) | all atomic + all composite_seen, then reach 34 | **35** |
| 18 (composite_seen start) | that block + composite_unseen, wrap to reach 0 | 33 |
| 34 (composite_unseen start) | that block + all atomic, wrap to reach 18 | **35** |

So 35 tasks per worker, and with `group_size: 1` that is 35 envs per worker — **280 envs**
across 8 workers, about 810 GB of env memory and 573k env-steps per training step. That is
why the full-task multi-teacher run of 2026-08-28 was configured with
`suite_balanced_sampling: False` and natural (horizon) weighting instead.

`random` sampling does not rescue it either: with 16/50 tasks in a suite, a rank of E envs
misses it with probability (34/50)^E — 0.31 at E=3, still 0.01 at E=12, and there are 24
rank-suite chances per epoch.

### Two corrections to earlier reasoning on this page

- **§5's claim that the assert protects against a missing suite is wrong.** It does not, and
  the message it prints is misleading.
- The 6-task round was configured at 48 envs on the strength of that reasoning. 48 is still
  the right number there — each worker takes 6 consecutive tasks out of 6, so every rank
  genuinely holds all three suites — but it was right for the wrong reason.

### What the fix should be

Derive `suites` from the **configured** teachers rather than the ones present, and then
either raise on a rank that is missing one, or redistribute that suite's quota and log the
realised global mix. Either way the failure has to become visible.

Not applied yet: `rlinf/` is shared over NFS with a second session that still has runs to
launch, and this touches the sampler those runs use.
