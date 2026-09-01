# M1 / M3 / M3-Hybrid / M4 on GR00T — what actually works

Checked 2026-08-23 on branch `opd-mopd-routed`, which inherits the four methods from
`opd-mechanism-study`. They were all built and validated against **OpenPI / pi0.5 /
ManiSkill**. This is what survived the move to GR00T N1.5 / RoboCasa365 — and what was done
about it on 2026-08-24 (see [What was implemented](#what-was-implemented-2026-08-24)).

**Status after the fixes:** all four methods are unblocked on GR00T. M1 now runs the
deterministic ODE rollout it is defined on, rather than M3-Hybrid's stochastic one.

## The four methods, as configured on OpenPI

| method | rollout | `noise_method` | `joint_logprob` | `loss_type` |
|---|---|---|---|---|
| M1 | ODE | `flow_ode` | True | `velocity_distill` |
| M3 | ODE | `flow_ode` | True | `analytic_local_kl` |
| M3-Hybrid | Hybrid SDE | `flow_sde` | False | `analytic_local_kl` |
| M4 | Hybrid SDE | `flow_sde` | False | `opd` |

## Status on GR00T (as found, before the fixes)

| method | rollout | loss | runnable? |
|---|---|---|---|
| M1 | ✗ cannot select pure ODE | ✓ `velocity_distill` works | **only with the wrong rollout** |
| M3 | ✗ same | ✗ assert: OpenPI only | no |
| M3-Hybrid | ✓ `flow_sde` + `joint_logprob: False` | ✗ same assert | no |
| M4 | ✓ same | ✗ assert: OpenPI only | no |

### Gap 1 — GR00T has no `flow_ode`, so M1/M3 cannot get a deterministic rollout

OpenPI picks the transition **per step**:

```python
sample_method = self.config.noise_method if idx == denoise_inds[0][idx] else "flow_ode"
```

and `flow_ode` is a real branch of `sample_mean_var_val`. GR00T does the same thing with
`mode` instead — `mode="train"` (applies `noise_method`) versus `mode="eval"` — and
**GR00T's `mode="eval"` branch is numerically exactly OpenPI's `flow_ode`**: both reduce to
the Euler step `x ± v·delta` under their own time conventions (OpenPI's tau runs 1 → 0,
GR00T's t runs 0 → 1, hence the sign).

But GR00T's train branch accepts only `flow_sde` / `flow_cps` / `reinflow` and raises
`ValueError` otherwise, and the eval branch is unreachable during training. So the two
available settings are:

| `joint_logprob` | GR00T behaviour | OpenPI behaviour at the same setting |
|---|---|---|
| `True` | `denoise_inds = arange` → **every step noisy** | with `flow_ode`: **fully deterministic** |
| `False` | one random step noisy (Hybrid SDE) | one random step noisy (Hybrid SDE) |

Note the two are **opposite** at `joint_logprob: True`. There is no setting that gives zero
noisy steps.

**Consequence for the current MOPD config:** `robocasa365_mopd_gr00t.yaml` runs `flow_sde` +
`joint_logprob: False` + `velocity_distill` — that is **M3-Hybrid's rollout carrying M1's
loss**, not M1. The comment in the config ("GR00T's train path has no fully deterministic
mode") described the code correctly but drew the wrong conclusion: the fix is to add
`flow_ode`, not to accept a stochastic rollout for M1.

**Fix:** mirror OpenPI — add a `flow_ode` branch to GR00T's train path using the same
weights the eval branch already uses (`x0_weight = 1-(t+delta)`, `x1_weight = t+delta`,
`x_t_std = 0`). `get_logprob_norm` already handles `sigma == 0` by masking to zero, provided
`safe_get_logprob` stays `False` (it is). Roughly six lines.

**Side effect worth noting:** with a true ODE rollout, `noise_level` becomes irrelevant to
M1 — no noise is drawn. The concern recorded in RESULTS.md about inheriting the PPO/GRPO
default of 0.5 dissolves for M1 and M3, and remains only for M3-Hybrid and M4, where the
noise is the mechanism under study.

### Gap 2 — `analytic_local_kl` is asserted OpenPI-only, but the maths already transfers

`fsdp_actor_worker.py:1594`:

```python
assert SupportedModel(...) in [SupportedModel.OPENPI], \
    "analytic_local_kl weights are derived for OpenPI's flow transition only."
```

and `flow_sde_kl_weights` reads `cfg.actor.model.openpi.noise_level`, which a GR00T config
does not have.

**But the assert is over-restrictive.** Transcribing GR00T's `flow_sde` branch and comparing
against `flow_sde_transition_coeffs` (num_steps=4, noise_level=0.5, float64):

| quantity | agree? |
|---|---|
| `w0 + w1` | **identical** |
| `sigma` | **identical** |
| `c` | **identical up to sign** (GR00T +, OpenPI −) |
| `w(tau) = c²/(2 sigma²)` | **identical, element for element** |

No reversal is needed: both arrays are indexed by the denoise step `k`, and step `k` is the
same physical point in both conventions (OpenPI's tau falls 1 → 0 as `k` rises, GR00T's t
climbs 0 → 1).

Analytically, `c = delta·(1 + noise_level²/2)` in both — constant in tau except at the
boundary step, where each implementation guards its own division. So `w(tau)` is already the
right weight for GR00T.

**Fix:** relax the assert and read `noise_level` per model family (GR00T:
`rl_head_config.noise_level`). This is plumbing, not a new derivation. Sign matters only for
M4, below.

### Gap 3 — M4's `adv_type: opd` is asserted OpenPI-only for the same reason

`fsdp_actor_worker.py:1609`. `_teacher_transition_logprobs` builds
`mu_T = x_k (w0 + w1) + v_T · c` from the OpenPI coefficients. Since `w0 + w1` and `sigma`
are identical and `c` differs only in sign, M4 needs the same plumbing as M3 **plus** the
correct sign of `c` for GR00T.

## What was implemented (2026-08-24)

All three fixes are in. Details of each, and how it was checked.

### 1. `flow_ode` added to GR00T's train branch

`gr00t_action_model.sample_mean_var_val` gained a `flow_ode` case using the same weights the
`mode == "eval"` branch already used (`x0_weight = 1-(t+delta)`, `x1_weight = t+delta`,
`x_t_std = 0`). It has to be a separate branch because the two are selected by different
things: `mode` is chosen *per denoise step* by `get_rl_action`, while `noise_method` is a
config value fixed for the run.

`get_rl_action` also gained a guard. Its `denoise_inds` selection switched on
`noise_method` with no `flow_ode` case, so `flow_ode` + `joint_logprob: False` would have
raised `UnboundLocalError`. It now raises a `ValueError` naming the fix — that combination is
meaningless (a deterministic rollout has no stochastic step to single out) and it matters
because `compute_teacher_velocities` reads `denoise_inds` to decide which flow state to
supervise.

Verified on the real checkpoint and real observations
(`toolkits/mopd/probe_flow_ode.py`):

| claim | result |
|---|---|
| `mode="train"` + `flow_ode` == `mode="eval"`, same seed | **bit-identical**, max abs diff `0.000e+00` |
| `denoise_inds` under `joint_logprob: True` | `[0, 1, 2, 3]` = arange, as intended |
| transition log-probs | exactly `0.0` (`get_logprob_norm` masks `sigma == 0`) |
| `flow_sde` regression | still stochastic, still one repeated index |
| `flow_ode` + `joint_logprob: False` | raises with the intended message |

One subtlety that briefly looked like a failure: under `joint_logprob: True` the log-prob
array has `num_steps + 1` entries and entry 0 is **not** a transition — it is
`log N(x_T; 0, I)` for the initial noise draw, nonzero by construction. OpenPI prepends the
same term (`openpi_action_model.py:991`), so the two families agree.

### 2. `analytic_local_kl` no longer asserts OpenPI-only

The assert was replaced by `_flow_sde_transition_spec`, a new helper returning
`(noise_level, action_chunk, action_env_dim, c_sign)` per model family — the only things
that genuinely differ. GR00T reads `rl_head_config.noise_level`, `num_action_chunks` and
`action_dim`; OpenPI keeps its `openpi.*` keys.

Checked against the real MOPD config: the helper returns `noise_level=0.5`,
`action_chunk=16`, `action_env_dim=12`, and `flow_sde_kl_weights` fed that `noise_level`
reproduces the GR00T-derived `w(tau)` exactly — `[0.125, 0.210938, 0.632812, 1.898438]` both
ways.

### 3. M4 carries GR00T's sign of `c`

`_teacher_transition_logprobs` now uses `c_sign * c`. This is the one place the sign cannot
be ignored: `mu_T = x_k (w0 + w1) + v_T * c` is affine in `c`, so the wrong sign would not
raise — it would move the teacher's transition mean the wrong way along the velocity and
quietly invert the reward `r_OPD = l_T - l_S`. With the real config, `c_sign * c` gives
`[0.25, 0.28125, 0.28125, 0.28125]`, matching GR00T's own branch; unsigned it would have been
the negation.

`w0 + w1` and `sigma` were confirmed identical, so nothing else in M4 needed changing.

### 4. The MOPD config is now actually M1

`robocasa365_mopd_gr00t.yaml` switched from `flow_sde` + `joint_logprob: False` to
`flow_ode` + `joint_logprob: True`, matching the mechanism study's
`maniskill_veldistill_openpi_pi05.yaml`.

Note `joint_logprob` means the opposite of what it means under a stochastic `noise_method`:
here it makes *every* step take the `mode="train"` branch, which under `flow_ode` is
deterministic. Downstream already handles both cases —
`compute_teacher_velocities` draws a uniform random supervised step when `joint_logprob` is
True (a pure ODE rollout has no distinguished step) and reads the rollout's sampled step when
it is False.

**`noise_level` is now unused by M1.** The concern recorded in RESULTS.md about inheriting
the PPO/GRPO default of 0.5 no longer applies to M1 or M3; it applies only to M3-Hybrid and
M4, where the noise is the mechanism being studied.

## What is still untested

The three fixes were verified at the level they act on — the `flow_ode` rollout end to end on
the real model, the coefficients numerically against GR00T's own branch, the helper against
the real config — plus a short MOPD training run to exercise the new
`joint_logprob: True` path through `compute_teacher_velocities`.

**But M3, M3-Hybrid and M4 have never executed on GR00T.** Their asserts are lifted and their
maths checked, and that is not the same as a run. Each still needs its own config and a smoke
run before any result from it is trusted. M4 additionally carries a known defect from the
mechanism study: it hung reproducibly five times on OpenPI, right after "Connected to Ray
cluster", and that was never diagnosed — see `.project_results/mechanism_study/README.md`.
Nothing here addresses that.

## End-to-end smoke run (2026-08-24)

`robocasa365_mopd_gr00t` with `runner.max_steps=2`, the config's own 64 training envs,
`status=OK`, no errors. This is the first time M1's full loop has executed on GR00T.

```
step 1   velocity_cosine=0.980   velocity_mse=0.038   velocity_norm=13.98
step 2   velocity_cosine=0.981   velocity_mse=0.035   velocity_norm=14.13
Routed teacher velocities over {'atomic_seen': 80, 'composite_seen': 40, 'composite_unseen': 40}
                               {'atomic_seen': 80, 'composite_seen': 60, 'composite_unseen': 20}
                               {'atomic_seen': 20, 'composite_seen': 60, 'composite_unseen': 80}
validation (48 trajectories, horizon 1664)   success_once=0.1875   success_at_end=0.1458
```

What this confirms: the `flow_ode` rollout feeds `chains` into
`compute_teacher_velocities`' `joint_logprob: True` path (uniform random supervised step per
sample) and that path runs; suite routing dispatches per sample, not as a fixed split — the
three teachers' shares differ across ranks; and the distillation signal is sane, with the
student's velocity field already well aligned in direction.

**These numbers are not comparable to the withdrawn 10-step run** (`velocity_mse`
0.0030 → 0.00086), which ran with the wrong `embodiment_id` and a stochastic rollout. This is
a fresh starting point.

### Open question: in-training validation vs the standalone baseline

The undistilled student measured **0.35** through the standalone harness (all six tasks,
horizon 1664, 40 trajectories). In-training validation after two steps gives **0.1875**
(48 trajectories).

That is a difference of 14/40 versus 9/48 successes — about **1.7 standard errors**, so it is
*not* on its own evidence of a problem, and it sits inside the run-to-run spread already
documented for this benchmark. But it also does not confirm that the two evaluation routes
agree, and they differ in ways that could matter: the in-training path evaluates the
FSDP-wrapped actor's weights after a weight sync, the standalone path loads a checkpoint
fresh.

**Cheap way to settle it:** run one validation at step 0, before any gradient step. If it
lands near 0.35 the drop is from training (or noise); if it lands near 0.19 the two routes
disagree and the in-training number cannot be compared against the standalone baselines —
which would matter for every MOPD result read off the training curve.

### Resolved: the two evaluation routes measure different things

The 0.35-vs-0.1875 gap above is **not** a weight-path or FSDP problem. Diffing the two
`env.eval` sections shows the cause outright:

| key | standalone eval | in-training validation |
|---|---|---|
| `episode_horizon_source` | **`task_horizon`** | **`max_episode_steps`** |
| `seed` | 7 | 0 |

Under `task_horizon` each task runs its own native horizon, so within a 1664-step rollout
epoch the ~450-step atomic tasks finish and `auto_reset` starts another episode — roughly
three each. That is why 24 envs produced **40** trajectories while 48 envs produced exactly
**48**.

The consequence is not just bookkeeping. `success_once` is an average over episodes, so
`task_horizon` **over-weights the short tasks** — and the short tasks are precisely the atomic
pair, where the pilot puts the student highest (0.75). `max_episode_steps` gives every task one episode per env, weighting the six
equally.

Rough arithmetic, using the per-suite student baselines:

```
task_horizon      ~24 atomic episodes @0.75  + ~16 composite @~0.03  ->  ~0.45   (measured 0.35)
max_episode_steps  16 atomic episodes @0.75  +  32 composite @~0.03  ->  ~0.27   (measured 0.1875)
```

**The two aggregates were never comparable.**

#### Confirmed by measurement

The same undistilled student was re-run through the **standalone** harness with the
in-training convention (`episode_horizon_source=max_episode_steps`, 48 envs, seed 0):

| measurement | convention | success_once | n |
|---|---|---|---|
| standalone, as originally run | `task_horizon` | 0.35 | 40 |
| **standalone, matched** | `max_episode_steps` | **0.2708** | 48 |
| in-training validation (after 2 steps) | `max_episode_steps` | 0.1875 | 48 |

The arithmetic above **predicted ~0.27 before the run**; the measurement came back 0.2708.
That is the mechanism being right, not two numbers happening to land closer. `n = 48` also
confirms exactly one episode per env under `max_episode_steps`.

Changing nothing but `episode_horizon_source` moves the same checkpoint from 0.35 to 0.2708.
The residual against the in-training number is 13/48 vs 9/48 — **0.97 standard errors**, not
significant, and consistent with sampling plus two gradient steps.

**There is no evaluation-route discrepancy.** The FSDP weight sync and the in-training
validation path are sound.

#### Method note

The step-0 diagnostic proposed above would have been actively misleading. It removes "the two
gradient steps changed the policy" by construction, but it cannot separate a route difference
from sampling noise at n=48 — and here it would have returned ~0.19, inviting exactly the
wrong conclusion. The decisive step cost nothing: diff the two configs. This is the same
failure mode as judging code-path equivalence from success rates, which already cost a round
earlier in this work.

#### What follows

`max_episode_steps` is the better convention **for this experiment** — equal weighting across
tasks, and composite tasks get their full horizon. The in-training validation is already set
that way. What needs fixing is the comparison: the **standalone aggregate baseline of 0.35
must not be used** as the reference for the training curve. The per-suite baselines
(atomic 0.75 / composite_seen 0.0625 / composite_unseen 0.0) are unaffected — within a suite
the two tasks have similar horizons, so repetition barely varies.
