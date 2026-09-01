# M4 on GR00T — what was missing, and what was changed

Written 2026-08-27 on the RTX 6000 Ada node. M1 stays untouched: everything here is inside
`adv_type == "opd"` branches.

> Historical implementation note. M4 subsequently completed on GR00T. The per-suite reward
> interpretation from that run was later invalidated by a sampled-row logging bug; see the
> correction at the top of [RESULTS.md](RESULTS.md).

---

## 1. First, a correction to the premise

The task was framed as "M4 already ran on ManiSkill single-teacher OPD, reuse it". **It did
not.** Two independent records say so:

`.project_results/mechanism_study/README.md`

```
| M4 | yes | passed (F4) | hangs, 5 attempts | not started |
```
> M4's plumbing has **never executed end to end**: five attempts have all hung silently
> right after "Connected to Ray cluster" and before any placement output.

`mechanism_study/m4_sampled_reverse_kl/README.md`
> **Not implemented.** Design agreed 2026-08-11.

There is no M4 run artefact anywhere in `outputs/`.

**What did run is a different thing**: RLinf's own OPD on LIBERO + OpenVLA-OFT
(`libero_10_opd_openvlaoft.yaml`), which shares the `opd` advantage and loss registrations
with M4 but scores **action tokens**, not a flow transition.

This distinction matters for how to debug. Treating the flow path as "already proven" would
mean assuming the plumbing is sound — and the plumbing is precisely what was missing.

## 2. Three gaps, none of which could have been caught before

M4 has never reached the code below, so nothing here had ever executed.

### 2.1 The dispatch sent GR00T to the OpenVLA path

`fsdp_actor_worker.py`, before:

```python
if self.cfg.algorithm.adv_type == "opd":
    if SupportedModel(...) == SupportedModel.OPENPI:
        self.compute_teacher_velocities()      # flow transition
    else:
        self.compute_opd_teacher_logprobs()    # OpenVLA action tokens
```

GR00T fell into `else`. The assert inside `compute_opd_teacher_logprobs` restricts it to
OPENVLA / OPENVLA_OFT, so this fails loudly rather than silently — but M4 on GR00T could not
start.

Lifting the OPENPI-only assert in `774ea54c` changed the assert *inside*
`compute_teacher_velocities` and left this dispatch alone.

**Fixed** with a module-level `FLOW_MODELS = (OPENPI, GR00T)`, so the condition states the
property that matters — the action head is a flow — rather than naming one model.

### 2.2 Three log-prob shapes, none of which agreed

| producer | shape |
|---|---|
| rollout `prev_logprobs` | `[T, B, num_steps(+1), chunk, dim]` — every denoise step is scored; under `joint_logprob: False` all but the sampled one have `sigma = 0`, so `get_logprob_norm` masks them to exactly 0 |
| `_teacher_transition_logprobs` | `[T, B, chunk, dim]` — only the sampled step exists |
| training forward | `[N, 1, chunk, dim]` — with `joint_logprob: False` the loop runs once, on `denoise_inds[:, 0]` |

`compute_opd_advantages` asserts the first two are equal, and the OPD loss asserts the
advantage matches the third. All three assertions would have failed.

**Fixed** by `_align_opd_logprobs`, which reduces the rollout's per-step log-probs and the
teacher's to the training forward's `[.., 1, chunk, dim]` by selecting the sampled denoise
step.

Two details that are easy to get wrong and are handled explicitly:

- **Selection, not summation.** The non-sampled steps are zero *today* because they are
  deterministic. Summing would give the same answer now and silently start mixing steps in if
  that ever changes.
- **The `joint_logprob` offset.** When it is True the rollout prepends `log N(x_T; 0, I)` for
  the initial noise, so denoise step `k` sits at index `k + 1`. Picking index `k` would
  quietly return the noise term instead.

Verified on CPU for both `joint_logprob` settings: shapes agree, the selected entry is the
sampled step, and the initial-noise term is never picked.

### 2.3 The rollout worker skips the expert model under OPD — correctly

`huggingface_worker.py` loads `rollout.expert_model` only when `not self.enable_opd`, because
OPD computes teacher quantities post-rollout on the actor. Routed MOPD uses
`rollout.expert_models` (plural) and `_attach_teacher_ids`, which are independent of that
flag, so the router keeps working. Checked, not changed.

## 3. What was reused unchanged

The whole M1 routing mechanism carries over: `expert_models`, `_attach_teacher_ids`,
`_get_routed_teacher_model`, `_teacher_velocity_pass` (one teacher on the GPU at a time), and
`_suite_balanced_sample`. M4 differs from M1 only in the loss, the advantage, and the rollout
being Hybrid SDE.

`_teacher_transition_logprobs` also already had the sign fix from `774ea54c`: it uses
`c_sign * c`, because GR00T's `c` is the negative of OpenPI's and `mu_T` is affine in it — the
wrong sign would not raise, it would invert `r_OPD`.

## 4. Config: `robocasa365_mopd_m4_gr00t.yaml`

Derived from the M1 config, three differences:

| setting | M1 | M4 |
|---|---|---|
| `adv_type` / `loss_type` | `gae` (placeholder) / `velocity_distill` | `opd` / `opd` |
| `noise_method` | `flow_ode` | `flow_sde` |
| `joint_logprob` | `True` | `False` |

`noise_level` is **live again** under M4 — M1's rollout is deterministic, so the value was
inert there. It is 0.5, inherited from the PPO/GRPO defaults, and the measured train-vs-eval
action deviation at that value is 50-90% relative. It is a real knob for M4 and has not been
chosen deliberately yet.

## 5. Status at the time of this note

- **The hang.** Five reproducible hangs in the mechanism study, before any placement output,
  on both healthy and unhealthy Ray, while M1 and M3 ran fine through the same runner.
  Undiagnosed. **Do not treat a hang as flaky Ray and retry.**
- At this point no M4 run had completed on GR00T. A run completed on 2026-08-28; see
  `RESULTS.md`, including the later diagnostic correction.
