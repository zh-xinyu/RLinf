# Flow-Based OPD Mechanism Study — Findings

Transferable findings, written for later writeup. Each entry says what was established, how,
and how much it is worth trusting. Plan: `.project_plan/Flow-Based OPD Mechanism Study.md`.
Branch: `opd-mechanism-study`.

Confidence legend: **[proved]** analytic + numerically checked · **[verified]** read from
code and confirmed by a run · **[observed]** seen in a run, not isolated · **[open]** claim
we cannot yet make.

The F8 comparison below describes the preliminary 50-step M1/M3/M3-Hybrid round. M4 later
completed in the final 25-step four-arm round; use `tables/03_summary.tsv` for that result.

---

## F8. At 50 steps, M1 ≈ M3 ≈ M3-Hybrid — the plan's Case A [observed]

`success_at_end` at step 50, 320 fixed-seed episodes: M1 **0.625**, M3 **0.681**,
M3-Hybrid **0.662**. All differences are under 1.5 SD (binomial SE 0.027; SE of a
difference 0.038) and the within-run spread across eval points (0.037-0.088) is as large as
the between-method gaps. No method climbs monotonically over the 50 steps. Training-side
convergence is identical: cosine 0.999, velocity_mse ~0.0025 for all three.

So neither the analytic local KL (M3) nor the Hybrid-SDE rollout (M3-Hybrid) buys anything
measurable over plain on-policy velocity distillation at this budget. For M3 this is what F1
predicts — it is M1 under a fixed timestep reweighting, so a large effect would have been
surprising. For M3-Hybrid it answers F3's narrowed question negatively: stochastic
exploration *in the environment* did not help here.

**What this does not license.** One seed per method: "within noise" is not "equal", and
separating the two needs more seeds. M4 is still missing, so Case A covers three of four
arms. And the SFT/teacher reference points have not been re-measured under this harness, so
nothing can yet be said about whether these runs match or beat the teacher — the old
0.550/0.691 figures belong to the pre-fix `flow_noise` setup (F5).

---

## F1. Analytic local reverse-KL on flow transitions is *not* a new mechanism [proved]

For the Flow-SDE transition kernel in `openpi_action_model.sample_mean_var_val`, the mean is
affine in the predicted velocity with a coefficient that teacher and student share:

```
mu = x_t * (w0 + w1) + v * c(tau),      c(tau) = -tau * w0 + (1 - tau) * w1
```

so `mu_S - mu_T = c(tau) * (v_S - v_T)`, and with a shared covariance the Gaussian KL is

```
D_KL(p_S || p_T) = w(tau) * ||v_S - v_T||^2,     w(tau) = c(tau)^2 / (2 * Sigma_tau)
```

Substituting `sigma_i = nl * sqrt(tau / (1 - tau))` gives `c = -delta * (1 + nl^2 / 2)`,
a constant, so the entire timestep dependence comes from `1 / Sigma_tau`:

```
w(tau) proportional to (1 - tau) / tau
```

**M3 is therefore exactly M1's velocity MSE under a fixed, known, parameter-free timestep
reweighting.** For `num_steps=4`, `noise_level=0.5`:

| tau | 1.00 | 0.75 | 0.50 | 0.25 |
|---|---|---|---|---|
| w | 0.125 | 0.211 | 0.633 | 1.898 |
| w / mean | 0.17 | 0.29 | 0.88 | **2.65** |

A ~15x spread, shifting supervision toward the late, near-data denoise steps.

*How established:* all four of the plan's required validations, offline and on CPU
(`scratchpad/m3_analytic_kl_validation.py`): means affine in v; KL finite; analytic KL
matches a 400k-sample Monte-Carlo estimate of `E_{A'~p_S}[log p_S - log p_T]` to <0.01% at
every timestep; analytic gradient equals the weighted velocity-MSE gradient elementwise.
Confirmed end-to-end in the M3 smoke: logged `kl_weight_mean = 0.713` vs theoretical 0.717.

*Why it matters for the writeup:* the M3-vs-M1 comparison answers one narrow question —
does emphasizing small-tau steps help — rather than "does probabilistic OPD help". Note the
caveat at `tau = 1`, where the schedule's denominator guard breaks the closed form and w
must be computed numerically.

---

## F2. Pi_RL's flow-SDE method *is* the Hybrid rollout [verified]

`_sample_actions_with_prefix_cache` uses `noise_method` only when
`idx == denoise_inds[0][idx]` and `flow_ode` otherwise. Under `joint_logprob: False` a single
`k` is drawn per rollout call, so the trajectory is ODE -> one SDE transition at k -> ODE.
Under `joint_logprob: True` every step is stochastic (full SDE).

The repo states the convention itself in `maniskill_ppo_openpi_pi05.yaml`:
`joint_logprob: True # False for flow_sde, True for flow_noise`.

So the study's "Hybrid SDE" needed **no new rollout code**, and M3-Hybrid/M4's rollout is
the same mechanism as the published flow-SDE baseline rather than something we invented.

---

## F3. M3-Hybrid does *not* broaden flow-state coverage — only environment coverage [verified]

Noise is added *after* the step-k mean is computed (`x_t = x_t_mean + noise * x_t_std`), so
`x_k` is reached by pure ODE and only `x_{k+1} ... x_n` carry the injected noise.

The supervised transition is (condition on `x_k` at `tau_k`, sample `x_{k+1}`), which is the
transition M4's sampled reverse-KL is defined on and the one the rollout's `prev_logprobs`
scores. Supervising there means **M3-Hybrid's flow-state distribution is identical to M3's**.
What differs is the *environment* state distribution `o_t`, because the noise changes the
executed action chunk.

*Consequence for interpretation:* if M3-Hybrid > M3, the plan's Case B reading
("stochastic state coverage helps") must be narrowed to **stochastic exploration in the
environment**, not stochastic coverage of the denoising trajectory. These are different
claims and the experiment only supports the former.

*Also:* the Hybrid rollout draws one `k` per rollout call shared across the whole batch,
while M3 draws one per sample. Same uniform marginal, so no bias, but higher per-update
gradient variance for M3-Hybrid.

---

## F4. M4 shares an estimator with the existing OPD, not an object [verified]

The score-function gradient of reverse KL is
`grad KL(p_S||p_T) = -E_{A~p_S}[grad log p_S(A) * (l_T - l_S)]`, so
`L = -log p_S^new * sg(l_T - l_S^old)` — which is literally `compute_opd_actor_loss`. That
layer is reusable regardless of distribution family.

Three things are **not** shared with RLinf's OpenVLA/LIBERO OPD:
1. distribution family — categorical over action tokens vs Gaussian over a flow transition,
   so `compute_opd_teacher_logprobs` (asserts OpenVLA) is unusable;
2. `compute_opd_advantages` reshapes by an OpenVLA token layout
   (`shape[-1] % num_action_chunks == 0`); pi0.5 logprobs are already `[T, B, chunk, dim]`
   and `7 % 5 != 0`, so it would assert;
3. **OpenVLA-OPD matches the teacher's final action distribution; M4 matches one denoising
   transition** (`A' = x_{k+1}` is an intermediate flow state).

*Estimator detail:* with diagonal Sigma the transition's components are independent, so both
the per-element form `-sum_i l_i * r_i` and the joint form `-(sum_i l_i)(sum_j r_j)` are
unbiased for the joint reverse KL; the per-element form drops zero-mean cross terms and has
strictly lower variance, so it is preferred. This Rao-Blackwell argument is *stronger* for
us than for OpenVLA, whose tokens are autoregressively dependent.

---

## F5. The flow-timestep pairing was wrong, and it inflated agreement [observed]

`compute_teacher_velocities` treated `chains`' step axis as `num_steps`, but it is
`num_steps + 1` (initial noise plus one state per step). With `num_steps=4` it sampled
`k in [0, 5)` and built `tau = linspace(1, 1/5, 5) = [1.0, 0.8, 0.6, 0.4, 0.2]`, while the
states correspond to `[1.0, 0.75, 0.5, 0.25, 0.0]`. Only `k=0` was right; `k=4` selected the
emitted action, where no velocity is ever evaluated. Fixed in commit `4ab7fafe`.

After the fix the measured teacher-student gap is ~100x larger:

| | before (flow_noise + wrong tau) | after (flow_ode + correct tau) |
|---|---|---|
| velocity_mse | ~7e-4 | 0.056 - 0.087 |
| velocity_cosine | ~1.000 | 0.987 - 0.989 |
| ‖v_S‖ vs ‖v_T‖ | 10.54 / 10.54 | 13.1 / 14.4 |

**[open]** This is *not* attributable to the tau fix alone: the rollout changed from
`flow_noise` to `flow_ode` in the same window and no ablation was run. The earlier writeup's
claim that teacher and student differ only by a "small-norm correction" should not be reused
without settling this — see `.project_plan/OPTIONAL_OPTIMIZATIONS.md`.

---

## F6. Standardization decisions [verified]

- lr stays **5.0e-5** for all four methods, *not* the reference PPO's 7.91e-6, which is tuned
  for a clipped policy-gradient objective rather than a supervised match. Fairness comes from
  all four sharing it.
- All four configs are otherwise field-for-field identical (checked): 50 steps,
  `val_check_interval`/`save_interval` 10, 320 envs, micro/global batch 32/5120, seed 0,
  `num_steps` 4, no value head, same teacher.
- M1/M3 use `flow_ode` + `joint_logprob: True`; M3-Hybrid/M4 use `flow_sde` +
  `joint_logprob: False`.
- Teacher is the composed dir `veldistill_teacher_RL-FlowSDE` (RL-FlowSDE weights, SFT
  norm stats via symlink) — teacher and student **must** share normalization.
- `Sigma_tau` is taken from the `flow_sde` closed form even for M3, whose rollout is
  deterministic ODE, because an ODE transition has `Sigma = 0` and an infinite KL.

---

## F7. `sync_module_states` [verified]

An uncommitted local hack had set `sync_module_states=False` after actor workers died at FSDP
init. Reverting to upstream `True` with `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` set passed
cleanly, so the hack was unnecessary and was dropped. Root cause of the original crash is
**[open]**. Relevant if it recurs: models load with `strict=False`, so a parameter absent
from the checkpoint keeps a per-rank random init and `sync_module_states=True` is what makes
ranks agree — disabling it can silently train an inconsistent model.
