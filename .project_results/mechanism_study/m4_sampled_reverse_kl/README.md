# M4 — Hybrid SDE rollout + Sampled Reverse-KL OPD

Config `examples/embodiment/config/maniskill_m4_openpi_pi05.yaml` ·
rollout `flow_sde` + `joint_logprob: False` (identical to M3-Hybrid)

`L = -log p_S^new(A') * stopgrad(l_T - l_S^old)`, `A' = x_{k+1}` sampled from the student's
own SDE transition at step k. No PPO clipping initially, per the plan.

## Status
**Not implemented.** Design agreed 2026-08-11:
1. generalize `flow_sde_kl_weights` to also return `w0, w1, c, sigma`, so
   `mu_T = x_k (w0 + w1) + v_T * c` can be built from the teacher velocity;
2. new `compute_teacher_transition_logprobs`: reuse the existing teacher forward for `v_T`,
   build `mu_T`, then `l_T = log N(x_{k+1}; mu_T, Sigma_k)` via the same formula as
   `get_logprob_norm`;
3. `r_OPD = l_T - prev_logprobs`, **per element** (lower variance, see FINDINGS F4), no
   OpenVLA token reshape;
4. reuse `compute_opd_actor_loss` for the objective itself.

`l_S^old` needs no new code: under `joint_logprob: False` the rollout's `prev_logprobs` is
already the log-prob of that single transition. `l_S^new` comes from the standard
`default_forward` path, which under `joint_logprob: False` scores exactly the same transition
(`get_log_prob_value` sets `num_steps = 1` and reads `denoise_inds[:, 0]`).

## What this shares with RLinf's existing OPD, and what it does not
Same estimator (score-function reverse KL), different object: OpenVLA-OPD matches the
teacher's final action distribution, M4 matches one denoising transition. The teacher
log-prob and the advantage reshape are NOT reusable. See FINDINGS F4.

Reusing `opd` is a deliberate engineering-light choice; splitting M4 into its own
`sampled_reverse_kl` is recorded in `.project_plan/OPTIONAL_OPTIMIZATIONS.md`.

## Validation to run before trusting results (from the plan)
teacher log-prob; student old/new log-prob; OPD reward distribution; analytic KL at the same
transition; sampled vs analytic KL; gradient norms; stochastic-rollout success vs
deterministic-ODE eval success. Assert teacher and student share `noise_level`, or `Sigma_k`
silently differs between `l_T` and `l_S`.
