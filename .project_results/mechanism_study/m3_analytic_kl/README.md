# M3 — Local Flow-SDE Analytic KL

Config `examples/embodiment/config/maniskill_m3_openpi_pi05.yaml` ·
loss `analytic_local_kl` · rollout `flow_ode` + `joint_logprob: True` (same as M1)

`L = D_KL(p_S || p_T) = w(tau) * || v_S - stopgrad(v_T) ||^2` at the student-visited state,
shared covariance schedule.

## Headline finding: this is M1 with a fixed timestep reweighting
Proved analytically and checked numerically — see FINDINGS F1. `w(tau)` is proportional to
`(1 - tau) / tau`; for num_steps=4, noise_level=0.5 the weights are
[0.125, 0.211, 0.633, 1.898] (relative 0.17 / 0.29 / 0.88 / 2.65). So M3 vs M1 tests exactly
one thing: does shifting supervision toward late, near-data denoise steps help?

## Status
Offline validation passed (all four of the plan's required checks). Smoke passed (3 steps).
50-step run **not started**.

## Offline validation
`scratchpad/m3_analytic_kl_validation.py` (CPU): means affine in v; KL finite; analytic KL
matches a 400k-sample Monte-Carlo estimate of `E_{A'~p_S}[log p_S - log p_T]` to <0.01% at
every timestep; analytic gradient equals the weighted velocity-MSE gradient.
Loss unit test: weighting, mask path, teacher==student -> loss 0 / cosine 1, gradient flows,
teacher detached.

## Smoke (3 steps)
| step | analytic_kl | velocity_mse | cosine | grad_norm |
|---|---|---|---|---|
| 1 | 0.031 | 0.087 | 0.986 | 0.140 |
| 2 | 0.024 | 0.074 | 0.988 | 0.105 |
| 3 | 0.021 | 0.064 | 0.988 | 0.097 |

`kl_weight_mean = 0.713` vs theoretical 0.717 — end-to-end confirmation of the weight
plumbing. Unweighted velocity_mse falls more slowly than M1's, as expected.
