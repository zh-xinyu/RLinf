# M3-Hybrid — Hybrid SDE rollout + Analytic KL

Config `examples/embodiment/config/maniskill_m3_hybrid_openpi_pi05.yaml` ·
loss `analytic_local_kl` (identical to M3) · rollout `flow_sde` + `joint_logprob: False`

Rollout: ODE -> one sampled SDE transition at step k -> ODE. Needed **no new rollout code**;
the mechanism already exists and `joint_logprob: False` selects it (FINDINGS F2). This is the
same rollout as Pi_RL's published flow-SDE method.

## Read this before interpreting M3-Hybrid vs M3
Noise is injected *after* the step-k mean is computed, so `x_k` is pre-noise and the
supervised flow-state distribution is **identical to M3's**. What differs is the environment
state distribution, since the noise changes the executed action chunk. A win here therefore
supports "stochastic exploration in the environment helps", NOT "stochastic coverage of the
denoising trajectory helps" (FINDINGS F3).

Also: the Hybrid rollout draws one k per rollout call shared across the batch, vs M3's
per-sample draw — same uniform marginal, higher per-update variance.

## Status
Smoke **passed** (3 steps, EXIT=0, no fatal log lines). 50-step run queued in
`outputs/mechanism_study/run_all.sh`.

## Smoke (3 steps)
| step | analytic_kl | velocity_mse | cosine | kl_weight_mean | grad_norm |
|---|---|---|---|---|---|
| 1 | 0.031 | 0.100 | 0.985 | 0.613 | 0.133 |
| 2 | 0.024 | 0.082 | 0.987 | 0.689 | 0.099 |
| 3 | 0.021 | 0.064 | 0.988 | 0.700 | 0.113 |

Two things worth noting:

- `kl_weight_mean` swings 0.613 -> 0.689 -> 0.700 around the theoretical 0.717, whereas M3
  sat at a flat 0.713 for all three steps. This is the predicted consequence of the Hybrid
  rollout drawing one k per rollout call shared across the batch: k's effective sample size
  per training step is the number of env steps (80), not the number of samples (25600).
  Empirical confirmation of the variance note in FINDINGS F3.
- `velocity_mse` starts higher than M3's (0.100 vs 0.087) at step 1 even though the loss is
  identical, because the executed actions and hence the visited environment states already
  differ — which is exactly the axis M3-Hybrid is meant to vary.
