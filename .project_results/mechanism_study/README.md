# Flow-Based OPD Mechanism Study — Results

The final four-arm comparison completed 25 steps for M1, M3, M3-Hybrid, and M4. The
authoritative run metadata and values are in [`tables/01_runs.tsv`](tables/01_runs.tsv) and
[`tables/03_summary.tsv`](tables/03_summary.tsv). The 50-step M1/M3/M3-Hybrid runs below are
an earlier, longer-horizon comparison retained as supporting evidence.

Cross-cutting mechanism conclusions are recorded in [`FINDINGS.md`](FINDINGS.md).

Branch `opd-mechanism-study`. Model pi0.5, env ManiSkill `put_on_plate_in_scene_25_main`,
student = SFT, frozen teacher = RL-FlowSDE. All four methods share every training setting
except the two axes under study (see FINDINGS F6).

## The comparison

| Method | Rollout | Training loss | Question |
|---|---|---|---|
| [M1](m1_velocity_distill/) | ODE | velocity MSE | is direct on-policy velocity distillation enough? |
| [M3](m3_analytic_kl/) | ODE | analytic local KL | does local probabilistic OPD add anything? |
| [M3-Hybrid](m3_hybrid/) | Hybrid SDE | analytic local KL | does stochastic exploration itself help? |
| [M4](m4_sampled_reverse_kl/) | Hybrid SDE | sampled reverse-KL | does sampled OPD beat stochastic exploration alone? |

Note (FINDINGS F1): M3's loss is provably M1's loss under a fixed timestep reweighting, and
(F3) M3-Hybrid's flow-state distribution equals M3's. The chain isolates less than the plan's
original framing suggested; read F1 and F3 before interpreting any result.

## Status

| Method | Implemented | Offline validation | Final 25-step run |
|---|---|---|---|
| M1 | yes | n/a | **done** (2026-08-17) |
| M3 | yes | **passed** (F1) | **done** (2026-08-17) |
| M3-Hybrid | yes (config + shared loss) | n/a | **done** (2026-08-17) |
| M4 | yes | **passed** (F4) | **done** (2026-08-18) |

## Preliminary results (50-step runs)

`success_at_end` at each validation, 320 fixed-seed episodes. M1 resumed from step 10, so it
has four eval points (steps 20-50) rather than five.

| eval @ step | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| M1 | — | 0.641 | 0.666 | 0.638 | **0.625** |
| M3 | 0.594 | 0.659 | 0.641 | 0.681 | **0.681** |
| M3-Hybrid | 0.653 | 0.659 | 0.625 | 0.625 | **0.662** |

All three converge equally well in training: `velocity_cosine` 0.999 and `velocity_mse`
0.0024 / 0.0029 / 0.0027 at step 50.

### The three methods are statistically indistinguishable

With 320 episodes the binomial SE is 0.027, so a difference between two runs carries
SE ≈ 0.038:

| comparison | difference at step 50 | significance |
|---|---|---|
| M3 − M1 | +0.056 | +1.5 SD — not significant |
| M3-Hybrid − M1 | +0.037 | +1.0 SD — not significant |
| M3 − M3-Hybrid | +0.019 | +0.5 SD — not significant |

The within-run spread across eval points (M1 0.041, M3 0.088, M3-Hybrid 0.037) is as large as
the between-method differences, which is the clearest way to see that none of these gaps is
real. No method shows a monotone climb over the 50 steps either.

**Reading against the plan's decision logic: this is Case A** — M1 ≈ M3 ≈ M3-Hybrid, i.e.
direct on-policy velocity distillation is sufficient and neither the analytic local KL nor
the Hybrid-SDE rollout buys anything measurable at this budget. That is exactly what
FINDINGS F1 predicts for M3 (it is M1 under a fixed timestep reweighting) and it narrows F3's
question for M3-Hybrid (stochastic environment exploration did not help here either).

**Caveats before this goes in a paper.** (1) One seed per method — the differences are within
noise, but so is any claim of equality; distinguishing "equal" from "too noisy to tell" needs
more seeds. (2) This preliminary 50-step round covers only three arms; M4 appears in the
final 25-step round recorded in the tables. (3) The SFT and teacher reference points have
**not** been re-measured under this harness, so
there is currently nothing to say about whether these runs beat the teacher — the old 0.55 /
0.691 numbers came from the pre-fix `flow_noise` setup and are not comparable (FINDINGS F5).

## Smoke results (3 steps, teacher = RL-FlowSDE)

Smokes check plumbing and metric sanity only; 3 steps is far too short for task success.

| | step | velocity_mse | cosine | analytic_kl | kl_weight_mean | grad_norm |
|---|---|---|---|---|---|---|
| M1 | 1 | 0.087 | 0.987 | — | — | — |
| M1 | 2 | 0.067 | 0.988 | — | — | — |
| M1 | 3 | 0.056 | 0.989 | — | — | 0.266 |
| M3 | 1 | 0.087 | 0.986 | 0.031 | 0.713 | 0.140 |
| M3 | 2 | 0.074 | 0.988 | 0.024 | 0.713 | 0.105 |
| M3 | 3 | 0.064 | 0.988 | 0.021 | 0.713 | 0.097 |
| M3-Hybrid | 1 | 0.100 | 0.985 | 0.031 | 0.613 | 0.133 |
| M3-Hybrid | 2 | 0.082 | 0.987 | 0.024 | 0.689 | 0.099 |
| M3-Hybrid | 3 | 0.064 | 0.988 | 0.021 | 0.700 | 0.113 |

M3's `kl_weight_mean` matches the theoretical 0.717 and is flat, confirming the weight
plumbing end to end. M3-Hybrid's swings (0.613 / 0.689 / 0.700) because the Hybrid rollout
draws one k per rollout call shared across the batch — empirical confirmation of FINDINGS F3.

Two sanity signals worth keeping:
- M1 and M3 agree exactly at step 1 (same seed, same deterministic ODE rollout), then
  diverge — as they must, since only the loss differs.
- M3's *unweighted* velocity_mse falls more slowly than M1's (0.064 vs 0.056), which is the
  expected cost of optimising the weighted objective instead.

M3's smaller grad_norm reflects `kl_weight_mean < 1`; not a confound here (Adam absorbs a
constant rescale, and both are far below `clip_grad: 1.0`).
