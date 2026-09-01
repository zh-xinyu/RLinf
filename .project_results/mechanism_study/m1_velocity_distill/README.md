# M1 — On-Policy Velocity Distillation (baseline)

Config `examples/embodiment/config/maniskill_veldistill_openpi_pi05.yaml` ·
loss `velocity_distill` · rollout `flow_ode` + `joint_logprob: True`

`L = mean || v_S(A_t^tau, o_t, tau) - stopgrad(v_T(...)) ||^2`, uniform over denoise steps.
No reward, advantage, critic, or PPO.

## Status
Smoke passed (3 steps). 50-step standard run **not started** — held until all four smokes pass.

## Important: the earlier 0.55 -> 0.74 result is NOT this baseline
That run (branch `opd-velocity-distill`, 2026-08-03) used `flow_noise` rollout and the
pre-fix flow-timestep pairing. Both changed, so its numbers are not comparable to this
study's M1 and it must be re-run. See FINDINGS F5.

## Smoke (3 steps)
| step | velocity_mse | cosine | ‖v_S‖ | ‖v_T‖ |
|---|---|---|---|---|
| 1 | 0.087 | 0.987 | 12.90 | 15.15 |
| 2 | 0.067 | 0.988 | 13.02 | 14.80 |
| 3 | 0.056 | 0.989 | 13.08 | 14.43 |

`grad_norm` 0.266, `lr` 5.0e-5, env/success_once 0.597 -> 0.734 -> 0.759 (3 steps is far too
short to read anything into task success).

## To record here later
50-step training curves, eval at steps 0/10/20/30/40/50 via the reusable harness
(`evaluations/eval_embodied_agent.py` + `evaluations/maniskill/maniskill_veldistill_openpi_pi05_eval.yaml`,
`runner.ckpt_path=<run>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt`),
plus the SFT and teacher reference points under the identical harness.
