# Canonical fixed-padding audit

`opd_canonical_fixed_padding: true` is an explicit new policy-input contract.
It requires `opd_fixed_text_padding: true`; the default is false. Before rollout,
NFT or flow refresh, valid text tokens are compacted to the left and padded on
the right to the configured fixed length (570 in this audit). This removes
batch-dependent processor left padding. Existing native/default behavior remains
unchanged. Fixed570 and native prompts are different policies: no equivalence or
success-rate claim follows from this option.

The bounded GPU audit uses four saved observations with differing prompt lengths:

```bash
CUDA_VISIBLE_DEVICES=<free-device> "$PROBE_PYTHON" -m \
  toolkits.libero_gr00t.fixed_padding_parity --run-gpu \
  --config /path/to/native-training/resolved_config.yaml \
  --observations /path/to/four_suite_observation_fixtures \
  --deterministic --output outputs/canonical_fixed570_parity
```

Alternatively use `--fixture-manifest /path/to/fixtures.json`; only its firstfour
entries are used, and at leasttwo true text lengths are required. The tool loads
the same base student/config and overrides only the two padding flags. It creates
no Ray cluster, simulator or optimizer. Default preflight requires40 GiB free and
caps PyTorch allocation at36 GiB; this is not a reservation. B4 backpropagation
requires substantially more memory than a B1 inference smoke.

The diagnostic temporarily intercepts exactly one initial `torch.randn` draw per
policy/refresh call, injecting the same recorded full16×32 noise for each item.
The patch requires the GR00T action-head `get_rl_action` callsite; an unrelated
same-shape draw fails rather than being replaced. Returned chain noise is checked
exactly against the injected tensor. The patch is local to the tool process and
is restored after each call. It checks:

- Legacy fixed-B1 versus canonical fixed-B1 actions and complete chains.
- True canonical B4 preprocessing/actions/chains versus four canonical B1 calls.
- B4 versus B1 NFT and FLOW_REFRESH with fixed inputs/noise.
- Seven selected action-head gradients from a mean B4 loss versus four B1 losses
  scaled by1/4, plus repeated B1/B4 backward controls.

Absolute/relative tolerances default to1e-5/1e-4 and are recorded before execution.
Preprocessing tensors always require exact equality. With `--deterministic`,
within-B1 and within-B4 repeat controls also require exact equality; cross-batch
model outputs/gradients retain the declared tolerances. Failures are grouped as
preprocessing, legacy-single, batch-forward, within-mode-repeat and batch-gradient,
so reproducibility failures are distinguishable from cross-batch numerical error.
Every measured error is retained; failed comparisons produce a nonzero exit and
`passed:false`. Investigate numerical differences rather than raising tolerances
merely to make a run pass. A batch-kernel numerical difference is possible even
when canonical text inputs match. The audit does not prove FSDP/world-size,
teacher-scoring, task-success or throughput equivalence.

Checkpoint/config/source/fixture identities and injected noise are saved with the
report. Changes to the model source invalidate previously prepared teacher
calibration source fingerprints; re-prepare an empty calibration manifest using
the reviewed source revision before collecting episodes. Do not relabel existing
calibration results as having used the new source. No current candidate training
recipe or evaluation guard is switched by adding this capability.


## Optional diagnostic precision control

Add `--fp32-head-control` with a **new output directory** to keep the frozen VLM
in BF16 and perform action-head computation in FP32. This temporarily wraps only
`action_head.forward` and `action_head.get_rl_action`, casts their floating boundary
inputs to FP32, disables autocast inside those calls, and disables TF32 for the
control. Original methods and TF32 flags are restored on exit. No production model
or training configuration is changed. Turning off outer autocast alone would not
provide this controlled head/backbone boundary.

The report records the execution override, effective TF32 flags and separate
B4/B1 frozen-backbone feature error. This helps determine whether batch numerical
differences remain upstream of the head. It retains the original strict tolerance
and exact-repeat requirements; the existing BF16 report is never overwritten.
Within each run, NFT/gradient comparisons use exactly the same flow state. Across
precision-control runs the state is drawn from that run's own student flow, so
changes in gradient discrepancy are diagnostic rather than a pure fixed-state
causal decomposition. This control is not a recommendation to change training
precision, padding, or the initialization, and it makes no performance claim.
