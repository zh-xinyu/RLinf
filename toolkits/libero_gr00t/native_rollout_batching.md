# Native rollout efficiency development

Branch: `experiment/libero-gr00t-rollout-batching`; isolated worktree
`outputs/worktrees/libero_gr00t_rollout_batching`. It inherits the exact T64
source snapshot. Existing T32/T64 experiment worktrees remain unchanged.

## Preferred exact-output candidate: skip frozen terminal slots

Use `--skip-finished-rollout`, or set
`rollout.model.opd_skip_finished_rollout: true`. This option keeps B1 inference
and cannot be combined with native batching. It requires frozen episodes,
no autoreset/ignored termination, deterministic ODE, disabled dropout, and no
critic. The environment worker passes a latched, batch-aligned active mask.

Live slots execute the unchanged policy. Terminal slots reuse their previous
finite storage payload, which existing terminal masks exclude from training;
the environment already ignores their actions. The original B1 initial and
zero-variance-step random draws are still consumed in slot order. A new round's
all-active mask resets the cache. No valid observation is dropped and physical
trajectory budgets are unchanged. Cached inactive payloads are not new
post-terminal demonstrations.

`benchmark_skip_finished` checks exact live actions, full live training payload
and CPU/CUDA RNG against the full-B1 reference at16/8/4/0live slots. It uses
real checkpoint/observations and synthetic latched masks; full distributed
rollout timing and closed-loop validation remain separate. The inherited T64
source is frozen and the action00LIBERO queue was stopped by the user.

## Experimental batching implementation

This opt-in optimization combines student policy forwards for compatible native
observations. The default is still B1. It changes neither the number of physical
environments nor T, B512, actor microbatch, loss or teacher parameters.

1. Preprocess every observation independently using the existing image/state/
   native-token contract. No policy-visible padding is added.
2. Bucket by prepared tensor shapes, dtype and device, including true text length
   and image layout. Tail groups use their actual size; no duplicate rows are added.
3. Presample each environment's initial flow noise and all original per-step
   noise draws in the original B1 order, then gather these tensors for each group.
   Current ODE execution draws transition noise even when its multiplier is zero;
   preserving these draws also preserves the next rollout's RNG state. Initial
   dtype is resolved from the actual head input, not guessed from autocast.
4. Batch the VLM backbone; keep the flow head at B1 by default. Restore original
   environment order for actions,
   full flow chains, log probabilities, token storage and raw-state metadata.
   Keep action unnormalization/physical conversion per observation.

Only deterministic `flow_ode`, disabled dropout and `model.eval()` are supported
by this first version. Batched BF16 kernels can still produce numerical
differences. Static checks and latency do not establish unchanged closed-loop
task success. The running formal experiments retain B1.

## Configuration

Configure only the rollout model:

```yaml
rollout:
  model:
    opd_native_rollout_batch_size: 4
    opd_native_rollout_batch_head: false
```

Or use the isolated branch's launcher option:

```text
--native-rollout-batch-size 4
```

Both `launch_production` and `launch_formal` accept the option; omission preserves
the existing recipe. Start with a maximum of2or4. Full head batching (`opd_native_rollout_batch_head:
true`, or the benchmark's `--batch-head`) is a separate experimental variant; it
has already produced a gripper flip in the initial static probe. Actor and routed teacher model
configs are unchanged. The actual rollout loader applies the override; the
independent checkpoint evaluator carries it in its derived policy config.
Its current one-environment evaluator still executes B1 and receives no speedup.

The allocation training signature records an enabled batch limit. Existing
teacher calibration data can be rebound through the verified reference importer,
but an old signature must not be silently reused. Strict resume rejects a
changed rollout recipe. Use a new run ID/group for a later batching experiment;
do not insert the optimization midway through a frozen formal comparison.

## Validation

Unit/regression tests cover exact sample coverage, independent input preparation,
no padded tails, full output ordering, CPU RNG equivalence (including discarded
ODE transition draws), rollout-only configuration, old native/fixed-padding
paths, strict checkpoint loading and resume. Real-model validation uses the
accepted offline-update50checkpoint and recorded teacher observations, without
additional physics or student updates.

On action10 with the isolated wrapper, explicitly select one suitable GPU:

```bash
OPD_ROOT=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try
OPD_REPO="$OPD_ROOT/outputs/worktrees/libero_gr00t_rollout_batching"
OPD_PYTHON="$OPD_REPO/outputs/runtime_libero_profile_action10_20261105_r2/bin/python"
cd "$OPD_REPO"
CUDA_VISIBLE_DEVICES=1 "$OPD_PYTHON" -m toolkits.libero_gr00t.benchmark_rollout_batching \
  --config "$OPD_ROOT/outputs/libero_gr00t_formal_T64_20261107/M1_Uniform-R2_seed0/resolved_config.yaml" \
  --pool "$OPD_ROOT/outputs/libero_gr00t_20261104/offline_distillation_20261107/teacher_observation_pool" \
  --output "$OPD_ROOT/outputs/libero_rollout_batching_20261107/gpu_probe_action10" \
  --sizes 1 2 4 --repeats 3
```

The command requires a new output directory, at least24GiB free and limits its
PyTorch allocator to16GiB. Choose GPU IDs using current availability; do not
rerun into the active output directory or displace another process.

Two16-observation patterns measure mixed four-suite tasks and a repeated-task
upper bound. Reports include actual bucket sizes, warmed end-to-end policy
latency, peak memory, exact initial noise/RNG/input checks, complete-chain error,
vectors at identical baseline flow states, physical action errors and gripper
flips. The declared static gate uses zero gripper flips, at most0.005physical
non-gripper absolute error and at most1%relative vector-field L2 error, alongside
exact inputs/noise/RNG. These are engineering tolerances, not a guarantee of
unchanged task success. A gate failure remains a diagnostic result.

Artifacts: `outputs/libero_rollout_batching_20261107/`, including
`gpu_probe_skip_finished/results.json` and `frozen_slot_waste.json`, plus
`regression_tests.log`, launch identity and GPU `results.json`. The research
entry under `docs/opd/libero_gr00t/` records the measured outcome and remaining
closed-loop checks. Formal rollout timing contains environment, synchronization
and model-management costs beyond this isolated policy benchmark; do not equate
its measured speedup with total-training speedup.

## Real closed-loop result (2026-11-08)

`benchmark_skip_closed_loop --config <resolved_config.yaml> --output <new_dir>`
uses one explicitly selected GPU and four real suite environments, one official
state each. The65decision action/flow/RNG traces and per-task outcomes/steps
matched exactly. Model forwards260→88, policy74.77→25.24s, rollout loop87.89→38.30s.
These are protocol/latency checks, not a success-rate estimate or full training
speedup. Raw artifacts are `closed_loop_skip_probe/` and `completion_audit.json`
under the common artifact root. The next engineering gate is Ray/FSDP training;
no existing formal training recipe was changed in place.

## Actor cache at the rollout handoff

The first T64 training round completed, but the following rollout reload hit OOM
on shared action10GPUs: the actor still occupied19.79GiB while the peer job used
12.41GiB. The ordinary20GiB retention bound kept idle cache below that bound.
Clearing in `recv_rollout_trajectories` was too late because it can overlap the
rollout model reload.

With cache retention enabled, the actor now completes weight transfer, offloads
parameters/gradients, drops temporary state-dict views and explicitly releases
unused cache back to its configured floor before its synchronization handle
returns. The runner waits for that handle before starting rollout generation.
The ordinary retention policy during training and the12GiBfloor are unchanged.
This adds no random draw and changes no learning parameter. Logs report actor
allocated/reserved memory after each handoff. Tests cover below-bound release,
live-tensor/RNG preservation and transfer→offload→release ordering.
