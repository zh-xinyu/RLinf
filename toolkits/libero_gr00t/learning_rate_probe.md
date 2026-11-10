# Native LIBERO learning-rate probe

`toolkits/libero_gr00t/launch_production.py` accepts `--learning-rate` for a
bounded fresh-start diagnostic. Omitting it preserves the selected YAML recipe.
A finite positive override changes only `actor.optim.lr` and
`actor.optim.value_lr`; `launch.json` records the requested override and both
effective rates. Warmup, scheduler, batch sizes, seeds and model precision stay
as configured. Strict checkpoint resume rejects a changed learning rate.

The first sensitivity probe uses the original weak student, native M1 inputs,
T32 / B64 / microbatch1 / Refresh-R2, LR **3e-5**, constant scheduling and the
existing four-update warmup. Add the following options to the normal launcher
command, retaining its runtime, checkpoint, GPU and output arguments:

```text
--updates 40 --save-every-updates 10 --learning-rate 3e-5
--sync-every-microbatch
--config-name libero_mopd_candidate_m1
--student-padding-profile native --rollout-steps 520
```

Do **not** pass `--resume-checkpoint`: this probe starts with a new optimizer and
scheduler. Use the same explicit training seed as the low-LR control. The strong
LR change is a diagnostic choice, not a validated formal recipe. The experimental
`canonical-fixed570` profile still requires LR3e-6.

Compare updates20/40 to the existing LR3e-6 updates20/40 and the original student using
all four suites, ten tasks each, trial1, evaluation seed1234 and the same native
FP32 action head / BF16 autocast contract. These 40 episodes are a quick paired
learning screen, not a formal performance estimate. Keep the existing fixed
initial-observation action diagnostic for comparing velocity and action error.
The current probe has a hard cap of40 updates, with evaluation at20 and40;
there is no automatic extension beyond40. Review the20-update signal while
training continues, and stop early if a numerical or rollout-chain failure is
confirmed. Changing LR during optimizer-state resume is intentionally unsupported.

## Changing the global batch

For a separate native batch-size probe, `--global-batch-size 256` preserves
microbatch 1 and uses 64 accumulated microbatches per rank on four GPUs. Valid
overrides are integer multiples of 16, at least 16, preserving equal four-suite
quotas on every rank. Omission preserves YAML B64. The `canonical-fixed570`
profile rejects this flag because it manages its own B256/micro4 recipe.
Strict resume requires the same effective batch: repeat the override when
continuing a B256 run. `launch.json` records both the override and effective
`batch_recipe`. Vary batch and LR in separately identified runs; do not alter
either halfway through a learning comparison.

## Optional allocator cache retention

Leave cache retention disabled for the current colocated diagnostic. A trial
with a 13 GiB floor and 18 GiB retention failed during the next rollout because
the retained actor cache overlapped rollout inference and simulator rendering.
The options below document available controls, not the recommended probe recipe.

On a shared GPU, the launcher can enable the existing actor cache retention:

```text
--actor-cache-retention-gib 18 --actor-cache-preallocate-gib 13
```

Both options default to zero, leaving the selected YAML unchanged. Preallocation
must be finite, nonnegative and no larger than retention. The actor creates
temporary allocations, then releases the tensors into its reusable allocator
cache; this does not draw random numbers or change the learning recipe. The
existing helper requires another8 GiB of free headroom for colocated workers
when filling the cache. Choose sizes from actual available memory and observed
peaks. This is not a GPU lock and does not guarantee protection against OOM.

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launch environment
to request expandable allocator segments; the launcher passes the effective
environment to its private Ray head and driver. `launch.json` records that value
and the effective actor cache sizes. Cache settings remain part of strict resume
validation, so repeat the same options when continuing that run. These controls
do not modify model precision, LR, B, microbatch, teacher routing or rollout reuse.

## Synchronize gradients every microbatch

`--sync-every-microbatch` sets the existing
`actor.fsdp_config.enable_gradient_accumulation` switch to `false`. Despite that
configuration name, mathematical gradient accumulation remains enabled: with
B64, microbatch1 and four actor ranks, each microbatch loss is divided by16 and
the optimizer and scheduler still advance once after16 microbatches per rank.
The flag disables FSDP `no_sync`, so each microbatch is reduce-scattered and its
gradient shard is accumulated. This trades more communication for lower gradient
memory. Floating-point reduction order changes; bitwise equality is not promised.

The option is off by default and leaves YAML unchanged when omitted. The manifest
records effective `gradient_synchronization.sync_every_microbatch` and
`optimizer_accumulation_steps`. FSDP wrapping, checkpoint tensor layout, model
precision, B, LR and loss are unchanged. Strict resume still requires the same
configuration: do not turn this flag on midway through an existing run. A fresh
two-update validation may be continued with the same flag and full checkpoint.
One rollout alone does not verify the next rollout after training; inspect the
first resumed round too. For a shared-card recovery, omit cache retention and
preallocation options so idle actor cache can be released before rollout.

Early round-boundary checkpoints now include the full ordered task catalog and
explicit zero occurrence counts for unvisited tasks. Legacy sparse counters are
accepted only when every completed round remains in the saved history and exact
replay reproduces its task identities, occurrences, reset seeds, cursors, quotas
and simulation count. Missing visited tasks, invalid counters and incomplete
history are rejected; the checkpoint file is never rewritten. A legacy file
cannot independently authenticate metadata for tasks it never visited. This
compatibility path therefore also relies on the launcher's unchanged environment
configuration check. New checkpoints compare the entire saved catalog, including
unvisited tasks. Allocator plan equivalence does not imply simulator/RNG replay.
