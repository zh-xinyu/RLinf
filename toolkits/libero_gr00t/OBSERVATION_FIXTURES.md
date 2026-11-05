# Fixed initial-observation bank

Use this bank only for paired-noise action/velocity diagnostics. It contains no
teacher/student policy trajectories and does not measure task success. The
standard initial-state indices are shared with training, not held out.

From the experimental worktree with the isolated LIBERO runtime and assets:

```bash
CUDA_VISIBLE_DEVICES=<available-device> "$PROBE_PYTHON" -m \
  toolkits.libero_gr00t.collect_observations --run-gpu \
  --initial-state 1 --seed 1234 --output outputs/libero_fixed_init1
```

The default collects all40 suite-labelled tasks. One GPU is used only for EGL
rendering; no model loads. Environments open and close sequentially. The free
memory preflight defaults to2 GiB and does not reserve memory. For a smaller
smoke use `--tasks 0`, retaining allfour suites.

Each fixture is captured after reset, assignment of the official initial state,
and exactly10 zero-motion/open-gripper settling calls, matching the existing
single-process evaluator. The reset seed is1234 for each environment, matching
`probe.create_env`; official initial-state index defaults to1. Camera images are
256×256. The collector stores raw image/proprioception arrays and task language,
without preprocessing them for a particular checkpoint.

On success, `fixtures.json` is the manifest for:

```bash
CUDA_VISIBLE_DEVICES=<available-device> "$PROBE_PYTHON" -m \
  toolkits.libero_gr00t.checkpoint_action_diagnostic --run-gpu \
  --config /path/to/training/resolved_config.yaml \
  --fixture-manifest outputs/libero_fixed_init1/fixtures.json \
  --teachers /path/to/spatial /path/to/object /path/to/goal /path/to/long \
  --seeds 2 --seed 1234 --output outputs/fixed_actions_update0.json
```

The manifest records suite, task index, official initial-state index, reset seed,
settling calls, artifact hash and the selected initial-state array's hash/dtype/
shape. Paths are relative to the manifest directory. Reuse the exact same bank
for baseline and trained checkpoints. Existing output directories are rejected.
An interrupted run retains partial artifacts and `complete:false` in its protocol;
it does not publish a complete `fixtures.json`.

CPU tests check reset/settling order, environment closure on failures, safe NPZ
loading and diagnostic manifest compatibility. Actual EGL collection requires a
separate GPU smoke; passing these tests alone does not verify rendering.
