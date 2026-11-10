# Native teachers through the production training collector

This diagnostic loads one frozen native LIBERO specialist per GPU and calls the
real `EnvWorker.interact` on `env.train` alongside
`MultiStepRolloutWorker.generate`. It starts no actor, optimizer, or FSDP group.
The inherited train reset, quota allocation, action preparation, chunk stepping,
terminal freezing, and HF sampling methods remain in use. Diagnostic subclasses
select one suite/teacher per rank and record contract checks.

Use a completed [teacher calibration capture](teacher_calibration.md) as the
reference. The launcher verifies its content hashes and actual task/trial/seed
protocol; a trial absent from that capture is rejected. Each teacher retains its
own scalar 8D state transform, fixed570 text, projector31, BF16 head, four ODE
steps, 16 generated actions, and eight executed 7D commands. Environment reset
seeds and per-episode policy noise seeds match the captured standalone reference.

From the experiment worktree, first prepare a four-episode smoke plan:

```bash
"$CLEAN_LIBERO_PYTHON" -m toolkits.libero_gr00t.launch_teacher_training_probe \
  --python "$CLEAN_LIBERO_PYTHON" \
  --config /absolute/path/to/native/resolved_config.yaml \
  --reference /absolute/path/to/completed/teacher_calibration \
  --output /absolute/path/to/new/teacher_train_smoke \
  --gpus 0 1 2 3 --tasks 0 --trials 0 \
  --memory-gib 10 --minimum-free-gib 12 \
  --capture-chunks 1 4 --timeout-seconds 1800
```

The default is a dry run: it writes a plan, checks reference/configuration
contracts, and starts no Ray or GPU workers. To execute after reviewing the plan
and available cards, repeat with a **new output directory** and add `--run`.
Select the four physical GPU IDs explicitly. Preflight requires at least12 GiB
free per card; the rollout allocator cap is10 GiB per process. EGL/driver memory
is outside that allocator cap. Inspect the smoke's measured peak and external GPU
usage before increasing episode coverage. The launcher uses only owner-scoped
cleanup for its private Ray cluster and bounds the driver lifetime.

For the full quick probe, omit `--tasks` and `--trials`: defaults select all ten
tasks per suite and initial-state indices0/1, for80 episodes in20 rounds. These
indices are checked against the reference rather than assumed available. Use a
new output directory and the default7200-second timeout, or another explicit
bound. The chosen native training source must retain frozen complete episodes,
ten settling steps, and the240/320/320/520 native suite horizons.

Acceptance requires exact unique coverage in `episodes.jsonl`, success or native
horizon exhaustion for every episode, and agreement of actual simulator counters
with policy steps plus ten settling steps per reset. `summary.json` reports
training and reference successes, paired disagreements, worker input/action
contracts, and peak allocator memory. Incomplete coverage raises an error and
withholds a completed summary. CPU tests and a dry-run plan do not establish GPU
or teacher-success acceptance.

Optional captures are written as `captures/<suite>_task<T>_trial<I>_chunk<K>.pt`.
They contain the exact pre-predict `env_obs`, selected `forward_inputs` including
the full `[1,5,16,32]` flow chains, identity/seeds, and physical actions. Chunk
indices are zero-based; the capture option accepts only noninitial indices.
`capture_manifest.json` is a list with relative paths, episode policy-step counts
(excluding settling), and eight Boolean `valid_controls` flags. Frozen post-done
captures have an all-false mask and must not be presented as active samples.

This probes native teachers as **acting policies** in the production train chain.
One suite per rank and one environment per rank isolate that question. It does
not establish mixed-suite actor-buffer routing, student-coordinate transport,
student-visited state coverage, or performance of a trained student. No live
training recipe, worker default, or existing run is modified.
