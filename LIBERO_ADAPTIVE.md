# Prepare, tune and publish the Adaptive recipe

## Current status

Adaptive now shares the manual training, post-training evaluation and NAS CSV
workflow with Uniform. The runtime profiles/dependencies are unchanged. The
collaborator assignment remains **nine Uniform runs**; Adaptive parameters are
owner-tuning candidates, not a new assignment or a claim of GPU acceptance.

The preferred entry is `python3 -m toolkits.libero_gr00t.run_experiment` with
`--method uniform|adaptive`. Existing `run_uniform` commands remain compatible
and still default to Uniform. R1/R2/R4 and training seeds0/1/2 are supported;
choose the eventual Adaptive matrix after local tuning rather than automatically
queuing every supported setting.

## 1. Tune only the Adaptive parameter file

Edit [libero_mopd_adaptive_release_common.yaml](examples/embodiment/config/libero_mopd_adaptive_release_common.yaml).
It explicitly contains allocator warmup/window/interval, coverage check, rho,
epsilon, priority power and quota-change cap. Current values are the previous
10/10/10, rho1, epsilon0, power2, cap8 candidate. They are not newly optimized.

Separate `libero_mopd_adaptive_release_r{1,2,4}.yaml` files select reuse and its
derived update/save budgets. Keep T64, B512, LR3e-5, four-update optimizer warmup,
100physical rounds and ten-round measurements aligned with the Uniform controls.
The admitted machine JSON selects the same training microbatch for both methods.
Do not tune the shared historical `adaptive_query_common` file for this release:
Uniform inherits that file, while the new Adaptive-only common file is isolated.

Use separate pilot output/CSV destinations while exploring parameter versions.
Existing run directories and conflicting checkpoint identities are deliberately
not overwritten. A completed run's resolved config and commit remain its identity.

## 2. Prepare a portable fixed teacher reference

The reference contains task SR counts and inference protocol, not controller
hyperparameters. Allocator-only tuning does not require collecting new teacher
trajectories. The actual teachers, normalization, execution lengths and horizons
must stay unchanged. The final agreed calibration is50episodes/task.

After calibration is audited, export its fixed counts and checkpoint content
hashes using the experiment runtime. Use a resolved config whose teacher paths
and inference signature match the source reference:

```bash
/absolute/runtime/bin/python -m toolkits.libero_gr00t.reference_bundle export \
  --config /absolute/path/matching/resolved_config.yaml \
  --reference /absolute/path/verified_teacher_reference.json \
  --minimum-episodes 50 \
  --output toolkits/libero_gr00t/configs/teacher_reference_release.json
```

This reads/checks files and creates no environment interaction. Export verifies
all40tasks and requires at least50episodes/task. The portable JSON omits local
checkpoint/capture paths, retains source-reference/provenance hashes and includes
the four teachers' exact file hashes. The raw calibration record remains a
separate reproducibility artifact. There is no fabricated reference committed
at that default path while final calibration/publication remains pending.

For owner-side tuning only, an explicitly labeled20-episode/task calibration can
be exported with `--minimum-episodes 20` to a private output path. Set optional
machine JSON `teacher_reference_bundle` to that absolute private path. Keep the
collaborator field null/absent so it will select the final repository artifact.

For actual Adaptive training/planning, the manual entrypoint automatically binds
the bundle into `<results>/teacher_reference_cache/`. It composes the actual
mixed-K inference signature, verifies local teacher shards/config/metadata by
SHA256, and changes only checkpoint locations. A signature/content mismatch
fails; editing absolute paths in an old reference is unnecessary. Binding is
serialized/idempotent for shared caches. It performs no teacher calibration or
simulator/GPU rollout, although hashing weights incurs CPU/disk work.

## 3. Use the same machine and command shape

Reuse `outputs/local/libero_machine.json` from Uniform, including its explicit
NAS `csv_dir`. Optional `teacher_reference_bundle: null` selects the published
repository JSON. A missing release bundle stops Adaptive before training.

```bash
# Plan after a valid release/private bundle is available.
python3 -m toolkits.libero_gr00t.run_experiment \
  --machine outputs/local/libero_machine.json --method adaptive --r 2 --seed 0
# Train100rounds, then automatically evaluate every saved round and export CSVs.
python3 -m toolkits.libero_gr00t.run_experiment \
  --machine outputs/local/libero_machine.json --method adaptive --r 2 --seed 0 --run
```

Names become `Adaptive-R2_seed0`, W&B group `T64_adaptive_seed0`, and checkpoint
directory `libero_adaptive/checkpoints/global_step_<R*round>`. Uniform names/layout
stay unchanged. Independent evaluation still uses the same runtime/rollout path,
40tasks x50states, evaluation seed1234 and K5/5/5/10. The initial student can be
shared after the existing identity/protocol/coverage checks.

Both methods write to the same configured NAS CSVs. `arm` distinguishes
`Uniform-R2` and `Adaptive-R2`; round0 remains one shared INITIAL row. The suite
macro is equal-weight, including when training allocation/query shares vary.
R/seed, source, snapshots, evaluation coverage and real interaction counters
remain independently auditable.

The existing modes also work:

```bash
# Recover the measurement queue after this training run is complete.
python3 -m toolkits.libero_gr00t.run_experiment \
  --machine outputs/local/libero_machine.json --method adaptive --r 2 --seed 0 \
  --evaluate-all --run
# Optional single checkpoint, automatically selecting update80.
python3 -m toolkits.libero_gr00t.run_experiment \
  --machine outputs/local/libero_machine.json --method adaptive --r 2 --seed 0 \
  --eval-round 40 --run
```

Evaluation recovery needs the saved resolved config/checkpoints, not rebinding a
newly changed teacher reference. `--train-only` remains available. CSV-only
recovery uses `collect_results --method adaptive`; see
[the command guide](LIBERO_COMMANDS.md#6-automatically-merged-csv-results).

## 4. Publish parameters, then update the collaborator checkout

After local GPU validation, commit the Adaptive-only parameter change and final
reference bundle, record its calibration provenance, and push
`release/libero-gr00t`. Keep the Uniform recipe and environment pins fixed.

After the active job using that checkout has finished, the collaborator can
update its tracking branch, including when earlier commands left it detached:

```bash
git fetch origin
git switch release/libero-gr00t
git pull --ff-only origin release/libero-gr00t
git rev-parse HEAD
```

Record the new commit, inspect the resolved Adaptive plan, then use the command
above. Parameter/reference-only updates require neither another venv nor edits
to the existing machine JSON or launch commands. Every new run records its actual
commit and resolved configuration. Keep existing Uniform/Adaptive checkpoints
attached to their original recipes; do not resume an optimizer under newly tuned
parameters. If a collaborator is asked to repeat an Adaptive setting with new
parameters, give that iteration a separate results/CSV namespace.
