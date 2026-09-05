# MOPD project tools

These are the small, reusable launch helpers and one-shot diagnostics referenced by the
RoboCasa365 MOPD research record. Runtime artifacts belong in the main repository's
ignored `outputs/train/`, `outputs/eval/`, or `outputs/smoke/`, one directory per run.

All launchers source [local_env.sh](local_env.sh), which reads the main repository's
Git-ignored `.opd.local.env`. Set `OPD_RUNTIME_ROOT` there; Python, venv, models,
assets, and caches are local dependencies derived from this root. `OPD_SHARED_ROOT`
is the persistent provisioning source, not a model-loading fallback. W&B credentials
remain in `WANDB_ENV_FILE` and are never copied into run snapshots. Explicit shell
values for these three variables override the local file.

Create `.opd.local.env` in the main repository with shell assignments such as
these; replace the placeholders with your machine's absolute paths:

```bash
export OPD_RUNTIME_ROOT=/absolute/nvme/path/opd
export OPD_SHARED_ROOT=/absolute/shared/path/opd_resources
export WANDB_ENV_FILE=/absolute/path/to/existing/wandb.env
```

The runtime and shared roots must be separate dedicated directories. The W&B
file is optional for setup and probes; train/eval use offline mode if it is
absent. Only source a local configuration file you trust. Historical research
runbooks may contain retired paths; this page is the current setup entry.

Provision a **new** runtime (Linux x86_64 / NVIDIA) from the prepared shared
models/assets, without training or starting Ray:

```bash
bash toolkits/mopd/setup_runtime.sh all --dry-run
bash toolkits/mopd/setup_runtime.sh all
```

[setup_runtime.sh](setup_runtime.sh) reuses `requirements/install.sh embodied
--model gr00t --env robocasa365`. It installs local CPython 3.11.14, pins the
GR00T/RoboCasa/Robosuite Git revisions, and installs the first two editable into
`$OPD_RUNTIME_ROOT/repos/`. Common Python dependencies follow the RLinf installer;
this is a source/compatibility-pinned recipe, **not** a complete package lock.
The GR00T source is NVIDIA's official repository. RoboCasa's `panda_omron`
configuration and the inference fixes remain in this project's RLinf adapter;
the benchmark fork is not an installation dependency. Use this project branch,
not an unmodified upstream RLinf checkout, with the recipe.
Host prerequisites are `git`, `uv`, `rsync`, `flock`, NVIDIA drivers and the
system libraries covered by `requirements/sys_deps.sh`. By default it passes
`--no-root`; opt into host package installation with `--system-deps`.

The stages can be run separately: `install` (software + CPU import checks),
`resources` (copy the four released checkpoints and RoboCasa assets from
`OPD_SHARED_ROOT`, then compare checksums), and `check` (offline CPU checks,
including the processor and model headers). Shared resources must already be
prepared; the script does not download datasets or model weights from the web.
It does not source W&B credentials. Logs and an installed-package inventory go
into `outputs/smoke/setup-*/`.

An existing venv is protected: use `check` on the current runtime. To test a fresh
install, set `OPD_RUNTIME_ROOT` to a new dedicated external directory. Only an
interrupted installation created by this script can use `install --resume-install`.
Resource copying resumes missing files but fails on conflicting existing files;
it neither overwrites them nor deletes extras. CPU checks do not certify GPU
inference, rendering, environment reset, or training.

`REPO` may select another worktree's code; outputs and local configuration still
belong to the main repository, resolved through Git's common directory. Legacy
worktree configs with retired hard-coded paths must be updated before use. The
current launchers do not rewrite historical configs automatically.

Preview a training launch without creating files, loading credentials, using GPUs,
or starting/stopping Ray:

```bash
OPD_DRY_RUN=1 bash toolkits/mopd/run_mopd.sh robocasa365_mopd_gr00t my_run
```

Train/eval launchers create `train.log`/`eval.log`, `status`, `paths.txt`, and
`command.txt` inside each run. W&B and TensorBoard use the same run directory.
RLinf currently nests saved checkpoints one level deeper:
`<run>/<runner.logger.experiment_name>/checkpoints/global_step_<N>/`.
Retries and resumes must use a new tag (or `OPD_RUN_ID`); existing runs are never
overwritten. `OPD_RUN_KIND=smoke` routes a short training sanity check to `smoke/`.
Evaluation takes `<tag> [wait_status_file] [Hydra overrides...]`; pass `""` for the
wait file when supplying overrides without waiting.

For direct Python commands, first source `toolkits/mopd/local_env.sh` and set an
absolute `OPD_RUN_DIR` for the config. Prefer [run_probe.sh](run_probe.sh), which
sets it automatically and runs from that directory. Relative probe output paths
then stay inside the run. It does not source W&B credentials.

The CPU-only path check imports the dependencies, loads the processor offline,
and reads the four checkpoint headers without starting an environment or Ray:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash toolkits/mopd/run_probe.sh probe_runtime_paths
```

No script in this directory runs automatically.

`probe_path_equiv.py` is a historical A/B script for reference-inference methods
that have since been removed. The probe launcher rejects it before allocating
resources; use `probe_flow_ode` for the current rollout-path comparison.
`traj_scale_round.sh` is also a historical experiment batch, not the default
training recipe. The base MOPD YAMLs still describe pilots: explicitly select the
suite, trajectory budget, learning parameters and checkpoint cadence for each
new experiment rather than treating those defaults as the latest successful run.

Before any new MOPD rollout, validate the mixed checkpoint lifecycle without creating an
environment or taking a training step:

```bash
bash toolkits/mopd/run_probe.sh probe_precision_init
```

The probe loads the configured student, checks that the backbone stays BF16 while every
trainable action-head master weight is restored in FP32, initializes AdamW with zero
gradients, and checks that its moments are FP32. Use `--skip-optimizer-state` for a lighter
model-only check. It does not run RoboCasa, perform a model forward, or update parameters.

Then measure the actual routed signal on fixed observations from two tasks per suite:

```bash
bash toolkits/mopd/run_probe.sh probe_supervision_matrix \
  --output artifacts/supervision_matrix.json
```

This resets four observations per suite and closes the environments immediately; it does
not step them or train. The student stays resident while each routed teacher is loaded and
released in turn. Student and teacher scoring use BF16 model copies to match FSDP compute,
and the report separates emitted-action disagreement from velocity-field disagreement at
each of the four denoise timesteps.

Finally, measure whether the three routed M1 objectives agree in parameter space:

```bash
bash toolkits/mopd/run_probe.sh probe_gradient_geometry \
  --output artifacts/gradient_geometry.json
```

This uses the same fixed observations and a separate BF16 rollout copy to freeze the
student flow states. By default, backward runs through a real single-rank FSDP wrapper with
FP32 student master weights and BF16 `param_dtype`; `--backend autocast` is available only
as a lighter approximation. The probe accumulates the expected uniform-timestep M1
gradient separately for each suite, copies the gradients to CPU, and reports their norms,
pairwise cosine, and `||sum g|| / sum ||g||` cancellation ratio. It never calls
`optimizer.step()`.

For Method 4, isolate sampled reverse-KL variance from transition or routing errors:

```bash
bash toolkits/mopd/run_probe.sh probe_m4 --output artifacts/m4_probe.json
```

The M4 probe uses the same fixed resets, forces each denoise timestep in turn, and reports
action drift plus conditional reward bias/variance at several Flow-SDE noise levels. It
then runs real single-rank FSDP backwards to compare finite-sample M4 gradients with the
analytic M3-Hybrid local-KL gradient at exactly the same states, including per-suite
signal-to-noise and multi-teacher gradient geometry. It never steps an environment,
constructs an optimizer, or changes model weights. Use `--skip-gradients` for the cheaper
reward/action-only part.
