# MOPD project tools

These are the small, reusable launch helpers and one-shot diagnostics referenced by the
RoboCasa365 MOPD research record. Runtime logs, status files, checkpoints, and retry
artefacts still belong in ignored `outputs/mopd/`.

The shell launchers derive the repository root from their own location. Machine-specific
locations can be overridden with `REPO`, `VENV`, and `WANDB_ENV_FILE`. Python probes derive
the repository root in the same way and accept `ROBOCASA365_CHECKPOINT_ROOT`.

No script in this directory runs automatically.

Before any new MOPD rollout, validate the mixed checkpoint lifecycle without creating an
environment or taking a training step:

```bash
PYTHONPATH="$PWD" \
  /localscratch2/liyifa11/venvs/gr00t_robocasa365/bin/python \
  toolkits/mopd/probe_precision_init.py
```

The probe loads the configured student, checks that the backbone stays BF16 while every
trainable action-head master weight is restored in FP32, initializes AdamW with zero
gradients, and checks that its moments are FP32. Use `--skip-optimizer-state` for a lighter
model-only check. It does not run RoboCasa, perform a model forward, or update parameters.

Then measure the actual routed signal on fixed observations from two tasks per suite:

```bash
MUJOCO_GL=egl PYTHONPATH="$PWD" \
  /localscratch2/liyifa11/venvs/gr00t_robocasa365/bin/python \
  toolkits/mopd/probe_supervision_matrix.py \
  --output outputs/mopd/supervision_matrix.json
```

This resets four observations per suite and closes the environments immediately; it does
not step them or train. The student stays resident while each routed teacher is loaded and
released in turn. Student and teacher scoring use BF16 model copies to match FSDP compute,
and the report separates emitted-action disagreement from velocity-field disagreement at
each of the four denoise timesteps.

Finally, measure whether the three routed M1 objectives agree in parameter space:

```bash
MUJOCO_GL=egl PYTHONPATH="$PWD" \
  /localscratch2/liyifa11/venvs/gr00t_robocasa365/bin/python \
  toolkits/mopd/probe_gradient_geometry.py \
  --output outputs/mopd/gradient_geometry.json
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
MUJOCO_GL=egl PYTHONPATH="$PWD" \
  /localscratch2/liyifa11/venvs/gr00t_robocasa365/bin/python \
  toolkits/mopd/probe_m4.py \
  --output outputs/mopd/m4_probe.json
```

The M4 probe uses the same fixed resets, forces each denoise timestep in turn, and reports
action drift plus conditional reward bias/variance at several Flow-SDE noise levels. It
then runs real single-rank FSDP backwards to compare finite-sample M4 gradients with the
analytic M3-Hybrid local-KL gradient at exactly the same states, including per-suite
signal-to-noise and multi-teacher gradient geometry. It never steps an environment,
constructs an optimizer, or changes model weights. Use `--skip-gradients` for the cheaper
reward/action-only part.
