# MOPD project tools

These are the small, reusable launch helpers and one-shot diagnostics referenced by the
RoboCasa365 MOPD research record. Runtime logs, status files, checkpoints, and retry
artefacts still belong in ignored `outputs/mopd/`.

The shell launchers derive the repository root from their own location. Machine-specific
locations can be overridden with `REPO`, `VENV`, and `WANDB_ENV_FILE`. Python probes derive
the repository root in the same way and accept `ROBOCASA365_CHECKPOINT_ROOT`.

No script in this directory runs automatically.
