# Validation record — 2026-11-05

## Captured source environment

A new Python 3.11.14 runtime was installed through the existing GR00T/LIBERO
installer without reusing a package overlay. The final installed versions were
read again after pinning Ray 2.56.1, PyArrow 25.0.0 and SymPy 1.13.1. This profile
captures all 265 installed distributions: 262 package pins, two editable source
repositories and one FlashAttention wheel. Both pinned source repositories had
no tracked diff. Local machine paths were excluded from the profile.

The source environment used uv 0.10.4. Isolated build constraints follow the
existing snapshot installer's backend policy; they are not a claim that every
upstream build was originally executed with those exact build-tool versions.

## Runtime evidence

- The initial four-episode native-specialist smoke succeeded on all four suites.
- The actual `env.train` / `HF.generate` native-specialist probe completed 80
  episodes: 65 successes, versus 61 in the selected reference, with 12 individual
  outcome disagreements. This establishes a functioning teacher rollout path,
  not outcome identity or student learning.
- A bounded target audit used eight captured, noninitial, teacher-visited
  observations. It ran fresh native student chains, production suite sampling
  and teacher routing, independent input/velocity comparisons, and physical
  integration checks. It did not inspect a live student refresh buffer.
- The diagnostic's initial metadata failure was an eager-import ordering error.
  Factory-first initialization fixed it; CPU reproduction validated the actual
  production metadata loader for all five checkpoints without loading weights.

Evidence artifact names from the experiment's `clean_environment_20261105` run:
`status.json`, `teacher_smoke_acceptance.json`, `teacher_train_80_acceptance.json`
and `real_rollout_target_audit_orderfix.json`. These are experiment outputs,
not required files for installing the portable profile.

## Fresh restore and remaining validation

A fresh independent runtime completed the full installer and automatic software
verification on 2026-11-05: all 265 distributions, Python 3.11.14, both clean
pinned Git revisions, installed editable provenance, import origins, Torch
CUDA 12.4 / C++ ABI and the local Eagle processor. Installation used a new
managed Python and private cache, with GPUs hidden, one-thread limits and
`MAX_JOBS=2`. Runtime-affecting source files stayed unchanged during the run.

The installer now runs from a Python 3.9 bootstrap interpreter. All **24 CPU
profile tests passed**, including both profile dry-runs under actual Python 3.9,
default compatibility, child selection propagation, private LIBERO config,
existing-runtime protection and strict editable-metadata shadowing cases.
Ruff checks passed. The broad installer and Dockerfile remain unchanged.

Two earlier terminal attempts were preserved. A GitHub LFS HTTP 502 affected
only GR00T's demo GIF; the pinned revision has no other LFS object, so only that
source skips smudging. A verifier lookup then selected source egg-info ahead
of installed dist-info. Verification now requires metadata from the requested
interpreter's site-packages, exact version and editable source location, plus
an independent import-origin check. Missing metadata is rejected. Shell startup
hooks are excluded with sanitized environment variables and startup-free Bash.

Successful restore artifacts are recorded under
`outputs/portable_libero_profile_review/fresh_restore_20261105_r3/`:
`status.json`, `verify.json`, `install.log` and `launch.json`. Earlier attempt
records distinguish download/bootstrap failures from model-runtime behavior.

This fresh restore performed no GPU test. No system-package installation,
Docker build, alternate driver validation, student optimizer update, improved
student success rate or formal training readiness is claimed. The broad CI
GR00T job resolves the combined `maniskill_libero` target and runs PPO; it is
not this frozen LIBERO OPD profile.
