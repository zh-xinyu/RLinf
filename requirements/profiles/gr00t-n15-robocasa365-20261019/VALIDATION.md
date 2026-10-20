# Validation record — 2026-10-19

## Completed

- Installed into a new, isolated directory on Ubuntu 24.04 / Linux x86_64.
  The installer downloaded Python 3.11.14 and packages from public sources and
  checked out all four pinned Git commits. No existing training venv was reused.
- Verified 281 package versions (280 captured packages plus the explicitly pinned
  setuptools-scm bootstrap tool), Git revisions, tracked source files, editable
  install locations, Torch CUDA 12.4 / C++ ABI and FlashAttention import.
- Imported the RLinf GR00T N1.5 action model, RoboCasa stack and local Eagle
  processor. Repeated verification with an isolated Hugging Face module cache
  and offline model loading.
- Prepared a checkout from tracked files plus the new public profile files;
  no private node env files or `toolkits/mopd/` were present. Software verification
  passed there, and the following CPU regression suite reported **76 passed**:

  ```bash
  PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' python -m pytest -q \
    tests/unit_tests/test_opd_profile.py \
    tests/unit_tests/test_flow_refresh.py \
    tests/unit_tests/test_mopd_config.py \
    tests/unit_tests/test_mopd_wandb_logging.py \
    tests/unit_tests/test_flow_opd.py \
    tests/unit_tests/test_gr00t_n1d5_compute_dtype.py \
    tests/unit_tests/test_gr00t_n1d5_precision.py \
    tests/unit_tests/test_robocasa365_lifecycle.py \
    tests/unit_tests/test_robocasa365_formal_eval.py
  ```

- Ruff and `bash -n requirements/install.sh` passed. The install-check heuristic
  reported the same pre-existing broad installer/Docker findings, with no new
  findings from this change. This profile adds no new model or environment type.

## Findings incorporated

A first installation attempt built evdev with runtime setuptools 75.8.2 and
failed on modern license metadata. Builds now run in isolated environments with
setuptools 80.9.0; runtime setuptools stays at 75.8.2. LeRobot's poetry-core build
backend is fixed at the observed 2.5.0. Runtime dependency metadata conflicts
inherited from the working stack are documented in `OPD_ENVIRONMENT.md`.

## Not established by these checks

- GPU kernels, EGL rendering, real task rollouts, teacher inference and optimizer
  updates in the new environment. All GPUs were occupied; no training jobs were
  interrupted. `verify --gpu` is available for a subsequent idle-GPU check.
- Fresh system-package installation, Docker builds or other OS/driver versions.
- Availability or identity of external task assets and experiment checkpoints.
- Bit-identical artifacts, deterministic success rates or a fully offline build.

The GitHub workflow `OPD Profile Contract` runs the six lightweight profile tests;
it does not claim to run a full installation or GPU training.
