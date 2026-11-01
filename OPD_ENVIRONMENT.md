# Fixed OPD environment

This guide and `requirements/profiles/` are versioned with `opd-project-main`.
A fresh clone does not require personal `.opd.*.env` files, `shs/`,
`toolkits/mopd/`, or local research notes. See the
[experiment guide](docs/opd/collaboration/README.md) for training and evaluation.

## Supported profile

**`gr00t-n15-robocasa365-20261019`** covers GR00T N1.5 + RoboCasa365 on
Linux x86_64 with NVIDIA GPUs. It pins Python 3.11.14, Torch 2.6.0 (CUDA 12.4,
old C++ ABI), FlashAttention 2.7.4.post1, Transformers 4.51.3, MuJoCo 3.3.1,
NumPy 2.2.5, and Ray 2.56.1. RoboTwin will use a separate environment profile.

| File | Purpose |
| --- | --- |
| [profile.json](requirements/profiles/gr00t-n15-robocasa365-20261019/profile.json) | Python, uv, four upstream Git revisions, and FlashAttention wheel/ABI |
| [requirements.txt](requirements/profiles/gr00t-n15-robocasa365-20261019/requirements.txt) | Exact versions of 275 runtime packages |
| [build-constraints.txt](requirements/profiles/gr00t-n15-robocasa365-20261019/build-constraints.txt) | Build-tool constraints |
| [opd_profile.py](requirements/opd_profile.py) | Installation, version/source checks, imports, and optional GPU verification |

This is a snapshot of a working environment, not a minimal dependency set.
Upstream GR00T, RoboCasa, and LeRobot dependency declarations conflict; historical
installation used `--no-deps` and explicit version overrides. The installer
restores that snapshot instead of resolving a different environment. `uv pip
check` can report these known conflicts; the supported scope is this project's
GR00T/RoboCasa365 path, not every upstream feature.

The upstream `requirements/install.sh embodied --model gr00t --env robocasa365`
installer does **not** promise to reproduce this fixed profile.

## Install from a clone

Prerequisites: Linux x86_64, Git, build tools, EGL/OpenGL libraries, and an NVIDIA
driver supporting CUDA 12.4 for GPU checks and training. The reference OS is
Ubuntu 24.04. `requirements/sys_deps.sh nvidia` installs system dependencies
with administrator privileges; the profile installer does not change system
packages. Installation needs access to PyPI, GitHub, and Python download hosts.

```bash
# Skip cloning again if you already checked out the experiment release.
git clone --branch opd-project-main https://github.com/zh-xinyu/RLinf.git
cd RLinf

# Keep bootstrap tooling separate from the training runtime.
python3 -m venv /absolute/path/opd-bootstrap
/absolute/path/opd-bootstrap/bin/python -m pip install 'uv==0.12.0'
export PATH="/absolute/path/opd-bootstrap/bin:$PATH"

# Use a new absolute directory. Dry-run does not write files.
python3 requirements/opd_profile.py install \
  --runtime /absolute/path/opd-runtime --dry-run
python3 requirements/opd_profile.py install \
  --runtime /absolute/path/opd-runtime
source /absolute/path/opd-runtime/activate.sh
```

The installer creates isolated Python downloads, a uv cache, a venv, and source
checkouts. It does not read personal node profiles or overwrite an existing
runtime. After a failure, retain its directory and error logs; retry in a new
empty path after resolving the cause. Do not upgrade an active training runtime.

Verification checks package versions, all four upstream revisions and tracked
source modifications, installed source locations, Torch CUDA/ABI, required
imports, and the local Eagle processor. `activate.sh` records the original
project checkout: source it **before starting Ray**, and update its repository
path if the checkout moves.

```bash
# Repeatable software checks, run from the repository root.
PYTHONPATH="$PWD" /absolute/path/opd-runtime/venv/bin/python \
  requirements/opd_profile.py verify --runtime /absolute/path/opd-runtime

# Also execute a real FlashAttention CUDA operation on an available GPU.
PYTHONPATH="$PWD" /absolute/path/opd-runtime/venv/bin/python \
  requirements/opd_profile.py verify --runtime /absolute/path/opd-runtime --gpu
```

## External experiment inputs

A clone does not include simulator assets, C10, or the three teachers. Passing
software verification alone does not establish experiment reproducibility.
Follow the [resource guide](docs/opd/collaboration/RESOURCES.md):

1. Prepare RoboCasa assets with the pinned checkout's setup/download scripts,
   or copy the agreed asset snapshot. Record asset provenance and checksums when
   available; upstream asset contents are not locked by this software profile.
2. Set `OPD_STUDENT_CHECKPOINT` to C10 and `OPD_MODEL_ROOT` to the directory
   containing `target_posttraining/{atomic_seen,composite_seen,composite_unseen}/checkpoint-60000`.
   Preserve each model's weights, `config.json`, and `experiment_cfg/metadata.json`.
3. Configure output storage, GPUs, and W&B authentication, then follow the
   [experiment guide](docs/opd/collaboration/README.md). Record the code commit,
   profile, input hashes, and resolved Hydra config.

Personal server profiles such as `.opd.action00.env` and `.opd.action10.env`
remain local. They explicitly select node-specific paths and are neither
published nor required on collaborators' machines.

## Validation and update policy

The 2026-10-19 isolated installation and clean-checkout verification passed
76 CPU regressions; see its [validation record](requirements/profiles/gr00t-n15-robocasa365-20261019/VALIDATION.md).
The subsequent experiment release has a separate
[validation record](docs/opd/collaboration/VALIDATION.md). Neither record replaces
GPU training/evaluation smoke tests on the receiving machine.

- Software versions and upstream commits are pinned; OS, drivers, and external
  assets are not a fully locked machine image.
- There is no complete wheel hash manifest or offline wheelhouse. Byte-identical
  builds and numerical determinism are not guaranteed. Missing upstream files
  cause explicit failures rather than automatic upgrades.
- Update the profile by creating a new dated directory, retaining old versions,
  and repeating fresh installation, software/GPU checks, and regressions.
- Verification claims apply only to the checks actually performed.
