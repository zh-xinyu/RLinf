# LIBERO on an approved platform image

## Agreed deployment route

The collaborator cluster does not permit importing custom container images.
Select an image from the platform's approved catalog. Prefer a small bootstrap
venv containing uv, followed by one isolated experiment runtime. Install through
the internal package repository plus artifacts imported through approved channels.
If uv is unavailable or disallowed, use the pure-pip fallback. Bootstrap is an
installation convenience; inference, training and evaluation use only the runtime.

**Status:** deployment plan, pending the image/software catalog and transfer
rules. The existing automatic installer cannot be used unchanged because it forces
public PyPI and uv-managed Python downloads. Neither a target-cluster uv restore
nor a pure-pip fallback restore has been tested; this is not a launch-ready recipe.
The numerical experiment contract stays in
[LIBERO_EXPERIMENTS.md](LIBERO_EXPERIMENTS.md).

Start the collaborator queue from [LIBERO_HANDOFF.md](LIBERO_HANDOFF.md):
environment admission, shared initial evaluation, then Uniform-R1/R2/R4,
each with training seeds 0/1/2. Adaptive is held for a later owner-tuned handoff. The team will debug the selected image together; internal pip
configuration is owned by the platform.

## Information needed to finish the manual commands

| Item | Needed information |
| --- | --- |
| GPU allocation | Model, memory/card, number of GPUs/node; CPU/RAM allocation |
| Approved image | Exact identifier, OS/architecture, Python version, supplied Torch/CUDA stack |
| System libraries | NVIDIA EGL/graphics access, Git/Git LFS, compiler; administrator route for missing system packages |
| Python installation | Whether user-owned venvs are permitted; available Python interpreters/modules |
| Internal package repository | pip configuration supplied by the platform; availability of exact version pins and build dependencies |
| Artifact transfer | Approved way to import missing wheels, pinned source/assets, model weights and experiment data |

Do not send repository credentials or tokens in the handoff. Repository URL and
authentication should use the platform's existing credential mechanism.

## Preferred first step: a small bootstrap venv

Use the platform image's available Python and its documented internal pip source.
Only the runtime needs the exact experiment Python version; bootstrap can use a
supported image-provided Python. The pinned uv version is 0.10.4. These commands
install only the tool and do not create Ray workers, query GPUs or run experiments:

```bash
export LIBERO_BOOTSTRAP=/your/writable/path/libero-bootstrap
# Run in a fresh location; do not replace an existing environment.
test ! -e "$LIBERO_BOOTSTRAP"
python3 -m venv "$LIBERO_BOOTSTRAP"
"$LIBERO_BOOTSTRAP/bin/python" -m pip install uv==0.10.4
"$LIBERO_BOOTSTRAP/bin/uv" --version
```

Ensure this pip invocation uses the platform's approved source/configuration.
If the source is configured only in a special shell command, apply the documented
index/configuration to the venv invocation as well. If uv is missing from the
internal repository, request an approved mirror/import of its wheel; fall back
to pip if necessary. No sudo or replacement container image is needed for these
user-owned venv commands, provided the image has working venv support.

Configure **uv's** internal index explicitly before installing runtime packages;
it does not automatically inherit pip configuration. Disable automatic Python
downloads and select an approved preinstalled interpreter, or arrange an approved
Python artifact import. Bootstrap installation success does not establish that
external Torch/FlashAttention URLs or Python downloads are available. Keep the
same package/source pins and hardware acceptance gates for either install tool.

## Environment organization

The platform image supplies the Linux userspace and access to host NVIDIA
drivers. One isolated runtime venv contains the project Python dependencies:
Torch, GR00T, LIBERO, RLinf dependencies, and both training/evaluation tools.
Use its own site-packages; do not implicitly inherit the image's Torch via
`--system-site-packages`. Image-installed CUDA/Torch version labels alone do not
identify the packages loaded by a project venv.

Prefer the exact reference Python3.11.14. The two existing GPU profiles are:

- Ampere/Ada: Torch2.6.0/CUDA12.4, FlashAttention2.7.4.post1, C++ ABI FALSE.
- Blackwell: Torch2.7.1+cu128/CUDA12.8, FlashAttention2.8.3.post2, C++ ABI TRUE,
  NCCL2.31.2 and the recorded SDPA policy.

Choose one profile based on available hardware. If the catalog requires other
versions, create and validate a separate profile; do not silently edit the old
snapshot or claim it has been reproduced. A different Python3.11 patch version
also needs an explicit profile decision because current verification is exact.

## Planned installation stages

1. **Select and inspect the approved image.** Record its identifier, Python and
   pip versions, host driver, GPU visibility and EGL libraries. Missing driver
   graphics/EGL support requires the platform administrator; pip cannot supply it.
2. **Create the isolated runtime.** Use the approved Python interpreter and its
   standard `venv` support. The final command will name that interpreter
   explicitly. Keep the venv on a writable path visible to all local Ray workers.
3. **Install pinned packages with uv pip, or pip as fallback.** Configure the
   chosen tool to use the approved internal repository. Install the selected profile's exact CUDA/Torch and extension
   wheels, captured requirements, and build dependencies in controlled stages.
   Preserve the intended NCCL override; an unconstrained resolver must not replace
   it. Verify the available pip version supports the required build controls.
4. **Install pinned source dependencies.** Obtain the fixed GR00T and RLinf-LIBERO
   revisions plus real assets/initial-state files. Preserve source provenance for
   the verifier. Git LFS pointer text is not a downloaded asset. Approved source
   archives/bundles can replace direct GitHub access, with verification adapted
   explicitly if Git metadata is absent.
5. **Prepare the existing scoped runtime wrapper.** After package/source install,
   run `python -m toolkits.libero_gr00t.prepare_runtime --runtime <absolute-path>
   --profile <selected-profile>` from this checkout. This helper uses the standard
   library and does not require uv. It installs no packages. It creates runtime-
   owned LIBERO config and the Python wrapper carrying EGL and Blackwell policies.
6. **Stage model artifacts separately.** Include all weight shards, configuration
   and original metadata for the four teachers and accepted student. Uploading
   the source repository alone is insufficient. Initialization/offline-baseline
   reproduction additionally needs its fixed teacher trajectory dataset.
7. **Verify and accept the allocated node.** Check exact packages/source/model
   identities, renderer-to-CUDA mapping, multi-GPU collectives, strict checkpoint
   resume and fixed-state rollout-worker evaluation before long experiments.

Keep exact installation commands and receipts after the target image is known.
The uv-specific `create_and_sync_venv` / `opd_profile.py install` entrypoints are
not pure-pip entrypoints. Do not call them while assuming the platform's pip
configuration will redirect all downloads: the current installer forces public
PyPI and uv-managed Python downloads.

## Package and artifact delivery

If the internal repository lacks an exact package/wheel, ask the platform to
mirror it or use their approved file-import mechanism. A prepared wheel directory
can support offline pip installation, provided it includes build dependencies
and the correct Linux/Python/GPU ABI. Internet access on compute nodes is not
required once all resources are present. W&B can remain offline; retain local
training and raw evaluation records.

The remaining hardware acceptance and initializer gates are unchanged. Neither
the image-import restriction nor replacing the installation tool requires a
change to the learning algorithm or experimental comparison.


## Manual worksheet for the joint setup session

Run these stages separately, keeping their outputs. Paths below are placeholders;
use persistent absolute paths visible to every local worker. This worksheet
installs packages/configuration only. The last GPU verification stage requires a
scheduled allocation. An image with a different software stack is acceptable only
after a separate profile decision and acceptance; do not improvise pin changes.

### A. Inspect the platform without importing Torch

Inside the selected image, record its identifier separately and run:

```bash
python3 - <<'PYINFO'
import importlib.metadata as m
import importlib.util
import platform
import os
print('platform:', platform.platform())
print('architecture:', platform.machine())
print('Python:', platform.python_version())
print('glibc:', platform.libc_ver())
print('CPU affinity:', len(os.sched_getaffinity(0)))
print('venv available:', importlib.util.find_spec('venv') is not None)
for name in ('pip', 'torch', 'torchvision', 'ray', 'mujoco'):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, 'not installed')
PYINFO
# Read-only; inside a GPU allocation, also retain:
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
```

This describes the image interpreter, not the future runtime. Do not send pip
configuration dumps or environment dumps: these can contain repository tokens.
The platform team only needs to confirm that its internal pip source works.

### B. Prepare bootstrap and a runtime using approved Python

Create the bootstrap using the earlier commands. Choose one existing profile
based on GPU family, then explicitly point at approved Python 3.11.14. If that
interpreter is absent, request it before continuing; do not trigger an external
Python download on an internal-only cluster.

```bash
export LIBERO_REPO="$PWD"
export LIBERO_BOOTSTRAP=/absolute/persistent/path/libero-bootstrap
export LIBERO_RUNTIME=/absolute/persistent/path/libero-runtime
export LIBERO_PROFILE=gr00t-n15-libero-20261105-r2
# Blackwell: gr00t-n15-libero-blackwell-cu128
export LIBERO_PROFILE_DIR="$LIBERO_REPO/requirements/profiles/$LIBERO_PROFILE"
export LIBERO_APPROVED_PYTHON=/absolute/path/to/python3.11
export LIBERO_UV="$LIBERO_BOOTSTRAP/bin/uv"
export PYTHONNOUSERSITE=1
export UV_PYTHON_DOWNLOADS=never

"$LIBERO_APPROVED_PYTHON" -c \
  'import platform; assert platform.python_version() == "3.11.14"'
test ! -e "$LIBERO_RUNTIME"
"$LIBERO_APPROVED_PYTHON" -m venv "$LIBERO_RUNTIME/venv"
export LIBERO_RAW_PYTHON="$LIBERO_RUNTIME/venv/bin/python"
```

Use a clean shell without an activated Conda/other venv or inherited PYTHONPATH.
Do not use `--system-site-packages`. The bootstrap's uv executable may operate on
this separate runtime through `--python`; bootstrap is not used to run experiments.

### C. Install pinned distributions through the selected package channel

The platform team configures `UV_DEFAULT_INDEX` for its approved internal source
without placing credentials in this document or Git. If only pip is configured,
either configure uv's index separately or choose the pip fallback below. Existing
public installer helpers must not be sourced to override these settings.

```bash
LIBERO_PIP=("$LIBERO_UV" pip install --python "$LIBERO_RAW_PYTHON"
  --no-deps --build-constraint "$LIBERO_PROFILE_DIR/build-constraints.txt")
# Alternative, only after confirming this pip supports --build-constraint:
# LIBERO_PIP=("$LIBERO_RAW_PYTHON" -m pip install --no-deps
#   --build-constraint "$LIBERO_PROFILE_DIR/build-constraints.txt")

"${LIBERO_PIP[@]}" setuptools==75.8.2 wheel==0.48.0 pip==26.3 packaging==26.3
```

If the fallback interpreter's initial pip is too old, install the pinned pip
wheel first without the build-constraint flag, then define the fallback array.
Do not interpret the fallback as a separately validated runtime restore.

For Blackwell, import the two exact `package_wheels` listed in `profile.json`
and install their local files first. For either profile, also import its exact
FlashAttention wheel. Print the approved acquisition list without downloading:

```bash
python3 - <<'PYWHEELS'
import json, os
from pathlib import Path
p = json.loads((Path(os.environ['LIBERO_PROFILE_DIR'])/'profile.json').read_text())
for url in p.get('package_wheels', []) + [p['flash_attention']['url']]:
    print(url)
PYWHEELS
# Blackwell only, using actual imported filenames:
# "${LIBERO_PIP[@]}" /approved/files/torch-2.7.1+cu128-....whl
# "${LIBERO_PIP[@]}" /approved/files/torchvision-0.22.1+cu128-....whl
"${LIBERO_PIP[@]}" -r "$LIBERO_PROFILE_DIR/requirements.txt"
# Both profiles, using the exact wheel just listed:
# "${LIBERO_PIP[@]}" /approved/files/flash_attn-....whl
```

The last three commented commands require the real imported filenames; replace
and execute them in that order. On Ampere/Ada ensure the imported/mirrored Torch
build is the reference CUDA12.4 build, not an identically versioned CPU build.
Blackwell requires the explicit cu128 builds. Full GPU verification checks the
loaded CUDA version/ABI. Package installation must finish before continuing.

If using a **complete, approved local wheel directory**, add `--no-index
--find-links /absolute/path/wheelhouse` to the chosen installation array before
any package installs. Include every transitive pin and the bootstrap/build
packages; `--no-deps` means the supplied snapshot must be complete. Explicit
Torch/FlashAttention local files still need the correct Python/Torch/CUDA/ABI.
A wheel directory prepared for one profile is not interchangeable with the other.

### D. Install supplied source checkouts and runtime launch files

Transfer or clone the two exact repositories to:

```text
$LIBERO_RUNTIME/repos/gr00t
$LIBERO_RUNTIME/repos/libero
```

Use the revisions in the selected profile: GR00T
`4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`, RLinf-LIBERO
`0c5e40cc4ae63e09c14e7df6f74481e9ee8585f7`. Preserve `.git` history/provenance and
real LIBERO assets/initial states. A Git bundle alone does not contain LFS blobs;
transfer those separately or transfer the complete verified checkout. GR00T's
non-runtime LFS weights may be omitted, but not its processor/tokenizer source.

```bash
git -C "$LIBERO_RUNTIME/repos/gr00t" rev-parse HEAD
git -C "$LIBERO_RUNTIME/repos/libero" rev-parse HEAD
"${LIBERO_PIP[@]}" --no-build-isolation --editable "$LIBERO_RUNTIME/repos/gr00t"
"${LIBERO_PIP[@]}" --no-build-isolation --editable "$LIBERO_RUNTIME/repos/libero"
cd "$LIBERO_REPO"
python3 -m toolkits.libero_gr00t.prepare_runtime \
  --runtime "$LIBERO_RUNTIME" --profile "$LIBERO_PROFILE"
export LIBERO_PYTHON="$LIBERO_RUNTIME/bin/python"
source "$LIBERO_RUNTIME/libero_env.sh"
```

Editable installation uses the already installed pinned build packages. Do not
run an unconstrained dependency resolver afterwards. Copying another machine's
venv is not the prescribed transfer method.

### E. Verify and admit before long jobs

Perform the import-free package check in
[environment guide, stage 7](LIBERO_ENVIRONMENT.md#7-check-software-then-accept-the-allocated-hardware).
Inside an allocated four-GPU job, retain the image identifier and execute:

```bash
"$LIBERO_PYTHON" requirements/opd_profile.py verify \
  --runtime "$LIBERO_RUNTIME" --profile "$LIBERO_PROFILE" --gpu
```

Then complete that stage's EGL, four-rank training/save/resume and fixed-state
rollout checks together. Package verification alone does not admit the node or
accept the new initialization. Follow [the experiment handoff](LIBERO_HANDOFF.md)
for artifacts, initial evaluation and the Uniform-first queue. Keep all install,
verification and failed-stage logs so joint debugging does not require guessing
which packages or files were changed.
