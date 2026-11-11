# LIBERO: install once on the approved platform image

Entry: [LIBERO_HANDOFF.md](LIBERO_HANDOFF.md). This is the first-time installation
worksheet; routine experiments use the short commands in that handoff. Installation
on the target image and joint GPU/student acceptance remain pending.

Request Linux x86_64, approved Python 3.11.14, four GPUs on one node, 32 CPUs and
preferably 256 GB RAM. The image needs Git/Git LFS, a compiler and NVIDIA EGL/graphics
access (`libEGL.so.1`, driver libraries, compute/utility/graphics capabilities).
Missing driver graphics support requires the platform administrator. GR00T N1.5
uses a Qwen-based VLM; ensure it is permitted on the target platform.

Use the platform image and internal package source. No custom image import is
required. A small bootstrap venv holds uv; a separate isolated runtime contains
all model/simulator/training/evaluation packages. Do not inherit the image's Torch
via system-site-packages. Profiles below are fixed snapshots; source pins, full
requirements and exact wheel URLs are in their `profile.json` files.

| GPU | Profile | Torch / CUDA | FlashAttention / ABI |
| --- | --- | --- | --- |
| Ampere/Ada | `gr00t-n15-libero-20261105-r2` | 2.6.0 /12.4 | 2.7.4.post1 /FALSE |
| Blackwell | `gr00t-n15-libero-blackwell-cu128` | 2.7.1+cu128 /12.8 | 2.8.3.post2 /TRUE |

Both use Python 3.11.14 and uv 0.10.4. Blackwell additionally pins NCCL 2.31.2 and
uses the scoped SDPA policy. Confirm hardware/image before selecting a profile.
If exact packages or sources are unavailable internally, the owner and platform
team will prepare approved imports; do not substitute versions during setup.
The existing public automatic installer is not the internal-cluster entrypoint.

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

Create the bootstrap using the platform's internally configured pip:

```bash
export LIBERO_BOOTSTRAP=/absolute/persistent/path/libero-bootstrap
test ! -e "$LIBERO_BOOTSTRAP"
python3 -m venv "$LIBERO_BOOTSTRAP"
"$LIBERO_BOOTSTRAP/bin/python" -m pip install uv==0.10.4
"$LIBERO_BOOTSTRAP/bin/uv" --version
```

The platform configures uv's internal index separately from pip if needed.
Choose one existing profile
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

Check installed versions without importing Torch or a renderer:

```bash
"$LIBERO_RAW_PYTHON" - <<'PYVERS'
import importlib.metadata as m
import json, os, platform
from pathlib import Path
p = Path(os.environ['LIBERO_PROFILE_DIR'])
profile = json.loads((p/'profile.json').read_text())
assert platform.python_version() == profile['python']
pins = [s for s in (p/'requirements.txt').read_text().splitlines()
        if s and not s.startswith('#')]
pins += profile['bootstrap']
pins += [s['name']+'=='+s['version'] for s in profile['sources']]
pins += ['flash_attn=='+profile['flash_attention']['version']]
for pin in pins:
    name, expected = pin.split('==', 1)
    assert m.version(name) == expected, (name, m.version(name), expected)
print('Python/package pins match; node acceptance is next.')
PYVERS
```

Inside an allocated four-GPU job, retain the image identifier and execute:

```bash
"$LIBERO_PYTHON" requirements/opd_profile.py verify \
  --runtime "$LIBERO_RUNTIME" --profile "$LIBERO_PROFILE" --gpu
```

Complete the following acceptance checks jointly with the owner: EGL rendering
on the assigned GPUs; four-rank collectives and a bounded R2 train/save/resume
check; fixed-state rollout inference; and the shared student evaluation. The owner
supplies the bounded check config. Package verification alone does not admit the
node or accept the initialization. Then return to
[LIBERO_HANDOFF.md](LIBERO_HANDOFF.md) for the nine-run queue. Keep installation
and failed-stage logs for joint debugging.
