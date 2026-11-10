# GR00T N1.5 / LIBERO: manual environment installation

**Collaborator cluster constraint:** importing custom container images is not
allowed. Use the [approved-platform-image route](LIBERO_CLUSTER_SETUP.md) for
this handoff: bootstrap/uv through the internal source when available, with a
pure-pip fallback. The public-network procedure below remains the reference
installation route, not the collaborator's final command list.

This guide prepares the experiment runtime from a Git checkout. Run each numbered
stage separately and retain its output. A coding agent is not required. Software
installation and target-machine acceptance are separate stages; the new cluster
image has not yet been tested. The experiment contract and outstanding
initialization checks are in [LIBERO_EXPERIMENTS.md](LIBERO_EXPERIMENTS.md).

## 1. Request an image and allocation

Request Linux x86_64, a writable persistent work directory, four GPUs on one node,
at least32CPU cores and preferably256GB host RAM. Existing training used four
48GB cards; allow at least30GiB free/card for the first acceptance run. Exact GPU
and CPU requirements must be confirmed on the target cluster, not inferred from
package installation. All four GPUs must be visible inside the allocated job.

| GPU family | Runtime profile | Python | Torch / CUDA wheels | FlashAttention / C++ ABI |
| --- | --- | --- | --- | --- |
| Ampere/Ada reference | `gr00t-n15-libero-20261105-r2` | 3.11.14 | 2.6.0 / 12.4 | 2.7.4.post1 / FALSE |
| Blackwell | `gr00t-n15-libero-blackwell-cu128` | 3.11.14 | 2.7.1+cu128 / 12.8 | 2.8.3.post2 / TRUE |

Possible base images are `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` and
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`. These are image-request suggestions,
not tested project image digests. A different cluster image is acceptable if the
isolated runtime below and hardware acceptance pass. Pin the final image digest
after admission. See the official [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/)
and [Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).

The image needs Git, Git LFS, Bash, a C/C++ compiler, `libEGL.so.1`, OpenGL/GLVND
libraries, normal system runtime libraries and NVIDIA driver access. For a
Debian/Ubuntu image, ask the image administrator to include `git git-lfs
build-essential libegl1 libgl1 libgles2 libglib2.0-0`; this list does not replace
the NVIDIA graphics driver libraries supplied by the container runtime. NVIDIA
container capabilities must include `compute,utility,graphics`. The tested
Blackwell node used an R580 driver; record the target driver and validate it.
The image CUDA toolkit and host driver are different components. Installing a
Python CUDA wheel does not supply the host driver or EGL renderer.

Confirm user-owned venv/pip installs and GitHub, PyPI, PyTorch wheel and Hugging
Face access. For an offline cluster, stage the exact wheels, pinned source
checkouts including simulator assets, and model artifacts on a connected machine
first. GR00T N1.5 includes a Qwen-based VLM; obtain cluster-policy clearance before
requesting compute if model-origin restrictions apply.

## 2. Choose paths and bootstrap tooling

Clone the agreed LIBERO branch/commit, then run in that checkout. Do not use the
RoboCasa release tag. Replace the example paths with writable absolute paths;
keep this checkout at the same path for the lifetime of a run.

```bash
export LIBERO_REPO="$PWD"
export LIBERO_RUNTIME=/your/persistent/path/libero-runtime
export LIBERO_BOOTSTRAP=/your/persistent/path/libero-bootstrap
export LIBERO_PROFILE=gr00t-n15-libero-20261105-r2
# For Blackwell use gr00t-n15-libero-blackwell-cu128 instead.
export LIBERO_PROFILE_DIR="$LIBERO_REPO/requirements/profiles/$LIBERO_PROFILE"

test -f "$LIBERO_PROFILE_DIR/profile.json"
test ! -e "$LIBERO_RUNTIME"
test ! -e "$LIBERO_BOOTSTRAP"
python3 -m venv "$LIBERO_BOOTSTRAP"
"$LIBERO_BOOTSTRAP/bin/python" -m pip install uv==0.10.4
export LIBERO_UV="$LIBERO_BOOTSTRAP/bin/uv"
"$LIBERO_UV" --version
```

The bootstrap environment contains only installation tooling. The separate
runtime venv contains Torch, GR00T, LIBERO and all training/evaluation packages.
The image's preinstalled Torch is not the experiment interpreter.

## 3. Create the runtime Python

Use a fresh shell without another active Conda/venv. These process-local settings
prevent accidental package overlays and user-wide LIBERO configuration changes.

```bash
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX
unset RLINF_PYTHON_EXECUTABLE BASH_ENV ENV GIT_LFS_SKIP_SMUDGE
export PYTHONNOUSERSITE=1
export UV_NO_CONFIG=1
export UV_DEFAULT_INDEX=https://pypi.org/simple
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_INSTALL_DIR="$LIBERO_RUNTIME/python"
export UV_CACHE_DIR="$LIBERO_RUNTIME/cache/uv"
export PATH="$LIBERO_BOOTSTRAP/bin:$PATH"
mkdir -p "$LIBERO_RUNTIME"

source "$LIBERO_REPO/requirements/install.sh"
VENV_DIR="$LIBERO_RUNTIME/venv"
PYTHON_VERSION=3.11.14
create_and_sync_venv create-only
export LIBERO_RAW_PYTHON="$LIBERO_RUNTIME/venv/bin/python"
"$LIBERO_RAW_PYTHON" --version
```

If inherited `PIP_*` or `UV_*` variables select a private index/config, remove
them or explicitly document the mirror. Do not run `uv sync` or `pip install -e .`
after this snapshot installation: resolving RLinf's broad dependencies again
can replace the pinned CUDA/model stack.

## 4. Install exact wheels and dependencies

Use Bash arrays so paths with spaces are preserved. `--no-deps` is intentional:
the profile contains the full captured package set, including the Blackwell
NCCL amendment. A resolver may otherwise downgrade the tested NCCL runtime.

```bash
LIBERO_PIP=("$LIBERO_UV" pip install --python "$LIBERO_RAW_PYTHON"
  --no-deps --build-constraint "$LIBERO_PROFILE_DIR/build-constraints.txt")
"${LIBERO_PIP[@]}" setuptools==75.8.2 wheel==0.48.0 pip==26.3 packaging==26.3

mapfile -t LIBERO_WHEELS < <(python3 - <<'PY'
import json, os
from pathlib import Path
p = json.loads((Path(os.environ['LIBERO_PROFILE_DIR'])/'profile.json').read_text())
print('\n'.join(p.get('package_wheels', [])))
PY
)
for LIBERO_WHEEL in "${LIBERO_WHEELS[@]}"; do
  if [[ -n "$LIBERO_WHEEL" ]]; then "${LIBERO_PIP[@]}" "$LIBERO_WHEEL"; fi
done
"${LIBERO_PIP[@]}" -r "$LIBERO_PROFILE_DIR/requirements.txt"
```

The reference Torch2.6 wheel's distribution pin is `torch==2.6.0`, although
`torch.__version__` includes `+cu124`. Do not substitute a different build on
that basis. Blackwell pins NCCL2.31.2; Torch's compile-time NCCL version can still
report2.26.2. Check the loaded library at hardware acceptance. Known dependency
metadata conflicts do not justify unreviewed package upgrades.

## 5. Install the two pinned source packages and FlashAttention

Run the commands from stage3's shell, where the shared installation functions
are loaded. Skip GR00T's non-runtime LFS weights only; do not disable LFS for
LIBERO's assets or downloaded model weights.

```bash
unset LIBERO_GR00T_SOURCE LIBERO_ENV_SOURCE
GIT_LFS_SKIP_SMUDGE=1 clone_or_reuse_repo LIBERO_GR00T_SOURCE \
  "$LIBERO_RUNTIME/repos/gr00t" https://github.com/NVIDIA/Isaac-GR00T.git --no-checkout
GIT_LFS_SKIP_SMUDGE=1 git -C "$LIBERO_RUNTIME/repos/gr00t" checkout --detach \
  4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
clone_or_reuse_repo LIBERO_ENV_SOURCE "$LIBERO_RUNTIME/repos/libero" \
  https://github.com/RLinf/LIBERO.git --no-checkout
git -C "$LIBERO_RUNTIME/repos/libero" checkout --detach \
  0c5e40cc4ae63e09c14e7df6f74481e9ee8585f7
"${LIBERO_PIP[@]}" --editable "$LIBERO_RUNTIME/repos/gr00t"
"${LIBERO_PIP[@]}" --editable "$LIBERO_RUNTIME/repos/libero"

LIBERO_FLASH_URL=$(python3 - <<'PY'
import json, os
from pathlib import Path
p = json.loads((Path(os.environ['LIBERO_PROFILE_DIR'])/'profile.json').read_text())
print(p['flash_attention']['url'])
PY
)
"${LIBERO_PIP[@]}" "$LIBERO_FLASH_URL"
```

The wheel is specific to Python, CUDA/Torch and C++ ABI. A missing wheel or ABI
error is a failed stage, not permission to install another FlashAttention version.
GR00T's pinned checkout includes the Eagle processor/tokenizer code; full VLA
weights and per-checkpoint metadata are separate artifacts.

## 6. Prepare scoped runtime entrypoints

This small configuration command installs nothing, imports no model and starts
no Ray/GPU work. It writes runtime-owned LIBERO paths, `libero_env.sh` and an
executable `bin/python` wrapper, rejecting conflicting existing launch files.

```bash
cd "$LIBERO_REPO"
python3 -m toolkits.libero_gr00t.prepare_runtime \
  --runtime "$LIBERO_RUNTIME" --profile "$LIBERO_PROFILE"
export LIBERO_PYTHON="$LIBERO_RUNTIME/bin/python"
source "$LIBERO_RUNTIME/libero_env.sh"
```

Always pass **this wrapper** as `--python` to training and evaluation launchers.
It ensures Ray subprocesses receive the same pinned runtime, thread limits, EGL
mapping and Blackwell SDPA policy. `CUDA_VISIBLE_DEVICES` is left to the job
allocation/worker. GPU IDs passed to the launchers must refer to the visible
device namespace in that job; never copy another node's GPU indices blindly.

The [adapters](toolkits/libero_gr00t/runtime_adapter/README.md) package the actual
node hooks previously used in successful experiments. Blackwell disables only
PyTorch memory-efficient SDPA because the retained FP32 reference test exposed
an error in that backend. NCCL P2P/IB are disabled consistently with the reference
single-node runtime. These choices must be recorded, not silently omitted.

## 7. Check software, then accept the allocated hardware

First, check versions without importing Torch, a model or a renderer:

```bash
"$LIBERO_RAW_PYTHON" - <<'PY'
import importlib.metadata as m
import json, os, platform
from pathlib import Path
p = Path(os.environ['LIBERO_PROFILE_DIR'])
profile = json.loads((p/'profile.json').read_text())
assert platform.python_version() == profile['python']
pins = [line for line in (p/'requirements.txt').read_text().splitlines()
        if line and not line.startswith('#')]
pins += profile['bootstrap']
pins += [s['name']+'=='+s['version'] for s in profile['sources']]
pins += ['flash_attn=='+profile['flash_attention']['version']]
for pin in pins:
    name, expected = pin.split('==', 1)
    assert m.version(name) == expected, (name, m.version(name), expected)
print('Python and distribution pins match; GPU/model acceptance remains pending.')
PY
```

Only inside a GPU allocation, retain `nvidia-smi` output and run full verification:

```bash
"$LIBERO_PYTHON" requirements/opd_profile.py verify \
  --runtime "$LIBERO_RUNTIME" --profile "$LIBERO_PROFILE" --gpu
```

The full verifier imports RADIO/model code and can query CUDA even without
`--gpu`; do not classify it as a CPU-only check or hide every GPU for that check.
It validates software/source revisions and a FlashAttention operation, not full
training acceptance. Before long jobs, separately require:

1. EGL rendering on every assigned GPU; compare rendered GPU identity with the
   worker's CUDA GPU. Fail on missing/ambiguous mappings or cross-device rendering.
2. Four-rank object collectives, FSDP and distributed-checkpoint save/load. On
   Blackwell verify the loaded NCCL runtime and SDPA reference agreement.
3. A bounded two-round MOPD run, at least one reuse update and strict resume from
   a completed round; verify finite loss, suite routing and metadata signatures.
4. Fixed-state inference through the shared rollout-worker evaluator; verify
   complete task/state coverage, loaded checkpoint and mixed-K execution.

These checks require scheduled GPU time and have **not** been rerun on the
collaborator's unknown image. Keep their admission records alongside the exact
Git commit, image digest, profile, driver and model hashes. A fresh image restore
has not been performed as part of the current source/documentation audit.

## 8. Model artifacts and experiments

Use [LIBERO_EXPERIMENTS.md](LIBERO_EXPERIMENTS.md) for model identities, the four
arms, commands and required results. The new initialization and portable
calibration bundle must be finalized before sending a launch-ready handoff.
No `.opd.action*.env`, personal paths, credentials or existing `outputs/` tree is
required by these installation steps. Keep authentication outside Git.
