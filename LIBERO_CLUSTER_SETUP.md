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
supported image-provided Python. The pinned uv version is0.10.4. These commands
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
