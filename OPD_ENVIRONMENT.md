# Reproduce the GR00T / RoboCasa365 environment

Use this guide for the current RoboCasa experiment. RoboTwin will have a
separate profile. The inherited RLinf `docs/` tree is not yet fully adapted to
this project; it is not the installation authority for these experiments.

The recommended workflow uses **one machine JSON and one command entrypoint**.
You do not need personal `.opd.actionXX.env` files, manual `export` statements,
manual venv activation, or private scripts. The complete experiment assignment
is in the separate email task sheet; the [technical guide](docs/opd/collaboration/README.md)
provides the public training and evaluation commands.

## 1. Prerequisites and fixed checkout

- Linux x86_64, Git, and host Python3.9+ with `venv` support for bootstrap tooling.
- NVIDIA driver supporting CUDA12.4; the recipe uses eight48GB GPUs on one node.
- System build tools and EGL/OpenGL libraries. On a fresh machine, ask its
  administrator to review and run `bash requirements/sys_deps.sh nvidia` if needed.
  The Python installer does not install drivers or system packages.
- Network access to PyPI, GitHub, Python downloads, HF, and RoboCasa asset hosts.
- Sufficient disk space for the runtime, assets, input models, and six local
  checkpoints per experiment. Output checkpoints need not be delivered, but
  must remain available until their independent evaluations finish.

```bash
git clone --branch robocasa-formal-joint-sft-full-150r-20261101 https://github.com/zh-xinyu/RLinf.git
cd RLinf
```

If already cloned, use the assigned tag without replacing a dirty checkout.
Do not switch to an advancing branch during a run.

## 2. Configure this machine once

Copy the versioned template outside the repository:

```bash
mkdir -p "$HOME/opd-config"
cp toolkits/robocasa_mopd/machine.example.json "$HOME/opd-config/robocasa.json"
hostname -s
```

Edit all placeholder values in that JSON:

| Key | What to enter |
| --- | --- |
| `hostname` | Exact output of `hostname -s`; the tool rejects the wrong node |
| `runtime_root` | New absolute directory for the fixed Python environment and upstream checkouts |
| `resource_root` | Absolute directory for downloaded teachers and the bound calibration reference |
| `student_checkpoint` | Absolute path to the received Joint-SFT-Full `checkpoint-60000` directory |
| `output_root` | Absolute directory for this checkout's experiment outputs |
| `wandb_entity` | Your own W&B username or chosen team slug for online logging; otherwise may be empty |
| `wandb_mode` | `disabled` (default: TensorBoard/JSONL only), `offline`, or `online` |

Do not put access tokens in this file. JSON values are read as data, not shell
commands. The template contains no training hyperparameters.

The tool derives teacher paths, reference paths, runtime Python, CUDA/EGL/Ray
settings, and Python import paths. It does not source a stale `activate.sh` from
another checkout. Every command explicitly selects the same machine JSON:

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" paths
```

`output_root` is bound through a new `outputs` symlink when preparation begins.
If `outputs` already exists, it must resolve to that same directory. The tool
never moves, deletes, or overwrites an existing output directory to make it fit.
Choose a fresh checkout or explicitly use the existing output location.

## 3. Install and verify software

```bash
# Print installation commands without creating directories or downloading.
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" --dry-run install

# Run once on a new runtime path.
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" install

# Repeatable verification; installation itself also runs software checks.
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" verify
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" verify --gpu
```

Installation creates a small host-Python bootstrap venv at `<runtime_root>-bootstrap`,
installs uv0.12.0 there, and delegates to `requirements/opd_profile.py` to create
the separate pinned runtime. Training always uses `<runtime_root>/venv/bin/python`.
An existing runtime is never reinstalled: use `verify`. After installation fails,
keep logs, resolve the cause, and choose a new empty runtime path for retry.
Do not fix conflicts by upgrading pinned packages.

## 4. Prepare resources and authenticate

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" download-teachers
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" assets
```

`assets` invokes the pinned RoboCasa macro setup and download scripts. Respond
to their interactive prompts; the downloader asks for confirmation before
fetching assets. It writes under
`<runtime_root>/repos/robocasa/robocasa/models/assets/`. If reusing an agreed
asset snapshot, place it at that expected location while retaining repository
files; do not point `DATASET_BASE_PATH` at assets (that macro controls datasets).
Asset-source contents are not fully hash-locked by the software profile.

Download Joint-SFT-Full from the public, fixed HF revision in
[RESOURCES.md](docs/opd/collaboration/RESOURCES.md#joint-sft-full-input) and set
`student_checkpoint` to its complete local directory, retaining weights and metadata.
The email task sheet includes a download command using the machine JSON. Then run:

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" prepare
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" login
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" preflight
```

`prepare` checks SHA256 for Joint-SFT-Full and all teachers and creates
`<resource_root>/teacher_reference.json`. It refuses an existing reference;
if already prepared, use `preflight` rather than preparing again. `login` uses
your own W&B account only in `online` mode; it is a no-op in `offline` and
`disabled` modes. A shared team is not required. The optional online project
name is fixed by the experiment YAML. TensorBoard/JSONL logs remain available
with W&B disabled, and result collection does not require W&B links.
`preflight` resolves and validates all four configs without starting Ray or a GPU
job. See [RESOURCES.md](docs/opd/collaboration/RESOURCES.md) for input provenance.

## 5. Prove the setup with smoke runs

On a dedicated, available eight-GPU node:

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine "$HOME/opd-config/robocasa.json" smoke
```

This executes two full physical rounds of M2 and a one-episode OpenDrawer
independent evaluation. It disables training W&B for the smoke and writes under
`outputs/smoke/` and `outputs/launches/`. Confirm both launcher statuses are `OK`
and the evaluation status is `OK_SMOKE`. Missing assets or GPU failures must be
resolved before long runs. Existing smoke directories are not overwritten.

Continue with the [four-experiment guide](docs/opd/collaboration/README.md).

## Pinned profile and reproducibility boundary

Profile **`gr00t-n15-robocasa365-20261019`** pins Python3.11.14, Torch2.6.0
(CUDA12.4, old C++ ABI), FlashAttention2.7.4.post1, Transformers4.51.3,
MuJoCo3.3.1, NumPy2.2.5, and Ray2.56.1.

| File | Purpose |
| --- | --- |
| [profile.json](requirements/profiles/gr00t-n15-robocasa365-20261019/profile.json) | Python, uv, four upstream revisions, FlashAttention wheel/ABI |
| [requirements.txt](requirements/profiles/gr00t-n15-robocasa365-20261019/requirements.txt) | Exact versions of275 runtime packages |
| [build-constraints.txt](requirements/profiles/gr00t-n15-robocasa365-20261019/build-constraints.txt) | Build-tool constraints |
| [opd_profile.py](requirements/opd_profile.py) | Existing versioned installation and verification implementation |

This is a working version snapshot, not a minimal dependency resolution.
Upstream GR00T/RoboCasa/LeRobot declarations conflict; installation explicitly
restores known versions with `--no-deps`. `uv pip check` may report those known
conflicts. The upstream general installer is not an equivalent environment.

Verification covers281 package versions including bootstrap/editable packages,
upstream revisions and source locations, imports, Torch ABI, and the local Eagle
processor; `--gpu` adds a real FlashAttention CUDA operation. It does not prove
a complete simulator/training run. OS, drivers, binary builds, and external
assets are not a fully locked machine image. Public-source availability and
bitwise numerical identity are not guaranteed.

The [initial installation record](requirements/profiles/gr00t-n15-robocasa365-20261019/VALIDATION.md)
and [release validation](docs/opd/collaboration/VALIDATION.md) describe completed
checks. A profile update requires a new dated directory and fresh installation,
software/GPU checks, and regressions. Preserve the current profile for this batch.
