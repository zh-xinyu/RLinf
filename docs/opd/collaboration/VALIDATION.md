# Release validation — 2026-10-31

## Formal launch resource settings — 2026-11-03

- Current-branch formal YAMLs use W&B project `FORMAL_robocasa_mopd` and the
  same optional allocator cache settings in all four arms: prewarm 26 GiB,
  retention bound 28 GiB per actor/GPU. Learning and evaluation controls are
  unchanged. The original v1 tag is preserved.
- Passed 78 targeted tests covering cache retention/preallocation, matrix
  controls, suite distillation, refresh, and the reproduction workflow.
- A real CUDA test reserved 26.018 GiB, reused a 2 GiB allocation without
  growing the reservation, and preserved the checked matrix product and CUDA
  RNG state. Temporary warmup tensors were released, leaving 0.008 GiB live.
- The existing action10 runtime passed 281-package profile verification and
  the FlashAttention GPU check. All four checkpoint inputs passed manifest
  verification. These checks do not replace the full eight-GPU smoke run.

## Completed

- Existing adaptive-sync, allocation resume/probes, suite metrics, and GR00T memory
  changes passed 115 targeted regression tests before consolidation.
- A clean snapshot exported from the Git index, with no private node env files,
  `shs/`, or `toolkits/mopd/`, passed **166 regression tests**. Coverage includes the four
  matrix arms, equal control parameters, 150-round/save25 conversion, full-pool
  refresh, weighted gradients, owner-scoped Ray cleanup, and 20/50-episode budgets.
- Real C10 and three teacher checkpoints were hashed. All 15 teacher inference
  files match the fixed official HF revision (including LFS weight hashes).
- The public resource command verified these four checkpoints, generated the
  machine-local reference, and passed adaptive M2 configuration dry-run.
- The existing fixed runtime passed software verification: Python 3.11.14,
  281 package versions, pinned upstream repositories and required imports.
- The clean snapshot also passed real-input adaptive dry-run and relative Markdown
  link checks. No doc-code or EN-ZH consistency issues found in checked scope
  (project READMEs, environment/development entries, handoff, evaluation guide).
- Ruff lint/format and Git whitespace checks cover the changed Python files.
  A final launcher-only adjustment uses a short temporary Ray directory to stay
  below Unix socket path-length limits; targeted launcher tests were rerun.

## Remaining machine acceptance

The current node has GPUs occupied by other jobs. This release pass does not
claim a new eight-GPU end-to-end run, a full fresh environment reinstall, or
completed formal experiments. Prior pilot runs validate the training mechanisms;
they do not replace the receiving machine's setup checks.

Before long runs, follow the handoff's runtime `--gpu` check, two-round training
smoke, and single-task independent evaluation smoke. Keep the resulting logs.
The repository locks software versions and model contents; the full simulator
asset tree is not yet identified by a published hash manifest.

## Unified reproduction entrypoint

The `robocasa-formal-150r-handoff-20261031` bundle adds `experiment.py`, a
machine JSON template, and the result collector. It fixes per-run JSONL and
TensorBoard isolation by assigning a separate logger directory per run. The
existing runner consequently stores local checkpoints under the nested run-name
directory; the new evaluation command resolves that path automatically.

- Confirmed the RLinf fork point from Git history:
  `e7609d4c9e2f33c5ffc10b67c61c8e4b73208b45`, the parent of the first OPD commit.
- Compared `rlinf/`, the training YAMLs, `requirements/`, and evaluation code
  against the original formal tag: no changes to algorithms, hyperparameters,
  pinned software, or evaluation semantics.
- Passed **180 relevant regressions**, repeated successfully in a clean snapshot
  exported from tracked files without private launchers or node profiles,
  including machine/hostname validation,
  output protection, per-run logging, all25 evaluation budgets, task-count score
  reconstruction, incorrect-checkpoint rejection, and TensorBoard timer reading.
- Invoked the new entrypoint from host Python3.9 without activating the runtime:
  the281-package verification and actual FlashAttention GPU operation passed.
- Verified real C10/teacher files through `prepare`; all four `preflight`
  configurations resolved and passed validation. Train/evaluation dry-runs
  selected the expected configs, six checkpoint paths, and50-episode mode.
- The clean snapshot also passed real-resource four-arm preflight and generated
  a25-row pending table through the same public entrypoint.
- Checked documentation links, shell syntax, and Ruff. The mail-only template
  has25 pending rows and keeps expected budgets separate from measured values.

This entrypoint validation did not reinstall the full runtime or run new
multi-GPU simulator training. The receiving machine must still complete the
installation/resource checks and the documented eight-GPU smoke sequence.

## Handoff review — 2026-11-01

Release tag: `robocasa-formal-150r-handoff-20261101`.

- W&B is optional: the machine template defaults to disabled logging, online
  users choose their own username/team, and offline/disabled modes skip login.
  Disabled mode removes the W&B backend while retaining TensorBoard and JSONL.
- Passed **178 targeted tests**, including all four formal configurations,
  flow losses/refresh, suite allocation/weighting, profile contracts, independent
  evaluation, and result collection. New tests check that disabling W&B changes
  only the logger backend and that a result with no W&B URL can be complete.
- C10 is available from the public project HF repository at revision
  `6d74e06da27996ae7802fb9b83b6697e07590a1f`. All five input file sizes and SHA256
  values match the existing checkpoint manifest; no model contents changed.
- Reviewed the current tracked tree (1,961 files): no matches in the scanned
  HF/GitHub/AWS credential, private-key, or literal API-token patterns; no tracked
  model weights, raw outputs, personal node env files, meeting documents, or mail
  handoff files; no individual tracked file larger than 10 MiB. This is a bounded
  current-tree scan, not a full-history or comprehensive security audit.
- RLinf attribution and Apache-2.0 license remain present. Third-party models,
  assets, and source repositories retain their own licenses. Historical mechanism
  study records remain tracked as provenance; their old machine paths are not
  installation inputs. Public pi0.5 config paths are environment inputs.
- The learning code, YAML recipes, runtime profile, evaluation protocol, and
  result CSV schema are unchanged by this handoff revision. The private task
  sheet and blank CSV remain outside Git and are sent separately.

Receiving-machine installation and eight-GPU smoke checks are still required.
This review does not claim a new full environment installation, formal training
run, or a fully hash-locked simulator asset snapshot.

## Joint-SFT-Full initialization release — 2026-11-01

Release tag: `robocasa-mopd-joint-sft-full-v1`.

The collaborator had not started training when the initialization was changed.
All four arms now start from **Joint-SFT-Full**, the full-data joint SFT model
trained on 50 tasks, 500 demonstrations/task, 25,000 demonstrations total, for
60,000 updates. The previous C10 releases and historical results are preserved;
they are not results for this new initialization.

- Uploaded only the five inference files to the project HF repository under
  `robocasa365/groot-n15/joint-sft-full/checkpoint-60000` at revision
  `ee4d8c4a3ba01e092ed91b8bd6e452e28831026e`. Checked local SHA256 and size,
  remote LFS SHA256/size for shards, and downloaded small-file contents.
- Updated the checkpoint manifest with model hashes and SFT provenance. The
  three teachers and their calibration statistics are unchanged; rerun
  `prepare` to bind the new manifest to machine-local paths.
- New run IDs and the W&B group identify the initialization. The baseline
  command is `eval joint-sft-full`; every CSV row includes `initialization`.
  The collector checks initial weight/config/metadata hashes as well as the
  trained checkpoint identity, preventing old-baseline results from being
  accepted as this release.
- All **181 regression tests passed in a clean tracked-file snapshot**, without
  private launchers, personal node env files, or email attachments. Coverage
  includes all four recipes, initialization hash rejection, resource binding,
  flow refresh, suite allocation/weighting, evaluation, and result collection.
- Real Joint-SFT-Full and three teacher inputs passed complete file verification.
  Four-arm preflight passed both in the workspace and the clean snapshot using
  the new input manifest and machine-local teacher reference.
- Public guides, English/Chinese READMEs, and the separate mail package use
  the new model and fixed release. The new blank template has 25 pending rows.
  Earlier validation sections above describe their original releases.

Learning algorithms, teachers, T96/B2304/micro32, LR3e-5/warmup10, R1/R2,
150 physical rounds, save-every-25-rounds and 50 episodes/task are unchanged.
No new training or simulator evaluation is claimed here; receiving-machine
installation checks and both smoke jobs remain required.

## Portable Slurm / Blackwell adaptation — 2026-11-03

Branch: `release/robocasa-formal`. The original tags and main branch are
preserved. Four/eight-GPU selection changes placement and accumulation, keeping
all four formal YAML recipes unchanged. Dedicated resource mode turns off the
optional CUDA cache reservation. Slurm GPU visibility is retained as an explicit
placement restriction; no peer processes are terminated.

- Installed `gr00t-n15-robocasa365-blackwell-cu128` into a fresh runtime directory:
  Python3.11.14, Torch2.7.1+cu128, torchvision0.22.1+cu128, FlashAttention2.8.3.post2,
  C++11 ABI. All282 version/source/import checks passed. The four upstream
  checkouts and non-CUDA scientific package pins match the original profile.
- Torch lists `sm_120`; the imported FlashAttention binary contains `sm_120`
  metadata. No actual Blackwell CUDA kernel execution is claimed locally.
- **121 targeted CPU regressions passed in each runtime** (original CUDA12.4
  and new CUDA12.8). They cover equal four/eight-GPU recipes and weighted-gradient
  accumulation, Slurm allocation validation, full matrix/budgets, flow refresh,
  cache settings, checkpoint download protection, evaluation and result collection.
- The same121 tests also passed from a clean staged-file snapshot with no
  private machine overlays, local handoffs or experimental launch scripts.
- Real-input four-arm preflight passed in the new runtime. Its independent
  evaluation runtime verifier accepted the pinned sources/packages and all50
  tasks, including editable RoboSuite metadata.
- Slurm formal launches require matching successful smoke records. The updated
  smoke verifies GPU forward/backward, runs M2 for two rounds/four updates, saves
  step4, and independently evaluates the saved weights on one task.
- Submission planning includes four training jobs plus25 checkpoint/baseline
  evaluation jobs. CSV provenance now includes GPU counts, profiles and commits.

**Remaining acceptance:** the target Blackwell node must pass the full documented
smoke, including EGL/simulator execution, FSDP training and saved-weight loading.
No Blackwell hardware was available during this adaptation. Existing GPUs were
occupied, so this pass did not run new GPU experiments or claim formal results.
The exact Slurm partition/account and site time limits must be confirmed locally.
