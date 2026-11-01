# Release validation — 2026-10-31

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
