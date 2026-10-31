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
