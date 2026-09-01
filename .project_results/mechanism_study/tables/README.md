# Mechanism-study tables (paste-into-Sheets)

Tab-separated. Open a file, select all, copy, paste into Google Sheets — tabs become columns.

## Paste these two

| file | what it is |
|---|---|
| **`01_runs.tsv`** | the run log — one row per **successful** run (7), with config, metric, wall clock, commit, log path, checkpoint dir |
| **`04_interim_comparison.tsv`** | the interim comparison — one row per method, `eval/success_once`, delta vs M1 in SD |

Optional third, if you want a chart: **`07_pivot_success_once.tsv`** (step × method, wide —
select and insert a line chart).

## Backup detail, paste only if needed

`00_shared_setup.tsv` (what all arms share) · `02_eval_curves.tsv` (long format, 240 rows,
every metric at every step) · `03_summary.tsv` (per-run first/final/best/mean) ·
`05_findings.tsv` (F1–F11 with confidence and caveat) · `06_legacy_reference_points.tsv`
(pre-fix numbers, quarantined so nobody reuses them) · `08_pivot_success_at_end.tsv`.

## Status: interim

All four arms — M1, M3, M3-Hybrid, M4 — were re-run **together at 25 steps**, eval every
step, commit `d04795ec` (2026-08-17→18), once the Ray startup failure that had blocked M4 was
understood. Metric: **`eval/success_once`**.

Mean over steps 11–25: **M3 0.834 ≈ M3-Hybrid 0.828 ≈ M1 0.808 > M4 0.787**. M1/M3/M3-Hybrid
are within ~1 SD of each other (SE of a difference 0.030); M4 is below M1 at 20 of 25 eval
points. **One seed per arm** — this is a snapshot to steer the next round, not a final result.

Failed and aborted attempts are deliberately not in the run log. The only one worth
remembering: M4's five startup hangs in the preliminary round were never M4's fault (F11).

Why `eval/success_once` and not the training-rollout `success_once`: eval forces
`flow_ode` for every arm (`denoise_inds = -1`), so all four are compared as the same kind of
policy, and its reset states are fixed across steps. The training rollout injects SDE noise
for M3-Hybrid and M4, which costs M3-Hybrid 0.31 success at step 1 and flips the ranking.

These tables are the canonical machine-readable record. The parent README summarizes the
final round and keeps the preliminary 50-step comparison in context.
