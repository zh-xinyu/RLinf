# Unfrozen formal LIBERO draft

**Executable draft, not an accepted or frozen experiment recipe. No formal runs
have been launched by these checks.** The four new YAMLs inherit
`libero_mopd_formal_draft_common.yaml`, which extends the existing candidate
common file. Provisional LR3e-5 and B256 are defined once in the new common file.
All arms share T32, microbatch1, constant LR with four optimizer-update warmup
calls, native student inputs, fixed570 teachers, per-microbatch gradient
synchronization, zero actor cache, and native240/320/320/520 horizons.

| Arm | Allocation / suite loss | Reuse | Physical rounds | Updates | Save interval |
|---|---|---:|---:|---:|---:|
| m0 | Uniform / equal | 1 | 150 | 150 | 25 updates |
| m1 | Uniform / equal | 2 | 150 | 300 | 50 updates |
| m2 | Adaptive / synchronized | 2 | 150 | 300 | 50 updates |
| m3 | Adaptive / synchronized | 1 | 150 | 150 | 25 updates |

The adapter `launch_formal_draft.py --arm m0..m3` registers one selected new
configuration by assigning `launch_production.CONFIG_NAME` **only in its own
process**, then invokes the unchanged `launch_production.main`. The real parser,
Hydra composition, checkpoint checks, private Ray ownership, driver execution
and cleanup remain in that launcher. No source file or running process is
patched. The adapter injects the table's update/save budgets and rollout520;
it rejects conflicting recipe overrides, duplicate options, nonnative padding,
and `--allocation-smoke`. Shortened adaptive integration checks remain separate
from this150-round matrix. A native B override was never restricted to M1;
only the canonical-fixed570 experimental profile has that restriction.

## CPU plan with real local paths

Run on **action11**, from the isolated worktree. This revised wrapper requires
that node and provides profile `gr00t-n15-libero-20261105-r2`. Other nodes require
their own verified wrapper; an identically named environment is insufficient.

```bash
ROOT=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try
cd "$ROOT/outputs/worktrees/libero_gr00t"
RUNTIME="$PWD/outputs/runtime_libero_profile_action11_20261105_r2/bin/python"
export OPD_WANDB_RUN_NAME=libero_formal_draft_m1_seed0
export OPD_WANDB_GROUP=libero_formal_draft_unfrozen
CUDA_VISIBLE_DEVICES='' "$RUNTIME" toolkits/libero_gr00t/launch_formal_draft.py \
  --arm m1 --seed 0 --python "$RUNTIME" \
  --models "$ROOT/outputs/libero_gr00t_20261104/models" \
  --student "$ROOT/outputs/libero_gr00t_20261104/models/student_fewshot" \
  --spatial-teacher "$ROOT/outputs/libero_feasibility_20261104/models/groot_spatial_rl" \
  --output "$ROOT/outputs/libero_gr00t_20261104/weak_student/formal_draft_m1_seed0_plan" \
  --gpus 4 5 6 7 --timeout-seconds 172800 --dry-run
```

Use `--arm m0`, `m2`, or `m3` and seeds0/1/2, with distinct output/run names.
M2/M3 additionally require `OPD_LIBERO_TEACHER_REFERENCE` pointing to an accepted
fresh-runtime reference. Historical `teacher_calibration_native20/teacher_reference.json`
may exercise CPU composition only; it is not accepted as the revised-runtime
formal calibration. The dry-run checks do not start Ray, initialize CUDA,
create W&B runs, or reserve GPUs. The48-hour timeout is a bound, not an estimate.

For a real Hydra entry check, use the generated plan directory:

```bash
CUDA_VISIBLE_DEVICES='' "$RUNTIME" examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/outputs/libero_gr00t_20261104/weak_student/formal_draft_m1_seed0_plan" \
  --config-name resolved_config --cfg job --resolve
```

After the gates below and explicit launch authorization, the execution path is
the same adapter command with `--run` and a **new** output directory. Set online
W&B authentication and a unique `WANDB_RUN_ID` per arm/seed, `WANDB_MODE=online`
and `WANDB_RESUME=allow`; retain that ID for strict continuation. All training
logs use project `libero_gr00t_pilot`. Config display names do not create distinct
W&B identities by themselves. The production launcher intentionally sets
`runner.logger.experiment_name=libero_production_smoke`; actual checkpoints are
`<output>/libero_production_smoke/checkpoints/global_step_<N>/`. The draft YAML
names do not rename that directory. No credentials belong in commands or artifacts.

## Gates still open

Learning and strict continuation under the revised runtime must pass before the
provisional common recipe is accepted. Freeze runtime/source/checkpoint identities;
refresh the20-state/task teacher reference; verify real adaptive quota activation,
synchronized suite weights and resume; and validate all2,000 independent evaluation
pairs under the chosen runtime. The historical12-plan matrix and200-pair scheduler
smoke do not satisfy these gates. Retain the original adaptive law and constants;
no extra arms or tuning grid are introduced here. The optional pipeline forensic
observer is not a required readiness gate.

Independent evaluation remains separate at rounds0/25/50/75/100/125/150. Saving
checkpoints does not schedule it. Follow `candidate_commands.md`'s full evaluation
commands using the accepted revised wrapper, **`--all-tasks`**, and a strict-load
validation protocol from each exact checkpoint. Each evaluation covers40 tasks
×50 official states0..49; use the same student evaluator/RNG protocol for baseline
and trained curves. Four arms ×six post-training evaluations ×three seeds require
144,000 episodes, plus shared student/teacher references. The matrix fixes physical
trajectories, not optimizer compute, and the final endpoint is round150 rather
than a checkpoint selected by test success. W&B dashboards do not replace raw
coverage and identity audits.
