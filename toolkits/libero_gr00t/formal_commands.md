# Historical T32 LIBERO GR00T experiments

The current primary plan is [T64 on action10](formal_t64_commands.md).
This page preserves the earlier T32/100-round recipe. Older `formal_draft_commands.md` and
candidate/K8 recipes are historical and must not be used for these runs.

| Arm | Allocation / suite loss | Refresh R | Rounds | Updates | Save every |
| --- | --- | ---: | ---: | ---: | ---: |
| M0 | Uniform / equal | 1 | 100 | 100 | 25updates |
| M1 | Uniform / equal | 2 | 100 | 200 | 50updates |
| M2 | Adaptive / synchronized | 2 | 100 | 200 | 50updates |
| M3 | Adaptive / synchronized | 1 | 100 | 100 | 25updates |

Initial execution queues **M1 then M2, seed0** on action10 GPUs0–3. M0/M3 and
additional seeds are planned, not implicitly launched by the entrypoint.
Each arm collects exactly3200student trajectories. Optimizer compute differs
between R1/R2; report both interaction and wall-clock budgets. Teacher calibration
and offline initialization have separate shared costs.

Shared settings: T32, B512, native text buckets with micro2/rank, LR3e-5,
four optimizer-update warmup calls, constant scheduler, original M3,
four ODE steps and full16×32 flow latent. Execute5/5/5/10actions with physical
horizons240/240/320/480. Longer storage prefixes do not add physics or extra
loss weight for Long. Each suite's tasks rotate through all ten tasks.

The initialized student is the accepted offline-distillation checkpoint at
update50 (confirmation122/200=61.0%). Keep its own grouped11D/native text
metadata. Teachers remain RLinf Spatial-Step400, Object-Step400, Goal-Step500,
and Long-Step300, with their own8D/fixed570 metadata and BF16 rotary buffers.
Use the verified800-episode official-prefix teacher reference; its score is a
calibrated controller reference, not a replacement for final teacher evaluation.

Adaptive settings are rho0.5, epsilon0.01, warmup20physical rounds, window20,
interval10, minimum5episodes/task and maximum quota change3/suite/decision.
Initial quotas8/8/8/8; minimum4/suite. The synchronized loss coefficient is
four times the suite's realized quota fraction. Sampling remains suite balanced
within each optimizer batch. No short-window `--allocation-smoke` overrides.

## Launch

Run on action10 using its independently restored profile. The wrapper must match
the current node; do not substitute a wrapper from another machine.

```bash
OPD_ROOT=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try
cd "$OPD_ROOT/outputs/worktrees/libero_gr00t"
OPD_PYTHON="$PWD/outputs/runtime_libero_profile_action10_20261105_r2/bin/python"
export OPD_LIBERO_TEACHER_REFERENCE="$OPD_ROOT/outputs/libero_gr00t_formal_20261107/official_teacher_reference.json"
export OPD_WANDB_RUN_NAME=M1_Uniform-R2_T32_B512_seed0
export OPD_WANDB_GROUP=T32_offline50_seed0
export WANDB_PROJECT=FORMAL_libero_gr00t
export WANDB_MODE=online

"$OPD_PYTHON" -m toolkits.libero_gr00t.launch_formal \
  --arm m1 --seed 0 --python "$OPD_PYTHON" \
  --models "$OPD_ROOT/outputs/libero_gr00t_20261104/models" \
  --student "$OPD_ROOT/outputs/libero_gr00t_20261104/offline_distillation_20261107/training/checkpoint-000050/hf" \
  --spatial-teacher "$OPD_ROOT/outputs/libero_feasibility_20261104/models/groot_spatial_rl" \
  --gpus 0 1 2 3 --cpus 32 --timeout-seconds 172800 \
  --log-cuda-memory \
  --output "$OPD_ROOT/outputs/libero_gr00t_formal_20261107/M1_Uniform-R2_seed0" \
  --run
```

Authenticate W&B through the existing local account. Never put credentials in
tracked files. The launcher defaults to a dry run; use a fresh output directory
for each plan or real launch. For M2 select `--arm m2` and distinct output/run ID.
Set a unique `WANDB_RUN_ID` for each arm and preserve it for strict resume.
Checkpoints are `<output>/libero_formal/checkpoints/global_step_<N>/`.
Both Adam/scheduler shards and full strict-load weights are saved. Global Ray
shutdown and unrelated process termination are prohibited.

## Independent evaluation and results

Save/evaluate at rounds0/25/50/75/100. Saving alone does not execute evaluation.
Evaluate the shared initial student once and each saved arm checkpoint using
all40tasks ×50official initial states0..49 =2000episodes/checkpoint. Use the same
seed1234and protocol for every arm. These are official initial states, not a
claimed held-out split from training resets. Use the [shared rollout-worker evaluation entry](parallel_evaluation.md) on
`experiment/libero-gr00t-rollout-batching`. It runs several exact states in
parallel through `LiberoEnv`, `EnvWorker`, `MultiStepRolloutWorker` and
`EmbodiedEvalRunner`, with episode-private noise streams. Execute the accepted
5/5/5/10 protocol; do not use the historical K8 distributed recipe.

```bash
"$OPD_PYTHON" -m toolkits.libero_gr00t.launch_parallel_evaluation \
  --python "$OPD_PYTHON" \
  --config "$OPD_TRAIN_CONFIG" \
  --weights "$OPD_FULL_WEIGHTS" \
  --gpus 0 1 2 3 --envs-per-task 5 --cpus 8 \
  --output "$OPD_EVAL_OUTPUT" --run
```

Use the runtime wrapper belonging to that worktree/node. Set `OPD_TRAIN_CONFIG`
to the selected run's resolved config and `OPD_FULL_WEIGHTS` to its saved
`actor/model_state_dict/full_weights.pt`. Omit `--run` for a dry run; use a new
output directory to execute. Replace `--weights ...` with `--baseline` for round0.
The default evaluates all40tasks ×50states. Preserve `episodes.csv`,
`summary.json`, configs, hashes and raw worker/driver logs with each result.
The previous `evaluate_full_checkpoint.py` remains the standalone parity reference.

Main plots use rollout rounds; every round has32trajectories for every arm.
Also provide real simulation steps and wall time. W&B internal step remains
student optimizer updates. Compare suite-equal success, per-suite MSE and
success, realized quotas/weights, task coverage and observation replacement.
Do not compare adaptive-mixture raw success as if it were suite-equal success.
Fill `checkpoint_results.csv` only with complete audited independent evaluation.
