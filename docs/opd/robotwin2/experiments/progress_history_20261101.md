# RoboTwin pilot live status

Updated 2026-11-01, action00. Goal active. Repaired dense-bucket pilot launched from the original base after verified patch-sync OOM; full effectiveness results remain pending.

- Worktree: `outputs/worktrees/robotwin4`
- Branch: `dev/mopd-robotwin2-four-task`; commits `8a9313fa`, `fad60d6c`.
- Root remains `opd-project-main`; existing RoboCasa code/runtime/outputs untouched.
- Activate `/localscratch2/liyifa11/XyFile/opd-robotwin4/env.sh`.
- Online W&B auth uses existing `WANDB_ENV_FILE` from `.opd.action00.env`; source its file without printing credentials. Isolated runtime alone does not load authentication.
- Active run: `outputs/robotwin4/train/robotwin4_uniform_refresh_r2_t64_b512_bucket_pilot_20261101`.
- Active launcher PID2976313, create_time1793536578.72; identity/command in `outputs/robotwin4/setup/uniform_bucket_pilot_launch.json`.
- Old failed-run W&B: https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/f6ve12mw . Active repaired-run W&B: https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/77se8lie .
- T64/B512/R2/micro8; LR2.5e-5, constant, warmup8updates; base student; FP32 actor masters + BF16 AMP. Initial 20 rounds / 40 updates; save every20 updates. NUMA interleave applies only to this job and descendants.

## Completed checks

- Four specialists actual successes in pilot calibration: adjust3/5, handover4/5, place5/5, stack4/5. Base reference0/3 each (small sample).
- Teacher source hashes, 5 conversions, strict loading, JAX/PyTorch agreement verified.
- 107 tests + Ruff passed.
- `robotwin4_fp32_r2_smoke_20261101` COMPLETE: 1round/2updates, MSE1.0939/1.1246, finite gradients, second LR3.125e-6. Peak~41.5GiB/GPU. Exact rollout/mask step accounting passed.
- Snapshot comparison verified actual FP32 updates in VLM, action expert and projections.
- Smoke checkpoint successfully loaded for native closed-loop evaluation.
- Old mixed-master smoke intentionally stopped after first rollout; original pilot failed only at W&B auth before training. Both have retained diagnostic artifacts; neither is an effectiveness result.

## Next autonomous work

1. Monitor `status`, `driver.log`, `optimizer_updates.jsonl`, `rollout_episodes.jsonl`; use `python -m toolkits.robotwin_mopd.summarize RUN_DIR`.
2. First two T64 rounds passed with finite losses/gradients. Monitor rounds3–5 and first save for host/GPU memory stability. Do not increase microbatch8; no global Ray stop or system memory changes.
3. Let initial20rounds finish if stable. Checkpoint folder is nested: `RUN_DIR/RUN_NAME/checkpoints/global_step_40/`.
4. Independently evaluate the trained student on all four tasks (small5episode/task pilot, explicit counts); compare to initial base and teachers. Use `toolkits.robotwin_mopd.evaluate --checkpoint PATH`, one GPU/task, only after training releases GPUs.
5. Based on learning/SR trends decide one useful continuation: extend uniform to80updates with round-boundary resume if still undertrained, or matched-budget Adaptive-Sync-R2 from base if capability gain and task differences are evident. No broad parameter scan. Pilot calibration lives in `outputs/robotwin4/eval/teacher_reference_pilot.json` (5episodes/task; not formal).
6. Update EXPERIMENTS_20261101.md and results. Do not mark active goal complete merely because the job was launched.

Actor cache has4teachers per rank; smoke host reclaim pressure was high around teacher/save, despite available RAM. NUMA interleave was applied for pilot; monitor actual timings before further memory changes. All checkpoints/outputs stay isolated.

## First two T64 rounds verified

Rounds1–2 / updates1–4 completed: raw velocity MSE1.14665 →1.12782 →1.08153 →1.07384; finite gradient norms3.23/2.93/2.97/2.88. Cumulative128episodes and73,085actual control targets. Per-round success3/64 then1/64, all adjust_bottle. These early small counts are not evidence of success-rate improvement. Round2 took about12m10s; roughly4h is a provisional20round estimate, excluding final evaluation.

NUMA interleave kept swap around6.6GiB with no sustained reclaim pressure; actor0 VmSwap0. GPU sampled peak43.35GiB. Host available fell257→204GiB across the first two round boundaries, so rounds3–5 still need memory-growth checks.

## Queued independent evaluations

Both queues are waiting without GPU children and never stop training or unrelated processes.

- Matching unfiltered/seen queue PID2910670: `outputs/robotwin4/eval/uniform_r2_step40/queue.json`. Waits for actual launcher exit+COMPLETE+step40 checkpoint, then student4×5episodes and base4×2extensionepisodes. Existing base3/task and teachers5/task are preserved. Seed overlap is explicit; adjust teacher starts at a shifted seed.
- Supplementary fixed feasible/unseen queue PID2933622: `outputs/robotwin4/eval/feasible_unseen_step40/queue.json`. Waits for the first queue to finish. Builds5scripted-expert-accepted scenes/task, then evaluates teacher/base/student on exactly the same seeds and prompts. Separate attempt paths and pool content hashes are retained. See [protocol audit](../EVALUATION_PROTOCOL_AUDIT.md) and [evaluation details](../FEASIBLE_EVALUATION.md).

The official screening rule is scripted planner `plan_success AND check_success`; it does not select scenes based on learned teacher/student success. No scripted action data enter training. Current training remains unfiltered/seen. All5episode results are pilot estimates, not formal benchmark scores.

New evaluation, fixed-seed adapter, queues and calibration budget accounting:35CPU tests, Ruff and git diff checks passed. Calibration records20teacher episodes,5379executed control targets, and219.39s approximate launch-manifest-to-last-result wall time; the478.01s sum of parallel episode durations is separately labeled.

## Round3 update

Updates5/6 completed: velocity MSE0.90990/0.47021, gradient norms2.1017/1.1633, all finite. Update6 task MSE: adjust0.37376, handover0.55558, place0.76869, stack0.18283. All four tasks show early optimization response; this alone does not establish closed-loop improvement. Round3 rollout SR1/64 was collected before these updates. Continue the unchanged20round pilot and independent evaluations.

## Round4 update

Updates7/8 completed: MSE0.41558/0.41516, gradient norms0.73483/0.99996. The first update uses fresh physical-rollout observations, so round3's reduction persists beyond reuse of its observation pool. Round4 rollout success remains1/64; capability gain is still unproven. Update8 task MSE: adjust0.33370, handover0.43812, place0.78303, stack0.10580. Cross-task MSE magnitudes are in distinct task-normalized action spaces and should not be interpreted directly as relative task proficiency.

## Failure after update8: weight-sync peak

The first four rounds/8updates completed with finite gradients. On entry to round5, rollout rank0 failed in collective IPC `tensor.clone()` while receiving a2GiB tensor. GPU0 had actor36.00GiB + environment3.58GiB + rollout7.47GiB; only284.5MiB remained. This was not backward OOM or another user's process. The collective queue thread failed while launcher remained alive, so the exact owned launcher PID2884880 was sent SIGTERM to invoke its scoped cleanup. Full traceback and operator_note.json retained. No pilot checkpoint had yet been saved (save interval20updates). Do not evaluate this as a completed40update run or silently resume from the unrelated smoke.

Both downstream evaluation queues are tied to this original launcher and must stop at its FAILED gate. A repaired run will need new queue/output paths and its own launcher identity. Investigate sync tensor lifetime/staging before restarting; retain FP32 actor masters.

## Repaired pilot launched

Commit fb4d714a changes only the RoboTwin config from patch to existing dense bucket sync:128MiB target, original transport dtype, immediate receiver loading. Largest single parameter~1.96GiB is not split.9CPU bucket tests passed;1GPU test skipped. No learning hyperparameter or FP32-master/BF16-AMP change. GPU peak still requires validation.

New run `robotwin4_uniform_refresh_r2_t64_b512_bucket_pilot_20261101` starts from the same base, max40updates, save_interval8. The failed8updates are not carried over or mixed into its curve. Five planned checkpoints replace the old interval20 to reduce recovery loss. PID2976313.

New matching queue PID2982573: `outputs/robotwin4/eval/uniform_bucket_r2_step40/queue.json`. New feasible/unseen queue PID2982576: `outputs/robotwin4/eval/feasible_unseen_bucket_step40/queue.json`. Both bind to the new run and use new paths. Original queues terminated at FAILED gates and did not evaluate nonexistent checkpoints.

## Dense-bucket repaired run: first round

First64episodes and2updates completed: raw MSE1.14772/1.14190, gradnorm3.2847/2.9510, all finite. First-rollout SR0/64 is an initial-base measurement. Initial sync53.32s; first round856.6s includinginitialsetup/teacher loading. Sampled training GPUpeak42.35GiB; nextsync sampled~18.8GiB withoutOOM. Actor0RSS45.2GiB, VmSwap0, PSI0. Continue validating later dense updates and firstcheckpoint.

## Dense-bucket repaired run: two rounds

Secondround/updates3–4 passed; MSE1.09855/1.05070 andfinitegradients. Round2SR2/64. Sync timings53.32/49.06s acrossfirsttwo rounds; sampledtrainingGPUpeak42.60GiB. Rollout-stagehostavailable~261GiB andactor0RSS46.35GiB. A transient post-round2 swap increase~6.6→12GiB had lowPSI (sampledpeak0.77%, then0.08%), still worthmonitoring. Nextgates:completecheckpoint atupdate8 andoriginalfailureboundary enteringround5.

## Dense-bucket repaired run: checkpoint8 and original failure boundary passed

Updates7/8 MSE0.40891/0.40426, finite gradients; round4SR0/64. Checkpoint8 has full_weights16,573,945,133bytes,8nonemptyDCPshards plusmetadata. Runner/allocation agree on8updates,4rounds,146,074actualcontroltargets. `checkpoint_step8_audit.json` records checks; this is artifact/accounting validation, not an actual fullresume test. Fifthrollout has started after successful sync, passing the previous patch-OOM boundary.

Save began08:30:24local and finished by~08:32. CPUreclaim briefly pushedPSIfullavg10~63% andswap~63.6GiB; afterwardavailable~252GiB andPSIavg10=0, noGPUOOM. Keep this as a checkpoint-time host-memory/latency limitation, separate from the fixed GPU weight-sync failure. Do not stop a healthy rollout solely for residual swapped coldpages.

## Dense-bucket repaired run: post-warmup and metric scope

Round5/updates9–10 completed: MSE0.38902/0.39066; gradientsfinite andLR2.5e-5. Round5SR1/64; no clear capability gain yet. Full fifthrollout passed aftertheoriginalsyncfailureboundary.

Read-onlylossaudit: OpenPI `nft_forward` returns25chunkpositions×32latentdims (`openpi_action_model.py`), whileenvironmentuses14actiondims. ExistingM3mask handlesvalidtemporalpositions, notpaddedlatentdims. Per-taskcount102400=128samples×25×32 confirms the loggedmetric scope. Full-latent matching is consistent with the model's32-dimensionalcoupledflow; it must not be reported as physical14-dimensionalactionerror. Do not altertheobjective midrun. IndependenttaskSR remains the effectivenesscheck; optionalfuturediagnostics can split actual andpaddedchannelMSE.
