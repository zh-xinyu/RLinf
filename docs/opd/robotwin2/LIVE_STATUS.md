# RoboTwin 四 teacher pilot：当前状态

更新：2026-11-01，action00。目标仍进行中；首轮训练及未筛选/seen独立评测已完成，完整flow诊断已完成，正在准备一组有界的full-horizon跟进实验。

## 当前运行

- 分支：`dev/mopd-robotwin2-four-task`；worktree：`outputs/worktrees/robotwin4`。
- 原根目录仍为 `opd-project-main`；RoboCasa 代码、runtime 和实验输出未改动。
- Run：`robotwin4_uniform_refresh_r2_t64_b512_bucket_pilot_20261101`。
- [W&B 当前运行](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/77se8lie)。
- Launcher PID：2976313，create_time：1793536578.72；身份与命令：`outputs/robotwin4/setup/uniform_bucket_pilot_launch.json`。
- 训练 COMPLETE：20 rounds / 40 optimizer updates，1,280 trajectories，734,389 executed control targets。Launcher 已退出。
- 最终 MSE：0.07117；末5轮（10次更新）平均 MSE：0.05592。所有已记录梯度和 loss 有限；当前 run 无 GPU OOM。

## Setting

四任务：adjust_bottle、handover_mic、place_container_plate、stack_bowls_two。四个独立 PACE specialist，student 从官方 π0.5 base 开始。

| 项目 | 当前值 |
|---|---|
| T / 每任务轨迹 | 64 / 16 |
| R | Refresh-R2，完整观测池，重新采样 noise |
| Global B / microbatch | 512 / 8 per GPU，8 GPUs |
| LR / warmup | 2.5e-5 constant / 8 optimizer updates |
| 总预算 | 20 rounds / 40 updates |
| 保存 | 每 8 updates；已完成 step8、step16、step24、step32、step40 |
| 精度 | FP32 actor masters + BF16 AMP；teacher/rollout 原推理精度 |
| 同步 | 现有 dense bucket，128 MiB target、原 dtype、即时加载 |
| 环境 | native clean ALOHA，seen instructions，未经过 scripted expert 筛选 |

原生 horizon：400/600/400/900。预测50步、执行25步、10步ODE；不提前缩短任务 horizon。

MSE 对前25个位置的全部32维 flow latent 求均值，环境执行前14维。它不是物理14维 action error，下降也不等于任务成功率提高。各任务还有独立归一化，不能用跨任务 MSE 大小直接比较能力。整个训练共有7/1,280次成功，仅来自前7轮的 adjust_bottle；第8–20轮均为0。末5轮四任务成功率均为0，独立测评尚未观测到能力收益。

## 已通过与限制

- 5个模型转换、源文件 hash、严格加载，以及 JAX/PyTorch 动作/velocity 一致性检查已完成。
- 107项原实现测试、35项新增评测相关测试通过；bucket测试9 passed、1 GPU test skipped。真实 GPU 同步与多轮训练已执行。
- Step8/step16/step24/step32/step40 均有完整权重、8个 DCP shard、metadata、runner/allocation 状态；计数核对一致。未在当前 checkpoint 上做实际全状态恢复测试。
- 已越过原 patch 同步 OOM 位置，第5轮完整 rollout 和更新通过。
- 同步约49–53秒，训练每次更新约44秒；采样 GPU 峰值约43.4 GiB。
- 保存 step8 约131秒，曾触发短暂 CPU reclaim：swap接近63.6 GiB，PSI full avg10约63%，随后回落。第二次保存也完成。保留为独立性能限制，不与 GPU patch OOM 混同。
- 初始 teacher 校准为3/5、4/5、5/5、4/5；base各0/3，均仅小样本 pilot。

## 自动接续评测

已确认本 run launcher 退出、COMPLETE 和完整 step40 checkpoint。第一条队列 COMPLETE，已退出。第二条因 MPLib 筛选适配风险被主动停止，未运行其 teacher/base/student 评测。

1. 已完成（原 PID2982573）：`outputs/robotwin4/eval/uniform_bucket_r2_step40/queue.json`。沿用未筛选/seen协议，student每任务5次，base补2次形成每任务5次；保留原teacher5次。记录实际seed重合情况。
2. 已停止（原 PID2982576）：`outputs/robotwin4/eval/feasible_unseen_bucket_step40/queue.json`。首个 MPLib scripted expert 候选失败并耗时121秒，随后持续 invalid-start/self-collision。原生 ALOHA 默认curobo，当前MPLib双臂关节映射与规划状态更新有适配风险；没有把筛选失败归因于策略。仅停止该自有队列及builder，保留partial pool、日志和`operator_stop.json`，状态为`STOPPED_PROTOCOL_UNVALIDATED`。该不完整pool不可用于评测或能力结论。

见 [评测协议](FEASIBLE_EVALUATION.md) 和 [协议核对](EVALUATION_PROTOCOL_AUDIT.md)。5 episodes/task 不足以支持正式论文统计结论。

## 复现与产物

以下路径均相对原 workspace 根目录，而非 worktree：

- Runtime：`/localscratch2/liyifa11/XyFile/opd-robotwin4/env.sh`。W&B凭据来自 `.opd.action00.env` 的 `WANDB_ENV_FILE`；激活runtime本身不加载凭据，不打印凭据。
- 训练：`outputs/robotwin4/train/robotwin4_uniform_refresh_r2_t64_b512_bucket_pilot_20261101/`；包含配置、源码快照、launch、JSONL指标和日志。
- Checkpoint：上述run目录下再进入同名run子目录的 `checkpoints/global_step_40/`。
- 曲线/CSV：`outputs/robotwin4/analysis/uniform_bucket_pilot/`，生成脚本 `outputs/robotwin4/analysis/plot_pilot.py`；图和metadata明确当前未完成状态。
- [历史进度记录](experiments/progress_history_20261101.md)、[实验方案](EXPERIMENTS_20261101.md)。

旧 `_retry` run 完成4轮/8updates后，在第5轮整包patch权重同步时OOM，已确认失败并清理自有进程。其日志、现场及旧W&B run `f6ve12mw` 保留；没有checkpoint，未与新run拼接。旧评测队列在FAILED gate停止。新的bucket配置提交为 `fb4d714a`。

## 下一步

启动并核验一组full-horizon温启动follow-up，完成后比较独立SR与相同flow probe；不扩展大规模扫描。不要仅凭loss下降宣布能力融合成功。只操作自有run；不执行全局Ray stop、不改变系统内存设置。

## 完整flow诊断与跟进

[完整 flow 诊断](FLOW_PROBE.md) 已在四任务各一个未筛选/seen场景上完成step40 student和base对照，共8份GPU report，每份4个观测。后25位置physical14误差明显较大；初始匹配seed/prompt/noise下，student endpoint误差小于base，但任务成功仍为0。诊断为描述性证据，不能直接推断失败原因。

新增默认关闭的`openpi.nft_full_horizon`开关，启用后仅扩大M1/M3 velocity监督到完整50×32，实际执行仍25×14。73项相关CPU测试通过（含RoboCasa配置回归），另外unfiltered probe入口9项测试通过，8份真实GPU诊断完成。旧默认不变。

准备单组温启动follow-up：step40权重、fresh optimizer，T32/R4/B256、micro8、LR2.5e-5/warmup4，新增40updates/10rounds，保存20/40。完整参数、累计预算和非因果消融限制见 [实验方案](EXPERIMENTS_20261101.md)。

只读核对：训练和评测按task名称选择归一化；NFT与ODE采样采用相同1.0→0.1时间网格。Bucket receiver调用load_state_dict并更新版本，已完成真实多轮训练与最终checkpoint评测。

## 已完成独立测评：未筛选 / seen

| Task | Base | Step40 student | Teacher |
|---|---:|---:|---:|
| adjust_bottle | 1/5 | 0/5 | 3/5 |
| handover_mic | 0/5 | 0/5 | 4/5 |
| place_container_plate | 0/5 | 0/5 | 5/5 |
| stack_bowls_two | 0/5 | 0/5 | 4/5 |

来源：`outputs/robotwin4/eval/uniform_bucket_r2_step40/summary.json`。Student为0/20，base为1/20，teacher为16/20。样本很小，且adjust_bottle参考seed部分错位；不能据此宣称student显著差于base，但本轮没有展示能力融合收益。
