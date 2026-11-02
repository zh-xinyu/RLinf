# RoboTwin 四 teacher：自动调试状态

更新：2026-11-01，action00。当前仍在调试，RoboCasa 主分支代码与环境保持隔离。

| 项目 | 状态 | 证据入口（根目录 outputs/robotwin4/） |
|---|---|---|
| teacher 真实训练推理路径 | 完成，21/32 成功 | analysis/teacher_train_t16_verified_20261101/ |
| specialist 初始化 joint MOPD 20 updates | 完成；MSE下降，冻结测评6/32→1/32 | analysis/specialist_init_final_summary_20261101.json |
| 同 teacher 自蒸馏控制 | 完成：place固定同场景6/8→6/8，末MSE3.56e-4；见SELF_DISTILL_RESULTS_20261101.md | setup/self_control_queue_20261101.json |
| 固定输入 flow 对照 | 四task完成，逐task分析中；每task一个初始观测，不能代表整条episode | analysis/fixed_reference_joint20_20261101/ |
| 更多 MOPD 优化 | self保留通过探索筛查，joint20→60全状态续训已启动，之后测评40/60 | setup/longer_opd_queue_20261101.json |
| SFT后续队列 | 已启动等待现有OPD结束；依次测评SFT100/200/500，固定500通过可行性筛查后8卡MOPD80，再测评40/80 | setup/joint_sft_followup_queue_20261101.json |
| 四任务联合 SFT | 官方200条示范下载/对齐完成；四task前向与官方FM相对loss差<0.32%；GPU0–3 micro2/micro4两步反向、保存通过，checkpoint已用推理工具重载；500updates联合SFT运行中 | setup/joint_sft/long_pilot_status_20261101.json；train/robotwin4_joint_sft_500_b128_micro4_pilot_20261101/ |

详细设计见 [自动调试计划](AUTONOMOUS_DEBUG_PLAN_20261101.md)。不把当前运行中状态、CPU测试或低loss当成效果结论；所有后续候选都需策略rollout验证。

## 早期 pilot 存档

以下为先前两轮已结束实验，不表示当前没有运行任务。

## 完成的实验

| Run | 设置 | 结果 / 预算 |
|---|---|---|
| `robotwin4_uniform_refresh_r2_t64_b512_bucket_pilot_20261101` | Base起点；T64/R2/B512；前25×32监督 | COMPLETE；40updates/20rounds；1,280 trajectories/734,389 controls；244.67min；独立0/20 |
| `robotwin4_fullflow_refresh_r4_t32_b256_warmstart_pilot_20261101` | 前run step40权重、fresh optimizer；T32/R4/B256；完整50×32监督 | COMPLETE；40updates/10rounds；320 trajectories/184,000 controls；100.34min；独立0/20 |

两段累计80updates、1,600 trajectories、918,389 executed control targets；这是模型谱系预算，不包含失败启动、smoke、校准或评测。不同T/B/R/初始化/监督范围，不能当单因素消融或直接拼接MSE。

- [首轮W&B](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/77se8lie)。
- [跟进W&B](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/02bn5rjr)，group=`robotwin4_followup_pilots`。
- 首轮checkpoint：steps8/16/24/32/40；跟进：20/40。完整weights、8个DCP分片、metadata、runner/allocation计数已核验；未做实际全状态resume测试。
- 训练目录均在根目录`outputs/robotwin4/train/`；每个run保存resolved config、source snapshot、commit/diff、launch、JSONL和日志。
- 身份/完整命令：`outputs/robotwin4/setup/uniform_bucket_pilot_launch.json`、`fullflow_warmstart_pilot_launch.json`。原launcher2976313/3332924已退出。

## 测评与诊断

- `outputs/robotwin4/eval/uniform_bucket_r2_step40/queue.json`：COMPLETE，原队列2982573退出。
- `outputs/robotwin4/eval/fullflow_r4_step40/queue.json`：COMPLETE，原队列3341569退出。
- 两队列student每任务5episodes均0成功，base补充后1/20，teacher参考16/20。各summary保留实际seed重合情况，adjust_bottle teacher参考seed部分错位。
- `flow_probe_step40_unfiltered/`：base/初始student共8份report；`flow_probe_fullflow_step40_unfiltered/`：最终student4份及joint/cached数值检查1份。均在`outputs/robotwin4/analysis/`。
- 初始flow后25位置误差较大；跟进后改善不一致。真实checkpoint joint/cached差异远小于student/teacher误差。见 [FLOW_PROBE](FLOW_PROBE.md)。
- `feasible_unseen_bucket_step40/`为**STOPPED_PROTOCOL_UNVALIDATED**：MPLib筛选反复碰撞，存在双臂适配风险；仅停止该自有队列/builder，保留partial pool、日志、`operator_stop.json`。没有使用该不完整pool评测。

## 已验证与边界

- 四teacher源revision/hash、五模型转换/严格加载、任务norm stats及真实原生评测已核验；跨JAX/PyTorch数值对照仅adjust_bottle。
- 原实现相关测试、73项full-horizon相关CPU测试（含RoboCasa配置回归）、9项probe入口测试通过；真实多轮训练/更新/保存/加载评测完成。测试组有交集，不将数量相加作为总覆盖数。
- `openpi.nft_full_horizon`默认false，开启只改变监督范围；生成仍50×32、执行25×14。新监督代码commit `c468dc66`，首轮默认行为不变。
- Bucket修正后的两段训练无GPU OOM；保存曾触发CPU短暂reclaim/swap。没有改系统VM参数或停止他人的进程。
- 旧`_retry` run在4rounds/8updates后因整包patch同步OOM而FAILED，现场保留，未接入有效模型谱系。
- 当前是unfiltered/seen、5episodes/task的pilot，**不具备正式RoboTwin benchmark协议或统计结论**。

## 代码与下一步

根目录仍`opd-project-main@5d1d1494`，tracked源码无改动；worktree为`outputs/worktrees/robotwin4`，分支`dev/mopd-robotwin2-four-task`。RoboTwin runtime、models和assets独立在`/localscratch2/liyifa11/XyFile/opd-robotwin4`，RoboCasa环境未改。

本轮停止参数扫描。若继续，优先重新评估联合SFT冷启动/更充分训练及正式评测协议；当前不能把失败归结为单一loss或full-horizon问题。
