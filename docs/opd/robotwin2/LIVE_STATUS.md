# RoboTwin 四 teacher：自动调试状态

更新：2026-11-01，action00。当前仍在调试，RoboCasa 主分支代码与环境保持隔离。

| 项目 | 状态 | 证据入口（根目录 outputs/robotwin4/） |
|---|---|---|
| teacher 真实训练推理路径 | 完成，21/32 成功 | analysis/teacher_train_t16_verified_20261101/ |
| teacher独立场景参考 | 完成：8/20、19/20、15/20、17/20；每task20个unique scenes；与旧pilot为不同seed集合 | analysis/teacher_confirm_stride20_t16_n20_20261101/verified_summary.json |
| specialist 初始化 joint MOPD 20 updates | 完成；MSE下降，冻结测评6/32→1/32 | analysis/specialist_init_final_summary_20261101.json |
| 同 teacher 自蒸馏控制 | 完成：place固定同场景6/8→6/8，末MSE3.56e-4；见SELF_DISTILL_RESULTS_20261101.md | setup/self_control_queue_20261101.json |
| 固定输入 flow 对照 | 四task的0/20/60相同缓存对照完成；place误差恢复，adjust不随训练loss同步下降；见FIXED_REFERENCE_RESULTS_20261101.md | analysis/fixed_reference_joint20_20261101/ |
| 更多 MOPD 优化 | 训练及40/60冻结评测已COMPLETE；place 6/8→1/8→1/8→4/8，其余task仍0/8；见LONGER_OPD_RESULTS_20261101.md | setup/longer_opd_eval_recovery_20261101.status.json；analysis/longer_opd_training_summary_20261101.json |
| SFT后续队列 | SFT100/200/500测评已完成（5/32、9/32、11/32）；固定500通过筛查，8卡MOPD80运行中，40个更新及20/40完整checkpoint已核验，再测评40/80 | setup/joint_sft_followup_queue_recovery_20261101.json |
| 四任务联合 SFT | 官方200条示范下载/对齐完成；四task前向与官方FM相对loss差<0.32%；GPU0–3 micro2/micro4两步反向、保存通过，checkpoint已用推理工具重载；500updates联合SFT已COMPLETE，固定CFM验证loss 0.167479→0.00440669 | setup/joint_sft/long_pilot_status_20261101.json；train/robotwin4_joint_sft_500_b128_micro4_pilot_20261101/ |

详细设计见 [自动调试计划](AUTONOMOUS_DEBUG_PLAN_20261101.md)。不把当前运行中状态、CPU测试或低loss当成效果结论；所有后续候选都需策略rollout验证。

## 23:35 最新结果

SFT500→MOPD40及保存已完成，训练仍在继续。末4步平均MSE为0.0277；最近3个在线rounds的成功数依次为adjust24/24、handover19/24、place14/24、stack18/24。出现了四任务正向信号，但冻结测评仍未进行，不能与独立基线直接比较。完整checkpoint与320条轨迹、119,276 controls恢复计数一致。详见[联合SFT记录](JOINT_SFT_WARMSTART.md)。

## 22:58 里程碑

SFT500→MOPD20及首次保存已完成，训练仍RUNNING。前4步/末4步平均MSE为0.1324/0.0512；完整权重、8个DCP分片、runner/allocation计数与160条轨迹、67,784 controls一致。暂未进行蒸馏后冻结评测。记录：`analysis/joint_sft500_mopd_step20_20261101.json`。

## 22:29 里程碑

SFT500→MOPD首轮4个更新已完成，MSE为0.147/0.131/0.112/0.140，梯度有限；32条轨迹和13,766个control targets计数一致。训练继续中，尚无蒸馏后冻结SR。逐更新JSONL在整轮R4结束后统一写出。详见[联合SFT记录](JOINT_SFT_WARMSTART.md)和`analysis/joint_sft500_mopd_startup_20261101.json`。

## 22:18 里程碑

SFT500闭环结果为adjust8/8、handover0/8、place3/8、stack0/8（11/32），记录与同场景序列已核验；与SFT200相比并非每task都改善。通过既定筛查后MOPD80已实际启动，仍待首个更新与后续效果验证。配置与证据见[联合SFT记录](JOINT_SFT_WARMSTART.md)。

## 22:10 里程碑

联合SFT500训练与完整checkpoint保存已完成，500条更新、11条固定验证和COMPLETE标记核验通过。最终固定CFM验证loss为0.00440669，不能与OPD MSE混比或据此推断SR。后续队列已实际进入SFT500策略评测；通过既定筛查后启动MOPD80。详见[联合SFT记录](JOINT_SFT_WARMSTART.md)。

## 21:45 里程碑

[Teacher独立场景参考](TEACHER_CONFIRMATION_20261101.md)已完成59/80，80条计数及每task20个unique scenes核验通过。GPU4–7已释放，SFT500训练与等待队列保持原计划。Teacher能力不均衡，adjust40%，其余75–95%；需同协议student测评才能比较。

## 21:23 里程碑

SFT100/200冻结测评分别5/32、9/32，详见[联合SFT记录](JOINT_SFT_WARMSTART.md)。主训练仍在运行，下游等待固定SFT500。已利用空档完成四task step60固定输入诊断，全部exit0、缓存哈希一致、reference复算完全一致；GPU4–7已释放。

## 21:05 里程碑

延长MOPD的完整结果见[报告](LONGER_OPD_RESULTS_20261101.md)。原place能力部分恢复，尚无新任务能力融合。恢复队列已COMPLETE且退出，SFT100评测已自动启动；joint SFT500仍在GPU0–3训练。

## 20:49 评测队列恢复

- 训练run本身为`COMPLETE`：60updates/30rounds/480trajectories/264,893control targets，逐episode计数守恒检查通过。没有重跑训练或改变参数。
- 原队列在训练结束后立即检查GPU空闲时失败，SFT后续队列随之停止。之后现场检查GPU4–7为1MiB且无compute进程；这与资源释放延迟相符，但失败瞬间没有逐卡遥测，不能把原因视为已证实。
- 原`longer_opd_queue_20261101.json`（FAILED）与`joint_sft_followup_queue_20261101.json`（REVIEW_REQUIRED）及日志保留。新的eval-only supervisor为`setup/run_longer_eval_recovery_20261101.py`，身份/状态分别是`longer_opd_eval_recovery_20261101.launch.json`/`.status.json`；先评测40，再60。20:49第40步评测进程已实际启动。
- 下游使用新tag=`recovery_20261101`，显式绑定上述上游；身份与失败队列分开。启用120秒严格GPU空闲等待，检查标准不变、到期停止，不终止其他作业。恢复CLI与有限等待有21项CPU测试，代码commit`573fc007`；说明在worktree的`toolkits/robotwin_mopd/JOINT_SFT_QUEUE.md`。
- SFT仍在独立GPU0–3训练。固定验证loss在steps200/250/300为0.00639/0.00716/0.00605，存在逐任务波动；尚无联合SFT策略成功率结论。

## 运行中里程碑：2026-11-01 19:48 Detroit

- 延长 MOPD 已到40/60 updates。step40的full weights、四个DCP分片与metadata、runner/allocation状态均已写出；runner记录40updates、20rounds、176,701 control targets。冻结策略评测仍待整段训练结束。
- 中途窗口（updates34–38）的平均velocity MSE为0.1144，对比原updates16–20的0.2268继续降低；四task均在拟合。续训前10轮实际episode中，adjust/handover/place/stack成功数为1/0/7/0，每task40episodes。这是不同场景的在线统计，不是冻结评测，也不说明因果。
- 联合SFT固定离线验证的CFM loss：step0/50/100/150/200分别0.16748/0.01600/0.00912/0.00778/0.00639。step200四task的physical14 loss均比step150降低。尚无joint-SFT策略成功率结果，不与MOPD的teacher velocity MSE横向比较。
- 原始证据：`analysis/longer_opd_interim_20261101.json`、`analysis/joint_sft_step200_validation_20261101.json`。此处为静态快照，最新进度以运行日志和supervisor状态为准。

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
- 本节早期pilot为unfiltered/seen、5episodes/task；当前训练链路小测评为8episodes/task（adjust仅6个unique scenes）。两者均**不具备正式RoboTwin benchmark协议或统计结论**。

## 代码与下一步

根目录仍`opd-project-main@5d1d1494`，tracked源码无改动；worktree为`outputs/worktrees/robotwin4`，分支`dev/mopd-robotwin2-four-task`。RoboTwin runtime、models和assets独立在`/localscratch2/liyifa11/XyFile/opd-robotwin4`，RoboCasa环境未改。

本轮停止参数扫描。若继续，优先重新评估联合SFT冷启动/更充分训练及正式评测协议；当前不能把失败归结为单一loss或full-horizon问题。
