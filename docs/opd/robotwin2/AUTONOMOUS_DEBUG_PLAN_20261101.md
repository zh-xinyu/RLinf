# RoboTwin 长程自动调试计划与完成记录（2026-11-01—02）

目标：检验充分优化与真正joint SFT初始化能否使四task MOPD产生任务收益，同时核对训练链路、保留负面结果，并将teacher可靠性作为有条件后备。**本轮预声明工作全部完成；结果入口：[联合SFT→MOPD](JOINT_SFT_MOPD_RESULTS_20261102.md)。**

| 阶段 | 实际完成内容 | 结果与证据 |
|---|---|---|
| teacher训练路径 | mixed-task硬路由，真实Env.interact/HF.generate；另完成独立20场景/task参考 | 初始21/32，独立59/80；TEACHER_CONFIRMATION_20261101.md |
| 正确性控制 | 同place teacher/student，M3 full50，T8/R4/B128/micro2/LR1e-5，20updates及冻结评测 | place 6/8→6/8；SELF_DISTILL_RESULTS_20261101.md |
| 固定输入诊断 | 四task同一缓存reference flow、全部10时间点，对照初始/20/60；拆分physical/padded与前后horizon | place误差先增后降；FIXED_REFERENCE_RESULTS_20261101.md |
| 更多specialist-init OPD | 从joint20完整恢复到60；T16/R2/B128/micro4/LR1e-5，额外40updates；冻结40/60 | place初始6/8→20步1/8→40步1/8→60步4/8，其余仍0；LONGER_OPD_RESULTS_20261101.md |
| 真正joint SFT | 官方四task各50demos，180/20训练/验证episodes；标准CFM；4GPU/B128/micro4/LR2.5e-5/warmup25，500updates | CFM验证loss 0.16748→0.00441；SFT100/200/500冻结pilot5/32、9/32、11/32；JOINT_SFT_WARMSTART.md |
| joint SFT→MOPD | 固定SFT500，fresh optimizer；8GPU/T32/R4/B128/micro8/LR1e-5/warmup4，80updates | 20rounds/640trajectories/213,912controls；冻结40/80分别18/32、28/32 |
| 独立确认 | 预声明SFT500和最终MOPD80，各20不同场景/task；T16×5、seed300000/stride20；与teacher同协议 | SFT50028/80→MOPD8068/80；三组实际场景全匹配、严格记录核验通过 |
| 不确定性后备 | 本地论文10/5步同noise一致性、prefix mask与梯度含义分析 | 已记录适用条件，当前未触发实现需要；TEACHER_RELIABILITY_FALLBACK.md |

## 实验约定与解释

- SFT500和最终MOPD80的独立确认计划在pilot结果之前固定；没有选最好checkpoint或删除失败场景。
- 保留原始负面pilot、失败handoff、恢复身份、准确命令、resolved config及源码快照。训练不因评测等待而重跑；所有控制与恢复仍只管理自有进程。
- 初始化/配置/预算与早期pilot存在差异；不将这轮作为初始化单因素消融、refresh-vs-cached消融或充分收敛SFT上限比较。
- 独立确认每task20episodes、单训练seed、seen instruction、unstable-only reset；action noise未固定，匹配scene的结果差异仅作描述。不是官方feasible/unseen benchmark成绩。
- 早期用户写teacher OOM，研究讨论中按teacher OOD/不可靠状态理解，未与GPU显存OOM混淆。此次结果支持优先充分拟合/联合初始化，未新增uncertainty loss。

## 执行收尾与证据

`outputs/robotwin4/setup/joint_mopd_evals_shared_20261102.status.json`为COMPLETE，pilot40、pilot80、独立SFT500、独立MOPD80四项完成，supervisor/遥测及各自Ray均已退出。旧队列状态和superseded说明保留；共享冻结评测用每卡30,000MiB空闲门槛，耗时不用于独占吞吐比较。根RoboCasa主分支源码保持原状。

- 训练核验：`analysis/joint_sft500_mopd_training_complete_20261102.json`。
- 独立确认：`analysis/joint_sft500_mopd80_confirmation_audit_20261102.json`，含三组原始记录哈希与scene匹配。
- 原预声明：`setup/joint_sft500_mopd80_confirmation_plan_20261101.json`。
- 最终结果、checkpoint和W&B入口：[结果报告](JOINT_SFT_MOPD_RESULTS_20261102.md)。

本轮没有剩余必需实验。多训练seed、充分SFT参考、50episodes/task及正式协议属于后续论文实验，不作为本次调试已完成的证据。
