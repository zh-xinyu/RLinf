# RoboTwin 2.0 四任务 MOPD

更新：2026-11-02，action00。**本轮自动调试已完成：联合SFT500→MOPD80在独立20场景/task上由35.0%提高到85.0%，三组实际场景与teacher参考匹配。** 完整训练、冻结pilot和独立确认均已核验，所有自有作业退出。详见[最终结果](JOINT_SFT_MOPD_RESULTS_20261102.md)、[计划与完成记录](AUTONOMOUS_DEBUG_PLAN_20261101.md)和[状态历史](LIVE_STATUS.md)。当前仍是单训练seed、seen/unstable-only reset的可行性结果，不是论文正式benchmark成绩。

前几轮负面结果完整保留：[初期 pilot](PILOT_RESULTS_20261101.md)、[specialist 初始化结果](SPECIALIST_INIT_RESULTS_20261101.md)。训练 loss 下降不等于任务能力融合成功。

## 已固定的 setting

模型为π0.5；早期student从base或place specialist初始化，最新student从真实四任务联合SFT500初始化。四个PACE specialists分别对应adjust_bottle、handover_mic、place_container_plate、stack_bowls_two。Native clean ALOHA，三相机，14维物理动作，50步latent预测、执行25步、10步ODE。按task路由teacher与固定归一化。

首轮T64/R2/B512训练40updates；一次温启动跟进使用完整50步监督、T32/R4/B256，再训练40updates。两组是可行性调参，不是单因素消融。当前评测为未筛选/seen小样本pilot，不能当作官方feasible/unseen benchmark成绩。

## 隔离与复现

- 根workspace：`opd-project-main@5d1d1494`，RoboCasa源码与runtime保持原状。
- 开发分支：`dev/mopd-robotwin2-four-task`，worktree：`outputs/worktrees/robotwin4`。
- 独立runtime/模型/assets：`/localscratch2/liyifa11/XyFile/opd-robotwin4`。
- 输出：根目录`outputs/robotwin4/`，训练、评测、分析和setup分别保存。
- 在worktree内参阅`toolkits/robotwin_mopd/README.md`和`toolkits/robotwin_mopd/setup/README.md`；精确命令、resolved config、source快照保存在每个run及setup launch JSON。

## 文档导航

| 文档 | 用途 |
|---|---|
| [联合SFT→MOPD结果](JOINT_SFT_MOPD_RESULTS_20261102.md) | 已完成的冻结/独立确认、预算、限制及checkpoint交接 |
| [结果与交接](PILOT_RESULTS_20261101.md) | 实测结果、预算、结论和限制 |
| [Teacher独立场景参考](TEACHER_CONFIRMATION_20261101.md) | 每task20个不同场景、真实训练路径59/80及协议边界 |
| [延长MOPD结果](LONGER_OPD_RESULTS_20261101.md) | 完整20→60续训、冻结测评与恢复信号 |
| [自蒸馏控制](SELF_DISTILL_RESULTS_20261101.md) | 同源teacher/student保留能力的冻结测评与限制 |
| [当前状态](LIVE_STATUS.md) | 运行身份、队列和checkpoint状态 |
| [实验方案](EXPERIMENTS_20261101.md) | 参数选择与单次follow-up设计 |
| [模型来源](CHECKPOINT_SURVEY_20261101.md) | 开源checkpoint调查与筛选依据 |
| [Actor精度](ACTOR_PRECISION.md) | FP32 masters/BF16 AMP验证 |
| [评测协议](EVALUATION_PROTOCOL_AUDIT.md) | 本地pilot与官方协议的区别 |
| [完整flow诊断](FLOW_PROBE.md) | 前后horizon、physical/padded维度及数值检查 |
| [补充筛选设计](FEASIBLE_EVALUATION.md) | 未验证路线；MPLib筛选已停止，不能直接用于正式成绩 |
| [初始设计存档](experiments/initial_design_20261020.md) | 保留早期计划，不代表当前状态 |

RoboTwin依赖继续与RoboCasa profile隔离。后续调试以训练路径验证和有限初始化试验为范围。后续是否推进正式实验应以任务成功率为依据。
