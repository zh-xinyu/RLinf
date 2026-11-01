# RoboTwin 2.0 四任务 MOPD

更新：2026-11-01，action00。**自动调试进行中，尚未展示四任务 MOPD 的成功率收益。** 已完成 teacher 训练路径验证（21/32）和 specialist 起点短程 MOPD（冻结测评6/32→1/32）。当前进行同 teacher 自蒸馏控制、固定输入误差诊断及真实四任务联合 SFT 初始化，见 [自动调试计划](AUTONOMOUS_DEBUG_PLAN_20261101.md)、[联合 SFT](JOINT_SFT_WARMSTART.md) 和 [当前状态](LIVE_STATUS.md)。

前几轮负面结果完整保留：[初期 pilot](PILOT_RESULTS_20261101.md)、[specialist 初始化结果](SPECIALIST_INIT_RESULTS_20261101.md)。训练 loss 下降不等于任务能力融合成功。

## 已固定的 setting

π0.5 base student；四个PACE specialists分别对应adjust_bottle、handover_mic、place_container_plate、stack_bowls_two。Native clean ALOHA，三相机，14维物理动作，50步latent预测、执行25步、10步ODE。按task路由teacher与固定归一化。

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
| [结果与交接](PILOT_RESULTS_20261101.md) | 实测结果、预算、结论和限制 |
| [当前状态](LIVE_STATUS.md) | 运行身份、队列和checkpoint状态 |
| [实验方案](EXPERIMENTS_20261101.md) | 参数选择与单次follow-up设计 |
| [模型来源](CHECKPOINT_SURVEY_20261101.md) | 开源checkpoint调查与筛选依据 |
| [Actor精度](ACTOR_PRECISION.md) | FP32 masters/BF16 AMP验证 |
| [评测协议](EVALUATION_PROTOCOL_AUDIT.md) | 本地pilot与官方协议的区别 |
| [完整flow诊断](FLOW_PROBE.md) | 前后horizon、physical/padded维度及数值检查 |
| [补充筛选设计](FEASIBLE_EVALUATION.md) | 未验证路线；MPLib筛选已停止，不能直接用于正式成绩 |
| [初始设计存档](experiments/initial_design_20261020.md) | 保留早期计划，不代表当前状态 |

RoboTwin依赖继续与RoboCasa profile隔离。后续调试以训练路径验证和有限初始化试验为范围。后续是否推进正式实验应以任务成功率为依据。
