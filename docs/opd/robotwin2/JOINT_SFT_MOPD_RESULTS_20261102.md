# RoboTwin：联合SFT初始化后的四teacher MOPD结果

更新：2026-11-02，action00。**本轮自动调试及预声明独立确认已完成：SFT500 → MOPD80的平均成功率由35.0%提高到85.0%。** 这是可行性验证结果，尚非论文正式benchmark成绩。

## 独立确认：每任务20个不同场景

| 策略 | adjust_bottle | handover_mic | place_container_plate | stack_bowls_two | 平均SR |
|---|---:|---:|---:|---:|---:|
| 对应task teacher | 8/20（40%） | 19/20（95%） | 15/20（75%） | 17/20（85%） | 73.75% |
| 联合SFT500起点 | 17/20（85%） | 0/20（0%） | 8/20（40%） | 3/20（15%） | 35.0% |
| **+ MOPD80** | **17/20（85%）** | **19/20（95%）** | **16/20（80%）** | **16/20（80%）** | **85.0%** |

- 固定选SFT500与最终MOPD80，在看到pilot结果之前预声明；未从40/80择优。
- 三组各80条实际episode、每task20个不同actual reset seeds，场景映射全部一致。T16×5rounds，seed300000、stride20；native不稳定reset仅在各自20-seed窗口内重试。
- 对SFT起点提高**50个百分点**：handover +95、place +40、stack +65个百分点；adjust的汇总SR保持85%。不过adjust有3个原成功场景变失败、另3个原失败场景变成功，不能称为逐episode无遗忘。Action noise没有固定，匹配场景的结果变化仅作描述。
- 本组MOPD平均SR高于teacher参考11.25个百分点，主要因为SFT起点本就拥有更强的adjust表现。Handover与teacher相同，place多1次成功，stack少1次；不据此宣称普遍超越teacher。

严格审计：根目录`outputs/robotwin4/analysis/joint_sft500_mopd80_confirmation_audit_20261102.json`。包含原始结果哈希、checkpoint、整数seed/occurrence/control计数、每task记录、协议及所有scene-map比较。审计器24项CPU自测通过，完整三组实际结果也通过。

## 同协议pilot：每任务8episodes

| 冻结策略 | adjust | handover | place | stack | 总计 |
|---|---:|---:|---:|---:|---:|
| SFT500 | 8/8 | 0/8 | 3/8 | 0/8 | 11/32（34.4%） |
| + MOPD40 | 8/8 | 2/8 | 4/8 | 4/8 | 18/32（56.3%） |
| + MOPD80 | 8/8 | 8/8 | 6/8 | 6/8 | 28/32（87.5%） |

三组实际scene seed序列相同；adjust因legacy reset重试只有6个unique seeds，其余各8个。此pilot使用seed200000，与上面的独立确认分开统计。结果目录：`analysis/joint_sft_step500_trainpath_t16_20261101/`和`analysis/joint_sft500_mopd_step{40,80}_trainpath_t16_20261101/`。

## 训练设置与预算

- 模型π0.5，四个PACE specialists按task硬路由。官方四任务demo_clean各50episodes，末5episode/task留出：180训练、20验证episodes。联合SFT为真实示范上的标准conditional FM，不是teacher伪标签；固定teacher任务归一化，50×32 latent动作监督，执行25×14动作。
- SFT500：4GPU，B128/micro4，LR2.5e-5恒定，warmup25。固定CFM验证loss **0.16748→0.00441**；这个值不能与OPD velocity MSE比较绝对大小。SFT500只是本次暖启动，不代表充分收敛的联合SFT上限。
- 从固定SFT500模型权重新建optimizer，joint MOPD：8GPU、**T32 / refresh-R4 / B128 / micro8 / LR1e-5**，恒定LR、warmup4，M3时间加权velocity MSE、完整50×32监督。没有对这次实验加入新的uncertainty masking。
- 共**80 optimizer updates、20 rollout rounds、640 trajectories、213,912实际control targets**；计数与runner/allocation状态一致，loss/梯度有限。前/末5轮（各20updates）平均plain velocity MSE **0.08170→0.01928**，最后一步0.02232。在线观测和flow状态分布会变化，不能把这看作固定输入误差的严格对照。
- 20/40/60/80 checkpoint结构核验通过；40/80已实际重载并完成冻结评测。训练源快照`df817aeb`，run内保留`source.tar.gz`、空`code.diff`和resolved config；后续改动为离线统计/文档。

## 对调试问题的回答

1. **链路已有闭环证据。** Teacher真实训练推理路径能完成任务；同teacher自蒸馏控制保留place 6/8→6/8；任务norm、动作/图像数据、FP32优化器、保存重载及联合SFT前向均已检查。SFT前向与官方实现为近似数值一致（最大相对loss差约0.313%），不声称bit-exact。
2. **更多训练并非对任意起点都立即有效。** 早期base-init两轮pilot没有得到四任务成功收益；specialist-init延长20→60仅使原place从1/8恢复到4/8，其他任务仍0。负面结果保留：[早期pilot](PILOT_RESULTS_20261101.md)、[延长优化](LONGER_OPD_RESULTS_20261101.md)。
3. **联合初始化后继续拟合的路线可行。** 从较好的joint SFT起点继续MOPD，低MSE已转化为冻结及独立场景SR收益。但初始化、配置、训练预算与早期pilot均有差异，不能据此单独证明初始化的因果贡献，也没有隔离refresh与cached的差别。
4. **早期较大的loss不是有界成功率指标。** Plain MSE比较归一化latent的teacher/student velocity，数值可大于1；base与task teacher的输出不匹配会造成较大初值。同源控制初始约5e-5是额外数值检查依据。没有跨模型通用的“足够低MSE”阈值。
5. **目前无需增加teacher可靠性截断。** 优先充分拟合与联合初始化已获得收益。参考论文的同noise 10/5步self-consistency、prefix loss mask及适用限制已分析于[Teacher可靠性后备](TEACHER_RELIABILITY_FALLBACK.md)，未将其混入当前实验。

## 结论边界与后续正式实验

当前为一个训练seed、每task20episodes、seen instructions、unstable-only reset过滤；不是官方feasible-scene/unseen-instruction协议。没有继续SFT580等额外训练对照、multi-seed复现或refresh/cached机制消融；因此本轮结论是四teacher MOPD链路与任务收益的可行性，不是这些机制各自的优越性。后续进入正式实验时，应先固定评测协议、补充训练seed与充分SFT参考，而不是继续添加复杂proxy。

## 复现与交接

- Worktree：`outputs/worktrees/robotwin4`，分支`dev/mopd-robotwin2-four-task`。根`opd-project-main@5d1d1494`源码保持原状。
- 独立runtime：`/localscratch2/liyifa11/XyFile/opd-robotwin4`；模型、资产版本见[模型来源](CHECKPOINT_SURVEY_20261101.md)及worktree的`toolkits/robotwin_mopd/setup/README.md`。当前仍是本地节点配置，不是新机器安装复验结论。
- SFT权重：`outputs/robotwin4/train/robotwin4_joint_sft_500_b128_micro4_pilot_20261101/checkpoints/global_step_500`。
- 最终MOPD权重：`outputs/robotwin4/train/robotwin4_joint_sft500_r4_t32_b128_pilot_20261101/robotwin4_joint_sft500_r4_t32_b128_pilot_20261101/checkpoints/global_step_80`。
- W&B：[MOPD run gi47axmt](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/gi47axmt)。训练日志确认同步完成。
- 预声明确认计划：`outputs/robotwin4/setup/joint_sft500_mopd80_confirmation_plan_20261101.json`；实际评测队列：`setup/joint_mopd_evals_shared_20261102.status.json`，四项全部COMPLETE且进程已退出，自有Ray清理已核验。
- 原GPU等待队列与替换原因保留。冻结评测按每卡至少30,000MiB空闲允许共存，记录了整卡遥测；没有向同事进程发信号。本轮共享资源耗时不用于独占吞吐比较。
