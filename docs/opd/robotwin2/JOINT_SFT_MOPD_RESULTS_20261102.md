# 联合SFT初始化后的RoboTwin MOPD结果

更新：2026-11-02。**pilot冻结评测完成；独立20场景确认正在运行。**

## 当前结论

联合SFT500之后，MOPD40/80在相同pilot场景上由11/32提高到18/32、28/32，保留adjust并在原本为0的handover/stack出现成功。最终MOPD80平均成功率87.5%，对起点34.4%提高53.1个百分点。这是明确的冻结策略正向信号，但仍需预声明的独立场景确认，不能当作正式benchmark结论。

| 冻结策略 | adjust | handover | place | stack | 总计 |
|---|---:|---:|---:|---:|---:|
| 联合SFT500起点 | 8/8 | 0/8 | 3/8 | 0/8 | 11/32（34.4%） |
| + MOPD40 | 8/8 | 2/8 | 4/8 | 4/8 | 18/32（56.3%） |
| + MOPD80 | 8/8 | 8/8 | 6/8 | 6/8 | 28/32（87.5%） |

上述三组已完成结果的实际scene seed序列相同；adjust因native reset retry实际只有6个unique scenes，其余各8个。Action noise未固定。来源：根目录`outputs/robotwin4/analysis/joint_sft_step500_trainpath_t16_20261101/`、`joint_sft500_mopd_step40_trainpath_t16_20261101/`与`joint_sft500_mopd_step80_trainpath_t16_20261101/`；MOPD40/80的`recovery_verified_summary.json`含完整计数/匹配核验。

## 初始化与优化预算

- 官方四任务demo_clean各50episodes，末5episode/task留出，180训练/20验证；联合SFT500updates、B128/micro4、LR2.5e-5、warmup25，标准conditional FM。固定验证CFM loss为0.16748→0.00441，不能与OPD velocity MSE比较绝对值。
- 固定选SFT500初始化，不从100/200/500中事后择优。随后fresh optimizer的MOPD：8GPU、T32/R4/B128/micro8、LR1e-5恒定、warmup4、完整50×32 flow监督，执行25×14动作。
- MOPD80已完成20rounds/640trajectories/213,912实际control targets。前/末5轮平均velocity MSE为0.08170→0.01928；最后一步0.02232。连续更新、有限梯度和checkpoint结构核验通过；MOPD40/80均已实际推理重载、完成评测并核验记录。
- 相比早期base/specialist初始化，配置和预算也有差异；这不是严格的初始化单因素消融。当前结果支持“联合SFT后继续MOPD”的可行性，不据此量化SFT初始化的独立因果贡献，也不证明refresh优于cached。

## 独立确认（预声明、待完成）

固定比较SFT500与最终MOPD80，不根据pilot40/80挑checkpoint。每任务20个不同场景：T16×5rounds、seed300000/stride20；与已完成teacher参考使用同一协议。Teacher为8/20、19/20、15/20、17/20（59/80）。须等两组student完成并核验实际场景匹配后才能比较。仍为seen instruction、unstable-only reset的探索性协议，不是官方feasible/unseen测试。

## 链路与后备方案

此前同teacher自蒸馏控制保留place 6/8→6/8，初始MSE约5e-5；task归一化、动作/观测、FP32优化器及完整horizon均已有专项检查。早期base-init的较大MSE是归一化latent velocity误差，不是[0,1]有界指标；低MSE本身也不是行为正确性的证明。

Specialist-init延长20→60仅使原place从1/8恢复到4/8，其他任务仍0，不能称作融合成功。完整负面结果保留在[延长优化报告](LONGER_OPD_RESULTS_20261101.md)。当前优先走充分优化与联合初始化，没有加入teacher uncertainty masking；论文借鉴和适用条件见[可靠性后备](TEACHER_RELIABILITY_FALLBACK.md)。

## 运行与复现

所有实验位于`outputs/worktrees/robotwin4`的`dev/mopd-robotwin2-four-task`，根RoboCasa主分支源码保持不变。MOPD训练源码快照为`df817aeb`（run内source.tar.gz/空code.diff）；训练W&B为`RoboTwin_MultiTeacher_Pilots/gi47axmt`。

当前评测supervisor：`outputs/robotwin4/setup/joint_mopd_evals_shared_20261102.status.json`。按实测单student显存允许与同事共存，启动前每卡至少30,000MiB空闲；只管理自有进程。本轮共享资源耗时不能用于独占吞吐对比。最终独立场景审核入口：`setup/audit_joint_mopd_confirmation_20261102.py`。
