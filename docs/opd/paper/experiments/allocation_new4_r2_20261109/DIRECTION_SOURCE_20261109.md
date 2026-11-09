# RoboMOPD allocation：目标与候选方向

2026-11-09讨论建议；未修改正文、训练代码或启动实验。依据当前`sec/0_abstract.tex`、`1_intro.tex`、`3_problem_formulation.tex`、`4_method.tex`、`5_experiments.tex`。本笔记不将候选方案视为已接受或已验证的方法。

## 1. 当前论文约束

- 正文主线是interaction-efficient MOPD，固定目标任务分布上的成功率；当前benchmark为RoboTwin与LIBERO，不能继续用旧RoboCasa计划替代正文。
- 最后核对时，当前Problem Setup已加入task-level最大teacher gap作为能力覆盖指标，并明确不是训练loss的直接优化保证。此处建议与该新增表述一致，不需要重复添加定义。
- Allocation与reuse共同管理环境经验；同步quota/loss已经写成方法机制，并有既往RoboCasa探索动机。保留同步执行方式，把主要变量集中在调度反馈与控制律。
- 不将全文改成纯minimax fairness论文。提出的allocation子目标是尽早形成完整、均衡的专家能力覆盖；平均SR仍报告，最坏能力缺口补充反映是否有任务长期掉队。

## 2. 推荐目标

表述：interaction-efficient, balanced capability consolidation。

为各group提前设定teacher-referenced目标c_i，例如S_i^T-delta；delta需在新实验前固定，不能按结果挑阈值。Teacher校准与student反馈应同分布。若initializer已高于该目标，其已有能力应单独报告是否退化，不能把teacher当性能上界。

关键评价是达到所有group目标所需的预算，以及固定预算下剩余的最差缺口。用多次预定测评持续达标判断，比单次first crossing更适合观察能力保持；同时保留连续指标，防止阈值选择主导结论。

连续指标：G(N)=max_i[S_i^T-S_i(N)]_+，其终点与预算曲线面积；共同报告固定分布宏SR与逐group成绩。Group含多个任务时，suite宏SR可隐藏suite内短板，需要明确只在group层调度/主张，并辅助逐task结果；不在首轮扩展层级控制器。

这些是可检验的目标，不是gap-only控制律的保证。训练反馈只用训练数据；held-out正式测评不可回流控制器。

## 3. 文献对应与边界

- [D³-MOPD](https://arxiv.org/html/2608.24987v1)：用归一化KL残量与近期下降相乘，偏向可获得的当前收益。其简化理论不能保证机器人任务SR改善。
- [Nash-MOPD](https://arxiv.org/html/2610.26124v1)：用按曝光数归一化的进展估计剩余训练需求，局部正速率模型下p正比g/a；目标更贴近共同完成。真实非线性训练、零速率任务和跨任务迁移不受该局部保证覆盖。
- [Open-MOPD](https://arxiv.org/html/2608.19098v1)：按实际loss份额再分配优化预算，对保留我们quota/loss同步有启发，不意味着两者实现相同。

## 4. 优先候选A：归一化蒸馏残差驱动的同步分配

第一轮只换需求信号，不加入速率：现有SR-gap版本保留作基准。取每轮第一次更新的未乘suite权重的velocity MSE L_i^r，避免将R1/R2不同数量的inner更新直接混成一个controller时间序列。

- 使用相同固定warmup阶段建立各group参考L_i^ref，之后冻结；用该阶段多点均值而非单个初始尖峰。下界避免极小参考除法。
- 平滑得到Lbar_i；m_i=Lbar_i/max(L_i^ref,epsilon_L)。m可超过1，不人为声称它是成功率或已恢复能力比例。
- 用m替代当前d进入现有uniform-mixture归一化，保留原有最低配额、变化上限、决策节奏、R2、B、LR和alpha_i=K q_i/T。
- 调度用的是原始误差而不是乘alpha后的total loss，避免控制器因自己增加权重而认为差距变大。

动机：信号已有日志、密集、直接衡量student访问查询上的teacher匹配需求；归一化避免任务动作尺度和初始误差直接决定份额。归一化不能消除任务对误差的不同敏感性，也不能修复模型访问状态分布变化。

目标仍是能力完整性；m只是surrogate。如果任务SR已经较好但残差大，或者反之，纯MSE调度可能分配错误。因此它是低成本候选，不是理论上优于SR的替代。不要第一轮同时增加SR门控、进展乘子和不确定性权重，否则难以判断有效原因。

## 5. 次级候选B：受约束的剩余学习需求调度

若候选A信号可用，再试Nash式思路：在同一种归一化残差上定义g_i和单位投入下降a_i，以g_i/a_i估计剩余需求。分子分母必须使用一致的残差，不能拿SR-gap除以MSE下降后声称具有完成时间推导。

实际建议用有下限、上限和平滑的提案；零/负进展、历史不足时显式回退而非无限增大逆速率。零进展可能是学不动，也可能是噪声/暂时退化，算法不应声称能区分。对于本项目短训练，不以该方向作为第一轮唯一押注。

投入分母是关键：在suite-balanced B下，每任务原始query数基本相同，而权重不同。可以记录E_i=sum_u alpha_i^u B_i^u为加权样本曝光proxy；若B_i=B/K、alpha_i=Kq_i/T，则每轮Delta E_i=RBq_i/T。固定B/R/T时，它与trajectory数成比例；改变R时不再可直接混用。

E_i不是实际梯度范数、Fisher信息或Adam参数更新量，归一化也不能消除共享模型迁移、不同状态和不同trajectory长度的影响。必须同时保留真实episode、env steps、query数和权重日志，不能把加权曝光当作真实环境交互。

## 6. 建议探索顺序

1. 先做离线固定历史检查：在已有日志上计算候选信号/配额，查看warmup敏感性、极端配额、切换频率及与SR的错配。它只检验控制器数值行为，不预测改变配额后的反事实性能。
2. 最小在线probe使用R2，同一个完整checkpoint/optimizer状态分叉三组：Uniform-sync、现有SR-gap-sync、normalized-MSE-gap-sync。统一处理proposal RNG、控制器warmup和历史来源；不能拿完整历史uniform直接当成恢复分支的精确配对结果。
3. 可用round10到30的短区间筛掉坏方向；如果parent checkpoint缺失必要状态，改为共同初始化。只作为开发probe，不把它用于完整50-round正式曲线。
4. 不改T/B/LR/R；100次正式测评池不用于选控制器阈值。少量预先固定的开发场景可筛查灾难性退化，最终仍需完整评测及独立training seed确认。新环境步预算和额外测评成本分别记录。
5. 只有A信号稳定且有值得追踪的方向时，再加B，不同时展开D³、Nash、SR×MSE及梯度冲突大网格。

## 7. 成功/失败判据与论文影响

成功不限于更高macro endpoint：也可以在平均性能取舍可接受的情况下，更早共同达标、降低最大gap曲线或减少特定能力退化；必须预先选定主要评价，不在结果出来后任选赢家。

仅MSE变低不够。若被分配较多的能力没有更早学会，或者别的能力出现更大损失，就不能解释为更完整融合。平均SR明显受损时必须报告trade-off，不能只用一个gap指标覆盖。

本文无需在allocation上另起一个复杂机器人专用理论。已有的robotics特征是：episode长度与物理成本不同，闭环状态分布随student变化，物理采集与flow学习可分开计量。不要同时新增success-only learning、event weighting、critic或自适应R。

当前Problem Setup已定义能力覆盖指标；进一步可在allocation节说明需求反馈，并在experiments解释joint attainment/最大gap曲线。当前正文写的是SR-gap；采用新方案前不能提前把它改写成已经运行的MSE/Nash控制器。
