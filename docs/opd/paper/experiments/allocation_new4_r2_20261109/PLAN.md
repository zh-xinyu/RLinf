# 新四任务 R2：allocation 需求反馈探索

依据 workspace `paper/notes/ALLOCATION_DIRECTION_20261109.md`。开发probe，不替代正式主矩阵；不修改论文正文。2026-11-09。

## 第一阶段（准备/运行中）

新组合没有现成的MOPD完整parent checkpoint，采用笔记允许的共同初始化方案：各组从同一SFT400、新建optimizer、相同seed0开始。固定 T32 / B256 / micro16×4GPU / LR1e-5 / constant / optimizer warmup4 / Refresh-R2 / 30 rounds（60 updates）。预测50、执行25、ODE10，native horizons700/600/600/900。四组相同5-round控制器warmup；同步权重 alpha=4q/32。

| 节点 | 第一条 | 后续候选 | 目的 |
|---|---|---|---|
| action30 GPU0–3 | Uniform-sync-R2 | SR-gap-sync-R2 | 同Blackwell硬件的SR基准 |
| action10 GPU0/2/3/5 | Uniform-sync-R2 | normalized-MSE-sync-R2 | 同Ada硬件的残差候选基准 |

两个Uniform副本不是两个training seeds，不将跨硬件差异归因于controller。候选取得正向信号后在共同硬件补直接比较。候选尚未实现/启动，不能将上述表视为已经运行三种方法。

- Uniform两条已排队，资源/实际训练状态以共享输出的各host状态JSON为准。
- action30新资源/ckpts在本地scratch2；action10本地仅余123GB，不删除既有实验，使用共享目录并引用不可变资源。
- 独立Ray6393/8273、owner-scoped清理；不停止其他作业。
- W&B project FORMAL_robotwin_pi05；group allocation_new4_sft400_r2_probe。
- 保存round10、30完整状态。采集每轮首次更新的未乘suite权重velocity MSE，完整rollout pool按R2刷新。

## 下一步实现/筛查

1. 离线重放旧历史及新Uniform历史：冻结最初5轮均值 Lref，用最近5轮原始MSE均值/Lref作为需求。epsilon_L=1e-8；允许比例大于1。比较warmup3/5与平滑窗口3/5/10仅检查数值敏感性，不把反事实配额视为效果预测。
2. 先复用SR的uniform mixture（rho0.5、epsilon0.01）、每worker suite覆盖、每次配额变化上限4及5-round决策节奏；只换需求信号。
3. 每个节点按实际训练分布重新做新四teacher 20 episodes/task校准。teacher SR与student在线反馈同场景/指令分布；不从开发或formal SR冒充校准。明确记录额外环境交互。
4. 残差controller状态需含warmup样本、冻结reference和历史，并测试保存/恢复。只消费每轮第一个student更新，不混入第二次flow刷新后的loss。
5. 第一轮不加入进展/逆速率、SR门控、不确定性、success-only学习或自适应R。只有A可用后才考虑B。

## 效果评价约定

正式100场景池不参与选择。预定开发场景使用cohort0前20条/task（已用于初始化开发，不称held-out正式结果）；控制器不可读取该评测反馈。各硬件使用匹配runtime重新测teacher与共同起点。

主要连续指标是最大正teacher gap在开发round0/10/30上的预算归一化梯形面积，同时报告endpoint gap、宏SR和逐task SR；宏SR较同机Uniform低超过5pp时明确标为trade-off警报，不只按gap宣布胜利。50-episode历史、20-episode开发和100-episode正式结果分开。

阈值型辅助目标预定 delta=0.10：所有task达到teacher SR−10pp。有限开发次数下达标噪声较大，只有连续两个预定测评点达标才记录持续达标；不得以单次first crossing下结论。若未共同达标，报告未达到而非外推所需预算。

同时记录真实rollout trajectories、executed control targets、原始query数、quota/alpha与加权曝光proxy；R2本阶段固定，不把加权曝光算作真实交互。seed0仅用于筛查，不作多seed稳定性证据。

输出：`outputs/robotwin4/allocation_new4_r2_20261109/`。训练由程序每60秒归档小型日志，模型约20–30分钟集中分析一次；故障立即检查。探索代码分支 `experiment/robotwin2-allocation-new4-r2`，worktree `outputs/worktrees/robotwin_allocation_new4/`。
