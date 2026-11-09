# 新四任务 R2：allocation 需求反馈探索

依据 workspace `paper/notes/ALLOCATION_DIRECTION_20261109.md`。开发probe，不替代正式主矩阵；不修改论文正文。2026-11-09。

## 第一阶段（准备/运行中）

新组合没有现成的MOPD完整parent checkpoint，采用笔记允许的共同初始化方案：各组从同一SFT400、新建optimizer、相同seed0开始。固定 T32 / B256 / micro16×4GPU / LR1e-5 / constant / optimizer warmup4 / Refresh-R2 / 30 rounds（60 updates）。预测50、执行25、ODE10，native horizons700/600/600/900。四组相同5-round控制器warmup；同步权重 alpha=4q/32。

| 节点 | 第一条 | 后续候选 | 目的 |
|---|---|---|---|
| action30 GPU0–3 | Uniform-sync-R2 | SR-gap-sync-R2 | 同Blackwell硬件的SR基准 |
| action10 GPU0/3/4/5 | normalized-MSE-sync-R2 | 仅在必要时补同机对照 | 优先验证残差候选 |

用户随后明确优先候选，取消action10重复Uniform（已发送owned launcher清理并保留日志）。现在并行action30 Uniform与action10 MSE候选；跨硬件仅作探索描述，不宣称严格controller因果比较。真有必要时再补同机对照。SR组尚待新teacher校准，未启动。

- Uniform与MSE候选已排队，实际训练状态以共享输出各host状态JSON为准。
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

## 实现与显存设置

MSE候选在独立worktree `outputs/worktrees/robotwin_allocation_residual/`、分支 `experiment/robotwin2-allocation-residual-r2` 开发，action30基线继续使用原冻结worktree。controller消费每轮首次update的原始suite velocity MSE，最初5轮均值作为冻结reference，最近5轮均值为平滑值，比例不截断；仍采用rho0.5/epsilon0.01的uniform mixture及原配额约束。controller历史/reference纳入checkpoint并验证恢复。该候选本身不需teacher SR作为调度输入；SR比较组及效果解释仍需训练分布校准。

候选启用已有的actor缓存预分配22GiB、清理阈值22.5GiB，保留8GiB设备空闲检查。预留的是可复用缓存，不保持无用live ballast；这不是进程间独占锁，也不将模型/teacher/仿真的峰值简单相加作为可行性保证。当前候选单测52项通过，GPU首轮峰值仍需实际日志确认。基线未重启去更改缓存设置，跨硬件wall time仅作描述。

W&B实证：action30 Uniform run `5fzkk7q3`，project `FORMAL_robotwin_pi05`，group `allocation_new4_sft400_r2_probe`。候选run ID待实际logger创建后记录。旧重复Uniform的消耗不合并为候选训练进度，不冒充新的seed。

## 已验证进展与运行修复

- Teacher训练分布校准完成，每task20条，SR90%/100%/90%/90%，80条轨迹、19335个实际control targets、630.12秒。已通过训练signature/checkpoint/task集合校验。参考见teacher_reference_action30.json。
- 独立开发评测reference（当前action30 OIDN233、cohort0前20条）teacher为80%/100%/90%/95%；SFT400为50%/45%/55%/70%，宏55%，max teacher gap55pp。开发参考与训练校准不是同一分布，分开使用。
- 候选首次缓存预留失败发生于初始化，零更新。重试完成3轮/6updates后，第4轮的可选缓存补回因headroom差约0.277GiB触发保护退出；已归档128条实际轨迹、54962 control targets。这不是模型CUDA OOM，也不能混成连续训练。
- 已修复运行中可选缓存补回：启动仍严格检查余量，运行中余量不足或可选补回OOM时跳过补回，不为无用cache中止训练。相关单测53项通过。
- 候选从相同SFT400重新开始，硬件microbatch改8/card，global B256/T32/R2/LR和controller配置不变。当前W&B fnq9kq7u（旧h52t2oq3为失败部分）。不能把两条run接成同一训练曲线。需继续验证新峰值。
- Uniform仍使用micro16/Blackwell；仅作跨硬件探索对照。主图效应与wall time的因果结论需后续匹配对照，不将不同micro与硬件下的噪声当controller收益。
- 开发评测队列修正了checkpoint查找路径：run_dir/run_dir.name/checkpoints/global_step_N。已完成的teacher/initial400结果保留，恢复队列等待/评测三组round10/30，不因某候选在已知重启期间暂停其他已就绪评测。
- W&B allocation/gap/* 对MSE候选表示归一化loss需求，不能当SR的G；真正SR最大gap保存在开发summary的max_teacher_gap。

当前状态与完整原始数据以共享OUT目录的status JSON/episodes/update记录为准。本文进展是阶段记录，不宣称候选已有效。

## 追加同硬件 MSE 对照（11/09 18:25）

用户要求充分利用action30已释放的训练资源，新增normalized-MSE-sync-R2 / Blackwell / micro16，对比该机已完成的Uniform与SR-gap。三者同SFT400、T32/B256/LR1e-5/R2/seed0/30rounds，保存round10、30；启动配置完整diff仅有allocation mode、demand_signal与输出/日志名称差异，见action30_matched_config_audit.json。MSE不需要SR校准作为控制输入，不启用action10的显存cache改动。

- 训练状态：OUT/action30_residual_matched_status.json；W&B run9mfsxt3k，同project/group。
- action10原MSE（Ada/micro8、fnq9kq7u）继续完成，作为补充探索，不再承担主要跨controller比较。
- 新测评状态：OUT/development_action30_matched/status.json；复用同一开发20场景/task、teacher/initializer参考，等待旧测评owner完全退出后再启动新服务，自动测round10/30。不使用formal100池选择方法。
- 本阶段最少完成主比较为action30 Uniform / SR-gap / MSE；同seed并不保证数值轨迹完全相同，仍需独立training seed确认稳健性。
- setup/summarize_exploration.py与plot_exploration.py已纳入第四条run。归档脚本可用--output-root指定OUT，在action30重新计算记录；主目录paper正文不变。
