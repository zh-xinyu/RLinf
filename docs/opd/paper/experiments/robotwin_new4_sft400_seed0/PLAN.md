# RoboTwin 新四任务：Joint-SFT400 / seed0 主实验矩阵

状态：实验列表已整理，未启动训练。2026-11-08。

## 目标与固定 setting

检验多teacher在线蒸馏能否提升三个待补强任务并保持stack能力，以及allocation+同步weighting、rollout reuse在固定交互预算下的增量。

- 任务固定为 open_laptop / handover_mic / dump_bin_bigbin / stack_bowls_two。
- 四组独立从同一 `student400_bundle` 初始化，新建optimizer；不按执行顺序串联训练。权重SHA256：`d2ef819ae9a7706accaa027edf406a1c2bc1c8f007e8bccbe695e63e4f77ce5a`。
- checkpoint、teacher与norm来源继承[初始化记录](../robotwin_initialization_20261107/README.md)。同一任务固定路由到对应teacher。
- 新组合没有可直接复用的Joint-SFT2000；旧四任务的SFT2000不能作为新组合的联合SFT baseline。

## 主矩阵：建议执行 M0 → M2 → M1 → M3

沿用最近原组合的50-round方案，四组共同固定终点，不根据正式成绩单独截取最佳round。

| ID | Allocation / loss权重 | Reuse | Rounds | Student updates | Student rollout trajectories | 状态 |
|---|---|---:|---:|---:|---:|---|
| M0 | Uniform / 等权 | R1 | 50 | 50 | 1600 | 待配置/启动 |
| M2 | Adaptive / 同步权重 | Refresh-R2 | 50 | 100 | 1600 | 待校准/配置/启动 |
| M1 | Uniform / 等权 | Refresh-R2 | 50 | 100 | 1600 | 待配置/启动 |
| M3 | Adaptive / 同步权重 | R1 | 50 | 50 | 1600 | 待校准/配置/启动 |

M0与M2先回答基础MOPD及完整方法是否有效；M0↔M1、M3↔M2对应reuse增量；M0↔M3、M1↔M2对应allocation+同步loss weighting增量。后者是一个联合模块，不拆称仅采样贡献。

## 共同训练参数

| 参数 | 约定 |
|---|---|
| T | 全局32条/round；Uniform每task8条；Adaptive改变配额但总和保持32 |
| Global B | 256 / student update；R2不除以2 |
| 训练设备与microbatch | 参考4GPU，micro16/card；这是训练microbatch，与推理batch不同 |
| LR / scheduler | 1e-5 / constant；warmup4 student updates，各组一致 |
| Loss / 参数范围 | 保留当前M3蒸馏loss及完整50步latent horizon监督；VLM与action expert均训练 |
| 控制 | 预测50步，执行H25，10步flow ODE；native horizons 700/600/600/900 |
| Seed0映射 | actor0；env/allocator42；rollout noise1000000000，沿用既有seed0定义 |
| Adaptive | rho0.5、epsilon0.01；warmup/window/interval均5 rounds；min_task_episodes5、max_quota_change4 |
| 保存 | round5/10/15/20/25/30/35/40/45/50；R1保存update5/10/…/50，R2保存update10/20/…/100 |

固定trajectory预算不等于固定simulation steps或计算量；R2使用更多优化。记录rollout rounds、optimizer updates、累计trajectories、实际control targets、各阶段与累计wall time、逐task SR/MSE、实际配额和loss权重。主横轴rollout rounds，辅助报告累计仿真步数与时间。

## 参考评测与运行前事项

| ID | 项目 | 规模 | 用途/状态 |
|---|---|---|---|
| C0 | 新四teacher的训练rollout分布校准 | 每task20 episodes，共80条 | Adaptive共享输入；待做，不能用开发评测SR替代 |
| E0 | SFT400共同起点 | 每task100 episodes，共400 | 正式round0，共用一次；待做 |
| E1 | Routed specialists | 每task100 episodes，共400 | 正式能力参考，共用一次；待做 |
| E-M | 四主组各10个保存点 | 每checkpoint每task100 episodes | 合计16000 episodes；待做 |

- 新四任务需要冻结新的正式100场景/任务池，避开本轮开发使用的cohort0前50场景；候选为未参与该新组合初始化选择的完整author cohort1。最终生成后锁定manifest/hash，所有参考和checkpoint共用，不按模型结果过滤。
- 所有正式评测复用标准RolloutWorker+EnvWorker与训练环境控制逻辑。固定场景、unseen指令是明确的held-out协议差异。
- 每卡并行8环境/推理batch8作为候选；需先完成多slot固定场景、noise、尾批mask和端到端验证。通过前，已有单环境worker入口可用。正式E0/E1和主曲线统一采用最终冻结的评测batch及noise规则，不混用历史串行成绩。
- 参考与主曲线评测合计16800 episodes，另加C0校准80条。评测交互与训练交互分别记录；校准成本单列，并在总交互成本口径中纳入。
- 核对新任务teacher/norm/train-eval注册表；删除旧adjust_bottle、place_container_plate条目。当前formal/recipe.json仍是旧四任务/SFT500/100 rounds，不能直接启动它代替本计划。
- 统一训练/评测runtime与实际仿真依赖版本；运行前检查完整rollout→teacher监督→student更新→保存/加载→评测链路。具体节点与GPU分配尚未指定。

## 日志与边界

W&B沿用 `FORMAL_robotwin_pi05`，使用独立group `robotwin_new4_sft400`，tags包含 `seed0`、`new4`、`rounds50`，名称显式包含method/R/T32/B256；输出另设 `outputs/robotwin4/formal_new4_sft400_seed0/`，本页列的是拟用路径，尚未创建运行。

本轮只安排seed0的四组主矩阵及必要参考。不同时铺cached、static-sync、R4、初始化或LR网格。后续若主张超过继续/充分联合SFT，需要另列新四任务的相应SFT对照；当前矩阵只直接支持共同cold start后的能力提升及内部效率比较。seed0不能替代多seed稳定性验证。
