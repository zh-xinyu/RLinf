# Specialist 初始化：延长 MOPD 到 60 updates

2026-11-01，action00。**更充分优化出现原有 place 能力恢复的信号，但仍未展示四任务能力融合。** 当前优先继续真实联合 SFT 初始化路线，不因本轮结果直接引入 teacher uncertainty 截断。

## 设置与预算

- Student 初始为 `place_container_plate` specialist，不是 joint SFT。四个任务等额 rollout，按 task 路由四个 teacher 与固定 normalization。
- T16/R2/B128/micro4、4GPU、LR1e-5、constant scheduler、warmup4、完整50×32监督；FP32 master/Adam 与 BF16 AMP。
- 从原 step20 **恢复完整训练状态**到60，R、B、LR及模型设置未改变；实际验证四actor的optimizer/scheduler/update counter恢复。rollout proposal RNG重新开始，因此不声称bit-exact replay。
- 本谱系累计60updates、30rounds、480trajectories、264,893control targets。逐episode身份无重复，trajectory/control计数与runner记录一致。续训增量为40updates/20rounds/320trajectories；预算不含其他pilot、smoke、控制或评测。

## 冻结策略结果

每checkpoint均使用真实训练推理链路，T16×2rounds、seed200000、每task8episodes。

| Checkpoint | adjust | handover | place | stack | 合计 |
|---|---:|---:|---:|---:|---:|
| 初始化（place specialist） | 0/8 | 0/8 | 6/8 | 0/8 | 6/32 |
| MOPD20 | 0/8 | 0/8 | 1/8 | 0/8 | 1/32 |
| MOPD40 | 0/8 | 0/8 | 1/8 | 0/8 | 1/32 |
| MOPD60 | 0/8 | 0/8 | 4/8 | 0/8 | 4/32 |

四个checkpoint的实际scene seed序列一致；action noise未显式固定。place从20→60有3个场景失败转成功、0个成功转失败；60的4个成功都属于初始化时的6个成功场景。它是小样本恢复信号，不是显著性证明，也不是新任务迁移。

原协议重试会产生重复场景：adjust的8episodes只有6个unique scene seeds；其余task为8个。当前为unfiltered/seen pilot，不是官方feasible/unseen成绩。

## Loss与解释边界

| 训练窗口 | velocity MSE | M3 weighted loss |
|---|---:|---:|
| updates1–5 | 0.44277 | 0.26016 |
| updates16–20 | 0.22678 | 0.22766 |
| updates21–25 | 0.18682 | 0.20345 |
| updates56–60 | 0.07408 | 0.07795 |

末5步逐task MSE为adjust0.06823、handover0.08629、place0.04808、stack0.09373。这里是各时刻student访问状态上的训练信号，状态分布会变化；不同窗口的下降不能独立证明全局策略接近teacher。

- 更长训练继续降低loss，且place出现部分恢复，说明不能笼统说“多训完全无效”。
- 三个新任务在本次冻结测评仍全为0；在该预算和specialist初始化下，尚未实现multi-teacher能力融合。
- 同teacher自蒸馏控制保留place6/8→6/8，为基础链路提供支持；但本轮不能证明遗忘仅由gradient conflict或teacher OOD导致。
- 当前联合SFT正在训练与评测。其示范CFM loss与上述teacher velocity MSE不是同一目标，不能横向比较绝对值。

## 证据与队列恢复

根workspace `outputs/robotwin4/`：

- `analysis/longer_opd_training_summary_20261101.json`：完整训练窗口、交互预算与计数核验。
- `analysis/specialist_full_trajectory_comparison_20261101.json`：0/20/40/60实际场景与成功率比较。
- `analysis/specialist_continue_step{40,60}_trainpath_t16_20261101/`：原始episodes、resolved config、COMPLETE。
- `analysis/longer_opd_summary_20261101.json`：恢复队列完成记录。
- 原`setup/longer_opd_queue_20261101.json`在训练结束后GPU空闲检查失败；**训练run为COMPLETE**。失败瞬间逐卡占用未捕获，后续检查4–7完全空闲，只能说与释放延迟相符。
- `setup/longer_opd_eval_recovery_20261101.status.json`已COMPLETE，原失败文件保留。只恢复40/60评测，训练没有重跑。
- 新下游`setup/joint_sft_followup_queue_recovery_20261101.json`已进入SFT100评测，再测200/500；固定500可行性筛查通过后才运行预定MOPD80。
