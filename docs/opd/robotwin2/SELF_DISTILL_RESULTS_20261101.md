# 同 teacher 自蒸馏控制（2026-11-01）

**结果：当前单 teacher 控制保留了原能力，支持继续检查 joint 优化与初始化；不证明多 teacher 路径没有任何问题。**

student 与 teacher 均从 place_container_plate 的同一 PACE SFT checkpoint 初始化。采用真实 MOPD 训练路径，full50×32 M3目标，T8/R4/B128/micro2，4GPU，LR1e-5、warmup4，20updates。没有修改原 M3 时间权重、teacher 或归一化。

| 指标 | 结果 |
|---|---|
| 初始／最终冻结 place SR | 6/8 → 6/8 |
| 场景配对 | actual_reset_seed 相同；8个place episode的成功/失败逐一一致 |
| place控制步数合计 | 1643 → 1637 |
| 在线训练5轮 | 7/8、8/8、7/8、7/8、7/8，总36/40 |
| MSE首次／最后 | 4.54e-5 → 3.56e-4 |
| 过程中最大记录MSE | 9.32e-4（step16），随后回落 |

冻结测评仍使用原四任务、T16×2 rounds、allocator seed200000的真实推理路径，确保place场景序列与原specialist step0一致。其余任务：adjust1/8、handover0/8、stack0/8；这一个成功不解释为稳定迁移收益。策略采样noise未固定，因此不是逐动作重放或统计等价试验。

结论边界：控制结果与“任何M3更新都会破坏策略”的解释不符；但不能排除joint特有的teacher/数据/优化问题，不能把此前6/32→1/32唯一归因于梯度冲突。固定输入probe显示新增任务velocity改善、place漂移，后续继续充分优化和真实联合SFT。

后续：探索性gate（place≥4/8、same actual seeds、末MSE<0.01）已满足，joint20→60全状态恢复队列已进入TRAINING。保持R2/T16/B128/micro4/LR1e-5，额外40updates/20rounds，冻结测评40/60。该gate只是决定是否值得继续，不是显著性检验。

证据位于根workspace `outputs/robotwin4/`：

- `analysis/self_control_summary_20261101.json`
- `analysis/self_control_step20_trainpath_t16_20261101/results.json`
- `analysis/specialist_step0_trainpath_t16_20261101/results.json`
- `controls/place_self_m3_full50_r4_t8_b128_pilot_20261101/optimizer_updates.jsonl`
- `setup/longer_opd_queue_20261101.json`

## 联合 SFT 后的资源配置预案

实际checkpoint由其冻结测评结果决定，尚未启动后续MOPD。已用存在的smoke checkpoint做CPU配置检查；它只是schema占位，不是后续训练起点。

若全8GPU空闲，候选为T32/R4/B128/micro8、80updates，每20保存，LR1e-5/warmup4，从选定SFT权重重新开始MOPD optimizer。当前allocator要求每env rank至少覆盖4tasks，T16/8不支持；T32/8每rank4slots有效。新初始化不应复用SFT optimizer。

T32/R4与此前T16/R2都是每optimizer update平均8条轨迹；但rollout轮次、局部数据池、硬件及初始化不同，不能作为单因素消融。两种最小覆盖配置均没有adaptive非均匀allocation余量。本阶段只验证uniform多teacher融合及rollout复用。
