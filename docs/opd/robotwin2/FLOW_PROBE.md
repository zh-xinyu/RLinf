# 完整 flow 诊断（训练后运行）

目的：解释当前 projected velocity MSE 与任务 SR 的差距，不改变训练 loss，也不把诊断当作成功率评测。

模型生成 50×32 的 latent trajectory，默认 M3 返回前 25 个位置的 velocity，实际执行前 25×14。这个工具在 student 访问的同一个 `(observation, x_t, t)` 上比较完整 student/teacher velocity，并比较相同初始 noise 下分别积分得到的完整 endpoint。

- 每次明确选择固定评测池或普通 seen reset；默认取前 4 个决策观测；最多执行 3 个 chunk / 75 个 control targets，原生终止时提前停止。
- 时间索引 `0 4 9` 对应 `1.0, 0.6, 0.1`；不是全时间积分，也不能直接当训练 MSE 的估计。
- 分开报告 first25/last25 × physical14/padded18 的 MSE、RMS、cosine，以及各时间点和 endpoint 的统计。
- 保存请求 seed、实际 reset seed、实际 prompt、协议、checkpoint、每个 noise seed、实际控制步数和样本数；pool 模式另存 pool hash。noise 根据实际 reset seed 和 probe index 可重复生成。
- “physical14”仍是**归一化 latent 坐标**，不是物理单位的关节误差。
- 不报告截短 episode 的成功率；不能仅凭未监督部分误差较大就宣称因果关系。

激活隔离 runtime，确认指定 GPU 已空闲后，在 robotwin4 worktree 运行：

```bash
CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/flow_probe.py \
  --task adjust_bottle \
  --checkpoint /absolute/path/to/global_step_40 \
  --evaluation-pool /absolute/path/to/evaluation_pool.json \
  --probes 4 --flow-indices 0 4 9 \
  --output /absolute/path/to/new_probe_adjust_bottle.json
```

`--checkpoint base` 可生成单独的基模参考。默认 `--episode-index 0`；指定另一个 accepted scene 可检查现象是否仅来自单场景。工具要求输出路径不存在，并在加载模型前拒绝有占用的 GPU。

## 普通 reset 的诊断模式

本机 expert screening 首个 `adjust_bottle` 候选 seed `100100000` 用时约 121 秒，结果为 `expert_failed`，同时出现大量 invalid start state / self-collision 警告。当前 MPLib 路径的可用性尚未确认，因此暂停构建 feasible pool，保留失败记录。只读检查发现：RoboTwin wrapper 向 MPLib 传入单臂 6 维 qpos，而当前 MPLib 的 `pad_move_group_qpos` 将其写入完整 qpos 的前缀；Aloha 两臂又共享一个 articulation。这是需要进一步验证的兼容性风险，**不是已经证明的库 bug**，也不能据此推断场景本身不可完成。

短 flow 诊断可显式改用普通 native reset，不调用 expert planner：

```bash
CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/flow_probe.py \
  --task adjust_bottle --checkpoint /absolute/path/to/global_step_40 \
  --scene-seed 200001 --probes 4 --flow-indices 0 4 9 \
  --output /absolute/path/to/new_unfiltered_probe_adjust_bottle.json
```

其他三个任务建议从 seed `200000` 开始。`--scene-seed` 与 `--evaluation-pool` 必须二选一；scene 模式不能指定非零 `--episode-index`。普通 reset 允许环境原有的 unstable-scene 重试，输出记录 `requested_scene_seed`、`actual_reset_seed` 和实际 seen prompt，`scene_seed` 对应实际场景。协议明确标记为 `unfiltered_seen`，不声称场景经过 expert 可行性筛选。固定 pool 模式仍严格复用 accepted seed/unseen prompt，拒绝重试换 seed 或 prompt 不一致。

两种模式都只做截短诊断，不产生 episode 成功率；不能把普通 reset 结果当作官方 expert-filtered benchmark 分数。训练 loss、训练 reset 和评测队列均不由本工具修改。

CPU 测试覆盖统计、同 noise/同 student path/真实时间网格、最多三个执行 chunk、原生终止、CLI 模式互斥、普通 reset 重试后的 seed/prompt 来源，以及 pool 模式的严格检查。真实 GPU probe 由调用方在确认资源空闲后单独启动。

## 可选的完整 horizon 监督

`actor.model.openpi.nft_full_horizon` 默认 `false`，保持现有前 `25×32` 的 M1/M3 监督。显式使用 Hydra override `+actor.model.openpi.nft_full_horizon=true` 后，student 与 routed teacher 的 NFT 输出均保留完整 `50×32`；要求 `reward_type=chunk_level`，以观测/chunk 的有效性 mask 广播到完整 latent。实际 action 执行仍是 `25×14`，ODE、时间权重和机器人 reset 不变。

在固定batch与有效mask时，完整 horizon 的 loss/velocity MSE 对50个位置取均值，分母翻倍，前25位置的单元素梯度系数减半；本次follow-up也改变了B，需另行考虑batch平均系数。因此新旧训练 MSE 数值不是同一口径，应使用同一 probe 的 first25/last25 分区统计比较。四任务短 probe 的尾段误差较大支持尝试此选项，但不证明它导致 SR 不响应；若从已有 student checkpoint 继续训练，这只是效率优先的跟进试验，不是控制其他条件的因果消融。


## 已完成 GPU 诊断（step40）

四任务各1个未筛选/seen场景、4个初始访问观测，学生与基模共8份report均完成；执行各75个control targets，不计作SR。初始seed、prompt、noise hash一致；后续观测随策略变化，不把后续aggregate当配对状态对照。

| Task | Student velocity MSE first25 / last25 | Student endpoint MSE first25 / last25 | 初始endpoint first25：base → student |
|---|---:|---:|---:|
| adjust_bottle | 0.1335 / 0.5701 | 0.0275 / 0.2046 | 0.6075 → 0.0643 |
| handover_mic | 0.3906 / 0.6388 | 0.0531 / 0.3068 | 0.4871 → 0.0890 |
| place_container_plate | 0.2535 / 0.7135 | 0.0391 / 0.2194 | 0.3942 → 0.1001 |
| stack_bowls_two | 0.2635 / 0.4888 | 0.0949 / 0.2163 | 0.5285 → 0.0703 |

表内均为normalized physical14坐标。证明拟合有进展，但不证明任务能力提升或后半段误差的因果作用。来源：`outputs/robotwin4/analysis/flow_probe_step40_unfiltered/summary.json`及逐任务reports。


## 温启动follow-up与数值检查

两段训练的最终独立SR都为0/20。下表只比较同一初始seed/prompt/noise的normalized physical14 endpoint MSE；teacher初始endpoint RMS也一致。后续观测受策略影响，因此不将四观测aggregate解释为严格配对比较。

| Task | 初始pilot first25 / last25 | Full-horizon follow-up first25 / last25 |
|---|---:|---:|
| adjust_bottle | 0.0643 / 0.3362 | 0.0234 / 0.0906 |
| handover_mic | 0.0890 / 0.3130 | 0.1539 / 0.5193 |
| place_container_plate | 0.1001 / 0.4265 | 0.0392 / 0.1907 |
| stack_bowls_two | 0.0703 / 0.2168 | 0.0690 / 0.3052 |

结果并非一致改善：adjust/place下降，handover上升，stack前25接近而后25上升；这组同时改变了训练量、T/B/R和监督范围，不支持full-horizon的单因素因果结论。

额外在真实最终checkpoint、adjust初始场景的3个ODE时刻比较joint/cached forward：physical14 MSE为1.4e-5–4.9e-5；joint加BF16 autocast为2.0e-5–1.8e-4。此差异远小于当前student/teacher误差；它是unsharded eval数值sanity check，不是完整分布式训练等价性证明。

新来源：`outputs/robotwin4/analysis/flow_probe_fullflow_step40_unfiltered/comparison_summary.json`、`joint_cached_adjust_bottle.json`；检查脚本保留在`outputs/robotwin4/analysis/check_joint_cached.py`。全部13份GPU诊断已完成，不需继续等待训练或排队。
