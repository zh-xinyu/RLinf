# Teacher 独立场景参考：每任务20 episodes

2026-11-01，action00。四个固定PACE teacher通过真实训练推理链路完成80个episodes，**59/80成功，suite等权73.75%**。这是unfiltered/seen参考，不是官方feasible/unseen benchmark成绩。

| Task | 成功 / episodes | SR | 不同实际场景数 | 使用reset重试的episodes |
|---|---:|---:|---:|---:|
| adjust_bottle | 8/20 | 40% | 20 | 2 |
| handover_mic | 19/20 | 95% | 20 | 0 |
| place_container_plate | 15/20 | 75% | 20 | 0 |
| stack_bowls_two | 17/20 | 85% | 20 | 0 |

## 协议

- 固定原四teacher、task norm、10步ODE、50步latent/执行25步；未训练teacher。
- 4GPU，T16×5rounds，seed300000，`--seed-stride 20`。task内第k个episode的请求seed为`300000 + task_id*1000003 + 20*k`。
- native reset仅在UnStableError时尝试请求seed至seed+19。请求间隔20使重试窗口互不重叠；20个episode的actual seed均唯一，并逐条检查落在对应窗口内。不会先看策略成败再换seed。
- 未启用expert/planner可行性筛选；instruction为seen。此协议的CPU测试已有覆盖，本次为真实GPU全80episodes确认。
- 与历史seed200000/T16×2的8episode pilot属于**不同场景集合**。不能直接用本次59/80与历史student成绩计算增益；后续student独立确认应同样使用seed300000、stride20、T16×5并核对actual seeds。
- action noise未固定；不同scene seed也不等于所有随机因素独立或统计显著。

## 当前解释

三个teacher在本组场景有较高成功率，adjust明显较弱。不要把teacher当作必然优于student的统一上限；尤其SFT200在另一组小样本中adjust为7/8，还需要同协议student确认才能比较。当前不据此改动已固定的SFT500→uniform MOPD80实验，也不挑选更弱的student checkpoint作为基线。

本次使用等待SFT500期间的GPU4–7空档，设置25分钟上限，约867.5秒完成，exit0；结束后四卡仅1MiB占用。没有改变SFT训练或后续队列。

## 证据

根workspace `outputs/robotwin4/`：

- `analysis/teacher_confirm_stride20_t16_n20_20261101/results.json`：80条实际episode与逐task成功率。
- 同目录`verified_summary.json`：完整task计数、唯一scene数、请求/实际seed窗口、success summary一致性检查。
- 同目录`resolved_config.yaml`与`COMPLETE`：运行配置和完成标记。
- `setup/teacher_confirm_stride20_t16_n20_20261101.process.json`：精确命令、进程身份、启动时SFT414步与时限。
- 同前缀`.status.json`：exit0、完成标记与实际耗时；`.launch.json`记录owned Ray身份。
