# RoboTwin 2.0：SFT500 正式实验（2026-11-05）

状态：action10 四卡 smoke 已通过，正式队列已启动（M0 → M1 → M2 → M3）；本页对应新初始化，不覆盖 SFT2000 历史结果或原交接分支。

- 分支：`experiment/robotwin2-sft500-formal`；worktree：`outputs/worktrees/robotwin_sft500_formal`。
- 起点：固定 500 updates 的联合 SFT cold start，每任务 45 条训练示范；完整权重及 metadata SHA256 见 `toolkits/robotwin_mopd/formal/resources.json`。不是 few-shot，也不声称充分 SFT。
- 四任务：adjust_bottle、handover_mic、place_container_plate、stack_bowls_two。
- 统一参数：4 GPU，T32，global B256，LR 1e-5，constant scheduler，warmup 4 student updates，100 rollout rounds，seed0。
- Microbatch：16/card；四卡完整模型 smoke 通过，NVML 峰值 41.03 GiB/card，global B256。
- 每 10 rollout rounds 保存完整 checkpoint：10/20/30/40/50/60/70/80/90/100。R1 对应 updates 10…100，R2 对应 updates 20…200。
- 每个保存点按原冻结 formal100 场景/指令池独立测评，每任务 100 episodes；固定执行 H25，10 ODE steps。Teacher 与初始 student 各测一次，共 42 行、168 个 task-checkpoint jobs、16,800 episodes。
- 四组：M0 Uniform-R1；M1 Uniform-Refresh-R2；M2 Adaptive-Sync-Refresh-R2；M3 Adaptive-Sync-R1。
- W&B project：`FORMAL_robotwin_pi05`；group：`robotwin_formal_sft500_100ep`。
- 执行顺序：完成 GPU smoke 后，先训练 M0，再评测 teacher、初始 student 和 M0 的所有保存点；然后依次训练并评测 M1、M2、M3。首轮不自动增加 seeds，不按结果改变训练预算。
- 共享输出入口：`outputs/robotwin4/formal_sft500_action10_20261105/`，机器配置 `machine.action10.json`，队列状态 `queue_status.json`，汇总 `deliverables/checkpoint_results.csv`。
- 存储：M0/M1 完整训练目录在共享输出下；M2/M3 在 action10 `/localscratch/liyifa11/XyFile/opd-robotwin4-action10-20261105/formal_sft500_20261105/train/`。所有测评、汇总和队列状态在共享输出入口。需到 action10 访问 M2/M3 权重，不误认为其已共享归档。
- 初始化比较：保留已有 SFT2000 成绩为历史参考；只有相同正式协议下的测评才合并到正式表格。SFT500→MOPD 提升不等价于优于充分 Joint-SFT。

验证：65 项针对性 CPU 测试已通过（formal 配置、checkpoint 计划、rollout refresh）；四卡通信、四 teacher 各 2 episodes（全部成功）、1 rollout / 2 updates / 完整 checkpoint 保存均通过；两次更新 loss/梯度有限。完整 checkpoint 实测 66.31 GB，四组共约 2.65 TB，按两组共享盘、两组本地盘存放。修复 formal 评测路径以匹配 runner 实际的 `<run>/<run>/checkpoints/` 布局。

## 首段运行（已中断，正式曲线保留 steps1–20）

- M0 已启动：[W&B t59ayqi2](https://wandb.ai/zhouxinyu2022-michigan-state-university/FORMAL_robotwin_pi05/runs/t59ayqi2)。
- 队列 PID：353662；机器：action10；GPU：0–3。
- 代码提交：`ac3d4da6`。共享输出中 `queue_status.json` 为实时状态，本页不是实时监控。

## 2026-11-06：M0 中断恢复

- 原 run `t59ayqi2` 完成 updates 1–24，第25轮采样后因显存被额外进程占用而 OOM。最近完整 checkpoint 为 round/update20；10、20 checkpoint 原样保留。
- 使用 GPU0–3 从 checkpoint20 恢复模型、Adam、scheduler、任务游标和累计有效交互计数，max_steps 仍100；LR/B/microbatch/评测协议不变。现有 rollout 噪声 RNG 重启，非逐位重放未中断训练。
- W&B 服务端拒绝 rewind（private preview 未开通）。续训 run：[ng4bjumu](https://wandb.ai/zhouxinyu2022-michigan-state-university/FORMAL_robotwin_pi05/runs/ng4bjumu)，同 project/group，名称后缀 `_resume20`。旧 run 在线历史未删除，本地和线上均保留审计信息。
- 有效曲线仅使用旧 run steps1–20 + 新 run steps21–100。丢弃的5轮（21–25）消耗160条轨迹、50,498 control steps，作为 recovery overhead 单独记录；最终100有效rounds对应105实际训练rollout rounds，不将这部分物理成本隐去。
- 恢复输出：`outputs/robotwin4/formal_sft500_action10_20261105/recovery_20261106_step20/`；队列 PID751031。checkpoint30–100 保存到续训目录，完成后在原评测入口链接；原checkpoint不覆盖。最终有效 JSONL 汇总到 `deliverables/m0_*.jsonl`，中断日志保留。
- 55项 refresh/allocation 测试通过；有效历史拼接和checkpoint链接的临时数据测试通过。实际 checkpoint 恢复已核验：四个 rank 的 optimizer/scheduler/sample_draw_count 均20，LR1e-5，每rank118个Adam状态条目；已继续rollout。证据 `recovery_20261106_step20/recovery_verified.json`，实时状态看共享入口的 `queue_status.json`。

## M0 训练完成与评测队列衔接（2026-11-06）

- M0有效100 rounds/updates已完成，checkpoint10/20/…/100齐全；有效JSONL已拼接，实际105轮（含5轮恢复开销）。
- 前10轮→末10轮：velocity MSE 0.10004→0.01328；训练rollout suite等权SR 54.69%→84.06%。这些是在线训练信号，不能代替100episodes/task独立测评。
- 末10轮逐task在线SR：adjust95.0%、handover83.75%、place76.25%、stack81.25%。
- 队列于2026-11-06 17:10 UTC在teacher评测启动前因GPU busy检查退出；未产生独立评测结果，M1–M3未启动。当时的占用者未记录，不能据此认定具体进程；当前GPU0–3空闲。
- 修复阶段切换：保持原显存空闲阈值，最多等待300秒，每5秒检查，仍繁忙则停止；不停止其他用户进程。应用到训练启动、评测启动和评测checkpoint之间，27项针对性测试通过。
- 续接入口：`outputs/robotwin4/formal_sft500_action10_20261105/continuation_20261106_after_m0/continue_queue.py`。跳过已完成M0训练，继续teacher→student→M0 checkpoints测评，再依次M1/M2/M3训练与测评。
