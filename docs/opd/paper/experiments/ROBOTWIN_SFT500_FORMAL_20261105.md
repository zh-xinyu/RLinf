# RoboTwin 2.0：SFT500 正式实验（2026-11-05）

状态：action10 启动检查中；本页对应新初始化，不覆盖 SFT2000 历史结果或原交接分支。

- 分支：`experiment/robotwin2-sft500-formal`；worktree：`outputs/worktrees/robotwin_sft500_formal`。
- 起点：固定 500 updates 的联合 SFT cold start，每任务 45 条训练示范；完整权重及 metadata SHA256 见 `toolkits/robotwin_mopd/formal/resources.json`。不是 few-shot，也不声称充分 SFT。
- 四任务：adjust_bottle、handover_mic、place_container_plate、stack_bowls_two。
- 统一参数：4 GPU，T32，global B256，LR 1e-5，constant scheduler，warmup 4 student updates，100 rollout rounds，seed0。
- Microbatch：先 smoke 16；如 48GB 卡容量不够，正式训练前统一调整，保持 global B256 不变，记录最终值。
- 每 10 rollout rounds 保存完整 checkpoint：10/20/30/40/50/60/70/80/90/100。R1 对应 updates 10…100，R2 对应 updates 20…200。
- 每个保存点按原冻结 formal100 场景/指令池独立测评，每任务 100 episodes；固定执行 H25，10 ODE steps。Teacher 与初始 student 各测一次，共 42 行、168 个 task-checkpoint jobs、16,800 episodes。
- 四组：M0 Uniform-R1；M1 Uniform-Refresh-R2；M2 Adaptive-Sync-Refresh-R2；M3 Adaptive-Sync-R1。
- W&B project：`FORMAL_robotwin_pi05`；group：`robotwin_formal_sft500_100ep`。
- 执行顺序：完成 GPU smoke 后，先训练 M0，再评测 teacher、初始 student 和 M0 的所有保存点；然后依次训练并评测 M1、M2、M3。首轮不自动增加 seeds，不按结果改变训练预算。
- 共享输出入口：`outputs/robotwin4/formal_sft500_action10_20261105/`，机器配置 `machine.action10.json`，队列状态 `queue_status.json`，汇总 `deliverables/checkpoint_results.csv`。
- 存储：M0/M1 完整训练目录在共享输出下；M2/M3 在 action10 `/localscratch/liyifa11/XyFile/opd-robotwin4-action10-20261105/formal_sft500_20261105/train/`。所有测评、汇总和队列状态在共享输出入口。需到 action10 访问 M2/M3 权重，不误认为其已共享归档。
- 初始化比较：保留已有 SFT2000 成绩为历史参考；只有相同正式协议下的测评才合并到正式表格。SFT500→MOPD 提升不等价于优于充分 Joint-SFT。

验证：65 项针对性 CPU 测试已通过（formal 配置、checkpoint 计划、rollout refresh）；GPU 实测待更新。修复 formal 评测路径以匹配 runner 实际的 `<run>/<run>/checkpoints/` 布局。
