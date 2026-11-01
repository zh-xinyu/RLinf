# RoboTwin 2.0：四任务多 teacher pilot

状态：完整八卡 smoke（含刷新、反向、保存、重载评测）通过；Uniform-R2 pilot 正在运行，尚无训练效果结论。独立于 RoboCasa 正式实验。

## Setting

- Student：官方 π0.5 base，**不先做目标任务联合 SFT**。这不等于无 demonstration：specialists 和归一化统计来自离线数据。
- Teachers：PACE 发布的四个独立单任务 π0.5 SFT checkpoints，JAX→PyTorch 转换，源文件 SHA256 校验。
- 同一 student 权重；已知 task identity 路由 teacher，同时选择固定任务归一化。student/对应 teacher 使用完全相同的坐标系。
- 原生 clean ALOHA、三相机、14维关节动作；50步预测、每次执行25步、10步 ODE。动作和归一化协议先对齐作者设置，不先压缩推理步数。

| Task / teacher | 原生 horizon | 选取动机 |
|---|---:|---|
| adjust_bottle | 400 | 短程物体调整 |
| handover_mic | 600 | 双臂交接 |
| place_container_plate | 400 | 容器放置 |
| stack_bowls_two | 900 | 接触与堆叠 |

成功率需本地测评；作者公开成绩仅用于筛选，不能作为本地 teacher reference。

## 首轮参数与实验顺序

| 参数 | 初始 pilot |
|---|---|
| 物理轨迹 / round（T） | 64，初始每任务16 |
| Student updates / round（R） | 2，第二次从完整观测池重采样并刷新 flow/noise |
| Global batch（B） | 512，每任务等额128 |
| Microbatch / GPU | 8；8卡、梯度累积，按真实峰值再调整 |
| LR | 2.5e-5，constant，warmup 8次更新 |
| 初始训练预算 | 20 rounds / 40 updates；YAML保留80-update上限，按信号决定延长 |
| 保存 | 每10 rounds / 20 updates |
| Loss | 现有 M3 analytic local KL，同时记录原始逐任务 velocity MSE |
| W&B | `RoboTwin_MultiTeacher_Pilots` |

T32只用于短 smoke：8 ranks下每任务每rank一个slot，没有动态分配余量。T64允许动态调整；每任务至少8轨迹，以保持每rank都能获得该teacher样本。

1. 原生四任务 reset/action、归一化与转换模型正确性检查。
2. 本地 teacher 短评测和 base 参考；首先确认能复现有效 teacher 行为。
3. 四 teacher Uniform-R2 短训练，核验 loss/梯度/刷新/保存。
4. Uniform-R2 pilot。若可行，使用相同预算测试 Adaptive-Sync-R2；根据短评测校准 teacher reference，不能用作者SR代替。
5. 少量调参仅围绕实测瓶颈/信号；不同时扫描大量组合。

解释横轴：updates、rollout rounds、真实episode数、实际执行 control targets、wall-clock均记录。这里 simulation_steps 的单位是执行的控制目标数，不是 SAPIEN 内部积分 substeps。终止后不继续执行动作。

## 隔离与复现

- 根 workspace 保持 `opd-project-main`。
- 开发分支：`dev/mopd-robotwin2-four-task`。
- worktree：`outputs/worktrees/robotwin4`。
- 环境/资产/模型：`/localscratch2/liyifa11/XyFile/opd-robotwin4`。
- 所有实验输出：根目录 `outputs/robotwin4/`（setup / eval / train）。
- 激活：`source /localscratch2/liyifa11/XyFile/opd-robotwin4/env.sh`。
- 训练入口：worktree 中 `python -m toolkits.robotwin_mopd.run --run-id NAME --output-root ROOT/outputs/robotwin4/train`。
- 每run保存配置、source快照、diff、commit、launch命令和独立日志。启动器拒绝已有Ray，不停止他人的进程。

## 进度

| 项目 | 状态 / 证据 |
|---|---|
| 固定独立 runtime、assets | 完成；不修改 RoboCasa runtime |
| 四任务原生 GPU reset/action | 通过 |
| 4个teacher下载 | 完成，源文件hash已校验 |
| 模型转换/跨框架检查 | 5模型完成；adjust velocity相对RMSE约0.08%，10步动作相对RMSE0.13% |
| 蒸馏/刷新/分配单测 | 91 passed；另有3项环境单测通过 |
| Teacher本地episode | 每任务5次：adjust 3/5，handover 4/5，place 5/5，stack 4/5；仅pilot校准 |
| Multi-teacher训练 | FP32 smoke全部通过；T64/B512/R2 pilot运行中 |

物理不稳定 reset 仅按原生异常重试（最多20次），记录requested与actual seed；不按teacher/planner成功与否筛选。评测episode完成后立即写JSON，失败保留部分记录。

训练实现注意：π0.5 在 checkpointed training 下不能沿用推理的 prefix KV-cache路径。pilot使用联合prefix/suffix前向计算给定x_t,t的velocity；teacher评测和rollout仍使用缓存。真实tiny Gemma测试确认预测/梯度一致、prefix有梯度且优化器更新有限。正式pilot改为actor-only FP32可训练参数和BF16 AMP；teacher/rollout维持原生推理精度。旧混合master smoke在完成首轮rollout与均衡采样后主动停止，不作为训练结果。

本地初始参考（小样本）：每任务teacher 5次，SR分别60%/80%/100%/80%；base每任务3次，均0%。reference文件为outputs/robotwin4/eval/teacher_reference_pilot.json。正式报告需要扩展独立评测。

## 当前运行与后续决策

- Run：`robotwin4_uniform_refresh_r2_t64_b512_pilot_20261101_retry`。
- W&B：[f6ve12mw](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/f6ve12mw)。
- 独立launcher PID记录：`outputs/robotwin4/setup/uniform_pilot_active.pid`。
- 使用自有进程NUMA interleave；不改系统设置。107项相关测试通过。
- Smoke两步MSE约1.094/1.125、第二步非零LR3.125e-6；参数快照确认VLM/动作expert/投影均有有限更新。两步不能用于判断学习趋势。
- 首次pilot启动仅因未加载本地W&B认证而失败，尚未训练；retry使用现有机器凭据，已在线同步。
- 后续先检查首两轮loss/梯度/显存及T64采样；完成20轮后独立评测。若仍在稳定学习可延长至40轮；若已出现明显收益与任务差异，再做相同预算Adaptive-Sync-R2。暂不做大规模参数扫描。
