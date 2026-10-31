# RoboCasa365 正式实验交接

给合作者及 coding agent 的执行入口。先完成环境、输入校验和 smoke，再跑四组。
本批次固定版本：**`robocasa-formal-150r-20261031`**。不要跟随开发分支更新，
不要自行改参数、复用旧 pilot 的 optimizer 或省略失败任务。

## 1. Clone 与环境

```bash
git clone --branch robocasa-formal-150r-20261031 https://github.com/zh-xinyu/RLinf.git
cd RLinf
git rev-parse HEAD
```

按 [固定环境入口](../../../OPD_ENVIRONMENT.md) 安装
`gr00t-n15-robocasa365-20261019`，完成软件 verify 和 `--gpu` 检查。
这是本批次唯一环境；不需要 `.opd.actionXX.env`、`shs/` 或 `toolkits/mopd/`。
安装失败保留日志并反馈，不通过升级 Torch/Transformers 解决。

按 [资源说明](RESOURCES.md) 下载 teachers、接收 C10、准备 RoboCasa 资产、校验
权重并生成机器本地的 teacher reference。新机器不重新校准 teacher；四组共享
已冻结的 20 episodes/task 参考，以免引入额外差异。

在仓库外建立自己的可信配置文件，例如 `/absolute/path/robocasa-node.env`：

```bash
export OPD_RUNTIME_ROOT=/absolute/path/opd-runtime
source "$OPD_RUNTIME_ROOT/activate.sh"
export OPD_STUDENT_CHECKPOINT=/absolute/path/C10/checkpoint-60000
export OPD_MODEL_ROOT=/absolute/path/teachers/gr00t_n1-5/foundation_model_learning
export OPD_TEACHER_REFERENCE=/absolute/path/inputs/teacher_reference.json
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
# 如需指定有写入权限的团队，在本地设置 WANDB_ENTITY；不要提交 token。
```

每次启动先 `source` 此文件并进入固定 checkout；移动 checkout 后修正 runtime
`activate.sh` 中的仓库路径。运行 `wandb login` 完成本机认证。
输出默认写在 checkout 的 `outputs/`，可在开始前将其链接到足够大的持久磁盘。
GPU 为单机 8×48GB、专用；另需充足 CPU、内存与检查点磁盘空间。

## 2. 固定实验矩阵

| ID | 物理 rollout allocation | Suite loss weights | R | 更新数 U | 保存间隔 U |
| --- | --- | --- | ---: | ---: | ---: |
| M0 | Uniform | 等权 | 1 | 150 | 25 |
| M1 | Uniform | 等权 | 2，Refresh | 300 | 50 |
| M2 | Adaptive | 与实际配额同步 | 2，Refresh | 300 | 50 |
| M3 | Adaptive | 与实际配额同步 | 1 | 150 | 25 |

共同约定：

- C10 权重 + 全新 optimizer；seed 0；全部 50 tasks，native horizons。
- **150 physical rounds，每轮 T96 = 48 env × 2 epochs**，合计 14,400 条训练轨迹。
- B2304，每 suite B768；8 ranks，每卡 micro32，9 次梯度累积/更新。
- LR3e-5，constant，warmup10 **student updates**；M3 `analytic_local_kl`、4-step
  ODE、chunk16、noise_level0.5 不变。这里 M3 loss 与实验 ID M3 是不同命名。
- R2 使用完整观测池、新 noise、当前 student flow 和新 teacher targets。
- 所有组共用 `suite_quota` collector；M0/M1 配额固定32/32/32。
- Adaptive：rho0.5、epsilon0.01，前20 **rounds** uniform；窗口20 rounds，
  每10 rounds 决策，每 task 至少5 episodes，每 suite 配额最低16，每次最多改变8。
  suite loss 权重 `alpha_s = 3*q_s/96`，本轮 R 次更新保持一致。
- 四组均开启 feature-only GR00T；`cuda_cache_retention_gib=0`，不用共享 GPU pilot
  的18GiB策略。microbatch 改动必须四组同步，不要单独调整一组。

W&B：project `RoboCasa_MultiTeacher_Diagnostics`，group `robocasa_formal_150r`。
第一批同 seed 的四组齐全后，再安排其余 seeds；当前入口固定 seed0。

## 3. 验证与启动

先做 CPU 配置检查（不启动 Ray、不占 GPU）：

```bash
python -m toolkits.robocasa_mopd.run train m2 --run-id check-m2 --dry-run
```

软件与资源都通过后，在空闲的8卡上做两轮完整训练 smoke，并做 C10 单任务评测
smoke。训练 smoke 保留全部50任务的采集器，不缩短任务 horizon；只运行2轮，关闭W&B。

```bash
python -m toolkits.robocasa_mopd.run train m2 --run-id setup-m2-smoke --smoke
python -m toolkits.robocasa_mopd.run eval --run-id setup-c10-eval-smoke --smoke
```

每条命令自动启动/清理自己拥有的 Ray；遇到已有 Ray 或 GPU 忙就退出，绝不 `ray stop`
其他任务。不在同一台机器并行执行两个入口。建议在 tmux 中顺序执行，错误时停止：

```bash
set -e
python -m toolkits.robocasa_mopd.run train m0 --run-id robocasa-M0-uniform-r1-seed0-formal
python -m toolkits.robocasa_mopd.run train m1 --run-id robocasa-M1-uniform-r2-seed0-formal
python -m toolkits.robocasa_mopd.run train m2 --run-id robocasa-M2-adaptive-sync-r2-seed0-formal
python -m toolkits.robocasa_mopd.run train m3 --run-id robocasa-M3-adaptive-sync-r1-seed0-formal
```

可在多台同规格机器各跑一组；启动顺序可按资源安排改变。每组固定参数不变。
训练 interrupted/OOM 后保留目录并反馈，不能当作完整正式 run；恢复需显式审核完整
rollout 边界和交互成本，入口不自动尝试续训或改 batch。

## 4. 独立评测

保存点按 **rounds 25/50/75/100/125/150**。训练入口只保存，不内联评测；可训练
完成后顺序评测，也可复制权重给另一空闲节点评测。不要因尚未评测而删除 checkpoint。

- 初始 C10（round0）共享评测一次，20 episodes/task。
- rounds25–125：每任务20episodes，全部50任务，总计1,000episodes/检查点。
- round150：每任务50episodes，总计2,500episodes。
- 训练外评测成本单列，每组7,500episodes，不含共享C10。

```bash
python -m toolkits.robocasa_mopd.run eval --run-id robocasa-C10-round0 --intermediate

# M2 round25 = U50；M0/M3 round25 = U25。
python -m toolkits.robocasa_mopd.run eval \
  --run-id robocasa-M2-seed0-round25 --intermediate \
  --weights "$PWD/outputs/train/robocasa-M2-adaptive-sync-r2-seed0-formal/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"

# M2 最终 round150 = U300，省略 --intermediate 才是50 episodes/task。
python -m toolkits.robocasa_mopd.run eval \
  --run-id robocasa-M2-seed0-round150 \
  --weights "$PWD/outputs/train/robocasa-M2-adaptive-sync-r2-seed0-formal/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
```

其他中间检查点和实验依此换算 `U=R*round`。20-episode 结果标记
`mode=intermediate` / `OK_INTERMEDIATE`，不能混称最终50-episode结果。
成功率以固定任务集独立评测为准，训练 `success_once` 只用于诊断和调度。

## 5. 验收与交回

保留完整 `outputs/train/<run-id>/`、`outputs/eval/<eval-id>/` 和
`outputs/launches/<run-id>/`。至少交回：

- Git SHA、profile、机器/GPU信息、资源 SHA256、resolved config、W&B URL。
- 每个 run 的 `optimizer_updates.jsonl`、`rollout_episodes.jsonl`、
  `rollout_allocation.jsonl`，全部6个检查点和 runner/allocation state。
- 每个独立评测的 `plan.json`、`summary.json`、`per_task_results.csv`、任务日志和状态。
- 异常/重启记录；不要只交 W&B 截图，不删除失败任务。

主图横轴 **rollout rounds**；同时保留 trajectories、真实 simulation/control steps、
student updates、wall time、teacher calibration成本。相同T/rounds不等于相同物理步数。
报告逐suite SR、suite等权宏平均，并保留50任务宏平均；18/16/16任务数使两者略有区别。
`time/step` 是整轮物理采集+R次更新；每更新摊销时间看 `time/amortized_update`。

## 版本边界

本次已有实现与新入口的验证记录见 [VALIDATION.md](VALIDATION.md)。正式8卡完整
训练仍须在接收机器先做 smoke，再启动长跑。RoboTwin 在新分支/新profile开发，
不修改本 tag、不升级这套 runtime，也不要求与 RoboCasa 共用 Python 环境。
