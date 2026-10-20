# Base MOPD 与 Refresh 开发入口

2026-10-19：`opd-project-main` 整合现有、已运行过实验的 Base MOPD 与
Refresh 实现，作为后续方法开发的起点。

Base/Refresh 整合保留在 `opd-project-main`；2026-10-20 的 adaptive 开发位于
`dev/mopd-adaptive-rollout-allocation`。本页是版本化的代码使用说明；共享工作区
的研究文档、实验记录与汇报材料统一从 [文档导航](docs/opd/README.md) 进入。
`docs/opd/` 为本地研究资料，单独clone时可能不存在；训练与评测不依赖该目录。

## 来源与保留范围

- 快进纳入 `dev/mopd-groot-robocasa365` 截至 `a3761fd5` 的9个提交：
  GR00T FP32 action-head 训练、checkpoint-specific 状态归一化、suite-balanced
  采样、完成 episode 的冻结处理、LR schedule 与运行路径整理等。
- 纳入评测提交 `628a5ae3`；它与 event-weighting 分支的 `4f0a5017`
  patch 相同。评测标准见 [冻结评测协议](evaluations/robocasa365/FORMAL_EVAL.md)。
- Refresh 核心来自 `dev/groot-robocasa365-decoupled-refresh` 的现有工作目录。
  已与 `multi_teacher_diagnostics` worktree 对照：完整观测池的 Refresh
  采样与 teacher 查询路径相同；后者额外的 Cached、fixed-batch probe 不进入主线。
- 纳入诊断版本中的 `time/rollout_round`、`time/amortized_update` 和 W&B group 支持。
- 保留原 M3 `analytic_local_kl`、suite hard router 和 teacher 坐标适配；未改 loss。
- Event weighting、Cached、固定观测 probe、SOPD 专用实验配置继续保存在原开发
  分支/worktree。本次不删除或重写这些目录，不包含新的训练实验。

## 推荐入口

| 配置名（`examples/embodiment/config/`） | T | B | R | 总更新数 | 保存间隔 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `robocasa365_mopd_base_t96_controlled_gr00t_full` | 96 | 2048 | 1 | 200 | 40 |
| `robocasa365_mopd_refresh_r2_t96_gr00t_full` | 96 | 2048 | 2 | 200 | 40 |

两者仅 `algorithm.flow_refresh_updates` 不同。共同使用 C10（显式提供路径）、
3个 suite specialists、LR3e-5、constant、warmup10 student updates、micro32、
8 actor ranks、4-step ODE、noise_level=0.5、原 task horizons。

T96 = 48 env × 2 rollout epochs；每 suite 32 个名义轨迹槽位。
B2048 沿用现有按 rank 的 suite 配额；需要严格每 suite B768 时，两组都设
`actor.global_batch_size=2304`。T/B/LR 应在对照之间同步修改。

R 可通过 `algorithm.flow_refresh_updates=4` 等覆盖；保持 B、LR 和总更新数相同。
总更新数、保存间隔及启用的评测间隔必须能被 R 整除。

### 保留的历史入口

- `robocasa365_mopd_gr00t_full`：原 Base T96/B8192/warmup3 配方，100轮，save25。
- `robocasa365_mopd_refresh_r2_gr00t_full` 与
  `robocasa365_mopd_base_controlled_gr00t_full`：T48/B2048/warmup10，200 updates，save20。
  本次将 Base 的 W&B project 与 Refresh 统一为 `Formal_Robocasa_Groot_MOPD`。
- `robocasa365_mopd_refresh_gr00t_full`：早期 R4/T96/B8192/warmup3，
  **50个物理轮次 = 200次更新**，save10物理轮次。此入口保留历史计步方式，
  新实验优先使用上面的 optimizer-update 入口。

## Refresh 的训练语义

1. 用当前 rollout policy 采集物理轨迹，保留本轮全部有效观测。
2. 首次更新使用 rollout 时的 flow states 和对应 teacher targets。
3. 后续 R−1 次更新从完整池重新进行 suite-balanced 抽样；使用更新后的
   student 与新 noise 生成 ODE flow，重新抽时间步并查询对应 teacher。
4. 每次 inner update 恰好更新 optimizer 一次；R次完成后同步权重并采集下一轮。

重复抽到的观测保留为不同样本，使用独立 noise/flow draws；有效 mask 排除
padding/已终止片段。刷新时关闭梯度，经过正常 FSDP `__call__` 入口；
controlled 模式隔离 flow RNG，避免刷新消耗训练 RNG。

## 指标与 checkpoint

- `runner.step_unit=optimizer_update`：W&B training loss、warmup、保存频率和
  `runner.max_steps` 都以 student 更新计数。
- `optimizer_updates.jsonl`：每次更新单独保存；`train/new_rollout_velocity_mse`
  仅记录每轮第一次更新。
- `rollout_count` 与 `rollout_success/*`：成功率按物理轮次记录。
- `time/step` 和 `time/rollout_round`：一轮物理采集 + R次更新的耗时。
- `time/amortized_update`：整轮耗时除以 R。
- checkpoint 带 `runner_state.json`，记录更新数、轮次与环境步数。
- 当前记录用于追溯；**optimizer-update 模式不支持精确 resume**，设置
  `runner.resume_dir` 会被拒绝。它需要同步 runner，不能使用 pipeline 模式。

`flow_reuse_mode=cached`、`flow_observation_sampling=fixed_batch`、启用
`flow_probe` 在主线会明确报错，防止旧实验配置被悄悄解释为 Refresh。

## 在已配置的单机环境运行

新 clone 请先按 [固定环境安装说明](OPD_ENVIRONMENT.md) 安装版本化 profile。

先进入本分支的 checkout，激活已有 GR00T/RoboCasa365 runtime，并设置实际路径：

```bash
export REPO_PATH="$PWD"
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export PYTHONPATH="$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export OPD_MODEL_ROOT=/absolute/path/to/foundation_model_learning
export OPD_STUDENT_CHECKPOINT=/absolute/path/to/C10/checkpoint-60000
export OPD_RUN_DIR=/absolute/path/to/new-run-directory
export MUJOCO_GL=egl
export RLINF_NODE_RANK=0

python examples/embodiment/train_embodied_agent.py \
  --config-name robocasa365_mopd_refresh_r2_t96_gr00t_full \
  'hydra.run.dir=${oc.env:OPD_RUN_DIR}' \
  runner.logger.experiment_name=refresh-r2-pilot \
  +runner.logger.wandb_group=mopd-refresh
```

Base 使用表中的 Base config 名称。W&B 默认 project 为
`Formal_Robocasa_Groot_MOPD`，可用 `runner.logger.project_name=...` 覆盖；
认证沿用本机配置，配置文件不包含凭据。

路径与 runtime 必须在启动 Ray 前准备，集群只使用本次分配的节点/GPU。
本地 `toolkits/mopd/` launcher、凭据和研究文档仍按既有规则 Git ignored，
不会随新 checkout 自动出现；上述入口直接使用版本化的训练脚本。

## 主线整合验证范围（2026-10-19）

CPU regression tests 覆盖：有效观测池采样、重复样本的独立 flow 与 teacher
labels、当前 student 参数、新 noise、RNG 隔离、Base 单次更新、R2/R4
更新数/rollout数/保存位置、逐更新日志、W&B group、配置限制、GR00T精度、
环境 episode 生命周期和冻结评测配置。

本次只做已有实现的整合与 CPU 验证，没有重跑完整 GPU 训练。
历史 GPU 证据来自原 worktree 的训练；原始 runs 和启动快照保留在共享 `outputs/`。


本次整合核验：**70 passed**，Ruff lint / format 通过。回归命令：

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/unit_tests/test_flow_refresh.py \
  tests/unit_tests/test_mopd_config.py \
  tests/unit_tests/test_mopd_wandb_logging.py \
  tests/unit_tests/test_flow_opd.py \
  tests/unit_tests/test_gr00t_n1d5_compute_dtype.py \
  tests/unit_tests/test_gr00t_n1d5_precision.py \
  tests/unit_tests/test_robocasa365_lifecycle.py \
  tests/unit_tests/test_robocasa365_formal_eval.py
```

## Adaptive rollout allocation pilot（2026-10-20）

入口：`robocasa365_mopd_adaptive_r2_gr00t_full`。T96、B2304、Refresh-R2、
200 student updates = 100 physical rounds = 9,600 trajectories；LR3e-5、constant、
warmup10 updates、save40。W&B project 为 `RoboCasa_MultiTeacher_Diagnostics`，
group 为 `adaptive_rollout_pilot`。`T` 是一轮 episode 槽位总数，不是固定仿真步预算。

- 前20轮均匀分配；此后每10轮，用最近20轮的 **task-macro** 成功率计算各 suite
  到对应 teacher 的非负能力差距。每任务至少5条，否则保留旧配额。
- `gap_s = max(teacher_success_s - student_success_s, 0)`；
  `p_s = (1-rho)/3 + rho*(gap_s+epsilon)/sum_j(gap_j+epsilon)`，
  `rho=0.5, epsilon=0.01`。整数配额和为96，每次每 suite 最多变化8。
- 8 workers × 2 epochs 的每个本地池至少保留各 suite 一条，因此每 suite 最低16条。
  suite 内按 task round robin；用 task occurrence 构造 reset seed，匹配不同分配的
  同任务初始状态序列。不同并行布局下不承诺相同 action noise。
- 仅改变新物理经验的分配；optimizer 的 suite-balanced B768/768/768、M3 loss、
  hard router、完整观测池和 flow 刷新均保持原逻辑。Signed progress 只记录，不参与调度。
- 首版禁用 bootstrap overlap，使用单 rollout stage、group_size=1、原生 task horizons。
  对照用同一入口设置 `runner.rollout_allocation.mode=uniform`，需要 R1 时再设置
  `algorithm.flow_refresh_updates=1`，避免新旧采集器差异混入 allocation 对比。

### Teacher reference 与启动

先按上面的运行环境设置 checkpoints、资源和 Python 路径。校准使用相同模型推理、
观测/动作转换和原生 horizon；每 task 20 episodes，独立的 reset-seed 序列：

```bash
python evaluations/robocasa365/calibrate_teachers.py \
  --output /absolute/path/to/new-calibration-directory \
  --gpus 0,1,2,3,4,5,6,7 --episodes 20 --parallel-envs 5

export OPD_TEACHER_REFERENCE=/absolute/path/to/new-calibration-directory/reference.json
export OPD_RUN_DIR=/absolute/path/to/new-training-directory
python examples/embodiment/train_embodied_agent.py \
  --config-name robocasa365_mopd_adaptive_r2_gr00t_full \
  runner.logger.experiment_name=adaptive-r2-T96-B2304-pilot
```

校准和训练顺序执行；校准 JSON 必须 COMPLETE、task set/horizons、checkpoint 路径与
有效 rollout 模型配置一致。校准成本单列，并在报告总交互成本时加回；这些轨迹不训练
student，也不作为最终独立评测。低样本 teacher reference 本身存在估计误差。

### 交互计数与追溯

| 字段 / 文件 | 含义 |
| --- | --- |
| `update_step` | student optimizer update；warmup 和 checkpoint 同此单位 |
| `interaction/rollout_rounds` | 完成的物理 rollout rounds |
| `interaction/rollout_trajectories` | 实际完成的 episodes；当前配置每轮96条 |
| `interaction/simulation_steps` / `interaction/control_steps` | 真正执行的环境控制步；不含终止后的填充、reset settling 或 MuJoCo 内部积分子步 |
| `interaction/env_steps` | 同上，保留已有曲线兼容性 |
| `interaction/total_simulation_steps` / `total_trajectories` | 训练累计数加上 teacher 校准；uniform/static 为纯训练成本 |
| `calibration/*` / `teacher_reference.json` | 校准轨迹数、实际步数、wall time 与参考快照 |
| `rollout_episodes.jsonl` | 每条轨迹的 task、suite、round、worker/slot、seed、policy version、success、实际步数 |
| `rollout_allocation.jsonl` | 每轮配额、决策依据与累计预算 |
| `rollout_allocation_state.json` | checkpoint 中的窗口、task cursors、计数；用于追溯，不支持精确 resume |

每轮环境实际步数必须与 actor 的有效 control mask 总数一致，否则报错。
第二次 flow 更新不增加交互计数。动态配额会改变 episode 长度，所以 **相同 T 和轮数
不代表相同 simulation steps**；需分别绘制效果对 student updates、实际环境步、wall time。
训练窗口成功率是自适应策略的输入，不能代替固定协议独立评测。
