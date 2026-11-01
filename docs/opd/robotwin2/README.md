# RoboTwin 2.0 四任务 MOPD 扩展

更新：2026-10-20。优先级低于 [RoboCasa rollout allocation](../adaptive_rollout_allocation/RoboCasa_MOPD_Adaptive_Rollout_Allocation.md)。
状态（2026-11-01）：已从 `opd-project-main@5d1d1494` 建立 `dev/mopd-robotwin2-four-task` 分支（当前根workspace）。尚未安装环境、下载权重、冻结四任务或启动训练。公开checkpoint调查见 [CHECKPOINT_SURVEY_20261101.md](CHECKPOINT_SURVEY_20261101.md)。

## 1. 目标与隔离方案

四个双臂操作tasks，各有对应specialist，一个共享student。保持task→teacher固定路由，先建立uniform MOPD，再测试自适应rollout allocation。不要求长任务或teacher近100%成功；需要teacher相对共同student初始化有可测量的能力优势和融合空间。

| 内容 | 建议 |
| --- | --- |
| 起点 | 当前 `opd-project-main`，不从旧probe/Cached分支起步 |
| RoboTwin开发 | 当前根workspace的 `dev/mopd-robotwin2-four-task`（已建立） |
| RoboCasa方法开发 | 另开 `dev/mopd-adaptive-rollout-allocation`（拟议名称） |
| Python runtime | 独立RoboTwin runtime / 固定profile，不向现有RoboCasa runtime追加包 |
| 节点配置 | 通用依赖与action00/action10本地assets、checkpoint路径分别记录 |
| 后续合并 | 适配、通用allocator与配置可合回主线；benchmark继续使用不同profile |

**同一代码主线不要求只有一个Python环境。** 分支隔离控制开发风险，runtime隔离控制依赖风险。
RoboTwin官方安装文档列出Python3.10、推荐CUDA12.1；现有RoboCasa固定profile为Python3.11 / Torch CUDA12.4。这不证明绝对不兼容，但建议先独立验证。[官方安装指南](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

## 2. 当前代码基础与缺口

核对基于 `opd-project-main@fdde9522`：

- 已有 [RoboTwinEnv](../../../rlinf/envs/robotwin/robotwin_env.py)、reset seed工具和若干单任务配置；环境读取单一`task_config.task_name`。
- 已有OpenPI / OpenVLA-OFT等配置，**没有当前GR00T N1.5 × RoboTwin的完整配置与observation adapter**。[GR00T simulation IO](../../../rlinf/models/embodiment/gr00t/simulation_io.py)的转换注册不含RoboTwin。
- 当前observation含图像、state、指令，尚未输出多teacher router要求的`task_groups`。
- 四任务混合执行、task identity传递、每个actor的本地任务覆盖需适配；已有单任务env不等于已支持四任务MOPD。
- [安装脚本](../../../requirements/install.sh)的`install_robotwin_env`包含SAPIEN / MPLib修改和cuRobo等依赖；GR00T安装组合目前没有RoboTwin分支。不能当作已验证的可复现GR00T-RoboTwin profile。
- 主线完整观测池Refresh当前限定GR00T N1.5。换Pi0.5不能假定Refresh直接可用。

## 3. 开始前固定的设置

1. **模型与checkpoints**：先确认四个specialists和共同student初始化是否可用。沿用GR00T有利于与RoboCasa对照，但需额外适配；是否用Pi0.5由资源和验证成本决定。
2. **动作/观测约定**：四个teachers使用相同embodiment、action参数化和归一化坐标。不同SFT统计量可能导致velocity坐标不一致，不能直接MSE。
3. **任务**：先看teacher可用性、运行稳定性和学习空间，再覆盖不同双臂交互。已有`handover_block`、`lift_pot`、`pick_dual_bottles`、`place_container_plate`单任务配置可作候选，不代表已验证或最终选定。
4. **数据/评测**：固定clean/randomized设置、初始化数据预算、各task horizon、训练/校准/最终评测reset分割和seed清单，不按新方法效果挑任务。
5. **预算**：先测rollout成本再定T/B/R；四task不照搬三suite的局部整除规则，确保每actor有所有激活teacher的有效数据。

## 4. 落地顺序

1. 独立runtime验证单task teacher inference、环境step、成功率和格式。
2. 跑通单task OPD，确认loss、mask与teacher动作坐标，再验证Refresh。
3. 四任务uniform MOPD，建立共同student与specialists对照。
4. 复用RoboCasa验证过的allocator接口，每个task成为一个group。
5. 整理固定安装profile与测试，再决定合并范围。

不因另开分支长期复制两套actor/loss/refresh；通用逻辑共享，benchmark差异放在环境适配与配置中。

评测协议：[筛选协议核对](EVALUATION_PROTOCOL_AUDIT.md)、[固定场景补充评测](FEASIBLE_EVALUATION.md)。训练进度和评测队列见 [LIVE_STATUS](LIVE_STATUS.md)。
