# 机器人策略的多教师 On-Policy Distillation

Built on [RLinf](https://github.com/RLinf/RLinf) at upstream commit [`e7609d4c`](https://github.com/RLinf/RLinf/commit/e7609d4c9e2f33c5ffc10b67c61c8e4b73208b45)。[English](README.md)

本项目将多个任务专家 VLA 的能力融合到一个 flow-based student 中。当前使用
GR00T N1.5、RoboCasa365 三个 suites（atomic seen、composite seen、composite
unseen），共 50 个操作任务、3 个 specialist teachers；student 从使用 10% 数据
联合 SFT 得到的 C10 checkpoint 出发。

研究两个互补机制：

- **Adaptive rollout allocation + 同步 suite loss weighting**：根据 teacher–student
  能力差距，调整物理经验分配和优化权重。
- **Decoupled flow refresh**：复用物理 rollout 的观测池，用更新后的 student
  重新生成 flow trajectories 并查询 teacher。

## 入口

| 目的 | 文档 |
| --- | --- |
| 安装当前固定环境 | [GR00T/RoboCasa365 profile](OPD_ENVIRONMENT.md) |
| 合作者或 coding agent 跑四个正式实验 | [交接与操作指南](docs/opd/collaboration/README.md) |
| 准备并校验模型和仿真资产 | [资源说明](docs/opd/collaboration/RESOURCES.md) |
| 理解训练实现与指标 | [开发说明](OPD_DEVELOPMENT.md) |
| 独立评测 | [评测协议](evaluations/robocasa365/FORMAL_EVAL.md) |

固定实验 tag：`robocasa-formal-150r-handoff-20261031`；`opd-project-main` 继续作为开发基线。
正式配方使用单机 8×48GB GPU；四组均从 C10 开始，每轮 96 条轨迹，训练 150 轮，
每 25 轮保存。完整参数和结果交付要求见交接指南。

模型和仿真资产另行准备。个人机器路径、凭据、原始运行数据与研究草稿不作为安装
前置条件。RoboTwin 将另开分支和环境 profile；当前 profile 仅覆盖 GR00T/RoboCasa365。

## 统一复现入口

四组训练 YAML 已提供；合作者只需复制
[machine.example.json](toolkits/robocasa_mopd/machine.example.json) 到仓库外，
一次填写机器名、路径与 W&B team。之后全部使用
`python3 toolkits/robocasa_mopd/experiment.py --machine /path/to/machine.json <command>`。
完整顺序是安装、资源、验证、训练、评测、汇总；无需分散设置 export 或手动激活环境。

除本 README 外，项目操作指南使用英文。`docs/` 中仍保留 RLinf 上游文档，暂未全部
适配本项目；当前实验以本页链接的 OPD 指南为准，后续再统一整理。

## 代码位置

- `rlinf/`：分布式训练、仿真、模型适配与蒸馏。
- `examples/embodiment/config/robocasa365_mopd_formal_*.yaml`：正式配置。
- `toolkits/robocasa_mopd/`：可移植的资源校验与启动工具。
- `evaluations/robocasa365/`：固定任务清单和独立评测。
- `requirements/profiles/`：版本化软件环境。
- `tests/unit_tests/`：协议、loss、allocation、refresh 和显存优化回归。

## 来源与许可

保留 RLinf 的 [Apache-2.0 许可证](LICENSE) 和上游版权声明。外部 checkpoints、
仿真资产和第三方软件分别遵循各自许可。开发规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。
