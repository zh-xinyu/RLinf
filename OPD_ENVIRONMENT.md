# OPD 固定环境

本页及 `requirements/profiles/` 随 `opd-project-main` 版本化。
新 clone 使用这里的入口；不需要 `.opd.*.env`、`shs/`、`toolkits/mopd/` 或
个人研究草稿。正式实验入口见 [合作者交接指南](docs/opd/collaboration/README.md)。

## 默认 profile

**`gr00t-n15-robocasa365-20261019`**：GR00T N1.5 + RoboCasa365，
面向 Linux x86_64 / NVIDIA GPU。Python 3.11.14、Torch 2.6.0（CUDA 12.4、
旧 C++ ABI）、FlashAttention 2.7.4.post1、Transformers 4.51.3、
MuJoCo 3.3.1、NumPy 2.2.5、Ray 2.56.1。

| 文件 | 固定的内容 |
| --- | --- |
| [profile.json](requirements/profiles/gr00t-n15-robocasa365-20261019/profile.json) | Python、uv、4个上游仓库的完整 commit、FlashAttention wheel / ABI |
| [requirements.txt](requirements/profiles/gr00t-n15-robocasa365-20261019/requirements.txt) | 已运行环境的275个 Python 包精确版本 |
| [build-constraints.txt](requirements/profiles/gr00t-n15-robocasa365-20261019/build-constraints.txt) | 源码包构建工具的独立版本约束 |
| [opd_profile.py](requirements/opd_profile.py) | 全新安装、版本与源码检查、关键导入、可选 GPU 检查 |

这是已运行环境的**版本快照**，不是最小依赖集。上游 GR00T/RoboCasa/LeRobot
的依赖声明相互冲突，历史安装也使用了 `--no-deps` 和版本覆盖；此入口显式恢复
快照，避免重新解析后悄悄改变训练环境。`uv pip check` 会报告这些已有冲突，
不能据此称整个上游软件栈的全部功能都兼容。检查范围是本项目的 GR00T/RoboCasa365 路径。

普通 `requirements/install.sh embodied --model gr00t --env robocasa365`
仍是 RLinf 的通用安装器，**不承诺恢复这个固定 profile**。

## 从 clone 安装

前置条件：Linux x86_64、Git、编译工具及 EGL/OpenGL 系统库；GPU 检查和训练需要
支持 CUDA 12.4 的 NVIDIA 驱动。当前参考机器为 Ubuntu 24.04。
系统依赖由已有的 `requirements/sys_deps.sh nvidia` 安装（需要管理员权限）；
profile 安装器不自动修改系统包。需能访问 PyPI、GitHub 和 Python 下载源。

```bash
git clone --branch opd-project-main https://github.com/zh-xinyu/RLinf.git
cd RLinf

# bootstrap 与训练 runtime 分开；也可使用已经安装的相同版本 uv。
python3 -m venv /absolute/path/opd-bootstrap
/absolute/path/opd-bootstrap/bin/python -m pip install 'uv==0.12.0'
export PATH="/absolute/path/opd-bootstrap/bin:$PATH"

# 必须指定尚不存在的绝对目录。dry-run 不写入文件。
python3 requirements/opd_profile.py install \
  --runtime /absolute/path/opd-runtime --dry-run
python3 requirements/opd_profile.py install \
  --runtime /absolute/path/opd-runtime
source /absolute/path/opd-runtime/activate.sh
```

安装器使用独立的 Python 下载目录、uv cache、venv 和源码 checkouts；不读取
个人节点配置，不覆盖已有 runtime。失败后保留目录与终端错误用于排查；修复后
选择新的空路径重试。不要对已有训练环境运行升级或重新解析依赖。

安装结束检查全部固定包版本、4个源码 commit / 已跟踪文件修改、实际安装来源、
Torch CUDA/ABI、关键模块和本地 Eagle processor。`activate.sh` 绑定生成时的
RLinf checkout，启动 Ray **之前** source；移动 checkout 后需调整该路径。

```bash
# 可重复运行的软件检查（从 clone 的仓库根目录）
PYTHONPATH="$PWD" /absolute/path/opd-runtime/venv/bin/python \
  requirements/opd_profile.py verify --runtime /absolute/path/opd-runtime

# 在空闲 GPU 上增加一次真实 FlashAttention CUDA 运算
PYTHONPATH="$PWD" /absolute/path/opd-runtime/venv/bin/python \
  requirements/opd_profile.py verify --runtime /absolute/path/opd-runtime --gpu
```

## 软件之外的实验输入

clone 不包含 RoboCasa 大型资产、C10 student 或三个 teacher checkpoints。
它们必须另行准备，不能把“软件安装通过”等同于“完整实验已复现”。

1. 在新环境中按固定 RoboCasa checkout 的说明准备资产：

   ```bash
   python -m robocasa.scripts.setup_macros
   python -m robocasa.scripts.download_kitchen_assets
   ```

   复现已有实验时优先使用同一份已归档资产，并记录校验和；上游下载服务内容
   不由本 profile 锁定。下载会占用额外磁盘空间。
2. 提供 `OPD_STUDENT_CHECKPOINT`（C10）和 `OPD_MODEL_ROOT`，后者包含
   `target_posttraining/{atomic_seen,composite_seen,composite_unseen}/checkpoint-60000`。
   每个 checkpoint 保留权重、`config.json`、`experiment_cfg/metadata.json` 等完整目录。
3. 设置输出路径、GPU 分配、W&B 认证，再按
   [训练入口](OPD_DEVELOPMENT.md#在已配置的单机环境运行)启动。
   正式比较时记录 RLinf commit、profile ID、checkpoint 与资产校验和，以及最终 Hydra config。

个人已有服务器仍用显式 `.opd.action00.env` / `.opd.action10.env` 指定路径和节点。
这些文件与通用 profile 分离，不进入 GitHub，也不作为新机器安装的前置条件。

## 验证边界与更新规则

2026-10-19：全新隔离安装、软件检查和公开 checkout 的 **76 项 CPU 回归通过**。
详细范围见 [验证记录](requirements/profiles/gr00t-n15-robocasa365-20261019/VALIDATION.md)。
本轮 GPU 正忙，没有中断训练或执行 GPU 端到端试验。

- 软件版本与 Git commit 固定；操作系统、驱动及外部资产不是完整镜像锁定。
- Python 包固定版本，当前未提供所有构建产物的 SHA256 清单或离线 wheelhouse；
  不承诺逐字节或数值完全相同。公共源下架会使安装明确失败，不自动升级版本。
- 此 profile 更新应创建新的日期目录，保留旧版本供实验追溯；同步修改默认入口并
  在空目录重跑安装、软件检查、GPU smoke 和项目回归。
- 检查通过只覆盖明确执行的检查，不替代完整 rollout / 优化更新验证。
