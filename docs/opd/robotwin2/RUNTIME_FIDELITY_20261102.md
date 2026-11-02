# RoboTwin4 渲染版本修正（2026-11-02）

当前四任务 profile 必须使用 **SAPIEN 3.0.0b1**。此前独立安装器固定了
3.0.1，导致腕部相机图像变化；不能把该版本下的旧 pilot 当作教师能力已经
正确复现的证据。此修正只涉及 RoboTwin worktree/profile，RoboCasa 环境不变。

## 已验证的因果链路

对 adjust_bottle 的三个固定场景 `100001/100008/100012`，原环境与本地的
物体位姿、完整机器人 qpos、相机内外参都逐元素一致。3.0.1 的腕部 RGB
与原环境 RMSE 约为 24/255。仅将同一 Python3.11/runtime 的 SAPIEN 导入
替换为 3.0.0b1，三场×三相机的九张图像全部逐像素一致；代码、MPLib、
SciPy、模型权重和归一化均未改变。

同十个作者缓存场景、固定 unseen 指令和逐查询噪声的闭环诊断：

| 路径 | SAPIEN | 成功数 |
|---|---|---:|
| 本地环境，原 JAX 教师 | 3.0.1 | 8/10 |
| 本地环境，转换后的 Torch 教师 | 3.0.1 | 8/10 |
| 本地环境，仅替换 SAPIEN，Torch 教师 | 3.0.0b1 | 10/10 |
| 原作者环境，原 JAX 教师 | 3.0.0b1 | 10/10 |

前两组逐场结果一致；更换 SAPIEN 后救回 `100008/100012`。这是有界修复
验证，不是正式 benchmark 分数，也没有在本地 MPLib 环境重新验证 expert
筛选协议。旧实验结果保留，需标记旧渲染版本。

证据保存在根 workspace 的 `outputs/robotwin4/analysis/teacher_reaudit_20261102/`：

- `sapien_snapshot_ablation_20261102.json`：几何与 RGB 比较。
- `cached_localenv_rlinf_sapien300b1_10/results.json`：版本修正后的闭环结果。
- `cached_originalenv_jax_explicit_noise_10/results.json`：匹配原环境对照。
- `scene_100008_rendering_comparison.png`：相机图像对照。

## 安装与运行约定

新安装使用隔离 `setup/runtime-pins.txt` 中的 `sapien==3.0.0b1`，不修改
通用 `requirements/embodied/envs/common.txt` 或其他 benchmark profile。
已有训练运行时，不替换它的 site-packages。可把对应 Python ABI 的官方
wheel 用 `pip install --no-deps --no-index --target NEW_VENDOR_DIR WHEEL`
装到新目录，再通过 `OPD_ROBOTWIN_SAPIEN_VENDOR` 为新进程选择它。

当前验证过的 cp311 wheel SHA256：
`d9497375c9dc26f07b82ba08f347f7ff5df063c057d2744aec0521568ffeb096`。
vendor 下 `opd_sapien_manifest.json` 记录来源与验证证据。其他 Python ABI
必须使用对应 wheel，不能复用 cp311 二进制。

`write_environment.sh` 生成的 profile 设置 `OPD_ROBOTWIN_PROFILE=pi05-four-task`，
可发现本机已验证 vendor，并向新子进程传播其优先导入路径。激活后必须
启动新的 owned Ray，不能继续使用捕获了旧环境的 Ray 实例。

四任务 env YAML 的 `required_runtime_profile` 是静态要求。环境 worker
在创建仿真之前验证实际 `sapien.__version__`、distribution version、可选
vendor 路径和 RoboTwin git revision；没有 profile 环境变量也不能跳过。
专用 launcher 也在启动之前检查，并保存 `runtime_profile.json`。
源码 hash 用于记录现有受控适配；它不等价于声明 checkout 无修改。

回归检查覆盖：正确版本/来源、实际模块和 metadata 不一致、vendor 被
遮蔽、错误源码版本、缺失环境变量时静态要求仍生效、通用环境无额外导入，
以及生成 profile 后子进程实际从 vendor 导入。真实图像和十场闭环证据
保留为 GPU 集成验证；CPU 单测不伪造成功率。
