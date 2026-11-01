# RoboTwin 仿真性能检查（2026-11-01）

当前仅做只读源码检查、运行中栈采样解读和 CPU 微基准。**尚未证明任何优化能加速真实训练 rollout**；本次没有修改生产代码，没有启动模型、模拟器或 GPU 作业。当前自动 pilot 保持原控制语义。

## 已有证据与边界

根 workspace 的 `outputs/robotwin4/analysis/specialist_step0_trainpath_t16_20261101/env_worker_stack.txt` 保存了 env worker 3557338 的一次快照：一个线程在 TOPPRA solver discovery 的失败 import/path stat，另一个在 `_entity_qf`。这说明这些路径确实执行，**不是耗时占比 profile，也不能据此排序主要瓶颈**。

源码位置如下；外部行号对应本机固定安装，未来依赖更新后可能变化：

| 来源 | 位置与含义 |
|---|---|
| worktree | `rlinf/envs/robotwin/multitask_env.py::reset/chunk_step/NativeTaskSlot.step_chunk`：串行 reset，线程池执行各 slot 原生动作，终止 slot 冻结 |
| worktree | `rlinf/workers/env/env_worker.py`、`rlinf/workers/rollout/hf/huggingface_worker.py`：固定 rollout chunk 数 |
| RoboTwin | `envs/_base_task.py:1538`：`take_action`；`:426` 更新渲染；`:462` 获取观察 |
| RoboTwin | `envs/robot/robot.py:105`：Aloha 双臂共享 articulation；`:639` passive force；`:643/:662` arm/gripper setters |
| MPLib | `mplib/planner.py:351`：每次 TOPP 构造 Spline、约束和 TOPPRA，再按 0.004 s 采样 |
| TOPPRA | `toppra/algorithm/reachabilitybased/reachability_algorithm.py:63` 无条件检查 solver；`solverwrapper/solverwrapper.py:9` 重复尝试 imports |

RoboTwin 根目录：`/localscratch2/liyifa11/XyFile/opd-robotwin4/repos/RoboTwin`，固定 commit `dca9ec682688821c42944eacd7f5dd9bfb15396f`，含项目记录的本地适配。Python 依赖位于同 runtime 的 `venv/lib/python3.11/site-packages/`。

## CPU 微基准：缓存 solver discovery

在同一 runtime、`CUDA_VISIBLE_DEVICES=''`、OMP/OpenBLAS 各1线程下，单独进程测试，设28秒超时，实际约2秒结束。只在该进程内临时替换 discovery 函数，没有修改安装文件。

- 可用列表：`seidel=True`，`hotqpoases/qpoases/ecos/cvxpy=False`。缺失模块重复 import 会触发路径查找。
- 200次 discovery：1线程 **46.20 ms**；4线程 **125.51 ms**。
- 完整 TOPP 测试采用6关节、两点 Spline、单位速度/加速度限制、`ParametrizeConstAccel`，包含轨迹求解和0.004 s的位置/速度/加速度采样；每组20条，重复两轮。

| 20次 TOPP 总 wall time | 原实现，两轮 | 缓存 discovery，两轮 |
|---|---:|---:|
| 1线程 | 110.20 / 112.23 ms | 106.82 / 106.53 ms |
| 4线程 | 154.89 / 154.12 ms | 150.99 / 150.25 ms |

对应平均改善约 **4.1% / 2.5%**，仅适用于此热缓存、短轨迹微基准。保存于内存的原实现与缓存输出 duration/q/qd/qdd 逐元素一致。4线程比1线程慢，提示短 TOPP 任务存在 Python/线程调度争用；不能直接推广为整个仿真线程池应串行化。

未来可在依赖固定的进程启动后缓存可用 solver 列表，保留 solver 选择顺序与 `output_msg` 行为。仅指定 `solver_wrapper='seidel'` 不会跳过当前无条件 discovery。进程运行中安装/修改依赖时需重新初始化缓存。

## 其他候选优化：尚待验证

- **Passive force 去重。** Aloha 的 `left_entity is right_entity`，每个 arm/gripper setter 却分别调用两次 `_entity_qf`。正常双臂 substep 最多执行8次相同 articulation 的 passive-force 计算，随后才 `scene.step()`；中间只调整 drive targets。先考虑每个 setter 内按实体 identity 去重，理论调用数最多8→4；进一步每物理 substep 一次可到8→1。不能跨 `scene.step()` 缓存，因为 q/qdot 已改变。尚无该部分计时或动力学等价性验证。
- **仅在需要观察时更新渲染。** 当前每 target 开头、每物理 substep、target 结束都会 `_update_render()`；`get_obs()` 本身也会刷新 wrist camera、scene 并取图。clean、无视频配置可能减少中间刷新，但必须保证实际观察前刷新；随机光照、视频和其他可能依赖 render 的路径不能直接套用。尚未证明删除中间调用不会改变结果。
- **冻结 slot 的无效推理。** 四任务 horizon 为400/600/400/900，固定900/25=36轮整 batch 推理。等量任务、均无提前成功时，名义 slot-target 中只有63.9%对应真实执行，其余至少36.1%为终止 padding；成功更早时浪费更多。这不是36.1%的整体 wall-time 可节省承诺。
- **reset 与重建。** 每 round 串行 reset 各 slot，关闭时清理 renderer/cache；同时每 target 双臂 TOPP 的物理 substep 数可变，control targets 相同不意味着仿真耗时相同。

## 后续最小验证顺序

先记录 reset、policy inference、TOPP、passive force、physics、render 的累计时间与调用数，再选择占比最大的项。单次只改一个优化：固定 scene seed、prompt、动作序列和执行顺序，比较每一步 qpos/qvel、物体姿态、终止/成功时刻，以及观察点 RGB；再进行同 noise 的闭环 teacher 对照并报告容差。Passive force 去重与 render 延迟都必须通过轨迹/观察等价性检查，再讨论真实 rollout wall-time 收益。当前 pilot 不应用这些候选修改。
