# Robotwin 独立评测统一到训练 worker 链路

状态：27 项 CPU 检查通过；action20 同场景逐 chunk 对照通过；四任务各 50 episodes 复测进行中（实时状态以输出目录的 `status.json` 为准）。

目标是复用 MOPD 的实际执行代码：`EmbodiedEvalRunner → MultiStepRolloutWorker / EnvWorker → RoboTwinMultiTaskEnv`。不修改训练环境的采样分布，也不改变原定的 held-out 场景、unseen 指令、native horizon、H25 或 10-step flow ODE。

- 分支：`experiment/robotwin2-task-init-probe`；worktree：`outputs/worktrees/robotwin_task_init_probe/`。
- 新入口：`toolkits/robotwin_mopd/worker_evaluate.py`。
- `FixedSceneRoboTwinEnv` 仅在 `fixed_evaluation_pool` 配置存在时选用，要求 `is_eval=true`、单任务、单环境、单 worker；严格重放场景，不重采 seed。
- rollout worker 继承标准 `predict`，仅在每个 episode 开始时按 scene seed 初始化噪声。推理、action transform、chunk 执行及终止处理均沿用现有实现。
- 每个 episode 保存 seed、指令、场景信息、实际执行的 control targets 与结果；汇总必须与标准 runner 的轨迹计数、成功率和 episode length 一致。
- 通过命名 node group 显式固定 Robotwin Python interpreter，避免复用 Ray 服务时混入其他实验的环境。每个输出目录使用独立 namespace。

## 验证安排

1. action20 两个相同的 stack teacher 场景：旧入口与新入口逐 chunk 对照观测哈希、动作、seed、指令、成功及控制步数。
2. 对照通过后，在相同 development cohort 0、前 50 个场景上，复测四个 teacher 与 SFT400 student。
3. 与原 action00 结果比较；跨机器差异与执行链路差异分别判断。此轮仍是开发集验证，不替代正式 held-out 评测。

| Task | Native horizon | 旧 teacher（50） | 旧 SFT400（50） |
|---|---:|---:|---:|
| open_laptop | 700 | 82% | 56% |
| handover_mic | 600 | 98% | 44% |
| dump_bin_bigbin | 600 | 92% | 46% |
| stack_bowls_two | 900 | 78% | 78% |

原始输出：`outputs/robotwin4/task_init_probe_20261107/shared_worker_eval_action20_20261107_r3/`。入口的 `--tasks` 必须显式声明整个场景池的任务注册表，`--norm-stats TASK=PATH` 必须覆盖所有这些任务；不会根据任务名猜测 normalizer。

不影响 action30 原正式任务组合的实验。新 setting 的训练收益仍需 MOPD 实验确认。

## 已完成的执行路径对照

`stack_bowls_two`，相同两个固定场景（seeds 100000、100001）：两入口均 2/2 成功，实际控制步数分别为 302、260。共 24 个有效 action chunks 的输入观测哈希完全一致，输出动作的最大绝对差为 0。标准 worker 在终止后仍有固定长度查询，但环境冻结，不计入实际控制步数；下一 episode 重新设定 seed，保证这些填充查询不影响下一场景的 flow noise。

证据：[shared_worker_parity.json](shared_worker_parity.json)。这验证了这两个场景的执行路径一致性，不能单独证明全部任务、全部场景的成功率一致。

复测队列：`outputs/robotwin4/task_init_probe_20261107/shared_worker_queue.py`；激活脚本：同目录 `shared_worker_env.sh`。action20 已复制原 action00 的 pinned RoboTwin 源码、assets、两个本地 teacher 和 tokenizer；继续使用原场景池要求的 canonical SAPIEN payload。GPU 0/1 各运行一个评测任务；推理 allocator 上限为设备显存的 32%。不启动 actor、optimizer 或新训练任务。

复测完成后，独立 finalizer 自动生成输出目录中的 `setting_comparison.csv` 和 `setting_comparison.json`，逐场景核对新旧记录的 seed、指令及场景属性，并报告成功结果与实际控制步数改变的场景。精确使用的队列、激活和分析脚本归档在 [shared_worker_scripts.tar.gz](reproduce/shared_worker_scripts.tar.gz) 中；它们保留当前服务器路径，作为本轮实验的复现记录。

## 中断恢复与阶段结果

首次完整队列在 `dump_bin_bigbin` 完成 6/50 时失去 Ray GCS 连接，日志为 owner/node crashed。尚不能仅凭该日志认定是物理机器故障；未见 CUDA OOM 报错。已改用独立、常驻的队列 supervisor 持有 Ray 服务，所有评测 driver 显式连接该地址，避免服务依赖某个先完成的评测 driver 的生命周期。

- 已完成并保留：teacher open_laptop 43/50（旧 41/50）；teacher handover_mic 50/50（旧 49/50）。两组 50 个场景的 seed、指令和场景属性逐条对应，分别有 2、1 个成功结果改变。
- 这是 action20 新入口与 action00 旧入口的跨机器比较，不表示 teacher 本身能力提升。
- 当前输出：`outputs/robotwin4/task_init_probe_20261107/shared_worker_eval_action20_20261107_r4/`。已完成两组通过链接复用；失败的部分结果保存在 r3，不拼接为完整成绩。
- 剩余：两个 teacher 重新各测 50 episodes，随后四个 student400 任务各测 50 episodes。
- 恢复队列：`shared_worker_continue.py`；汇总：`finalize_shared_worker_continuation.py`，均在 study 目录。精确脚本见 [shared_worker_continuation.tar.gz](reproduce/shared_worker_continuation.tar.gz)。
