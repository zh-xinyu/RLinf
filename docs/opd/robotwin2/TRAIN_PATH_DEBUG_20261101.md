# RoboTwin 2.0：训练 rollout 验证与初始化调试

**最新状态：20-update pilot及测评已完成，MSE下降但SR从6/32降至1/32；见 [完成结果](SPECIALIST_INIT_RESULTS_20261101.md)。以下运行中状态为历史记录。**

更新：2026-11-01，action00。状态：teacher和shared-specialist step0验证完成，20-update MOPD pilot训练中。

## 问题与顺序

1. 用冻结、逐任务路由的四个 teacher 验证真实训练的 HF `generate` → Env `interact` 路径，保留混合任务 batch、原生 horizon 和 action transforms。
2. 分别统计初始化、policy 推理、环境 bootstrap 和 chunk 仿真耗时。四个 teacher 依次分组推理的耗时不能直接等同一个 student 的批量推理耗时。
3. 若通过，评测 `place_container_plate` SFT specialist 作为共享 student 的四任务起点，再做小预算 MOPD。它是单 specialist 初始化，不是联合 SFT；必须同时观察原有任务保留和其他任务学习。

## 版本与资源

- RoboTwin **2.0**，官方仓库 RLinf_support commit `dca9ec682688821c42944eacd7f5dd9bfb15396f`。
- worktree：`outputs/worktrees/robotwin4`，分支 `dev/mopd-robotwin2-four-task`；根目录仍 `opd-project-main`。
- 本轮限定物理 GPU **4–7**；初查0–3各占约17GB，4–7空闲。随后其他作业退出时，本轮尚无GPU模型加载/训练，未执行进程终止或Ray清理。系统内核日志不可读，不能确定他人退出原因。
- RLinf 的设备发现不受 Ray/CUDA mask 充分约束，因此显式 placement 为4–7，诊断另核对GPU UUID。第一次placement预检、第二次UUID前缀格式检查均在模型加载前退出。

## 诊断约定

- T16，4 workers，每worker4个混合任务slot；2 rounds，每task8episodes。
- 沿用训练的uniform allocator、seen instructions、未筛选reset、10步ODE、预测50/执行25、四套固定task normalization。
- 无actor、无optimizer、无weight sync；仅替换执行策略为对应teacher。
- 结果写入 `outputs/robotwin4/analysis/`，精确启动记录在 `outputs/robotwin4/setup/`。
- 小样本诊断不能替代正式50episodes/task benchmark。与旧独立测评的seed/noise不同，不要求逐轨迹相同。

## 已完成代码审计

- 未发现明确的train/eval相机、动作或norm错配；实际GPU诊断仍是必要验证。
- 原生训练固定36次action-chunk交互；终止slot不再模拟，但仍参加policy推理。四task无提前成功时，有效control-target比例仅63.9%。这是潜在优化点，本轮验证不修改该行为。
- `total_loss`记录除gradient accumulation后的micro loss，解释旧B512/8卡/micro8日志的8倍差异；实际梯度只归一化一次，无确认的缩小8倍bug。
- 旧base-init已有velocity/endpoint拟合进展，但两段独立student均0/20；不可仅凭loss断言任务能力有效。

## 条件性后续 pilot

若teacher训练路径通过：先测shared specialist step0，再从相同checkpoint进行T16/R2/B128、micro4/GPU、20updates；constant LR建议1e-5、warmup4，完整50×32 velocity监督（沿用上一跟进轮选项），每10步保存，便于保留中间结果。根据原任务保留和其他任务成功率再决定是否延长。该组同时改变起点和预算，是可行性调试，不能当初始化/参数单因素消融。

## Teacher训练路径实测

两轮T16共32episodes：adjust 4/8、handover 5/8、place 4/8、stack 8/8，总计21/32（65.625%）。四任务均有成功，动作finite，无OOM。轮耗时199.82/184.37秒；每worker两轮累计predict97.46–101.30秒、env chunk144.83–193.07秒、bootstrap/send32.82–37.16秒。计时包含异步等待/不同worker并发，不直接相加解释全局wall time。每次predict按teacher分组，每task子batch=1；当前验证覆盖混合task通信与并行环境，不是大teacher子batch数值等价性测试。

来源：`outputs/robotwin4/analysis/teacher_train_t16_verified_20261101/results.json`，source/config/资源采样同目录；COMPLETE标记与owner-scoped进程清理已确认。旧独立测评16/20和此处21/32的seed/noise不同，不能把差值解释为训练路径性能退化。

资源采样：teacher诊断单卡总占用峰值33,012MiB（约32.24GiB），CPU available最低683.9GiB；4张48GiB卡足够该验证。0–3仅有驱动/渲染枚举的小额上下文占用，未放置训练/teacher worker。

Step0正在使用同一个place specialist作为四任务共享policy，在原始HF `predict` 混合batch路径验证；T16×2 rounds，allocator seed=200000，与新训练seed0分开。之后对应final checkpoint使用相同诊断seed/预算比较；这仍不是正式benchmark测评。

## Shared-specialist step0结果与当前训练

T16×2 rounds：adjust 0/8、handover 0/8、place 6/8、stack 0/8，总计6/32。源文件：`outputs/robotwin4/analysis/specialist_step0_trainpath_t16_20261101/results.json`。轮耗时226.73/212.60秒；第一轮各worker推理21.05–22.92秒，环境chunk166.85–181.07秒。当前主要耗时在原生环境，不应根据GPU并行数量推断仿真会很快。

新训练：`robotwin4_specialist_init_r2_t16_b128_pilot_20261101`。四卡4–7，T16/R2/B128/micro4，LR1e-5 constant/warmup4，完整50×32监督，20updates/10rollout rounds，每10updates保存。初始化是place单任务SFT，fresh optimizer。W&B project仍为RoboTwin_MultiTeacher_Pilots，group=robotwin4_specialist_init_pilots。队列在训练完成后自动对step20进行相同seed200000、T16×2轮的冻结policy诊断；缺失checkpoint、训练失败或step0原任务0成功时会停止而不继续。

队列状态：`outputs/robotwin4/setup/specialist_debug_queue.json`；启动身份与脚本在同目录。资源采样：`outputs/robotwin4/analysis/specialist_debug_resources_20261101.jsonl`。尚不能报告本轮MOPD能力提升。

## 速度检查的边界

一次真实env线程栈采样同时看到TOPP solver discovery与robot passive-force计算；这不是统计profiling结论。CPU热缓存微测发现solver discovery缓存只让完整TOPP快约2.5–4%，不值得为此改当前实验。Aloha两臂共享articulation，setter会重复计算passive force；未来可验证按实体/physics substep去重，但未经轨迹等价性验证，不在本轮修改。原生控制、仿真步长与成功判据保持不变。

新增诊断/四GPU launcher及环境相关29项CPU测试通过，Ruff与diff check通过。根workspace的RoboCasa代码未改。

更详细的仿真源码审计和微基准见 [仿真性能备忘](SIMULATION_PERFORMANCE_NOTES.md)。

## 运行中快照（不是最终结论）

截至4/20次更新、2轮rollout：完整teacher打分、反向、flow refresh、权重同步和下一轮rollout已通过。MSE约0.43–0.48，warmup刚结束，暂无新增任务成功。原任务训练rollout为3/4、2/4，其余任务0/4；不同场景且样本很少，不能认定遗忘。第一次同源teacher/student的place velocity MSE=5.03e-5，其他task约0.44–0.70，支持监督路由与前向匹配。

单次反向约25.5秒，refresh更新准备约43.8秒（其中重采flow约10.6秒）。首轮整体429.3秒=2updates，摊销214.7秒/update；rollout242.4秒、首次teacher scoring56.7秒、训练/refresh合计94.8秒、sync35.4秒。首次加载等成本不应直接推广到所有轮次。

采样训练峰值升至约47,507MiB/卡（46.4GiB），尚未OOM，但仅余约1GiB，四卡方案需要这些卡独占使用，不能再与17GB任务同卡共存。峰值仍需包含后续保存等阶段；资源采样持续运行。

[W&B run](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/oipp0xze)。队列PID3565733，状态/精确启动身份以setup中的JSON为准，避免仅凭旧PID操作进程。训练20步完成后自动评测step20，比较输出目标为`outputs/robotwin4/analysis/specialist_init_comparison_20261101.json`；该文件仅在完成后生成，当前尚无final SR。代码提交`1071ad4a`。

## 后续确认评测：避免 reset 重试导致重复场景

仅诊断工具新增 `--seed-stride`；默认 `1` 完整保留旧 allocator seed 序列及 modulo 行为，现有队列命令无需变化。`NativeTaskSlot.reset` 最多尝试请求 seed 起的20个连续值；旧相邻 occurrence 可能在重试后落到同一实际 seed，不能把episode行数直接当作独立场景数。

未来若pilot出现收益，可另开如下确认评测（**这里只记录示例，不启动任务**；在worktree中激活本机runtime后，使用已有owner-scoped launcher检查GPU并管理独立Ray生命周期）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python \
  /absolute/path/to/workspace/outputs/robotwin4/setup/launch_teacher_train_diagnostic.py \
  joint_sft500_confirmation_seen_stride20 \
  --trajectories 16 --rounds 5 --seed 300000 --seed-stride 20 \
  --student-checkpoint /absolute/path/to/checkpoint
```

stride≥20时，请求seed为 `base_seed + task_id*1000003 + occurrence*stride`，不做wrap；2–19及其他无效stride拒绝。每个请求的完整20-seed重试窗口必须落在uint32范围内，否则失败。每轮对累计记录验证实际seed位于自己的窗口、同task实际seed不重复；异常会报错，不静默去重。改变的是该诊断的pending plan，仍经原allocator的episode身份检查，共享训练allocator和环境均未修改。

示例成功完成时为80episodes、每task20个不同的实际reset seeds；resolved config和results记录 `seed_protocol`、`seed_stride`、`reset_retry_span`、语言类型及reset筛选方式。这仍是 **unfiltered / seen** 的训练路径确认，不是官方 expert-feasible / unseen 评测；场景seed不重复也不等于完整轨迹统计独立，或两个checkpoint使用完全相同的policy noise。
