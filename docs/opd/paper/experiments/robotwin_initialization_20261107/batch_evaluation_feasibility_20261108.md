# SFT400 初始化与并行评测可行性

## 初始化

保留新四任务 Joint-SFT400 作为共享 cold start。它使用每任务45条训练示范、开发集选择的400-step checkpoint；不称 few-shot、充分 SFT，或预先固定且未经选择的400步预算。当前统一 worker 评测平均55%，teacher90%，三个任务仍有明显差距。

LIBERO 使用已接受的离线蒸馏 checkpoint-000050（确认集平均61%）。不同 benchmark 可以采用不同离线初始化配方；每个 benchmark 内所有 MOPD 对照共用同一权重、任务和评测协议。报告各自的数据来源和离线成本。该设计检验从有基础能力的共享策略进行在线能力融合；不能仅凭 cold-start 后提升宣称优于充分联合SFT。若论文强调相对继续SFT的优势，应补匹配起点的继续SFT对照。

## 现有执行链路

训练与新独立评测都复用 MultiStepRolloutWorker、EnvWorker 和 RoboTwinMultiTaskEnv 的推理/控制实现。固定评测另加场景选择和按episode设置noise seed的适配。训练的seen指令/随机场景与固定评测的unseen指令/author-cache场景是有意的协议差异；共用执行代码不等于两者状态分布完全相同。

训练模型已支持真正的batch forward。T32是全局环境数；四个rollout workers、pipeline stage=1时，每worker处理8个环境。当前固定场景评测出于对照验证目的强制单环境，需要扩展其调度，不能仅把total_num_envs改成32。

## action20 单H100 NVL实测

使用同架构stack teacher、一个真实场景的观测复制为不同batch；H25、10步flow ODE、完整predict_action_batch。每个batch一次warmup、三次计时，取中位数。

| Batch | 整批推理秒数 | 每观测秒数 | PyTorch峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|
| 1 | 0.289 | 0.289 | 7.14 | 7.26 |
| 2 | 0.338 | 0.169 | 7.29 | 7.50 |
| 4 | 0.403 | 0.101 | 7.57 | 8.00 |
| 8 | 0.536 | 0.067 | 8.13 | 8.73 |
| 16 | 0.891 | 0.056 | 9.26 | 10.40 |
| 32 | 1.742 | 0.054 | 11.52 | 13.77 |

测量包含模型权重与推理临时张量，不包含多个SAPIEN场景的显存；只有一个常驻场景。这不是端到端32环境吞吐测量，也不是跨batch动作等价性测试。仿真、reset、IPC、完成较早环境的等待都会影响端到端加速。

建议：评测优先每GPU4–8个环境和对应batch；8作为候选默认。四GPU可组成32全局环境；两GPU不必强求32，可以先16全局环境。单卡batch32相对16吞吐仅提升约2.3%，不作为默认配置。

并行固定评测仍需处理：每slot的固定场景与指令校验、明确且可复现的noise规则、终止冻结、精确50/100条计数的尾批mask，以及episode记录与runner指标一致性。当前逐episode全局RNG重置不能原样推广到多slot；不能承诺改变batch后仍与旧单环境评测逐动作一致。部署前应做小规模端到端单环境/并行对照，并统一复测参考checkpoints。

本次仅完成代码核对与模型侧测速，没有修改训练逻辑或启用并行正式评测。原始数据见 [JSON](batch_inference_profile_20261108.json)，测速脚本见 [记录](reproduce/profile_eval_batch_20261108.py.txt)。
