# 单specialist初始化 MOPD：完成结果（2026-11-01）

**结论：训练与冻结checkpoint测评完成，velocity MSE下降，但未出现能力融合收益；原任务能力明显退化。不能把管线跑通等同算法有效。**

## 设置与预算

- RoboTwin 2.0、4任务；共享student从place_container_plate单任务PACE π0.5 SFT初始化，非联合SFT。
- 物理GPU4–7；T16、R2、B128、micro4、LR1e-5 constant、warmup4、完整50×32 velocity监督；执行25×14。
- 20次更新、10轮rollout、160条轨迹、86,558个实际control targets；checkpoint10/20。
- W&B：[oipp0xze](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/oipp0xze)。
- 队列与训练均COMPLETE，step20冻结测评完成。检查时8张GPU均空闲。采样峰值48,263MiB（约47.1GiB），无OOM，四卡配置余量很小。

## 测评（同顺序task/实际reset seeds）

| Task | Step0 | Step20 | MSE前5更新均值 | MSE后5更新均值 |
|---|---:|---:|---:|---:|
| adjust_bottle | 0/8 | 0/8 | 0.693981 | 0.285251 |
| handover_mic | 0/8 | 0/8 | 0.479934 | 0.307631 |
| place_container_plate | 6/8 | 1/8 | 0.000124 | 0.132337 |
| stack_bowls_two | 0/8 | 0/8 | 0.597024 | 0.181888 |

合计6/32→1/32（18.75%→3.125%）。每task只有8episodes，不是正式benchmark；场景seed一致，但未锁定逐步policy noise，不能视为完全配对轨迹。训练过程中adjust偶有成功，最终冻结测评未复现，不能宣称已获得新任务能力。

## Loss的口径与初始量级

- 当前全局velocity MSE：前5均值0.4428→后5均值0.2268，下降约48.8%；首/末点0.4264→0.1957。
- 真正M3加权objective（analytic_kl）：前5均值0.2602→后5均值0.2277，仅下降约12.5%。handover逐task加权objective均值0.3161→0.3535，虽然该task未加权MSE下降。
- total_loss为gradient accumulation缩放后的日志数；本轮等于analytic_kl/8。此前fullflow跟进为/4，因此不能拿不同B/micro/world的total_loss纵轴直接比较。梯度累积检查未发现多除一次。
- 初始place与其teacher同源，第一步MSE=5.03e-5，其余任务约0.702/0.437/0.566，等权平均得到0.426。起点不是四任务联合SFT，初始总loss不应接近零。
- 早先base-init pilot第一步MSE=1.148，末步0.071；fullflow温启动第一步0.115，来源是该已训练step40，而非base。当前specialist起点不等同这个温启动。旧base pilot监督前25，本轮监督50，跨run量级还有horizon/状态分布差别。
- velocity MSE是归一化latent坐标中的逐元素平方误差，非成功率/物理距离，没有[0,1]上界。与RoboCasa/GR00T不同norm、模型和维数的数值不可直接比较。

## 对链路与机制的判断

冻结routed teachers通过真实训练generate/interact路径21/32成功；共享specialist的原始混合batch推理起点6/8成功；同源teacher/student初始误差接近零；20次真实更新、refresh、权重同步、保存和重新加载测评完整完成。这些支持关键链路可用，但不等于所有训练细节/数值/benchmark协议均已证明无误。

总体MSE下降掩盖原任务MSE从接近零升到0.1量级、SR下降。该模式符合任务间干扰/遗忘，也可能包含student状态分布变化及teacher在偏离状态上监督质量不足；尚无梯度冲突或严格同状态probe，不能断言唯一原因。训练loss是在变化的on-policy分布上测量，不是固定验证集。

下一项最有判别力的小测试是single-teacher self-distillation保留控制（同source student/teacher、同精度与更新逻辑），配合固定(o,x,t)的逐task/时间段loss探针。若单teacher自身也丢能力，优先查训练数值/目标；若能保留而joint丢失，再聚焦多任务优化。当前没有自动启动额外实验，也不凭本轮直接宣布M3无效或提高R能解决。

## 原始证据

根workspace相对路径：
- `outputs/robotwin4/analysis/specialist_init_final_summary_20261101.json`
- `outputs/robotwin4/analysis/specialist_init_comparison_20261101.json`
- `outputs/robotwin4/train/robotwin4_specialist_init_r2_t16_b128_pilot_20261101/optimizer_updates.jsonl`
- `outputs/robotwin4/analysis/specialist_step0_trainpath_t16_20261101/results.json`
- `outputs/robotwin4/analysis/specialist_step20_trainpath_t16_20261101/results.json`
