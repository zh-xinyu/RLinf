# Teacher reliability：保留的后备路线

**当前优先级仍是更充分的 MSE/M3 优化与 joint SFT 初始化。** 只有固定输入诊断也显示优化受阻，且出现 teacher 在 student 状态上不可靠的证据时，再考虑 uncertainty/OOD masking；本笔记不改变训练，不安排新 GPU 实验。

本轮完成状态（2026-11-02）：固定输入诊断、充分优化和joint SFT→MOPD均已完成，独立确认35.0%→85.0%。因此没有触发新的uncertainty实验；下方最小诊断清单是当时设计及后备判据，已执行部分的结果见[最终报告](JOINT_SFT_MOPD_RESULTS_20261102.md)。

## 本地论文实际做了什么

来源：根 workspace 的 `25_When_to_Trust_Your_Teacher_.pdf`，*When to Trust Your Teacher: Uncertainty-Aware On-Policy Distillation for Embodied Policies*。关键依据为 §III-B/Eq.(3–4)、§IV-C/Eq.(6–7)、§V Practical Implementation 和 Appendix B。

- **基础 OPD：** 在 student 生成的同一 flow 状态 `x_i` 上比较 teacher/student 的局部 Gaussian transition；共享 covariance 时化为时间加权 velocity MSE。其 mixed ODE-SDE 概率推导不能自动解释为“必须用纯 SDE 采样”，也不能据术语推断与我们所有采样实现完全相同。
- **不确定性：** 冻结 teacher，在同一观测 `o`、同一初始 noise `z` 下，分别用10步和5步确定性 flow solver 得到 `a10`、`a5`。不是独立 noise 的动作方差、多个 teacher 的分歧或训练 critic。
- **排除规则：** `U > τ` 的触发 chunk 及同一 episode 后续 chunks 全部不参与蒸馏。Appendix B 明确的是 loss masking；不能据此断言实现提前停止了物理仿真，或获得了多少环境步节省。
- **惩罚：** Algorithm 1 提出 `r_tilde = -L_OPD - λu` 的 uncertainty-penalized reward；消融比较 penalty 与 truncation。但本地 PDF 没有充分给出该 penalty 到具体 optimizer/梯度估计的实现细节，不能擅自补成某一种 PPO 或乘性 weighting。
- **没有训练期自动纠错接管。** teacher takeover 是验证 reliability 的实验：从 student 访问状态交给 teacher 执行到结束，统计成功率。不是论文训练流程中的 teacher takeover 机制，也不保证 teacher 能救回高不确定状态。

Eq.(6) 为每个 action 元素在不同 discretization 间的 Lp dispersion，除以该元素的 RMS 幅值，再对 horizon×dimension 平均。实际采用 `p=1`、两个 discretization，故可写成：

$$
U(o,z)=\frac{1}{HD}\sum_{h,d}
\frac{|a^{10}_{h,d}-a^5_{h,d}|/2}
{\sqrt{((a^{10}_{h,d})^2+(a^5_{h,d})^2)/2}}.
$$

PDF 公式没有明确零分母的 epsilon/处理规则，也未充分说明与本项目一致的 latent/反归一化坐标选择；不能直接照搬阈值。Appendix B：RoboTwin `τ=0.025, λ=0.05, B=512, chunk=20`；LIBERO `τ=0.1, λ=0.01, B=512, chunk=10`。这些是对方设置，不是我们的建议参数。

论文报告 teacher takeover 中低 U 组成功率更高，并在 Adjust Bottle 用近满分 reference teacher 得到 uncertainty 与 reward gap 的 Spearman 相关0.683。理论保证依赖 uncertainty 上界 reward error 的假设；这个相关性并不证明实际 proxy 满足逐点上界。低 U 也可能是“一致地错”，高 U 可能反映数值积分曲率或小幅值归一化，并不等同已识别 OOD。

## 对当前 π0.5 MOPD 的适用条件

概念上可复用 frozen flow teacher 的10/5步 self-consistency，但需保持同一 noise、prompt、normalization、latent horizon；我们生成50、执行25×14，不应为估计 U 偷改执行语义。还需定义统计前25还是全50、physical14还是包含 padded18，并处理近零分母。先用少量 student 状态验证 U 是否与 teacher 后续成功率相关，再校准各任务尺度；当前 teacher 正常起点的成功率不是偏离后可靠性的证明。

**直接把 `+λU(o,z)` 加到现有 supervised MSE 并不能复制 reward penalty：** 在固定 rollout 状态/noise、teacher 冻结的更新中，U 对 student 参数没有梯度。要改变优化，必须明确使用 sample mask/weight，或另外设计能把 reward 反馈到策略的估计器；这些都应标为我们的实现选择。最小后备方案可只试 prefix masking，但它不是完整 UA-OPD，且可能减少晚期任务状态覆盖。两次额外 teacher flow 求解也有真实成本，不能先验声称更高效。

## 先解释当前 MSE 与加权目标的差距

[当前 specialist-init 结果](SPECIALIST_INIT_RESULTS_20261101.md)：前5→后5更新平均，plain velocity MSE `0.4428→0.2268`（约−48.8%），实际 M3 objective `0.2602→0.2277`（约−12.5%）；handover 的加权目标反而上升。这不足以证明已经优化到平台，更不能直接证明 teacher OOD。

此前机制记录 `.project_results/mechanism_study/FINDINGS.md` 的 F1 已确认：共享 covariance 下，M3 是固定时间权重的 velocity MSE。当前 `num_steps=10, noise_level=0.5` 的源码 `rlinf/algorithms/losses.py::flow_sde_kl_weights` 给出：

| ODE t | 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w(t) | .0200 | .0281 | .0633 | .1085 | .1688 | .2531 | .3797 | .5906 | 1.0125 | 2.2781 |

均值0.49027，最后两个时间点占权重总和67.12%。若误差主要在高 t 下降、低 t 不降，plain MSE 可以显著下降而 weighted MSE 改善很小。另一方面，训练逐步改变观测/flow 状态分布，且每样本随机取一个时间点；时间点计数、任务组成、valid mask 都会改变观测到的均值。这些目前是待检验解释。

## 最小判别诊断（只记录计划）

1. 每任务选固定观测，保留原任务与其他任务的样本；固定 prompt、norm、noise。用一个固定 reference policy 生成并缓存完整10个 `x_t`，多个 checkpoint 都在**同一 `(o,x_t,t)`** 上前向，与同一 teacher 比较。
2. 报告每 task×全部10个时间点的误差 `E[j,k]`、有效元素数；统一汇总 `mean_k E[j,k]` 与 `mean_k w[k]E[j,k]`。同时拆分 first25/last25 与 physical14/padded18，避免 horizon 或 padded 分母变化掩盖问题。
3. 固定输入误差下降 → 支持继续优化/更好初始化；低 t 或原任务误差持续不降/升高 → 再定位时间重点与任务干扰。另做 checkpoint 各自 flow path 的 probe 可看状态漂移，但不能与固定 `x_t` 测试混为一谈。现有前4观察×3时间点 probe 不是完整训练分布或10点加权目标估计。
4. 若更充分优化/joint SFT 后仍卡住，再评估 uncertainty 的排序可靠性与 prefix masking。此前不改变生产 loss、采样或 truncation。

`total_loss` 还受 microbatch accumulation 的日志缩放影响，跨设置比较应使用 `analytic_kl`、plain MSE 和上述固定输入分解，不能将 `/8` 的日志差额解释成额外梯度衰减。
