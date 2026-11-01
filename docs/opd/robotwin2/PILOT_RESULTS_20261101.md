# RoboTwin 2.0 四 teacher pilot：结果与交接

日期：2026-11-01；action00，8×RTX A6000 48GB。

**工程管线已跑通；当前尚未获得可用于论文的能力融合结果。** 首轮和跟进student独立测评均为0/20，全部训练、测评和计划内诊断已完成。不要把velocity拟合改善等同于任务能力融合。

## 固定 setting

- 同一个 π0.5 student，从官方base出发；没有先做目标任务联合SFT。
- 四个独立PACE单任务specialists：adjust_bottle、handover_mic、place_container_plate、stack_bowls_two。
- Native clean ALOHA，三相机，14维物理关节动作，50步latent预测/执行25步，10步ODE。
- 已知task identity hard-route teacher及对应归一化；student/teacher使用相同坐标。Teacher和norm stats来自离线数据，不能称为无demonstration设置。
- 当前评测为**未经过scripted expert筛选的seen指令pilot**，每任务5episodes；不是官方feasible/unseen成绩复现。

模型下载、源文件hash、5模型严格加载均已核验；adjust_bottle JAX/PyTorch velocity相对RMSE约0.08%，10步动作相对RMSE约0.13%。四个teacher都实际执行并获得成功。来源和固定revision见 [checkpoint调研](CHECKPOINT_SURVEY_20261101.md) 与 [环境/评测协议](EVALUATION_PROTOCOL_AUDIT.md)。

## 两段完成的训练

| 项目 | 初始pilot | 温启动follow-up |
|---|---:|---:|
| 起点 | 官方π0.5 base | 左侧step40权重，fresh optimizer |
| T / R / B | 64 / Refresh-R2 / 512 | 32 / Refresh-R4 / 256 |
| Microbatch / GPU | 8 | 8 |
| LR / warmup | 2.5e-5 constant / 8updates | 2.5e-5 constant / 4updates |
| M3监督范围 | first25×32 | full50×32 |
| 新增updates / rounds | 40 / 20 | 40 / 10 |
| Trajectories | 1,280 | 320 |
| Executed control targets | 734,389 | 184,000 |
| 保存steps | 8/16/24/32/40 | 20/40 |
| 整run用时，含启动/保存 | 244.67min | 100.34min |
| Rollout中位数 | 388.91s | 177.22s |
| Optimizer update中位数，排除前8updates | 44.21s | 25.28s |
| Flow refresh准备中位数，排除前8updates | 63.67s | 52.60s |
| 最后10次更新平均raw MSE | 0.05592（前25） | 0.08311（全50） |
| 末5轮训练SR | 0 | 0 |

这两段是**可行性调参，不是受控消融**：初始化、监督范围、T/B/R/warmup/保存频率共同变化，loss口径不同，不能直接拼接MSE曲线或归因于full-horizon。有效模型谱系累计80updates、1,600 trajectories、918,389 control targets、约345分钟训练；失败启动、smoke、teacher校准和测评/诊断预算另存各自记录，未计入该谱系预算。这里control targets不是SAPIEN内部积分substeps。

来源：`outputs/robotwin4/analysis/completed_training_summary.json`，以及两个run的JSONL、配置、launch、source快照与checkpoint状态。

## 独立测评

| Task | Base | 初始step40 | Follow-up step40 | Teacher |
|---|---:|---:|---:|---:|
| adjust_bottle | 1/5 | 0/5 | 0/5 | 3/5 |
| handover_mic | 0/5 | 0/5 | 0/5 | 4/5 |
| place_container_plate | 0/5 | 0/5 | 0/5 | 5/5 |
| stack_bowls_two | 0/5 | 0/5 | 0/5 | 4/5 |

原始结果：`outputs/robotwin4/eval/uniform_bucket_r2_step40/summary.json`；跟进结果：`outputs/robotwin4/eval/fullflow_r4_step40/summary.json`。adjust_bottle的teacher参考seed部分错位，各summary显式保存实际seed重合情况。5episodes/task只适合pilot判断，不能支持正式显著性结论。

## 机制检查与限制

- 四任务各选一个固定seed/seen prompt，比较前4个student访问观测和3个ODE时刻。初始seed、prompt、noise hash匹配；初始teacher endpoint RMS也完全一致。后续状态随student变化，不能当严格配对状态对照。
- 初始pilot后25位置physical14的velocity MSE约为前25的1.6–4.3倍，endpoint MSE约2.3–7.4倍。跟进后并非一致改善：adjust/place初始endpoint误差下降；handover增大；stack前25近似、后25增大。
- 真实checkpoint的joint/cached前向检查：同一初始场景3个ODE时刻，physical14 MSE约1.4e-5–4.9e-5；joint加BF16 autocast约2.0e-5–1.8e-4，远小于观察到的student/teacher误差。没有发现大幅路径不一致；该检查是unsharded eval数值检查，不是全分布或分布式backward证明。
- 可选`openpi.nft_full_horizon`默认关闭；开启时只扩大监督，环境执行仍25×14。73项相关CPU测试（含RoboCasa配置回归）、9项probe入口测试通过；真实GPU两段训练、完整checkpoint保存及13份flow/数值diagnostic已执行。实际全状态resume尚未测试。
- 补充的feasible/unseen筛选已停止：MPLib反复invalid-start/self-collision，首候选失败耗时121秒，原生ALOHA默认curobo，而当前MPLib双臂映射/状态同步存在适配风险。未将此失败归因于student，也未使用不完整pool评分。保留`feasible_unseen_bucket_step40/operator_stop.json`和全部日志；未改planner依赖。

完整数值表见 [FLOW_PROBE](FLOW_PROBE.md)。这里能确认拟合变化和工程正确性检查通过，**不能确认“遗漏后25步”是失败原因**，也不能排除冷启动不足、状态覆盖或多teacher干扰等其他原因。

## 后续使用

- 本轮只完成首轮及一组跟进，不继续盲扫参数。两次最终SR均无收益，不建议直接投入正式RoboTwin实验；优先考虑小规模联合SFT冷启动或更充分训练，再与当前基模起点作区别，具体原因仍需验证。
- 分支：`dev/mopd-robotwin2-four-task`；worktree：`outputs/worktrees/robotwin4`；runtime：`/localscratch2/liyifa11/XyFile/opd-robotwin4`。根目录仍为`opd-project-main`，RoboCasa源码/runtime未改。
- 激活runtime后，在worktree使用`python -m toolkits.robotwin_mopd.run`。精确启动命令和起点预算：`outputs/robotwin4/setup/uniform_bucket_pilot_launch.json`、`fullflow_warmstart_pilot_launch.json`。本地W&B认证按action00 profile加载，不在文档中保存凭据。
- [首轮W&B](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/77se8lie)；[跟进W&B](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/02bn5rjr)。

旧patch同步pilot在4rounds/8updates后OOM；该run标记FAILED并保留现场，未拼接进以上模型谱系。Bucket修正后的两段训练无GPU OOM。checkpoint保存有CPU临时reclaim/swap开销，保留为性能限制；没有调整系统VM参数或停止他人的进程。
