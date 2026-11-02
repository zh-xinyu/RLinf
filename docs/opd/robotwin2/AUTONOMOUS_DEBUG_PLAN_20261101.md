# RoboTwin 长程自动调试计划（2026-11-01）

目标：优先检查充分优化与真正的joint SFT初始化能否使4-task MOPD产生任务收益。不能把MSE下降、teacher正常起点成功或程序无报错单独当作成功。

## 已知起点

- 先前joint20-update specialist-init：global MSE前5/后5约0.443→0.227，M3目标0.260→0.228；冻结测评6/32→1/32。
- self teacher/学生初始同源任务MSE5e-5，训练推理teacher21/32成功；关键链路有依据，仍需行为保留控制。

## 阶段安排与完成状态（更新至2026-11-02）

| 阶段 | 内容 | 状态/资源 |
|---|---|---|
| 正确性控制 | place单任务、student=teacher同SFT，M3 full50、T8/R4/B128/micro2、LR1e-5、20updates，随后原4-task eval seed200000 | 完成；place同场景6/8→6/8，末MSE3.56e-4；见SELF_DISTILL_RESULTS_20261101.md |
| 真正joint SFT | 官方4task demo_clean、50episodes/task；逐task最后5episode留出，seen instruction；标准conditional FM，非M3，非teacher生成伪数据 | 完成500updates，B128/micro4/LR2.5e-5/warmup25；固定CFM验证loss 0.16748→0.00441；SFT100/200/500冻结pilot为5/32、9/32、11/32。数据/前向/保存重载验证见JOINT_SFT_WARMSTART.md |
| 固定输入诊断 | 固定place reference policy的同一(o,x_t,t)、全10时间点；比较初始与joint20，拆分physical/padded与first/last25 | 四task初始/20/60对照完成；place误差先增后降，adjust未随在线MSE同步改善；见FIXED_REFERENCE_RESULTS_20261101.md |
| 更多OPD优化 | 正确性控制通过后考虑从joint20恢复更多更新，或优先从jointSFT起点继续；保存阶段性checkpoints检验低MSE是否转化为SR | 已完成specialist-init20→60及冻结40/60：原place能力6/8→1/8→1/8→4/8，其他task仍0/8。之后固定SFT500通过预声明筛查，完成T32/R4/B128/micro8/LR1e-5的80updates；冻结40/80及独立20场景确认待评测 |
| 不确定性后备 | 同noise的teacher10/5步ODE self-consistency、可能prefix mask | 仅文献分析；优先级低于充分拟合/联合初始化 |

以下为最初预算依据，已完成训练不再追加或重跑。各阶段以GPU实测显存和吞吐确定最终训练预算。SFT实测micro2约47s/update，micro4约27s/update；500updates训练本体约3.7h，保存/测评另计。每100保存，先检查100/200的拟合及rollout，再判断是否需要额外训练。保持RoboCasa主分支/环境隔离，不终止其他用户进程。

若同teacher自蒸馏也持续丢失能力，先定位训练数值/目标实现，不以增加joint训练量掩盖问题。若self保留而joint退化，优先充分优化与jointSFT；uncertainty不提前加入。失效/负面结果也保留，所有run记录精确初始化和累计预算。

本次paper解读见 [Teacher可靠性后备](TEACHER_RELIABILITY_FALLBACK.md)。原user写teacher OOM，此处暂按teacher OOD/不可靠状态理解，不将其与显存OOM混为一谈。

历史续训队列（已完成训练，旧handoff失败记录保留）：`outputs/robotwin4/setup/longer_opd_queue_20261101.json`；完整argv为`joint20_to60_command.json`。先核对上游PID/create_time真实退出，再检查GPU空闲；不启动第二个Ray作业、不绕过resume计数、不停止他人作业。若控制不满足筛查条件，进入`CONTROL_REVIEW_REQUIRED`，保留现场供分析。

## 剩余验收项

1. SFT500后的MOPD40/80各8episodes/task冻结评测，对照同协议SFT500的11/32；原pilot允许native reset retry产生重复场景，需报告实际seed序列。
2. 按2026-11-01 21:56预声明计划，对固定SFT500与最终MOPD80各测每task20个不同场景（seed300000、stride20），与已完成teacher参考59/80核对。无论pilot40/80结果如何，均不挑选最好checkpoint替代80。
3. 审核实际episode数量、整数seed/occurrence/control计数、场景匹配和协议；action noise未固定，不能声称确定性配对。独立确认仍属seen/unstable-only-reset pilot，不能充当官方feasible/unseen结果。
4. 汇总正负结果、初始化与累计预算，核对根主分支未改变、所有自身作业退出。当前无需引入uncertainty masking。

当前执行入口：`outputs/robotwin4/setup/run_joint_mopd_evals_shared_20261102.py`。仅冻结评测，基于历史实测峰值，每卡至少30,000MiB空闲时允许共存运行；不会重跑训练，耗时不作独占吞吐比较。身份/状态在同目录`joint_mopd_evals_shared_20261102.{launch,status}.json`。之前严格空闲等待队列已验证无子进程后被替换，旧记录保留，详见LIVE_STATUS。
