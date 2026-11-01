# RoboTwin 长程自动调试计划（2026-11-01）

目标：优先检查充分优化与真正的joint SFT初始化能否使4-task MOPD产生任务收益。不能把MSE下降、teacher正常起点成功或程序无报错单独当作成功。

## 已知起点

- 先前joint20-update specialist-init：global MSE前5/后5约0.443→0.227，M3目标0.260→0.228；冻结测评6/32→1/32。
- self teacher/学生初始同源任务MSE5e-5，训练推理teacher21/32成功；关键链路有依据，仍需行为保留控制。

## 当前安排

| 阶段 | 内容 | 状态/资源 |
|---|---|---|
| 正确性控制 | place单任务、student=teacher同SFT，M3 full50、T8/R4/B128/micro2、LR1e-5、20updates，随后原4-task eval seed200000 | 完成；place同场景6/8→6/8，末MSE3.56e-4；见SELF_DISTILL_RESULTS_20261101.md |
| 真正joint SFT | 官方4task demo_clean、50episodes/task；逐task最后5episode留出，seen instruction；标准conditional FM，非M3，非teacher生成伪数据 | 数据/26项工具CPU测试通过；GPU前向相对官方FM loss差<0.32%；micro2/micro4 GPU smoke及保存通过；checkpoint重载通过；物理GPU0–3已启动500updates，B128/micro4/LR2.5e-5/warmup25，每100保存 |
| 固定输入诊断 | 固定place reference policy的同一(o,x_t,t)、全10时间点；比较初始与joint20，拆分physical/padded与first/last25 | 四task完成；同输入新任务误差下降，place遗忘；见FIXED_REFERENCE_RESULTS_20261101.md |
| 更多OPD优化 | 正确性控制通过后考虑从joint20恢复更多更新，或优先从jointSFT起点继续；保存阶段性checkpoints检验低MSE是否转化为SR | 已排条件队列：self最终同seed place≥4/8、MSE<0.01后，从joint20完整恢复到60；保持R2/T16/B128/micro4/LR1e-5，额外40updates/20rounds，冻结测评40/60；gate仅为探索性筛查 |
| 不确定性后备 | 同noise的teacher10/5步ODE self-consistency、可能prefix mask | 仅文献分析；优先级低于充分拟合/联合初始化 |

各阶段以GPU实测显存和吞吐确定最终训练预算。SFT实测micro2约47s/update，micro4约27s/update；500updates训练本体约3.7h，保存/测评另计。每100保存，先检查100/200的拟合及rollout，再判断是否需要额外训练。保持RoboCasa主分支/环境隔离，不终止其他用户进程。

若同teacher自蒸馏也持续丢失能力，先定位训练数值/目标实现，不以增加joint训练量掩盖问题。若self保留而joint退化，优先充分优化与jointSFT；uncertainty不提前加入。失效/负面结果也保留，所有run记录精确初始化和累计预算。

本次paper解读见 [Teacher可靠性后备](TEACHER_RELIABILITY_FALLBACK.md)。原user写teacher OOM，此处暂按teacher OOD/不可靠状态理解，不将其与显存OOM混为一谈。

续训队列：`outputs/robotwin4/setup/longer_opd_queue_20261101.json`；完整argv为`joint20_to60_command.json`。先核对上游PID/create_time真实退出，再检查GPU空闲；不启动第二个Ray作业、不绕过resume计数、不停止他人作业。若控制不满足筛查条件，进入`CONTROL_REVIEW_REQUIRED`，保留现场供分析。
