# Strong-SFT 起点：正式预算开发验证（2026-11-03）

状态：Uniform-R2与Adaptive-R2的20轮训练、DEV5/10/20及继续SFT2500/3000控制全部完成，所有子任务正常退出并完成自身进程清理。MOPD预算已据DEV选为20 rounds，配置见[正式配置](FORMAL_RECIPE_20261103.md)。正式主实验尚未启动。

目的：验证 MOPD 能否在 Joint-SFT2000 上保留并改善能力，并确定能显示学习过程、但评测成本适中的正式训练长度和间隔。SFT2000 是强起点，不声称已经收敛。

## 固定测试

|组别|共同起点|训练预算|保存|开发评测|
|---|---|---|---|---|
|Uniform-Refresh-R2|Joint-SFT2000|20 rounds / 40 updates|每10 updates|round 5/10/20|
|Adaptive-Sync-Refresh-R2|同上|同上|同上|同上|
|继续 Joint-SFT|同上，新 optimizer|额外1000 updates|新增500/1000|累计2500/3000|

每次开发评测：四个任务各20 episodes，固定 DEV 场景和提示；round0复用同协议SFT2000结果（87.5%）。共8个新开发评测点、640 episodes；训练过程中 success_once 另记录，不替代独立评测。

MOPD共同参数：T32，global B128，单卡 microbatch4，4卡，LR1e-5、constant、warmup4 optimizer updates，R2；全50步flow监督、执行25步action chunk。FP32可训练参数和优化器状态，BF16 AMP；保持已验证的梯度同步配置。种子 actor0 / rollout1000000000+rank / env与allocation42。

Adaptive：teacher参考来自独立训练校准80 episodes；rho0.5、epsilon0.01，warmup/window/interval均5 rounds；窗口内每任务至少5条记录才允许决策（`min_task_episodes`），每轮单任务quota下限是4（4 env workers × 1 epoch）；单次quota变化最多4，同步loss权重。Uniform显式清空teacher参考并关闭同步loss权重。每组20 rounds均为640条student trajectories；Adaptive额外报告校准80条及18318 control targets。仿真control steps按实际累计记录。

继续SFT：原45 train + 5 holdout demos/task，B128、micro4、LR2.5e-5、constant、warmup25、新seed44。它是增加离线预算的控制，不是精确恢复optimizer，也不是独立SFT训练seed。

执行：action00 GPU0–3先Uniform，查看其DEV后启动匹配Adaptive；GPU4–7并行继续SFT。W&B项目 `RoboTwin_MultiTeacher_Pilots`，group `strong_sft_mopd_budget_probe`。所有输出在 `outputs/robotwin4/`，不修改RoboCasa运行代码。

## 事先固定的决策规则

- 先确认协议、运行和交互预算完整，再比较逐任务结果、同场景新增/丢失成功。
- 若至少一组在后段仍恢复或改善且未趋稳，两组都可延长到40 rounds；不只延长胜出组。
- 若两组持续退化且没有后段恢复，不自动开启参数网格，保留负结果并收敛结论。
- DEV共80 episodes，1场成功对应1.25个百分点；差1–2场视为暂不能区分，不据此宣称Adaptive优越。
- SFT2500/3000检查离线预算趋势，不自动更换共同起点并重跑全部组别。
- 保留全部检查点与失败证据；无需对每个已保存checkpoint做独立评测。

## 已选择的MOPD正式预算

两组round10→20均未继续提高，按本轮DEV不延长到40 rounds。固定20 rounds，每5 rounds保存，正式测round0/5/10/20、每任务50 episodes。六份M0/M1/M2×两seed配置已通过实际CLI dry-run；继续SFT控制已完成，共同起点保持不变。详见[正式配置](FORMAL_RECIPE_20261103.md)。

准备文件：`outputs/robotwin4/setup/strong_sft_mopd_tests_20261103/` 与 `outputs/robotwin4/setup/strong_sft_continuation_20261103/`。实际状态、耗时和结果以其status及原始日志为准。

## 首组开发结果（2026-11-03）

|Round|adjust|handover|place|stack|等权SR|相对起点新增/丢失成功|
|---|---:|---:|---:|---:|---:|---:|
|0，SFT2000|19/20|17/20|18/20|16/20|87.5%|0/0|
|5，Uniform-R2|20/20|19/20|18/20|15/20|90%|9/7|
|10，Uniform-R2|20/20|14/20|20/20|17/20|88.75%|10/9|
|20，Uniform-R2|20/20|12/20|19/20|17/20|85%|10/12|

终点退化集中于 handover，其他三任务较起点改善；目前不支持强 SFT 起点上稳定增益的结论。后续匹配Adaptive与继续SFT控制均已完成，完整结果见下表；没有由首组结果单独延长或调参。

Uniform 完成40 updates /20 rounds /640 student trajectories /155995 control targets。训练启动至清理约8498s（2.36h），round平均418.6s；三个 DEV 点共240 episodes、约2217s（37min，四任务并行）。MSE首10次更新均值0.0350、末10次0.0130；loss下降不能替代闭环成功率判断。原外层监督器在训练正常退出时触发PID存活检查竞态；保留原FAILED记录，独立核验训练及清理完成后仅恢复DEV，未重训。全部12个DEV子任务正常退出，结果/场景/检查点核验与owned cleanup通过。

详细逐场结果：`outputs/robotwin4/analysis/strong_sft_budget_20261103/summary.{json,md}`；监督恢复及Adaptive入口：`outputs/robotwin4/setup/strong_sft_mopd_recovery_20261103/`。

### Warmup 的运行波动限制

Adaptive round5 DEV为85%（20/13/18/17），Uniform同点为90%；此时两组仍采用均匀配额，不能将差异归因于动态分配。首轮32条episode的完整记录一致，但actor各rank收到的有效观测池大小为Uniform `[61,98,66,89]`、Adaptive `[61,66,89,98]`；按rank播种的观测/时间步/noise采样因此可能不同。当前是共同初始化、相同名义seed/配置/预算的独立训练，不是共享实际warmup前缀的严格配对消融。不能由记录一致推断每个像素/动作逐位一致；正式效果需多training seeds。未修改运行源码或重新开始实验。证据：`outputs/robotwin4/analysis/strong_sft_budget_20261103/warmup_variability_audit.json`。


## 完整DEV结果与结论

每任务20场；以下均为开发集，不与formal50结果混用。

|模型 / 轮次|adjust|handover|place|stack|等权SR|相对SFT2000新增/丢失成功|
|---|---:|---:|---:|---:|---:|---:|
|SFT2000|19/20|17/20|18/20|16/20|87.50%|0/0|
|Uniform-R2 / round5|20/20|19/20|18/20|15/20|90.00%|9/7|
|Uniform-R2 / round10|20/20|14/20|20/20|17/20|88.75%|10/9|
|Uniform-R2 / round20|20/20|12/20|19/20|17/20|85.00%|10/12|
|Adaptive-R2 / round5|20/20|13/20|18/20|17/20|85.00%|6/8|
|Adaptive-R2 / round10|20/20|20/20|20/20|17/20|96.25%|9/2|
|Adaptive-R2 / round20|20/20|17/20|19/20|17/20|91.25%|9/6|
|SFT2500|20/20|20/20|17/20|18/20|93.75%|9/4|
|SFT3000|20/20|18/20|17/20|17/20|90.00%|8/6|

Adaptive终点比Uniform多5/80场（配对新增11、丢失6），但比继续SFT2500少2场、比SFT3000多1场，不能据此声称稳定优于继续SFT。round10的96.25%提供正向信号，仍需多training seeds检验；不能用正式保留集挑选峰值替代预定round20终点。

SFT2500相对SFT2000净增5场，说明SFT2000并非已知离线上限；SFT3000回落，不足以证明收敛。控制使用新optimizer、相同45条训练示范/task，论文需如实说明。

两组MOPD各20 rounds / 40 updates / 640 student trajectories，训练各约2.3h。Adaptive另有一次80条teacher校准（18,318控制步、约16.3分钟）；继续SFT新增1000步/128,000样本约7.37h，新增训练环境交互为0。新增DEV共640场，累计151,229控制步，与训练交互分开统计。

最终核验：`outputs/robotwin4/analysis/strong_sft_budget_20261103/completion_audit.json`。正式三方法×两seed的训练和独立测评命令已准备，未自动启动。
