# RoboTwin 四任务联合 SFT 冷启动

目的：在四任务共享策略的 MOPD 前加入真正的联合示范 SFT，检验 base student 缺少任务先验是否限制了短程 OPD。这里不是单 specialist 初始化，也不是把固定 observation 上的 teacher velocity 回归改名为 SFT。

实现位于隔离 worktree 的 `toolkits/robotwin_mopd/joint_sft.py`，不改变已有 MOPD 默认代码。独立 `torchrun` 启动，不需要环境、teacher、Ray 或 rollout。

## 数据与坐标

- 官方 [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186/dataset)，固定 revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`。
- `adjust_bottle`、`handover_mic`、`place_container_plate`、`stack_bowls_two` 各 50 条 `demo_clean.zip`，内部 embodiment 为 `aloha_agilex`。合计 1,590,824,998 bytes；四个 ZIP 均通过 HF LFS SHA256 校验，只提取 HDF5、JSON、TXT。
- 数据放在 `/localscratch2/liyifa11/XyFile/opd-robotwin4/data/official_clean_3dc3b798_20261101`；该目录 `manifest.json` 记录完整来源、哈希和提取清单。
- 使用官方更新过的 RGB/合法关节数据，**不声称与 PACE 当年训练的 50 episodes 完全相同**。不修改 pinned 仿真源码，不使用有待验证的本地 planner 重新采集。
- 默认每 task 的 episode 0–44 用于训练，45–49 用于固定离线验证；ID 按数值排序并检查完整 0–49。只采样 `seen` instruction，不使用 `unseen`。
- 标签遵循 [官方 Pi_05 转换器](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/openpi/scripts/process_data.py)：`action[t] = state[t+1]`，最后一帧重复末状态；从该 action 序列取 50 步，不足部分重复末 action。
- 原始 200 episodes 的所有非末帧均满足这一对齐；52 条的原始末 action 与末 state 有微小差异，最大 0.001070。读取时按官方 converter 重复末 state，原文件保持不变；协议记录每 task 的差异数量和最大值。
- 三相机使用 head/left wrist/right wrist。读取器只接受 `XPL-RGB1` 标记的标准 JPEG；旧版无标记图像主动报错，避免颜色反转。官方 decoder 来源固定于 [RoboTwin ea8b2112](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/data/decode_image_bit.py)。
- 保留已有四套 teacher norm；使用生产 `input_transform`：`adapt_to_pi=False`、12 个 arm joints 转 delta、两侧 gripper 保持 absolute、quantile normalization、14→32 padding。不得静默重算 norm。

固定 norm 审计：每 task 50 episodes × 16 frames，完整 50-step chunk；全部有限，physical14 的 `abs > 2` 和 `abs > 10` 比例都为 0，padded18 为 0。

| Task | Normalized action RMS | Min | Max |
|---|---:|---:|---:|
| adjust_bottle | 0.574 | -1.565 | 1.745 |
| handover_mic | 0.594 | -1.814 | 1.740 |
| place_container_plate | 0.739 | -1.731 | 1.911 |
| stack_bowls_two | 0.453 | -1.521 | 1.802 |

Canonical 审计记录在根 workspace `outputs/robotwin4/setup/joint_sft/teacher_norm_compatibility.json` 和 `data_schema_and_rgb_audit.json`。

## 训练约定

以官方 `pi05_base_torch` 为默认共同起点，训练 VLM 与 action expert 的全部参数。目标为标准 conditional flow matching：

\[
\epsilon\sim\mathcal N(0,I),\quad
t=0.999\operatorname{Beta}(1.5,1)+0.001,\quad
x_t=t\epsilon+(1-t)a,\quad
\mathcal L=\mathbb E\|v_\theta(o,x_t,t)-(\epsilon-a)\|^2.
\]

- 完整 `50×32` 平均 MSE；没有 M3 时间权重，也没有 teacher forward 或 flow trajectory refresh。
- 每个 global batch 对 task 等额；task 内均匀取 episode，再均匀取 frame。每次重采样 noise/time 和 seen instruction。
- 训练使用 OpenPI 的图像增强；离线验证关闭增强，并固定 observation/frame/prompt/noise/time seed。验证 loss 和带增强的 training loss 不应直接比较绝对大小。
- FP32 master/Adam moments、BF16 AMP、full-shard、gradient checkpointing。每个 microbatch 同步梯度，避免累计未分片 FP32 gradient；因此 microbatch 太小会增加通信次数。
- 当前已启动方案：4 GPU，B128，microbatch4，LR2.5e-5，warmup25，constant scheduler，500 updates，每100保存、每50离线验证。microbatch4先通过2步smoke再启用；根据中间checkpoint策略评测决定是否干预长跑。
- 文件中只绑定本次实例的 demonstration forward，复用 `get_joint_velocity`，绕过旧 `sft_forward` 禁用 checkpointing 的行为；参数名称和 MOPD checkpoint 接口不变。
- `--verify-only` 在四 task 各一个 batch 上使用同一 observation/noise/time/augmentation RNG，比较该 forward 与原 OpenPI `PI0Pytorch.forward` 的 full-horizon loss。两者使用相同数学目标；当前 FP32 master/BF16 AMP 下是近似数值一致，不是 bit-identical。

## 执行顺序

在 worktree 根目录执行；示例中的输出目录必须尚不存在。GPU 编号须由当前调度确认，不能照抄到有人占用的卡。

```bash
source /localscratch2/liyifa11/XyFile/opd-robotwin4/env.sh
JOINT_DATA=/localscratch2/liyifa11/XyFile/opd-robotwin4/data/official_clean_3dc3b798_20261101
JOINT_ROOT=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try/outputs/robotwin4

# CPU 数据与对齐审计
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu python -m toolkits.robotwin_mopd.joint_sft \
  --data "$JOINT_DATA" --output /tmp/robotwin_joint_inspection --inspect-data

# 单 GPU：先验证四任务 official forward 近似数值一致，不能跳过
CUDA_VISIBLE_DEVICES=0 python -m toolkits.robotwin_mopd.joint_sft \
  --data "$JOINT_DATA" --output "$JOINT_ROOT/setup/joint_sft/forward_equivalence" \
  --verify-only

# 独立端口，四 GPU：2 updates + optimizer/checkpoint smoke
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 --master-port=29674 \
  -m toolkits.robotwin_mopd.joint_sft \
  --data "$JOINT_DATA" --output "$JOINT_ROOT/train/joint_sft_smoke" \
  --steps 2 --batch-size 128 --microbatch 2 --warmup 1 \
  --save-interval 2 --validation-interval 2

# 前两项通过后才进入候选完整训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 --master-port=29674 \
  -m toolkits.robotwin_mopd.joint_sft \
  --data "$JOINT_DATA" --output "$JOINT_ROOT/train/joint_sft_500_pilot" \
  --steps 500 --batch-size 128 --microbatch 4 --warmup 25 \
  --save-interval 100 --validation-interval 50 \
  --wandb-project RoboTwin_MultiTeacher_Pilots \
  --wandb-group robotwin_joint_sft_initialization \
  --wandb-name robotwin4_joint_sft_500_pilot
```

记录：`launch.json`、`resolved_config.yaml`、逐更新 `metrics.jsonl`、固定 `validation.jsonl`、`checkpoints/global_step_N/actor/`、结束后 `COMPLETE`。Metrics 包含逐 task 全维/physical14/padded18 loss、样本数、LR、gradient norm、时间。每更新汇总各 rank 的累计 allocated/reserved 峰值与跨 rank 最大值；单位是十进制 GB，不能直接与 nvidia-smi 的 GiB 数字比较。完整 checkpoint 使用现有 RLinf FSDP 保存器；入口暂不实现 optimizer resume，`--model-path` 仅为显式 weights warmstart。

W&B 默认关闭，显式设置 `--wandb-project` 后仅 rank0 上传，`train/*`、`validation/*` 都以 `optimizer_step` 为横轴；`wandb_run.json` 保存链接。无论是否开启，JSONL 始终保留。若计划训练500步，应一次启动500步并检查100/200等中间结果，避免200步结束后意外用新optimizer接着训练。

后续判断必须使用策略 rollout SR：offline validation loss 改善只能证明示范拟合；不能直接声称已实现四任务能力融合。联合 SFT 后的 MOPD 比较须使用同一 joint 初始化和归一化。

## 验证状态

- 数据下载、SHA256、全200episode非末动作对齐、末帧规范化记录、固定norm范围审计已完成。
- `tests/unit_tests/test_robotwin_joint_sft.py`：11 个 CPU tests 通过，包含显式W&B开关/更新轴和跨rank峰值汇总。
- Hydra compose 与 FP32/BF16 配置检查通过；GPU四任务 official-forward 检查通过，逐task相对loss差异为0.313%、0.169%、0.048%、0.063%。现有joint helper将embedding转为FP32，而官方forward在FP32 master时保留部分AMP产出的BF16 embedding；RMSNorm/residual路径因而存在数值差别，不能称完全相同。MSE计算本身两边均为FP32；具体归因还未通过额外GPU控制实验验证。
- 4 GPU、B128、microbatch2的2步FSDP smoke已通过，约47秒/update，rank0 allocated峰值30.1GB，完整checkpoint保存成功；保存权重已通过 `fixed_flow_probe` 重新加载。
- microbatch4的2步smoke也已通过：27.0 / 26.5秒/update，跨rank最大allocated 30.128GB、reserved 47.928GB；包含初始化、验证和保存的总耗时180.6秒。两步样本显示更新耗时约降低43%，尚不代表长跑或仿真训练的同等收益。
- 500-step长跑已启动：`robotwin4_joint_sft_500_b128_micro4_pilot_20261101`，GPU0–3，W&B [33rml11k](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/33rml11k)。启动时四rank FP32 master/Adam验证通过。这是启动时的核验记录；最终500步训练及策略评测均已完成，结果见下文。
- 启动后前5步检查正常：平均26.8秒/update，loss为0.1692、0.1604、0.1378、0.1609、0.1392，gradient norm均有限（1.26–1.55）。最大allocated稳定30.128GB、reserved稳定48.331GB。LR按25-step warmup从0增至4e-6；W&B已启动且内部日志无ERROR。这里只确认启动/数值健康，不把少数不同训练batch的loss变化当作验证集或SR提升。
- 长跑状态：根workspace `outputs/robotwin4/setup/joint_sft/long_pilot_status_20261101.json`；原始训练和验证指标位于 `outputs/robotwin4/train/robotwin4_joint_sft_500_b128_micro4_pilot_20261101/`。日志和状态文件优先于本页的静态进度。

## 第50步离线验证（历史记录）

固定留出观测、prompt、noise/time（每task16个样本，关闭图像增强）的CFM loss：0.16748→0.01600，下降90.4%。四task都改善：adjust0.1690→0.01056，handover0.2015→0.01647，place0.1180→0.02194，stack0.1813→0.01504。physical14也均下降，改善并非仅来自padded坐标。

这是示范条件flow matching拟合证据；它与MOPD在student flow上计算的teacher velocity MSE目标不同，**不能横向比较绝对loss，也不能据此宣称成功率改善**。策略rollout尚待checkpoint测评。结构化记录：根`outputs/robotwin4/analysis/joint_sft_step50_validation_20261101.json`。

## 第200步离线验证补充

固定留出CFM loss在steps0/50/100/150/200为0.16748/0.01600/0.00912/0.00778/0.00639。第200步physical14 loss依次为adjust0.00823、handover0.01439、place0.01725、stack0.01758，均比第150步降低。该样本仅覆盖每task16个固定观测；支持继续拟合，不证明闭环操作已成功。记录该阶段时，500步长跑及下游策略评测仍在进行；最终结果见下文。结构化记录：`outputs/robotwin4/analysis/joint_sft_step200_validation_20261101.json`。

## 已固定的下游测评与 MOPD 队列

入口：`toolkits/robotwin_mopd/joint_sft_followup.py`；本机wrapper为根`outputs/robotwin4/setup/run_sft_then_mopd_queue.py`。队列依赖顺序为：joint20→60及其测评完全退出，再依次测评SFT100/200；SFT500需训练正常完成并退出后测评。全部使用原训练推理链路，T16×2 rounds、seed200000，每任务8episodes。

固定选SFT500作为后续初始化，100/200仅用于观察过程，不挑选成绩最高的checkpoint。探索性可行性筛查要求四任务均有完整8episodes，至少两个任务出现成功；不满足时停止为`REVIEW_REQUIRED`，先复查初始化。这不是统计显著性标准。

通过后，须8GPU均空闲且无compute进程，才启动新optimizer的uniform MOPD：T32/R4/B128/micro8，LR1e-5、warmup4、full50×32，80updates，每20保存；再冻结测评40/80。不会恢复SFT optimizer，不改teacher、loss或task norm。源checkpoint固定500，run名含pilot，W&B group=`robotwin_joint_sft_mopd_pilots`。

状态：根`outputs/robotwin4/setup/joint_sft_followup_queue_20261101.json`。GPU忙时停止待review，不终止他人进程；整个队列限时10h，异常仅清理本队列启动的子进程。启动防重、精确PID/create_time和完整episode校验有CPU测试。

评测限制：实际重试可能产生重复seed。目前adjust的8episodes仅6个unique scene seeds，其余三任务8个；新summary会分别记录episode数/唯一场景数，不能称所有任务都有8个独立场景。比较报告明确actual scene seed序列是否匹配，action noise未固定。该小样本队列用于pilot判断，若出现收益还需独立、更充分的确认。

## 策略测评进展（22:18更新）

SFT100/200/500均已完成真实训练推理路径测评：

| Checkpoint | adjust | handover | place | stack | 总计 |
|---|---:|---:|---:|---:|---:|
| Joint SFT100 | 4/8 | 0/8 | 0/8 | 1/8 | 5/32 |
| Joint SFT200 | 7/8 | 0/8 | 1/8 | 1/8 | 9/32 |
| Joint SFT500 | 8/8 | 0/8 | 3/8 | 0/8 | 11/32 |

实际scene序列一致，action noise未显式固定；adjust为6个unique scenes，其余为8个。已看到部分任务的SFT初始化能力，但不是MOPD收益或统计显著性结论。小样本中adjust已高于原teacher参考4/8，不据此断言student显著优于teacher；后续需以同一SFT500为蒸馏前后基线，不能把较弱的SFT100拿来夸大收益。

主训练已完成预先确定的500步；steps350/400固定验证loss为0.004826/0.004988，有正常波动。未按测评结果改选初始化。原始记录在根`outputs/robotwin4/analysis/joint_sft_step{100,200,500}_trainpath_t16_20261101/`。

原下游队列在上游评测衔接失败后已停止，恢复后的当前状态入口为`setup/joint_sft_followup_queue_recovery_20261101.json`，已完成SFT500测评并进入MOPD80训练。原失败记录保留；恢复参数与身份见[LIVE_STATUS](LIVE_STATUS.md)。

## 独立场景确认预注册（SFT500/MOPD80结果出来前）

若既定SFT500可行性筛查通过且MOPD80正常完成，独立确认固定比较**SFT500与最终MOPD80**，无论8episode pilot是否显示收益都进行；不从40/80中挑最好者。使用seed300000、stride20、T16×5、每task20个不同实际场景，与[teacher参考](TEACHER_CONFIRMATION_20261101.md)一致。仍需逐条核对actual seeds，action noise不固定。

精确路径、命令及每checkpoint30分钟上限记录于根`outputs/robotwin4/setup/joint_sft500_mopd80_confirmation_plan_20261101.json`。在现有主队列完成并释放GPU后执行，检查完整checkpoint与GPU空闲状态；本声明时尚未启动确认评测。如果SFT500筛查失败或MOPD80未完成，先诊断该问题，不改选其他checkpoint冒充预定比较。

## 500步训练完成（22:10）

完整500次更新、64,000次样本呈现和11次固定验证已核验；`COMPLETE`及step500完整权重、4个DCP分片和metadata均已保存，训练正常退出。固定64样本CFM验证loss从0.167479降至**0.00440669**（约97.4%）；最后训练batch loss为0.00297036。最终physical14验证loss：adjust0.004583、handover0.009924、place0.013535、stack0.011663。验证过程有波动，不等同于闭环成功率改善。

长跑跨rank累计allocated峰值34.139 GB、reserved峰值48.331 GB（十进制），高于早期micro4 smoke的allocated值；未发生OOM。训练记录及最终checkpoint文件尺寸审计见根`outputs/robotwin4/analysis/joint_sft_step500_validation_20261101.json`。下游恢复队列已自动进入SFT500训练推理路径评测；MOPD尚未开始，仍按既定可行性筛查决定。

## SFT500闭环基线与MOPD启动（22:18）

SFT500已COMPLETE：adjust8/8、handover0/8、place3/8、stack0/8，总计11/32（34.375%）。32条实际episode与汇总一致，和SFT100/200的actual scene序列相同；adjust仍仅6个unique scenes，action noise未固定。SFT200→500的总体改善来自adjust和place；handover仍0，stack由1/8变0/8，不能把总体loss下降解释为四任务同步改善。结构化审计为`analysis/joint_sft500_policy_verified_20261101.json`。

固定SFT500通过原定“至少两个任务出现成功”的探索性筛查。恢复队列已实际启动`robotwin4_joint_sft500_r4_t32_b128_pilot_20261101`，T32/R4/B128/micro8、8GPU、LR1e-5/warmup4、80updates，与预注册命令一致；使用SFT500模型权重、新optimizer，`resume_dir=null`。本快照仅确认launcher启动，尚未确认首个优化更新或MOPD效果。后续与SFT500比较，重点同时看能力保留和新任务学习。

## MOPD首轮完整更新核验（22:29）

首轮32条轨迹/13,766个实际control targets已完成，随后4个更新记录齐全；plain MSE为0.146931、0.131413、0.111765、0.139576，M3加权loss及gradient norm均有限。LR依warmup为0、2.5e-6、5e-6、7.5e-6；首个LR=0不产生参数变化，后续已有正LR更新。8个actor均通过FP32 Adam检查。每个更新保留独立指标，但runner等待整轮R4完成才统一写出，不能用短暂无JSONL判断未优化。

第一轮采样使用更新前策略：adjust8/8、handover0/8、place4/8、stack2/8；与冻结评测场景不同，不作为蒸馏收益。首步逐task MSE依次0.0416、0.0374、0.1544、0.3543；handover低MSE与零成功率并存，跨task的误差绝对值不可当作难度或SR排序。仍执行固定80步及后续评测。证据：`analysis/joint_sft500_mopd_startup_20261101.json`；[W&B gi47axmt](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/gi47axmt)。

## MOPD20阶段记录（22:58）

20次更新、5轮、160条轨迹、67,784个control targets的日志与checkpoint恢复计数一致。前4次/末4次更新的平均plain MSE为0.132421/0.051230，step20为0.045281；20步loss与gradient norm均有限。这是变化采样分布上的在线拟合信号，尚无蒸馏后冻结SR结论。

step20已发布完整权重（16,573,945,133 bytes）、8个DCP分片及metadata、runner与allocation状态。这里只做结构和计数核验；不打断当前8卡训练额外加载中间checkpoint，后续按既定40/80测评。结构化记录：`analysis/joint_sft500_mopd_step20_20261101.json`。

## MOPD40阶段记录（23:35）

半程40次更新、10轮、320条轨迹、119,276 controls已完成；完整权重、8个DCP分片及metadata、runner/allocation状态均已发布，计数一致。前4步/末4步平均plain MSE为0.132421/0.027710，step40为0.024005，loss与梯度均有限。

最近3轮（rounds8–10，采样策略分别为update28/32/36）的在线成功数：

| task | 在线成功 / episodes | updates37–40 平均MSE |
|---|---:|---:|
| adjust_bottle | 24/24 | 0.018725 |
| handover_mic | 19/24 | 0.041370 |
| place_container_plate | 14/24 | 0.032447 |
| stack_bowls_two | 18/24 | 0.018296 |

四任务均出现连续成功，handover相较先前rounds4–6的0/24明显改善，支持继续完成训练。但这些窗口使用不同策略和场景、动作噪声未固定，不能直接与SFT500冻结11/32或teacher独立59/80相减，也不能据此宣称因果收益。保持固定80步计划；40/80冻结测评和预先指定的SFT500/MOPD80独立确认仍待完成。证据：`analysis/joint_sft500_mopd_step40_20261101.json`。

## MOPD60阶段记录（2026-11-02 00:10）

60次更新、15轮、480条轨迹、166,274 controls及完整checkpoint结构和恢复计数已核验。末4步平均MSE为0.019834；最近rounds13–15在线成功数为adjust22/24、handover19/24、place17/24、stack18/24。采样策略依次为update48/52/56，仍是变化场景上的在线诊断；冻结对照尚待训练80完成后执行。证据：`analysis/joint_sft500_mopd_step60_20261101.json`。

## 2026-11-02：MOPD80训练完成，冻结评测等待资源

联合SFT500后的80次MOPD更新已完成，20轮/640轨迹/213,912实际control targets计数一致。前/末5轮平均MSE为0.081703/0.019279，末一步0.022324，全部梯度有限；最终checkpoint结构齐全。在线四任务均有成功，但冻结40/80及独立20场景的SFT500/MOPD80比较尚待执行。原队列因其他用户GPU作业停止，新的eval-only队列严格等待4–7空闲；训练不重跑。详细资源状态、界限与证据见[LIVE_STATUS](LIVE_STATUS.md)。
