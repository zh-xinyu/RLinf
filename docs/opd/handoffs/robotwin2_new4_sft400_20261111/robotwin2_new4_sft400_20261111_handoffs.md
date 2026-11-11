# 2026-11-11 实验交接：RoboTwin 2.0 / π0.5 新四任务 Joint-SFT400

2026-11-11。本交接替代旧四任务 SFT500/SFT2000 配方。**每组训练完成后，立即测评round5/10/…/50的全部10个checkpoints，100 episodes/task；结果写入CSV后再启动下一组。** 只需发送本文；`checkpoint_results.csv`、`experiment_results.csv`由运行代码自动生成，无需发送CSV模板。脚本及完整资源 manifest 在仓库。

- 分支：`release/robotwin2-new4`；固定 tag：`robotwin2-new4-train-eval-20261111`。记录实际 `git rev-parse HEAD`。
- 初始化：新四任务 **Joint-SFT400**，固定预算冷启动；四任务各45条训练示范，所有8组从同一权重新建 optimizer。不是旧SFT500，也不从上一组终点续训。
- 任务顺序：`open_laptop` / `handover_mic` / `dump_bin_bigbin` / `stack_bowls_two`，逐任务硬路由至对应固定PACE专家。
- 目标：4×RTX PRO6000 Blackwell 96GB，32 CPUs、建议384GB RAM；同一时间一个四卡训练进程组。输出盘建议至少2.5TiB可用，输入约75GB，runtime/assets另留余量。
- 目标机必须先完成 `validate`。本机已做资源内容校验、八组配置预检查、训练/刷新/恢复与测评协议的CPU测试及四卡Blackwell小模型FSDP/Adam完整保存与恢复（含全新optimizer；单rank峰值reserved小于70MiB）；**完整VLA的目标机训练/恢复smoke仍须执行，不得跳过。**

## 1. 固定实验清单

| 顺序 | Job ID | 方法 | Seed | Physical rounds | Student updates |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | uniform_r2_s2 | Uniform-R2 | 2 | 50 | 100 |
| 2 | adaptive_r2_s2 | Adaptive-R2 | 2 | 50 | 100 |
| 3 | uniform_r1_s2 | Uniform-R1 | 2 | 50 | 50 |
| 4 | adaptive_r1_s2 | Adaptive-R1 | 2 | 50 | 50 |
| 5 | uniform_r4_s1 | Uniform-R4 | 1 | 50 | 200 |
| 6 | adaptive_r4_s1 | Adaptive-R4 | 1 | 50 | 200 |
| 7 | uniform_r4_s2 | Uniform-R4 | 2 | 50 | 200 |
| 8 | adaptive_r4_s2 | Adaptive-R4 | 2 | 50 | 200 |

本批合计400 rounds、12800条训练轨迹、1100次更新。本机另跑的seeds0/1主矩阵与seed0 R4不重复；合并后六种方法均有seeds0/1/2。全部有效重复都保留。

共同设置：**T32、global B256、micro16/card、LR1e-5 constant、warmup4 optimizer updates**。4个actor ranks、4个env workers，每worker8个仿真slot；不能改成8个env workers，否则T32下allocator没有可调配额。Microbatch是执行配置；如显存不足，保留失败记录，用新output_root重新验证更小microbatch，再统一八组。不得改B/LR以绕过部署问题。跨microbatch比较原始 `velocity_mse` / suite-macro MSE，不能直接比较含累积缩放的 `total_loss`。

Loss为现有M3 `analytic_local_kl`；原flow时间权重、ODE10、预测50步并固定执行H25，native horizons依次700/600/600/900；FP32 master/Adam、BF16 AMP、gradient checkpointing。全参数student训练方式沿用当前配方。CUDA cache预分配/保留均为0，FSDP `limit_all_gathers=true`。

Adaptive同时改变 **rollout轨迹配额与每次更新的query数**，额外suite loss coefficients都为1；不是旧的同步loss weighting。

- 前10 rounds均匀8/8/8/8；window10、每5 rounds调整，首次用于round11。
- 正teacher-SR gap平方：rho1、epsilon0、power2；固定20 episodes/task校准reference为0.9/1.0/0.9/0.9，不重新估计。
- 每任务窗口至少40 episodes，最低轨迹配额4，单次最大变化6，总T32不变。
- R包含第一次更新。后续更新从完整最新rollout观测池重新抽观测、noise和flow transition，再用当前student刷新flow、对应teacher监督。

Seed映射与本机一致，不能沿用旧handoff的大stride：

| Seed | actor.seed | env/allocator seed | rollout noise seed |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 10042 | 1000010000 |
| 2 | 2 | 20042 | 1000020000 |

每5物理rounds保存，即5/10/15/20/25/30/35/40/45/50；R1/R2/R4对应updates5k/10k/20k。保留全部policy权重与runner/allocator JSON；每次保存完整恢复状态，只有新DCP校验成功后才移除本run旧的托管DCP，始终保留最新Adam/scheduler/RNG状态。每个新组启动前须有300GiB可用；不足即停，不删旧checkpoint。

## 2. 获取固定代码与安装

```bash
git clone --branch release/robotwin2-new4 --single-branch https://github.com/zh-xinyu/RLinf.git
cd RLinf
git checkout --detach robotwin2-new4-train-eval-20261111

# 所有/data路径替换为自己的绝对路径；用专用持久磁盘。
bash toolkits/robotwin_mopd/formal/install.sh /data/robotwin-new4-runtime blackwell-cu128-candidate
source /data/robotwin-new4-runtime/env.sh
python -m toolkits.robotwin_mopd.formal.blackwell install --runtime /data/robotwin-new4-runtime
source /data/robotwin-new4-runtime/env.sh
python toolkits/robotwin_mopd/setup/download_assets.py
(cd "$ROBOTWIN_PATH" && python script/update_embodiment_config_path.py)

git clone https://github.com/niejunnan25/PACE.git /data/PACE
git -C /data/PACE checkout fd2e624cc19325cc9b20a620c82d51c0d8e25cc1
```

已有旧Blackwell runtime可以复用，先核对相同payload，再应用缺失的overlay；不要对正在运行任务使用的环境修改安装。固定Torch2.7.1/cu128、SAPIEN3.0.0b1、OIDN2.3.3、NCCL2.31.2，以及固定RoboTwin/OpenPI版本。`preflight`验证实际文件指纹和NCCL加载版本；不要关闭fingerprint/signature检查或自行升级依赖来绕过。

本次 `formal/` 仅复用安装、Blackwell修复与小模型通信probe。**训练使用下面的 `new4.experiment`；不要运行旧 `formal.experiment` 或旧 `formal/slurm.sh`，它们的任务和自动测评计划不同。**

## 3. 下载新student与四teacher

固定HF revision：`c8b4f7559ad189921598c0a33ba82d8ff3dda5a6`。

[模型仓库与固定版本](https://huggingface.co/theo-zhou/vla-mopd-checkpoints/tree/c8b4f7559ad189921598c0a33ba82d8ff3dda5a6/robotwin2/pi05)。Student位于 `joint-sft-new4/checkpoint-400/`，teacher位于 `teachers/<task>/`。已转换为PyTorch，不需要重新下载JAX再转换。

```bash
source /data/robotwin-new4-runtime/env.sh
python -m toolkits.robotwin_mopd.new4.download_resources --resource-root /data/robotwin-new4-resources
```

下载器按代码manifest校验24个文件的大小与SHA256，固定revision，拒绝覆盖不同内容。大文件链接到HF cache，实验结束前不要删除cache。输入总计74446916018 bytes。Student权重为16573945133 bytes，SHA256：

```text
d2ef819ae9a7706accaa027edf406a1c2bc1c8f007e8bccbe695e63e4f77ce5a
```

准备阶段根据已校验的内容重新绑定模型/norm路径，保留原teacher校准计数。使用新resource_root，不与旧任务SFT500目录混用。

## 4. 机器配置与目标机 smoke

```bash
mkdir -p /data/opd-config
cp toolkits/robotwin_mopd/new4/machine.example.json /data/opd-config/robotwin-new4.json
```

填写实际绝对路径、可用CPU数、机器短名和W&B entity。例如：

```json
{
  "runtime_root": "/data/robotwin-new4-runtime",
  "resource_root": "/data/robotwin-new4-resources",
  "output_root": "/data/robotwin-new4-collaborator",
  "pace_reference_root": "/data/PACE",
  "profile": "blackwell-cu128-candidate",
  "wandb_mode": "online",
  "wandb_entity": "填写有写权限的用户或团队",
  "actor_micro_batch_size": 16,
  "ray_port": 6451,
  "dashboard_port": 8331,
  "num_cpus": 32,
  "node_label": "collab",
  "minimum_gpu_free_mib": 60000,
  "evaluation_cpus_per_task": 8,
  "evaluation_gpu_memory_fraction": 0.32,
  "evaluation_minimum_gpu_free_mib": 30000
}
```

W&B project固定 `FORMAL_robotwin_pi05_NEW`。主组group为 `new4_sft400_main_rounds50_seed2`，R4为 `new4_sft400_reuse_r4_rounds50_seed1/seed2`；run name包含方法/R/T/B/rounds/seed/机器。使用本人获授权的W&B登录，不把token写入JSON或交回材料。已有团队写权限时优先使用 `zhouxinyu2022-michigan-state-university`。

```bash
source /data/robotwin-new4-runtime/env.sh
unset WANDB_RUN_ID WANDB_RESUME WANDB_RESUME_FROM
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 替换成获分配的四张卡
python -m toolkits.robotwin_mopd.new4.experiment --machine /data/opd-config/robotwin-new4.json plan > /data/robotwin-new4-plan.json
python -m toolkits.robotwin_mopd.new4.experiment --machine /data/opd-config/robotwin-new4.json validate > /data/robotwin-new4-validation.log 2>&1
```

Validate依次进行：24文件hash校验与路径绑定、8组CPU/config/reference核对、四卡小模型NCCL/FSDP/Adam保存与全新optimizer精确恢复、teacher走实际训练 `generate/interact` 链路的一轮T32闭环（8 episodes/task）、完整π0.5的R2训练两轮/4updates与完整保存、然后用新进程加载完整状态再训练一轮至update6，最后用此checkpoint在独立的dev2场景池上完成2 episodes/task的四任务测评。

历史20场/task校准是共享setup，不重复执行。本次部署验证另产生32条teacher、96条student训练轨迹及8条dev2测评轨迹，记录为smoke/setup成本，不混入八组各1600条正式训练轨迹。

部署smoke不是正式成绩，不加入正式曲线；任一阶段失败即停，保留日志，不能手工伪造PASS或绕过gate。失败重试用新output_root；已有输出不覆盖。Teacher若0/8即停止排查。模型/Adam/scheduler恢复计数、有限loss/gradient和实际query配额均需通过核验。

`prepared/gpu_validation.json`必须为 `TARGET_GPU_VALIDATION_PASS`。它绑定源码、runtime和microbatch；这些改变后须重新验证。既有独立运行的四卡Ray实例不受本脚本清理，脚本只终止自己创建并标记owner的进程；不运行全局 `ray stop`。

## 5. 开始八组队列与返回材料

通过validate并检查日志后，在同一代码版本与machine JSON下另行执行：

```bash
python -m toolkits.robotwin_mopd.new4.experiment --machine /data/opd-config/robotwin-new4.json run > /data/robotwin-new4-training.log 2>&1
```

按第1节顺序执行 **训练 → 校验 → 10个checkpoints独立测评 → 下一组训练**。同一checkpoint的四个任务各占一张卡并行测评，每卡一个常驻仿真环境、串行100场，使用训练的EnvWorker/MultiStepRolloutWorker推理链路，不加载teacher/actor/optimizer。可指定 `run --job uniform_r2_s2` 只跑一组。队列重启仅跳过已完成且产物核验通过的组；遇到失败/部分输出即停，不覆盖、不静默重跑。每完成一个checkpoint的四任务测评，立即原子更新 `deliverables/checkpoint_results.csv` 的对应行；同时更新 `experiment_results.csv` 与 `training_summary.json`。前者记录全部80个点，后者每组一行、固定报告round50成绩与已测点数，**不自动挑最好checkpoint**。缺失结果为空，部分任务完成时不计算macro SR。需要手动汇总时：

```bash
python -m toolkits.robotwin_mopd.new4.experiment --machine /data/opd-config/robotwin-new4.json summarize
```

如训练已完整结束，只需恢复/单独补测，不重新训练：

```bash
python -m toolkits.robotwin_mopd.new4.experiment --machine /data/opd-config/robotwin-new4.json evaluate --job uniform_r2_s2
```

重启 `run` 也会跳过已核验的完整训练，复用已经通过场景/配置/计数核验的任务测评；失败/部分测评输出及日志不覆盖，须先归档失败任务再重试。每个结果必须匹配checkpoint路径、场景ID、instruction、runtime、norm与ODE10/H25。训练和测评使用不同的私有Ray端口，测评GCS为ray_port+10…13、dashboard为dashboard_port+10…13，各任务worker端口范围从ray_port+20000起每任务128个；这些端口须空闲；不要在同一output_root并发启动两个队列。

有Slurm时，从仓库根目录提交，仍只启动一个四卡launcher：

```bash
sbatch toolkits/robotwin_mopd/new4/slurm.sh /data/opd-config/robotwin-new4.json validate
# 上一个验证完成且检查通过后，另行提交：
sbatch toolkits/robotwin_mopd/new4/slurm.sh /data/opd-config/robotwin-new4.json run
```

partition/时长按集群修改；Slurm必须提供四卡CUDA mask，CPU/RAM与JSON匹配。不要在同一output_root并发运行两个队列。

中断后保留最近完整DCP、全部权重与JSON/日志，交回负责人安排续训。Full checkpoint可恢复Adam/scheduler/actor RNG及allocator历史/游标；物理rollout proposal RNG仍会重启，不承诺逐位重放。没有完整状态时，不能静默把weights-only当作完整续训，也不能把重跑算成新seed。

请返回：

- `deliverables/checkpoint_results.csv`、`experiment_results.csv`、`training_summary.json` 和所有W&B链接；自动生成的checkpoint表覆盖80个正式测评点，每行包含四任务SR、等权macro SR、pool SHA、原始结果目录与完成状态。
- `prepared/`、全部smoke结果、validation/Slurm日志，以及全部 `eval/<job>/round<N>/<task>/` 的summary/config/episodes和日志。
- 每个 `train/<run>/` 的resolved config、launch/runtime、代码快照、JSONL、driver/Ray日志；包括实际rollout trajectories、simulation steps、update/round计数、时间和显存。
- 八组的全部10个policy保存点及最新完整恢复状态；至少保留至正式测评与负责人确认归档。
- 实际机器/GPU/driver、代码commit、HF revision及所有失败/恢复说明。不要只交最好checkpoint。

训练rollout的总success_once受访问配额影响，只作在线诊断，不代替等任务权重的独立测评。

正式独立测评固定为 **round5/10/15/20/25/30/35/40/45/50 × 100 episodes/task**。本批80个policy点、32000 episodes；与本机10组训练合并后180个训练policy点，加共同初始化round0为181点、72400 episodes。Teacher正式参考另行统筹。

场景池随代码冻结在 `new4/formal100.json`：新四任务、PACE作者cache的cohort2、按原顺序取全部100场/任务、unseen instructions；pool SHA为 `4c0fe3c18b68d747d6045a8e830f20b6d6f6a082af3da6273c7beb5e2e6f43ba`。与初始化调试用cohort0及本批训练seed范围分离。所有方法/训练seeds/保存点使用同一池与相同的逐episode推理noise规则；不重新筛选成功场景，不允许不稳定场景静默换seed。`PACE_REFERENCE_ROOT`可改变本地路径，但不得重写场景、protocol或pool SHA。`dev2.json`只用于部署smoke，不入正式CSV/论文曲线。

本机训练管理goal仍暂停；本机原51点清单稍后恢复管理时迁移到101点，不能据此宣称这些新增测评已经完成。

## 给接收 coding agent 的 prompt

```text
请使用release/robotwin2-new4分支及robotwin2-new4-train-eval-20261111固定tag，阅读本handoff。不要沿用旧SFT500四任务或旧formal.run入口。新任务为open_laptop/handover_mic/dump_bin_bigbin/stack_bowls_two，统一SFT400，T32/B256/micro16、LR1e-5 constant、warmup4 updates，50物理rounds、每5轮保存。先跑new4.experiment validate，通过后按recipe中的8组队列：seed2 U2→A2→U1→A1，之后seed1 U4→A4、seed2 U4→A4；每组独立从SFT400开始。Adaptive同步rollout/query数，suite loss coefficients为1，10/10/5窗口与gap平方law；不是同步loss weighting。W&B project用FORMAL_robotwin_pi05_NEW，使用自己有权限的entity，group和name由入口生成。每组训练完立即按round5/10/…/50、每任务100场测评，再跑下一组；四任务四卡并行，用new4/evaluation.py调用worker_evaluate训练链路，固定cohort2 formal100池，不得重筛选或改seed。每个checkpoint四任务完成即原子更新checkpoint_results.csv的一行；experiment_results.csv每组一行、固定round50，不挑最好结果。已完成训练/测评可核验后复用，部分输出保留且停止，不覆盖。先检查machine JSON绝对路径和四卡分配、环境fingerprint与资源SHA，保留所有smoke/失败记录；不要跳过验证、覆盖部分run、改变T/B/LR或自动挑结果。完成后交回两份CSV、prepared、全部eval原始结果、完整日志/W&B及checkpoints。
```
