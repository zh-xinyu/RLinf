# RoboTwin 2.0 / π0.5：seed1 正式实验交接

更新：2026-11-06。本次只跑 **seed1，M0 → M2 → M1 → M3**；seed0由Xinyu执行。此文替代旧SFT2000交接，代码与下载权重都必须更新。交付附件仅本文件和 `checkpoint_results.csv`；所有脚本在仓库中。

- 代码分支：`experiment/robotwin2-mopd`。
- 固定release tag：`robotwin2-sft500-seed1-save5to50-20261106`（包含代码、测试和本版交接文档；运行时记录`git rev-parse HEAD`）。
- 初始化：**Joint-SFT500**，每task45条训练示范、固定500 updates；不是SFT2000，不是few-shot；四组共用同一权重，OPD optimizer重新初始化。
- 已验证：54项CPU测试、seed1四组配置preflight和评测seed隔离；action30已通过四卡真实π0.5更新及完整checkpoint保存。**合作者仍须先在自己的4卡Blackwell节点完成下述validate，不能直接跳过。**

## 1. 固定实验与预算

| 执行顺序 | 方法 | Rollout分配 / loss权重 | R | Rounds | Student updates |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | M0 | Uniform / 等权 | 1 | 100 | 100 |
| 2 | M2 | Adaptive / 同步loss权重 | 2 | 100 | 200 |
| 3 | M1 | Uniform / 等权 | 2 | 100 | 200 |
| 4 | M3 | Adaptive / 同步loss权重 | 1 | 100 | 100 |

- 四tasks：`adjust_bottle`、`handover_mic`、`place_container_plate`、`stack_bowls_two`；各路由对应固定PACE teacher。
- **T32**为每round四任务合计；Uniform每task8条。每组3200条student轨迹；R2每轮复用环境观测并用当前student刷新flow。Teacher校准成本独立记录。
- 4GPU；**global B256、microbatch16/card、LR1e-5 constant、warmup4 student updates**。不要自行改变T/B/LR/R、训练轮数或采样/weighting规则。
- 前50 rounds每5轮保存，之后每10轮：**5/10/15/20/25/30/35/40/45/50/60/70/80/90/100**。R1按相同update编号保存；R2对应updates10/20/30/40/50/60/70/80/90/100/120/140/160/180/200。
- 所有保存checkpoint独立测评 **100 episodes/task**；固定formal100场景/指令，执行H25、10 ODE steps。不得减episodes、换seed或挑最佳checkpoint后只交该结果。
- W&B project：`FORMAL_robotwin_pi05`；group：`robotwin_formal_sft500_100ep`；run名带`seed1`，自动分别创建run。四组都从SFT500出发，不从上一组终点接着训练。
- 单份完整checkpoint约66.3GB；四组60份约3.98TB，另加约75GB输入权重、仿真资产、环境与日志。建议训练输出盘准备至少4.5TB可用。不要自行删除尚未评测或未交接的checkpoint。

`training_seed=1`已实际映射为actor seed=1、环境/allocator基准seed=10000042、rollout noise seed=1001000000；不仅改变run名。四组使用相同seed映射，评测场景/noise规则保持固定。控制器继续使用冻结20-episode/task校准，不用正式100-episode成绩替换。

## 2. 获取代码、安装并补齐Blackwell修复

资源假设：4×RTX PRO6000 Blackwell、32 allocated CPUs、384GB RAM，driver580；Slurm partition默认`hpg-rtx6e0e`，可按集群实际资源名修改。只运行一个launcher，它负责四卡，不启动四份训练主程序。

```bash
git clone --branch experiment/robotwin2-mopd --single-branch https://github.com/zh-xinyu/RLinf.git
cd RLinf
git checkout --detach robotwin2-sft500-seed1-save5to50-20261106

# 全新专用目录。若已装过旧交接runtime，可直接激活它并执行下面的Blackwell修复。
bash toolkits/robotwin_mopd/formal/install.sh /data/robotwin-runtime blackwell-cu128-candidate
source /data/robotwin-runtime/env.sh
python -m toolkits.robotwin_mopd.formal.blackwell install --runtime /data/robotwin-runtime
source /data/robotwin-runtime/env.sh

python toolkits/robotwin_mopd/setup/download_assets.py
(cd "$ROBOTWIN_PATH" && python script/update_embodiment_config_path.py)
git clone https://github.com/niejunnan25/PACE.git /data/PACE
git -C /data/PACE checkout fd2e624cc19325cc9b20a620c82d51c0d8e25cc1
```

将所有`/data/...`替换为自己的绝对路径。修复只应用到本次专用、未在运行任务的runtime；不改系统驱动或其他项目环境。固定Torch2.7.1/cu128、SAPIEN3.0.0b1；额外使用内容hash锁定的 **OIDN2.3.3、NCCL2.31.2** 独立overlay。旧OIDN和NCCL在action30分别触发过渲染及checkpoint通信错误，不要省略修复。环境指纹须与action30验证版本一致；不能关闭fingerprint检查或自行升级包来绕过。

## 3. 下载SFT500和四个teacher

固定HF revision：`9e973cf5ff2cd606e962db6d94b93d2cf9b7bdde`。仓库：[theo-zhou/vla-mopd-checkpoints](https://huggingface.co/theo-zhou/vla-mopd-checkpoints/tree/9e973cf5ff2cd606e962db6d94b93d2cf9b7bdde/robotwin2/pi05)。

```bash
# 在仓库根目录、已激活专用runtime后执行。
python -m toolkits.robotwin_mopd.formal.download_resources   --resource-root /data/robotwin-resources-sft500
```

下载器按本次代码的manifest逐文件核对大小和SHA256；使用固定revision，teacher和tokenizer复用已有HF缓存，不需要重新转换JAX。Student来自 `robotwin2/pi05/joint-sft/checkpoint-500/`。**不要再使用旧HF download_resources.py默认的SFT2000清单。** 建立新的resource_root，避免覆盖旧SFT2000目录；资源为指向HF缓存的链接，实验结束前不要删除缓存。

## 4. 配置seed1并先做目标机validate

```bash
mkdir -p /data/opd-config
cp toolkits/robotwin_mopd/formal/machine.example.json /data/opd-config/robotwin-seed1.json
```

编辑JSON，保留其他配置并填写：

```json
{
  "runtime_root": "/data/robotwin-runtime",
  "resource_root": "/data/robotwin-resources-sft500",
  "output_root": "/data/robotwin-formal-sft500-seed1",
  "pace_reference_root": "/data/PACE",
  "profile": "blackwell-cu128-candidate",
  "wandb_mode": "online",
  "wandb_entity": "填写有权限的W&B用户或团队",
  "actor_micro_batch_size": 16,
  "training_seed": 1
}
```

使用全新output_root；不要与seed0/旧SFT2000或smoke输出混用。W&B使用自己有权限的登录，不在JSON中填写token。若驱动Vulkan ICD路径不同，增加`vulkan_icd`绝对路径。

```bash
source /data/robotwin-runtime/env.sh
# 按实际内容验证runtime，只重绑定fingerprint，保留所有原始场景与指令。
# 此命令会给machine JSON添加runtime_pool_dir。
python -m toolkits.robotwin_mopd.formal.blackwell bind   --machine /data/opd-config/robotwin-seed1.json
python -m toolkits.robotwin_mopd.formal.experiment   --machine /data/opd-config/robotwin-seed1.json plan > /data/robotwin-seed1-plan.json

unset WANDB_RUN_ID WANDB_RESUME WANDB_RESUME_FROM
# 从仓库根目录提交；Slurm必须提供四张已分配卡的CUDA_VISIBLE_DEVICES。
sbatch toolkits/robotwin_mopd/formal/slurm.sh /data/opd-config/robotwin-seed1.json validate
```

Validate依次检查资源hash、四组CPU配置、四卡FSDP/通信/精确checkpoint恢复、四个teacher各2场闭环，以及完整π0.5的两次更新和checkpoint保存；全部通过才继续。smoke不计正式结果。先检查渲染没有OIDN错误，loss/gradient有限，保存路径完整。任何阶段失败保留日志并反馈，不反复覆盖输出试到成功。

## 5. 跑seed1队列与交回结果

Validate完成并检查后，另行提交：

```bash
source /data/robotwin-runtime/env.sh
unset WANDB_RUN_ID WANDB_RESUME WANDB_RESUME_FROM
sbatch toolkits/robotwin_mopd/formal/slurm.sh /data/opd-config/robotwin-seed1.json run
```

实际顺序为：**M0训练→M0全部15个checkpoints评测→M2训练/评测→M1训练/评测→M3训练/评测**。共60个task-macro checkpoint结果、240个task-checkpoint jobs、24,000 episodes。每个checkpoint四任务并行评测，一次只评一个checkpoint；失败即停，不跳过或覆盖已有结果。

共同SFT500/teacher/SFT2000的正式baseline评测由Xinyu单独统筹，本队列不重复；teacher部署smoke仍必做。脚本最终summarize生成62行CSV，其中60行为本次主实验，额外student/teachers两行允许PENDING，之后合并共同参考成绩；附件模板已标记这两行为`SHARED_REFERENCE`。

交回：`deliverables/checkpoint_results.csv`、完整`eval/` JSON/log、`train/`中的resolved config/launch/runtime/code快照/训练JSONL/driver与Ray日志、`prepared/`、Slurm日志、W&B链接、运行机器信息和所有异常/恢复说明。模型与optimizer checkpoint至少保留到评测及负责人确认归档。训练中断后先报告，由负责人决定续训；不要把中断run当成额外seed或静默重跑挑结果。

## 6. seed2：第二批可选，不自动运行

推荐最终3个training seeds（seed0/1/2），比两seed更能反映训练波动，但不保证统计显著性。优先完成seed1四组；有时间再补seed2。资源紧时seed2先M0/M2，再M1/M3，报告各比较实际重复数，不混成所有方法都有三seed。

seed2需要负责人确认后另建machine JSON，改`training_seed: 2`和全新output_root，重复bind/validate/run；共享同一SFT500/teachers和固定评测池，无需重新SFT。代码会使用actor seed2、环境/allocator20000042、rollout noise1002000000，并拒绝在已冻结seed1输出下换seed。本次交付的默认machine为seed1，没有自动衔接seed2或参数扫描。

保存点补充：本seed1交接已增加round5/15/25/35/45，训练总预算不变；正在运行的seed0仍沿用原保存计划，不为增加checkpoint重启。跨seed主曲线先在共同rounds10/20/…/100汇总，seed1的5/15/25/35/45作为额外早期观察，缺失点不伪造。

## 给原coding agent的更新prompt

```text
请更新到release tag robotwin2-sft500-seed1-save5to50-20261106，并重新阅读本交接文档。seed1执行顺序仍为M0→M2→M1→M3，SFT500、T32、B256、micro16、LR1e-5和100rounds不变。保存计划改为前50rollout rounds每5轮保存，之后每10轮，即5/10/15/20/25/30/35/40/45/50/60/70/80/90/100；R2按round×2映射student update。每checkpoint仍为100episodes/task，使用新版CSV。新tag包含此前的实际training seed切换、HF SFT500下载及Blackwell修复。先做目标机validate再运行；seed2不自动启动。如果已经启动旧配方，请报告进度并保留输出，不要直接改动运行中的配置或覆盖旧结果。
```
