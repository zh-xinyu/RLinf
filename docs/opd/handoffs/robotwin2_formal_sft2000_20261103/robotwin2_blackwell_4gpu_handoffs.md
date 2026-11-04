# RoboTwin 2.0 四任务正式实验交接

交付代码：`experiment/robotwin2-mopd` 分支的四组修正版；按第3节获取并记录 commit，全部实验保持同一版本。旧 `efbac9d6` 仅含三组，不能用于本交接。本指南包含实验设计、安装、权重下载、目标机验收、运行及交回结果的完整流程。随附 `checkpoint_results.csv` 为结果模板。74.45 GB 模型/tokenizer 通过 Hugging Face 单独下载，命令见第3节。

状态：交接候选版，本机配置、协议与小规模GPU检查已通过；Blackwell 实机验收未完成；代码通过下述实验分支交付。不要把本文件理解为已经验证过 Blackwell 的正式 release。

代码分支：`experiment/robotwin2-mopd`。从 `opd-project-main@5d1d1494` 建立，fast-forward 纳入已验证的 RoboTwin 开发提交至 `9345146b`，再添加本交接入口。原 workspace 和原 pilot worktree 不切换、不改训练程序。

> **合作者必读：必须先在你们自己的 4×RTX PRO 6000 Blackwell 计算节点完成 smoke test，才能启动正式实验。** 当前已准备专用 cu128 环境与四卡/32 CPU Slurm 入口，但我们的验证硬件是 RTX A6000，尚未在你们的 Blackwell 节点验证。
>
> 按第3节完成安装与资源准备后，先提交 `sbatch toolkits/robotwin_mopd/formal/slurm.sh /data/opd-config/robotwin.json validate`。检查四卡 NCCL/FSDP smoke、四个 teacher 的闭环 smoke，以及**完整 π0.5 的两次 student 更新和 checkpoint 保存**全部完成；同时确认图像和推理正常。仅通过安装、CPU preflight 或小模型检查不够。
>
> 验收通过并查看日志后，再单独提交同一命令的 `run` 阶段。若失败，保留日志并反馈，不绕过检查启动正式实验。验收数据不计入正式100 episodes/task结果。

## 1. 实验清单

四任务：`adjust_bottle`、`handover_mic`、`place_container_plate`、`stack_bowls_two`。π0.5，四个固定 task specialists，按任务路由。共同 student 为 **Joint-SFT2000**，每组重新创建 OPD optimizer；不恢复 SFT optimizer。保留现有全参数训练，不冻结 student VLM。

| ID | 分配 / loss 权重 | R | Rollout rounds | Student updates | 保存 / 测评 updates |
|---|---|---:|---:|---:|---|
| M0 | Uniform / 等权 | 1 | 20 | 20 | 5 / 10 / 15 / 20 |
| M1 | Uniform / 等权 | 2，Refresh | 20 | 40 | 10 / 20 / 30 / 40 |
| M2 | Adaptive / 同步权重 | 2，Refresh | 20 | 40 | 10 / 20 / 30 / 40 |
| M3 | Adaptive / 同步权重 | 1 | 20 | 20 | 5 / 10 / 15 / 20 |

四组构成 Uniform / Adaptive-Sync × R1 / R2 的完整 2×2 矩阵；M0/M1、M3/M2比较复用，M0/M3、M1/M2比较联合分配。这里实验编号 M3 与下文的 M3 loss 名称是两个概念。

共同参数：4 GPUs，4 env workers，T32（每轮32轨迹），B128，microbatch4，LR1e-5、constant scheduler、warmup4 **student updates**。每组640条训练轨迹。M3 `analytic_local_kl` loss；全50步action horizon监督，执行H25，10步ODE；FP32可训练参数/optimizer，BF16 AMP，FULL_SHARD、gradient checkpointing。保持已验证的 `enable_gradient_accumulation=false`：仍累积8个microbatches/rank/update，但每个microbatch同步并分片梯度。

Adaptive：rho0.5、epsilon0.01、warmup/window/interval均5 **rollout rounds**，min_task_episodes5，单次quota变化上限4，最低quota4/task，权重 `alpha_i=4*q_i/32`。使用原20/task训练校准；正式teacher评测不回填控制器。校准成本80条轨迹、18,318个控制步单列。

第一轮只执行 **seed0**。完成并检查训练/评测链路后，再设计额外training seeds；暂不自动排队seed1/seed2。不同评测场景不等于不同training seeds。

## 2. 所有正式评测均为100 episodes/task

- 共同SFT起点round0、teacher参考、四组的round5/10/**15**/20，均每task100次；每份四任务共400场。
- **18份结果、72个task-level jobs、7,200 episodes。** 旧方案跳过round15；本交接按“所有保存点都评测”纳入round15。
- 冻结池为 `toolkits/robotwin_mopd/formal/formal100.json`：作者cache cohort2，顺序取0–99，固定unseen指令、固定每episode Torch RNG。此前DEV20和formal50成绩属于不同池，不能直接作为这次round0/teacher参考。
- 所有方法、checkpoint、后续training seeds使用同一池。固定H25、10 ODE，task horizons依次400/600/400/900；保留真实失败，不换seed、不根据teacher成功与否挑场景。
- 协议是作者已接受场景的cache replay，不声称本地重新通过scripted expert筛选，也不运行PACE动态horizon算法。
- 主终点固定round20；中间点用于学习曲线，不按正式成绩挑选峰值代替终点。
- 四任务各用一张卡并行；不同checkpoints顺序评测，训练和评测不争用GPU。

论文横轴为rollout rounds；必须保留实际trajectories、simulation/control steps和wall time。训练success_once不能替代独立评测。不同硬件的pilot wall time不和正式Blackwell直接混算。

## 3. 资源与环境准备

合作者目标：Slurm单节点4张Blackwell，32分配CPU，384GB RAM。现有完整训练checkpoint约66GB，四个保存点约265GB/run，首轮四组约1,060GB；另外需要74.45GB起点/teacher资源、环境和仿真资产空间。GPU更快不保证32 CPU下同比提速，以目标机实测为准。`ntasks=1`，启动器读取`SLURM_CPUS_PER_TASK`并限制Ray CPU数量；线程默认OMP/MKL/OpenBLAS各1。保留Slurm提供的CUDA_VISIBLE_DEVICES。当前训练入口要求四个**数字GPU ID**；若集群提供UUID或其他映射，应先适配/检查，不手工猜测物理卡号。

从实验分支获取代码，记录本次交付的commit，并在全部实验期间保持不变。在仓库根目录执行后续命令：

```bash
git clone --branch experiment/robotwin2-mopd --single-branch https://github.com/zh-xinyu/RLinf.git
cd RLinf
git checkout --detach origin/experiment/robotwin2-mopd
git rev-parse HEAD
```

代码可获取不代表Blackwell实机验收已完成；仍须先执行目标节点的validate阶段。

```bash
# 新目录安装；脚本拒绝覆盖已有环境。
bash toolkits/robotwin_mopd/formal/install.sh /data/robotwin-runtime blackwell-cu128-candidate
export ROBOTWIN_RUNTIME_ROOT=/data/robotwin-runtime
source "$ROBOTWIN_RUNTIME_ROOT/env.sh"
python toolkits/robotwin_mopd/setup/download_assets.py
(cd "$ROBOTWIN_PATH" && python script/update_embodiment_config_path.py)

# 固定作者参考；也可接收负责人提供的同commit checkout。
git clone https://github.com/niejunnan25/PACE.git /data/PACE
git -C /data/PACE checkout fd2e624cc19325cc9b20a620c82d51c0d8e25cc1

mkdir -p /data/opd-config
cp toolkits/robotwin_mopd/formal/machine.example.json /data/opd-config/robotwin.json
```

编辑machine JSON，填写runtime_root、resource_root、output_root、pace_reference_root。`profile`保留`blackwell-cu128-candidate`；W&B默认disabled，也可offline/online，entity由执行者填写，不存token。启用时project沿用`RoboTwin_MultiTeacher_Pilots`，group为`robotwin_formal_sft2000_100ep`。

环境固定Python3.11.14；Blackwell候选使用Torch2.7.1/cu128、torchvision0.22.1；SAPIEN3.0.0b1及UTF-8补丁、固定RoboTwin/OpenPI commit、Transformers4.53.2 overlay沿用已验证协议。公共包snapshot在`runtime-common.lock.txt`，GPU依赖按官方wheel及本次实际安装生成的`cu128.constraints.txt`固定，安装后另存实际freeze。该snapshot继承原环境中非必需包，并非最小依赖集合。系统须有相应编译库、FFmpeg、Vulkan/EGL驱动；脚本不安装驱动。若Vulkan ICD不在`/usr/share/vulkan/icd.d/nvidia_icd.json`，应在machine JSON增加`"vulkan_icd": "/actual/path/nvidia_icd.json"`，入口会设置`VK_ICD_FILENAMES`和`VK_DRIVER_FILES`。

安装参考：[PyTorch固定版本](https://pytorch.org/get-started/previous-versions/)、[NVIDIA驱动兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。Blackwell的GPU计算、渲染、NCCL/FSDP与保存恢复必须实机验收，CPU preflight不能替代。

### 3.1 下载已转换 checkpoints

仓库：[theo-zhou/vla-mopd-checkpoints](https://huggingface.co/theo-zhou/vla-mopd-checkpoints/tree/ebf161f8743f80c9eb3fbb5a25886d172144d571/robotwin2/pi05)

固定 revision：`ebf161f8743f80c9eb3fbb5a25886d172144d571`。19 个训练资源文件，总计 74,446,894,047 bytes；本地和远端 SHA256 均已核对。

#### 下载

在已安装 `huggingface_hub` 的环境中执行（若仓库访问需要认证，先使用自己的 HF 登录）：

```bash
hf download theo-zhou/vla-mopd-checkpoints \
  robotwin2/pi05/download_resources.py \
  --revision ebf161f8743f80c9eb3fbb5a25886d172144d571 --local-dir ./robotwin-release

python ./robotwin-release/robotwin2/pi05/download_resources.py \
  --revision ebf161f8743f80c9eb3fbb5a25886d172144d571 \
  --resource-root /absolute/path/robotwin-resources
```

将 machine JSON 中 `resource_root` 设为 `/absolute/path/robotwin-resources`，随后按第4节运行 `prepare`、`preflight` 和目标机器 smoke test。下载脚本将 HF 中的层级映射为训练入口需要的 `teachers/`、`student/`、`openpi-data/`。映射文件指向 HF 缓存；不要删除该缓存。

#### 内容

- `robotwin2/pi05/teachers/<task>/checkpoint-30000/`：四个 PACE 专家已转换的 PyTorch 权重、任务 normalization 和来源记录。无需自行执行 JAX 转换。
- `robotwin2/pi05/joint-sft/checkpoint-2000/`：四任务联合 SFT 初始化，累计 500+1500 updates；保留原始 metadata，其中 step=1500 指第二阶段局部步数。仅交付模型权重，不含 SFT optimizer。
- `robotwin2/pi05/openpi-data/`：配套 tokenizer。
- `robotwin2/pi05/resources.json`：精确大小、SHA256 以及训练目录/HF 目录映射。
- `robotwin2/pi05/README.md`、`TERMS.md`、`NOTICE`：使用说明、修改说明与上游模型条款。


文件路径及SHA256在`toolkits/robotwin_mopd/formal/resources.json`；prepare逐文件读取验证，不能用仅文件名或文件大小代替。teacher来源为`niejunnan25/pi05-robotwin@17ef545270cf26ed81169893b60f845f80ce6557`，原始来源与转换说明见[checkpoint说明](https://github.com/zh-xinyu/RLinf/blob/efbac9d69d8b1ef6cbe6d61f1263352c707cebda/docs/opd/robotwin2/CHECKPOINT_SURVEY_20261101.md)。SFT2000是45训练+5离线验证示范/task，累计500+1500 updates，中间重启optimizer；不能写成充分收敛的全量SFT。

## 4. 先验收，再跑首轮

```bash
entry=toolkits/robotwin_mopd/formal/experiment.py
machine=/data/opd-config/robotwin.json
python3 "$entry" --machine "$machine" plan > /data/robotwin-plan.json
python3 "$entry" --machine "$machine" prepare
# 必须在已分配四卡的计算节点执行；继承CUDA_VISIBLE_DEVICES。
python3 "$entry" --machine "$machine" preflight
python3 "$entry" --machine "$machine" collective-smoke
python3 "$entry" --machine "$machine" teacher-smoke
python3 "$entry" --machine "$machine" smoke m1
```

prepare保留原始校准，只重绑定已验证权重/normalization路径，并把tokenizer实体文件准备到输出目录，兼容资源符号链接和只读挂载；preflight检查四组Hydra配置、实际runtime指纹、100/task评测池、训练/评测seed隔离及teacher校准协议。collective-smoke使用极小模型检查四卡NCCL/FSDP/Adam及文件保存，不代表完整VLA训练验证。smoke跑2次student updates（R2为1个rollout round）并保存checkpoint，输出在独立smoke目录，不计正式结果。它只验证uniform刷新链路，不证明adaptive反馈收益。

`teacher-smoke`使用固定DEV20池前2场/task，共8场，只做部署验收、不计正式结果。目标机还须检查其teacher闭环推理/三路图像正常，再运行正式参考；不要通过选择正式池中成功场景调试。任何runtime或推理行为更改都需重新验收，不能静默绕过fingerprint检查。

验收通过后：

```bash
python3 "$entry" --machine "$machine" eval teachers
python3 "$entry" --machine "$machine" eval student
for method in m0 m1 m2 m3; do
  python3 "$entry" --machine "$machine" train "$method"
  python3 "$entry" --machine "$machine" eval "$method"
done
python3 "$entry" --machine "$machine" summarize
```

`slurm.sh`分为验收和正式执行两个阶段，避免手工验收后再次运行同名smoke导致输出冲突。也可以用validate阶段替代上面的手工smoke命令；不要两种方式重复执行。先提交validate并检查日志，确认Blackwell部署通过后再提交run：

```bash
sbatch toolkits/robotwin_mopd/formal/slurm.sh /data/opd-config/robotwin.json validate
# validate完成并检查结果后，另行提交：
sbatch toolkits/robotwin_mopd/formal/slurm.sh /data/opd-config/robotwin.json run
```

partition名称沿用合作者提供的`hpg-rtx6e0e`，提交前核对集群实际名称/时限。不要同时手工跑相同任务。启动器不删除其他人的Ray，发现占用会拒绝启动。已有训练或评测输出不覆盖；故障后保留现场，由负责人明确恢复/重排，不能无记录重复试到成功。

## 5. 交付与进度

交回：

1. `output_root/deliverables/checkpoint_results.csv`：18行，逐task SR和等权macro SR，分母均100，必须全部COMPLETE。
2. 全部`eval/`原始JSON及log，包含逐episode实际seed、指令、成功标记、控制步数、checkpoint身份和pool hash。
3. `train/`非checkpoint资料：resolved config、launch/runtime/code快照、JSONL counters、TensorBoard/W&B本地日志、driver/Ray日志，及异常/重启说明。
4. `prepared/`校准路径绑定、资源校验与preflight记录；目标机实际包freeze、GPU/driver信息、Slurm资源记录及验收日志。

空白CSV模板不是结果。summarize缺测留空；拒绝不同checkpoint、不同池、重复/换seed、错分母和不一致成功计数。当前CSV主要汇总独立评测，训练时间/交互预算由必须交回的原始JSONL提取，不伪造未运行数据。模型在本地至少保留到独立测评完成；不要求交回optimizer。

首轮完成后，冻结是否延续同一recipe，并设计匹配的额外training seeds（优先总计3个seed）；M0/M1/M2/M3同时增加，不能只重复表现最好的方法。初始化与teacher参考同协议可共享，多seed不重训SFT。不要把新超参数run混入原配置seed统计。

## 6. 已完成的验证与适用范围

### 已通过

- 新分支以opd-project-main为祖先，继承既有RoboTwin实现；root主分支与pilot代码未改动。
- 19个模型/tokenizer文件，74,446,894,047 bytes，逐文件SHA256核验通过。
- cohort2 formal100：4任务各100场；与DEV20无交集；保守覆盖640次task occurrence及0–19 reset retries的seed0训练范围与formal100无交集。
- 原runtime下完整CPU preflight通过：三组真实Hydra CLI、校准重绑定及训练签名、SAPIEN安装内容/源码指纹、固定场景/指令协议。
- 新建临时runtime，从零安装Torch2.7.1+cu128、torchvision0.22.1、SAPIEN3.0.0b1及固定补丁；未更改原runtime。
- 新runtime下41项测试通过，包含真实小型OpenPI velocity/梯度一致性、scheduler CPU约束、配置、场景池、结果身份、tokenizer迁移回归测试。
- shell语法、Ruff与文档命令/配置核对通过（最终状态由本次提交记录固定）。

### Smoke发现并修复

暂存资源中的tokenizer是外部符号链接；OpenPI resolve后要求文件仍在缓存目录内，首次smoke失败。prepare现将已校验的tokenizer实体复制至output_root/prepared/openpi-data，权重仍可保留链接。回归测试覆盖外部链接、幂等准备和损坏文件拒绝。

失败日志原样保留在`outputs/robotwin4/handoff_20261103/local_validation/shared_gpu_smoke/`，不当成任务失败或正式结果。

### GPU检查已完成（A6000）

- 原runtime：GPU4，四teachers和SFT2000各任务2个DEV episodes，共16场，串行；显存fraction0.35，每次模型启动前要求至少24GiB free。
- 新cu128 runtime：GPU7，teacher adjust_bottle和SFT2000 handover_mic各2场，共4场；同样显存限制。
- 原runtime共16场、新runtime共4场，全部正常退出且成功；只作为链路smoke，不估计正式SR。峰值PyTorch allocated约7.12GiB、reserved约7.23GiB，未出现OOM。
- 新runtime四卡NCCL/FSDP/Adam/保存的小模型检查通过，每卡allocated约17.4MiB；不是完整VLA训练。
- 新runtime完整CPU preflight也通过，同一formal100与校准签名均匹配。
- 两个输出目录为`local_validation/shared_gpu_smoke_retry1/`与`candidate_validation/candidate_shared_smoke/`，相对于`outputs/robotwin4/handoff_20261103/`。
- allocator统计不包括Vulkan/上下文开销；nvidia-smi余量另行观察。不会停止其他人的进程。

### 尚不能宣称已验证

- Blackwell硬件上的渲染、GPU计算及四卡NCCL/FSDP训练。
- 新handoff四卡2-update训练/保存smoke：2026-11-03 本机检查时，其他任务占用所有卡，完整 FP32 参数/optimizer 峰值余量不足。
- 100/task正式训练曲线和独立测评结果：尚未启动；新formal100不能复用旧formal50得分。
- 多training seeds：首轮完成后再设计，不自动启动。

此处是研究交接profile，非新增通用RLinf model/env安装目标；未改Docker/CI矩阵。新的候选环境已有安装与CPU测试证据，但不等价于目标Blackwell节点部署验收。

完整本机完成记录：`outputs/robotwin4/handoff_20261103/validation_summary.json`。上述记录对应 2026-11-03 的安装/CPU/GPU 检查，不代表收件方目标节点的当前运行状态。


2026-11-04：模型资源发布至第3节指定的 Hugging Face revision，19 个文件的远端大小与 SHA256 均验证通过；下载脚本的路径映射、重复运行和覆盖保护检查通过。原 RoboCasa 资源未改动。

2026-11-04 四组修正：补齐 Adaptive-Sync R1（M3），配方、CLI plan/train/eval、Slurm 队列与结果汇总同步支持完整 2×2 矩阵。其余训练参数、HF 资源 revision 和 formal100 场景池保持不变；本次未启动训练或正式评测。

四组修正验证：10 项正式交接回归测试通过；真实 runtime 下四组 Hydra 配置均成功解析，M3 和 M2 的 teacher 校准签名与 rates 一致。M3 的完整 GPU 训练尚未执行，仍须完成目标机验收。
