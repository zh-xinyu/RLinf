# RoboTwin 2.0 四任务正式实验交接

状态：交接候选版，本机配置、协议与小规模GPU检查已通过；Blackwell 实机验收未完成，尚未发布远端固定版本。不要把本文件理解为已经验证过 Blackwell 的正式 release。

代码分支：`dev/robotwin2-formal-handoff`。从 `opd-project-main@5d1d1494` 建立，fast-forward 纳入已验证的 RoboTwin 开发提交至 `9345146b`，再添加本交接入口。原 workspace 和原 pilot worktree 不切换、不改训练程序。

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

共同参数：4 GPUs，4 env workers，T32（每轮32轨迹），B128，microbatch4，LR1e-5、constant scheduler、warmup4 **student updates**。每组640条训练轨迹。M3 `analytic_local_kl` loss；全50步action horizon监督，执行H25，10步ODE；FP32可训练参数/optimizer，BF16 AMP，FULL_SHARD、gradient checkpointing。保持已验证的 `enable_gradient_accumulation=false`：仍累积8个microbatches/rank/update，但每个microbatch同步并分片梯度。

Adaptive：rho0.5、epsilon0.01、warmup/window/interval均5 **rollout rounds**，min_task_episodes5，单次quota变化上限4，最低quota4/task，权重 `alpha_i=4*q_i/32`。使用原20/task训练校准；正式teacher评测不回填控制器。校准成本80条轨迹、18,318个控制步单列。

第一轮只执行 **seed0**。完成并检查训练/评测链路后，再设计额外training seeds；暂不自动排队seed1/seed2。不同评测场景不等于不同training seeds。

## 2. 所有正式评测均为100 episodes/task

- 共同SFT起点round0、teacher参考、三组的round5/10/**15**/20，均每task100次；每份四任务共400场。
- **14份结果、56个task-level jobs、5,600 episodes。** 旧方案跳过round15；本交接按“所有保存点都评测”纳入round15。
- 冻结池为 `toolkits/robotwin_mopd/formal/formal100.json`：作者cache cohort2，顺序取0–99，固定unseen指令、固定每episode Torch RNG。此前DEV20和formal50成绩属于不同池，不能直接作为这次round0/teacher参考。
- 所有方法、checkpoint、后续training seeds使用同一池。固定H25、10 ODE，task horizons依次400/600/400/900；保留真实失败，不换seed、不根据teacher成功与否挑场景。
- 协议是作者已接受场景的cache replay，不声称本地重新通过scripted expert筛选，也不运行PACE动态horizon算法。
- 主终点固定round20；中间点用于学习曲线，不按正式成绩挑选峰值代替终点。
- 四任务各用一张卡并行；不同checkpoints顺序评测，训练和评测不争用GPU。

论文横轴为rollout rounds；必须保留实际trajectories、simulation/control steps和wall time。训练success_once不能替代独立评测。不同硬件的pilot wall time不和正式Blackwell直接混算。

## 3. 资源与环境准备

合作者目标：Slurm单节点4张Blackwell，32分配CPU，384GB RAM。现有完整训练checkpoint约66GB，四个保存点约265GB/run，首轮三组约795GB；另外需要74.45GB起点/teacher资源、环境和仿真资产空间。GPU更快不保证32 CPU下同比提速，以目标机实测为准。`ntasks=1`，启动器读取`SLURM_CPUS_PER_TASK`并限制Ray CPU数量；线程默认OMP/MKL/OpenBLAS各1。保留Slurm提供的CUDA_VISIBLE_DEVICES。当前训练入口要求四个**数字GPU ID**；若集群提供UUID或其他映射，应先适配/检查，不手工猜测物理卡号。

准备代码的固定commit/git bundle后，在仓库根目录运行。当前分支尚未发布到GitHub，不能直接把未发布branch当作可clone的release。

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

负责人提供74.45GB模型/tokenizer资源（精确总量见resources.json）：

- `teachers/<task>/`：已数值验证的PyTorch转换，含各task normalization和conversion receipt。
- `student/actor/model_state_dict/full_weights.pt`、`student/metadata.json`：SFT2000初始化；无需SFT optimizer和重复权重副本。
- `openpi-data/big_vision/paligemma_tokenizer.model`。

文件路径及SHA256在`toolkits/robotwin_mopd/formal/resources.json`；prepare逐文件读取验证，不能用仅文件名或文件大小代替。teacher来源为`niejunnan25/pi05-robotwin@17ef545270cf26ed81169893b60f845f80ce6557`，原始来源与转换说明见[checkpoint说明](../../robotwin2/CHECKPOINT_SURVEY_20261101.md)。SFT2000是45训练+5离线验证示范/task，累计500+1500 updates，中间重启optimizer；不能写成充分收敛的全量SFT。

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

prepare保留原始校准，只重绑定已验证权重/normalization路径，并把tokenizer实体文件准备到输出目录，兼容资源符号链接和只读挂载；preflight检查三组Hydra配置、实际runtime指纹、100/task评测池、训练/评测seed隔离及teacher校准协议。collective-smoke使用极小模型检查四卡NCCL/FSDP/Adam及文件保存，不代表完整VLA训练验证。smoke跑2次student updates（R2为1个rollout round）并保存checkpoint，输出在独立smoke目录，不计正式结果。它只验证uniform刷新链路，不证明adaptive反馈收益。

`teacher-smoke`使用固定DEV20池前2场/task，共8场，只做部署验收、不计正式结果。目标机还须检查其teacher闭环推理/三路图像正常，再运行正式参考；不要通过选择正式池中成功场景调试。任何runtime或推理行为更改都需重新验收，不能静默绕过fingerprint检查。

验收通过后：

```bash
python3 "$entry" --machine "$machine" eval teachers
python3 "$entry" --machine "$machine" eval student
for method in m0 m1 m2; do
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

1. `output_root/deliverables/checkpoint_results.csv`：14行，逐task SR和等权macro SR，分母均100，必须全部COMPLETE。
2. 全部`eval/`原始JSON及log，包含逐episode实际seed、指令、成功标记、控制步数、checkpoint身份和pool hash。
3. `train/`非checkpoint资料：resolved config、launch/runtime/code快照、JSONL counters、TensorBoard/W&B本地日志、driver/Ray日志，及异常/重启说明。
4. `prepared/`校准路径绑定、资源校验与preflight记录；目标机实际包freeze、GPU/driver信息、Slurm资源记录及验收日志。

空白CSV模板不是结果。summarize缺测留空；拒绝不同checkpoint、不同池、重复/换seed、错分母和不一致成功计数。当前CSV主要汇总独立评测，训练时间/交互预算由必须交回的原始JSONL提取，不伪造未运行数据。模型在本地至少保留到独立测评完成；不要求交回optimizer。

首轮完成后，冻结是否延续同一recipe，并设计匹配的额外training seeds（优先总计3个seed）；M0/M1/M2同时增加，不能只重复表现最好的方法。初始化与teacher参考同协议可共享，多seed不重训SFT。不要把新超参数run混入原配置seed统计。

具体检查状态见[VALIDATION.md](VALIDATION.md)。
