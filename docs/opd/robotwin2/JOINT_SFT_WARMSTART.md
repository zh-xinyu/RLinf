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
- 500-step长跑已启动：`robotwin4_joint_sft_500_b128_micro4_pilot_20261101`，GPU0–3，W&B [33rml11k](https://wandb.ai/zhouxinyu2022-michigan-state-university/RoboTwin_MultiTeacher_Pilots/runs/33rml11k)。启动时四rank FP32 master/Adam验证通过。**状态是运行中，不是500步已完成，也没有联合策略SR提升结论。**
- 启动后前5步检查正常：平均26.8秒/update，loss为0.1692、0.1604、0.1378、0.1609、0.1392，gradient norm均有限（1.26–1.55）。最大allocated稳定30.128GB、reserved稳定48.331GB。LR按25-step warmup从0增至4e-6；W&B已启动且内部日志无ERROR。这里只确认启动/数值健康，不把少数不同训练batch的loss变化当作验证集或SR提升。
- 长跑状态：根workspace `outputs/robotwin4/setup/joint_sft/long_pilot_status_20261101.json`；原始训练和验证指标位于 `outputs/robotwin4/train/robotwin4_joint_sft_500_b128_micro4_pilot_20261101/`。日志和状态文件优先于本页的静态进度。
