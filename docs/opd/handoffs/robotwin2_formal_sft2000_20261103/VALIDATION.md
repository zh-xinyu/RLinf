# 交接验证记录（2026-11-03）

## 已通过

- 新分支以opd-project-main为祖先，继承既有RoboTwin实现；root主分支与pilot代码未改动。
- 19个模型/tokenizer文件，74,446,894,047 bytes，逐文件SHA256核验通过。
- cohort2 formal100：4任务各100场；与DEV20无交集；保守覆盖640次task occurrence及0–19 reset retries的seed0训练范围与formal100无交集。
- 原runtime下完整CPU preflight通过：三组真实Hydra CLI、校准重绑定及训练签名、SAPIEN安装内容/源码指纹、固定场景/指令协议。
- 新建临时runtime，从零安装Torch2.7.1+cu128、torchvision0.22.1、SAPIEN3.0.0b1及固定补丁；未更改原runtime。
- 新runtime下41项测试通过，包含真实小型OpenPI velocity/梯度一致性、scheduler CPU约束、配置、场景池、结果身份、tokenizer迁移回归测试。
- shell语法、Ruff与文档命令/配置核对通过（最终状态由本次提交记录固定）。

## Smoke发现并修复

暂存资源中的tokenizer是外部符号链接；OpenPI resolve后要求文件仍在缓存目录内，首次smoke失败。prepare现将已校验的tokenizer实体复制至output_root/prepared/openpi-data，权重仍可保留链接。回归测试覆盖外部链接、幂等准备和损坏文件拒绝。

失败日志原样保留在`outputs/robotwin4/handoff_20261103/local_validation/shared_gpu_smoke/`，不当成任务失败或正式结果。

## GPU检查已完成（A6000）

- 原runtime：GPU4，四teachers和SFT2000各任务2个DEV episodes，共16场，串行；显存fraction0.35，每次模型启动前要求至少24GiB free。
- 新cu128 runtime：GPU7，teacher adjust_bottle和SFT2000 handover_mic各2场，共4场；同样显存限制。
- 原runtime共16场、新runtime共4场，全部正常退出且成功；只作为链路smoke，不估计正式SR。峰值PyTorch allocated约7.12GiB、reserved约7.23GiB，未出现OOM。
- 新runtime四卡NCCL/FSDP/Adam/保存的小模型检查通过，每卡allocated约17.4MiB；不是完整VLA训练。
- 新runtime完整CPU preflight也通过，同一formal100与校准签名均匹配。
- 两个输出目录为`local_validation/shared_gpu_smoke_retry1/`与`candidate_validation/candidate_shared_smoke/`，相对于`outputs/robotwin4/handoff_20261103/`。
- allocator统计不包括Vulkan/上下文开销；nvidia-smi余量另行观察。不会停止其他人的进程。

## 尚不能宣称已验证

- Blackwell硬件上的渲染、GPU计算及四卡NCCL/FSDP训练。
- 新handoff四卡2-update训练/保存smoke：当前其他任务占用所有卡，完整FP32参数/optimizer峰值余量不足。
- 100/task正式训练曲线和独立测评结果：尚未启动；新formal100不能复用旧formal50得分。
- 多training seeds：首轮完成后再设计，不自动启动。

此处是研究交接profile，非新增通用RLinf model/env安装目标；未改Docker/CI矩阵。新的候选环境已有安装与CPU测试证据，但不等价于目标Blackwell节点部署验收。

完整本机完成记录：`outputs/robotwin4/handoff_20261103/validation_summary.json`。安装/CPU/GPU检查均已结束，无正式训练/测评队列在后台运行。

文档核对范围为本交接与对应代码/配置；研究文档无EN/ZH双语对应页。No doc-code or EN-ZH consistency issues found in checked scope.
