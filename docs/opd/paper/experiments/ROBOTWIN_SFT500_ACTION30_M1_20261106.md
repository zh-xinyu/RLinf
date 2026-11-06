# RoboTwin SFT500 M1：action30 重新启动（2026-11-06）

当前目标仅M1；先做四卡完整模型smoke，通过后自动从SFT500启动正式训练。独立测评暂缓，不自动接M2/M3。

- 代码：`experiment/robotwin2-sft500-formal@f4db2f31`；worktree `outputs/worktrees/robotwin_sft500_formal`。
- 节点：action30，物理GPU0/1/2/3（用户批准迁移），4张RTX PRO6000 Blackwell；GPU启动阈值55,000MiB空闲/卡，不要求清空他人进程。
- 方法：M1 Uniform-Refresh-R2；100 rollout rounds，200 student updates，每10轮保存（updates20/40/…/200）。
- 初始化：相同SFT500权重，从头新建OPD optimizer；`resume_dir=null`，不接action10失败M1。
- 参数：T32、B256、microbatch16/card、LR1e-5、constant、warmup4 updates、seed0；任务/teacher/normalization不变。
- 环境：Torch2.7.1/cu128、SAPIEN3.0.0b1+隔离OIDN2.3.3，场景池按当前实际软件指纹显式绑定；详见[节点验收](../../robotwin2/ACTION30_ENVIRONMENT_20261106.md)。旧M0的渲染环境不同，保留此差别，不声称逐位等价。
- W&B project：`FORMAL_robotwin_pi05`；group：`robotwin_formal_sft500_100ep`；新run ID，`resume=never`；tags包括action30/blackwell/oidn233/fresh_start。
- 共享入口：`outputs/robotwin4/formal_sft500_action30_20261106/`，正式配置 `machine.action30.json`，smoke配置 `machine.smoke.action30.json`（W&B disabled）。
- 本地完整训练和smoke：`/localscratch/liyifa11/XyFile/opd-robotwin4-action30-20261106/formal_sft500_action30_20261106/`。共享入口的train/smoke是指向该节点的符号链接，其他机器未必能访问。
- 每分钟将正式训练顶层日志/配置/JSONL复制到共享 `training_records/m1/`，checkpoint位置另有JSON指针；不把完整optimizer重复写入共享盘。
- 队列：共享入口 `setup/gpu0123/run_m1.py`，实时状态 `queue_status.json`，队列日志 `setup/gpu0123/queue.log`。仅在smoke状态COMPLETE、两次loss/gradient均有限、无OIDN错误、完整模型/optimizer checkpoint落盘且空间足够时进入正式M1；失败即停。
- `setup/*_gpu_memory.jsonl`记录整张卡使用量，包含其他作业，不能当成本实验单卡独占峰值。
- action10失败M1不计入此次正式曲线；没有保存checkpoint，仅保留小体积故障日志并写入 `superseded_by_action30.json`。不删除已完成M0或参考资源。

正式启动及smoke结果见后续补充。本页不是实时监控，`queue_status.json`为状态入口。

## 启动故障与重试

原完整smoke已完成两次更新，但NCCL2.26.2在DCP保存阶段illegal memory access；不是OOM，未创建正式W&B run。失败smoke保存在本地smoke目录的 `failed_nccl2262_` 前缀下，原日志保留。

action30独立runtime固定加载NCCL2.31.2+cuda12.9（LD_PRELOAD overlay），Torch/模型/算法/正式参数不变。小规模旧/新通信库对照及实际FSDP模型、Adam、scheduler精确保存恢复已通过，证据 `setup/nccl_probe/`；实际运行库版本以ncclGetVersion为准。

2026-11-06 22:09 UTC：新队列PID1483642开始prepare→preflight→完整smoke→M1；独立测评仍暂缓。用户再次确认T32。全模型smoke和正式启动结果待追加。

## 完整smoke通过，正式M1等待显存

2026-11-06 22:19:52 UTC，完整smoke COMPLETE并通过队列检查：32 trajectories（每task8）、10,448 simulation steps、两次更新loss/gradient有限；DCP optimizer、full weights和runner状态合计66,315,442,526 bytes。smoke的success为22/32，仅用于部署检查，不计入正式成绩。实际四个actor的`/proc/maps`均确认加载NCCL2.31.2。

正式M1计划从SFT500重新开始，已预留W&B ID（尚未确认服务端创建）：[4oge6bkt](https://wandb.ai/zhouxinyu2022-michigan-state-university/FORMAL_robotwin_pi05/runs/4oge6bkt)，队列PID1483642。smoke启动commit为`af4f240a`（包含smoke改进与记录）；训练代码未因NCCL修复改变。T32/R2/B256/micro16/LR1e-5、100rounds/200updates、每10rounds保存不变。当前仅排队M1，未排队M2/M3或独立评测。

启动状态澄清：22:19:52 UTC进入的是formal launcher；GPU4/5随后仅约52GB空闲，低于55,000MiB门槛，因此尚未创建正式训练目录、未开始student更新。原队列1483642保持GPU2–5等待。改用GPU0–3的操作被自动审批拒绝，未执行；已向用户请求明确授权。无其他用户进程被停止。

## GPU迁移获批，正式训练实际启动

用户明确批准改用GPU0–3。旧GPU2–5启动器于22:24 UTC等待超时退出，未创建正式训练目录，预留ID `4oge6bkt` 未用于实际训练。新四卡的通信与生产FSDP模型/Adam/scheduler精确保存恢复均PASS，见 `setup/nccl_probe/production_gpu0123/rank*.json`。

2026-11-06 22:38:48 UTC新队列PID1518174启动；正式run目录已创建、状态RUNNING、四卡actor/rollout正在初始化，已通过显存门槛。实际CUDA_VISIBLE_DEVICES=`0,1,2,3`，launch commit=`d3dfd7a6`。W&B新ID：[0iwqz1g9](https://wandb.ai/zhouxinyu2022-michigan-state-university/FORMAL_robotwin_pi05/runs/0iwqz1g9)。实际resolved config已核对T32/R2/B256/micro16/LR1e-5/200updates，每20updates保存，resume=null；证据 `setup/gpu0123/resolved_config_verified.json`。未改训练算法、未停止其他用户作业；仅M1，独立测评继续暂缓。
