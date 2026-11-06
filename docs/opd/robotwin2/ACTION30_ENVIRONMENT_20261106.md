# action30 RoboTwin 2.0 环境（2026-11-06）

用途：在新节点复用 Blackwell 交接环境，承接 SFT500 正式实验。当前阶段只配置和验收环境，**未启动正式训练或正式独立测评**。

## 代码与固定版本

- 以 `experiment/robotwin2-sft500-formal@83844b91` 为基线，另提交本次运行时绑定与测试修复；worktree：`outputs/worktrees/robotwin_sft500_formal`。
- 该分支继承交接分支 `experiment/robotwin2-mopd@a84ecc7c`；继续使用其 `blackwell-cu128-candidate` 安装方案，保留 SFT500 起点、100 rounds、每10轮保存等后续约定。不是退回 SFT2000 交接配方。
- 根目录 `release/robocasa-formal` 未切换，其他模型环境未修改。
- Python3.11.14，Torch2.7.1+cu128，torchvision0.22.1，SAPIEN3.0.0b1，Transformers4.53.2；GPU为6×RTX PRO6000 Blackwell 96GB，driver580.126.20。
- RoboTwin：`dca9ec682688821c42944eacd7f5dd9bfb15396f`；OpenPI：`c5dc4b9296a1a4739bf52828f28a579f12dce763`；PACE reference：`fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`。
- 使用已交接的 Transformers overlay、SAPIEN UTF-8、MPLib 和 native-step logging 补丁。

## 路径与激活

- 本地 runtime：`/localscratch/liyifa11/XyFile/opd-robotwin4-action30-20261106`。
- teacher / SFT500 / tokenizer：上述目录的 `resources/`，从action10复制，按当前分支 `formal/resources.json` 逐文件SHA256校验。
- 仿真资产：上述目录的 `repos/RoboTwin/assets/`，从已验证的action10环境复制，并重新生成指向action30的embodiment配置。
- 共享验收记录：`outputs/robotwin4/action30_environment_20261106/`。
- 节点验证配置：上述共享目录的 `machine.action30.oidn233.json`，W&B disabled；这不是正式训练输出配置。

从 workspace 根目录激活：

```bash
source .opd.robotwin.action30.env
cd outputs/worktrees/robotwin_sft500_formal
python toolkits/robotwin_mopd/formal/experiment.py --machine "$OPD_ROBOTWIN_MACHINE_JSON" plan
```

`.opd.robotwin.action30.env` 带 hostname 检查；不自动选择GPU、不自动启动任务。共享盘可见此文件，不代表其他节点具有action30的本地runtime。

## 验收记录

- 安装通过；实际依赖freeze保存在本地 `runtime_freeze.txt`，概要见共享验收目录 `runtime_versions.json`。
- 39项正式配方/运行时指纹/GPU检查测试通过：`runtime_tests.log`。
- GPU2/3/4/5四卡NCCL/FSDP/Adam/保存小模型检查通过，每卡PyTorch峰值约17.4MiB：`smoke/tiny_fsdp/rank*.json`。这不代表完整π0.5训练已验收。
- 权重哈希、CPU preflight与闭环smoke结果见下方后续补充。

## 正式训练前仍需完成

- 当前存在其他作业，显存动态变化；环境配置不授权占满或停止其他作业。
- 在四张卡显存足够时做完整π0.5 smoke（2 student updates + checkpoint保存），实测Blackwell峰值和吞吐。action10同参数历史峰值41.03GiB/card，仅作选卡参考。
- 正式运行另建输出配置，保留 `FORMAL_robotwin_pi05`、SFT500和已冻结配方。先训练M1；按用户要求独立测评暂缓。
- action10的M0已完成；M1因OOM中断，原始日志保留在旧输出。不同GPU/软件栈的wall time需分开标注；不可当成同机速度对比。
- 本节点本地盘当前约1.7TiB可用，所有M1–M3共30份完整checkpoint约1.99TB（十进制）；开正式队列前重新安排本地/共享存储并检查容量，不能直接沿用action10的路径配置。

## Blackwell 去噪兼容性修复

原交接环境CPU与CUDA检查通过，但真实闭环发现旧OIDN2.0.1报 `unsupported device type: CUDA` / `invalid handle`。旧smoke虽完成推理，不能作为渲染验收通过；原日志和结果保留在顶层 `smoke/`。

- 上游依据：[SAPIEN issue243](https://github.com/haosulab/SAPIEN/issues/243)、[OIDN2.3.3 release](https://github.com/RenderKit/oidn/releases/tag/v2.3.3)。后者明确增加Blackwell支持。
- 未升级SAPIEN/PhysX或改camera、shader、相机姿态；将完整SAPIEN包复制到本地 `vendor/sapien-3.0.0b1-utf8-oidn233`，添加OIDN2.3.3三份GPU库，并把 `_oidn_tricks.py` 加载版本改成2.3.3。原venv的SAPIEN保留。
- 下载包与三份库的SHA256、补丁说明见vendor中的 `opd_sapien_manifest.json`；激活文件 `env.sh` 显式选择该vendor，`env.before_oidn233.sh`保留旧激活方式。
- **去噪版本改变可能改变像素，不能声称与action10完全相同的渲染。** 保留冻结formal100/dev20的全部场景、指令、horizon、seed、噪声规则，仅将运行时指纹重新绑定为当前实际vendor，生成共享目录 `runtime_pools_oidn233/`。原仓库池文件不修改。
- machine配置增加显式 `runtime_pool_dir`。正式入口验证该目录中的池除了 `protocol.runtime_profile` 与对应内容hash外，和原池完全一致；拒绝场景/指令/协议变化。仍由preflight和evaluator检查真实加载包的指纹，不跳过任何校验。新绑定hash进入正式配方冻结记录，不能在同一输出目录静默换回旧去噪器。
- 原20/task校准成绩保留其原始runtime provenance，不冒称已在新去噪器上重测。正式teacher/student/checkpoint参考需按同一新运行时测评；旧M0的跨环境可比性需在正式启动前明确，尤其不能直接混比较训练wall time。
- 修复后独立验收目录：`oidn233_validation/`；全部部署检查只使用DEV池，每个teacher2场，student handover_mic2场。GPU5串行、PyTorch allocator上限15%，保存三路相机图片并拒绝任何OIDN错误。
- 新增运行时绑定完整性/冻结测试后，共41项测试通过（`runtime_tests_oidn233.log`）。完整π0.5训练smoke仍待四卡余量允许时运行。

重建补丁的脚本保存在共享验收目录 `install_oidn233_overlay.py`，需提供 `--runtime <已按交接方案安装的新runtime>`。脚本校验固定下载hash并拒绝覆盖现有vendor；之后仍需显式激活、生成新运行时绑定并完成GPU验收，不因复制成功就视为可训练。

## 本次验收结论

- 19个资源文件SHA256全部匹配；OIDN2.3.3新绑定下CPU preflight通过，41项回归测试通过。
- 四卡tiny NCCL/FSDP/Adam/保存通过；实际进程映射确认加载三份OIDN2.3.3动态库（`oidn233_loaded_libraries.json`）。
- 修复后DEV smoke：teacher adjust_bottle 2/2、handover_mic 2/2、place_container_plate 2/2、stack_bowls_two 1/2；student SFT500 handover_mic 1/2。全部轨迹正常结束、动作有限，失败轨迹原样保留，不据此调整场景或估计正式成功率。
- 修复后无OIDN错误，三路图像已保存，adjust_bottle的头部和双腕相机图片已人工查看。图片和逐episode记录在 `oidn233_validation/smoke/`。
- PyTorch reserved峰值约7.24GiB，仅代表单卡推理smoke，不包含Vulkan/context，不代表训练峰值。
- 总状态：**部署/推理验收通过，四卡完整模型训练与checkpoint保存验收待做。** 无正式训练/正式独立测评在本轮启动。机器可用显存变化时再次检查，不自动排队。
- 汇总：`outputs/robotwin4/action30_environment_20261106/validation_summary.json`。

## NCCL checkpoint 修复（2026-11-06）

完整模型 smoke 的两次更新成功，但原 NCCL 2.26.2+cuda12.2 在 DCP 保存的 object collective 中发生 illegal memory access；未进入正式训练。这不是 OOM。小规模四卡复现也失败。

仅在 action30 独立 runtime 的 `vendor/nccl-2.31.2/` 安装 `nvidia-nccl-cu12==2.31.2`，通过 `env.sh` 的 `LD_PRELOAD` 加载。原 Torch2.7.1/cu128 和原 NCCL wheel 保留，旧 env 备份为 `env.before_nccl2312.sh`。上游参考：[NVIDIA NCCL issue2418](https://github.com/NVIDIA/nccl/issues/2418)。本机库路径/hash与对照结果见正式输出 `setup/nccl_probe/runtime_overlay.json`。

- 同一探针：原库 FAIL；新库的四卡 all-reduce/scatter（至4MiB）及2000 tensors的DCP save/load PASS。
- 增强 `formal/collective_smoke.py`：实际 FSDPStrategy 的 DCP/full-weight 保存、修改模型与Adam状态后恢复、逐值核对模型/Adam/scheduler全部 PASS。四卡实际 `ncclGetVersion=23102`。Torch的 `torch.cuda.nccl.version()` 仍报告编译时2.26.2，不能据此判断实际运行库。
- 正式超参数不变：T32/B256/micro16/LR1e-5/R2；完整模型 smoke 仍须成功后才能启动M1。
