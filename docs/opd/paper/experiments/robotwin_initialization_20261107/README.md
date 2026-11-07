# RoboTwin 初始化探索：推荐新四任务 + Joint-SFT400

本轮只筛选适合 MOPD 的共同初始化，没有切换 action30 的原定正式实验。推荐任务为 `open_laptop`、`handover_mic`、`dump_bin_bigbin`、`stack_bowls_two`，分别覆盖开合、交接、倾倒和堆叠。

## 决策依据

同一固定开发集、每任务50 episodes，预测50步、固定执行25步、10步 flow ODE：

| 任务 | Teacher | Joint-SFT400 | Joint-SFT500 | Teacher − SFT400 |
|---|---:|---:|---:|---:|
| open_laptop | 41/50，82% | 28/50，56% | 35/50，70% | 26 pp |
| handover_mic | 49/50，98% | 22/50，44% | 22/50，44% | 54 pp |
| dump_bin_bigbin | 46/50，92% | 23/50，46% | 22/50，44% | 46 pp |
| stack_bowls_two | 39/50，78% | 39/50，78% | 34/50，68% | 0 pp |
| **等权平均** | **87.5%** | **56%** | **56.5%** | **31.5 pp** |

优先推荐400步：每任务已有基础能力，三个任务相对专家仍有明显差距。500步平均分近似相同，主要改变了任务间能力分布；不主张400步平均性能统计上更好。堆叠已与对应teacher持平，因此此任务检验能力保持，不能宣称四个任务都存在巨大蒸馏空间。Teacher是能力参考，不是理论上界。

## 为什么不只削减原 Joint-SFT 的步数

原任务组合的同场景快速开发评测（每任务20 episodes）：

| 初始化 | adjust_bottle | handover_mic | place_container_plate | stack_bowls_two |
|---|---:|---:|---:|---:|
| 原 Joint-SFT200 | 90% | 5% | 20% | 15% |
| 原 Joint-SFT500 | 100% | 50% | 65% | 30% |

减少更新后，容易任务仍然很强，其余任务先失去基础能力。替换原先接近饱和的两个任务，比统一削弱起点更符合本轮目标。旧任务此前formal50的teacher/student成绩与本表20场协议分开保留，不混算差距。

新组合的20场筛选曲线见 [screening.csv](screening.csv)。100/200步过弱；300步交接仍仅10%；400步已形成较完整的能力基础。400与500又分别补测同一组额外30个开发场景，确认结果并非只来自前20场。选择使用开发集，不使用formal测试集。

另测试了四个同源专家的朴素等权参数平均：四任务累计71条轨迹，0成功后停止。它不是完成的20场/任务正式估计；此结果只排除这次朴素平均作为初始化，不排除其他融合或离线蒸馏方法。本轮未进行离线蒸馏。

## 初始化来源与复现

- 从官方 `pi05_base_torch` 进行联合示范 SFT，VLM和action expert全部训练；不是专家初始化，也不是只训练action head。
- 官方 `TianxingChen/RoboTwin2.0` revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`，四任务各50条 `demo_clean`；0–44号训练、45–49号离线验证，共180条训练示范。数据不是few-shot设定，也不声称与PACE当年的示范逐条相同。
- 任务均衡采样，标准 conditional flow matching；标签 `action[t]=state[t+1]`，预测完整50步，不足时重复末动作。
- B128、microbatch4/GPU、2张H100 NVL；LR 2.5e-5、constant scheduler、warmup25、seed42；FP32 master与Adam moments、BF16 AMP、FSDP和gradient checkpointing。
- 实际从base连续训练500步，每100步保存；推荐其中400步checkpoint。复现时保留这条500步训练轨迹并取400步，不把后100步算作初始化的更新。
- Teachers来自 `niejunnan25/pi05-robotwin` revision `17ef545270cf26ed81169893b60f845f80ce6557`，每task一个30k-update/EMA0.99专家；只使用权重，不使用PACE动态执行。转换与norm校验均保留。

训练软件版本见 [training_versions.json](training_versions.json)，仿真运行时内容指纹见 [evaluation_runtime.json](evaluation_runtime.json)。

原始配置、模型/数据哈希与可移植训练入口见 [reproduction.json](reproduction.json)、[train_joint_sft.py](reproduce/train_joint_sft.py)。入口依赖本实验分支的现有 `joint_sft.py` 和数据读取器，需要配置运行时、tokenizer、base与官方数据路径；不是独立安装脚本。原始实际启动器和失败修复记录完整保存在研究输出中。`reproduce/assets/`提供经上游SHA256核对的四套norm；训练入口须使用新输出目录。

使用Python 3.11复核已保存结果（不运行模型）：

```bash
export OPD_INITIALIZATION_STUDY=/egr/research-actionlab/liyifa11/XyCode/Intern/Intern_experiments/Pi_OPD_try/outputs/robotwin4/task_init_probe_20261107
python reproduce/audit400.py --final
python reproduce/audit500.py --final
```

需从本报告所在目录执行，或改成对应脚本的绝对路径。

## 可用产物

共享输出根目录：

`outputs/robotwin4/task_init_probe_20261107/`

- **推荐模型**：`student400_bundle/`，含真实权重文件、四套norm、metadata、manifest、可移植训练入口和审计结果；仅weights，不含optimizer，适用于新建MOPD optimizer，不用于精确续训SFT。
- 对应原始checkpoint：`new_student_sft500/checkpoints/global_step_400/`。
- 已验证备选：`student500_bundle/`。
- 配置绑定：推荐模型下 `configuration_binding.action00.json`，是配置片段而非完整可直接提交的MOPD实验。
- 逐episode评测、原始完整checkpoints和所有日志均保留；本页的紧凑证据见 [confirmed.csv](confirmed.csv)、[audit400.json](audit400.json)、[audit500.json](audit500.json)。

权重SHA256：`d2ef819ae9a7706accaa027edf406a1c2bc1c8f007e8bccbe695e63e4f77ce5a`；16,573,945,133 bytes。norm SHA256见manifest，逐项匹配固定上游teacher manifest。推荐包也已通过真实RLinf `get_model` 的CPU strict reload及两份可移植配置检查，见 [bundle_load.json](bundle_load.json)。训练用FP32 master；推理loader保留既有的混合dtype行为。

## 接入后续实验时的约定

1. M0–M3、各seed共用同一选定初始化与任务组合，不混合原四任务的正式结果。
2. 同时更新train/eval任务、teacher硬路由、逐task norm，删除旧 `adjust_bottle` / `place_container_plate` 的teacher与norm项。原生任务horizon依次700/600/600/900；执行chunk仍统一H25。
3. Adaptive所需teacher参考应在新任务的训练rollout分布重新校准；不要直接复用旧四任务reference文件。这里的开发成绩也不冒充训练校准成绩。
4. 正式测评仍每checkpoint每task100 episodes，固定正式场景；新组合需要相应formal pool。本轮没有运行新组合的MOPD，也未证明其MOPD增益。
5. 本轮开发评测使用action00的规范运行时。移到action30 Blackwell后，应按其当前正式运行时重测共同参考；本页不是该运行时的formal结果。

SFT400是固定预算冷启动，不是充分联合SFT或收敛上限。它适合检验多teacher能力补强与保持，以及后续rollout efficiency；不能仅凭从此起点的提升声称优于充分Joint-SFT。

后续：独立评测开始统一到标准训练 worker 执行链路，见 [shared_worker_evaluation.md](shared_worker_evaluation.md)。原开发集结果保留；新入口复测另存，不覆盖历史记录。
