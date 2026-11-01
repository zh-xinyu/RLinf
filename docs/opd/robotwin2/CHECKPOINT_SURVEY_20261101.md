# RoboTwin 2.0 checkpoint 调研

> 调研阶段存档：后续已选定四个PACE specialists并完成两段pilot。当前结果见 [PILOT_RESULTS_20261101](PILOT_RESULTS_20261101.md)，下文候选建议不代表待执行任务。


更新：2026-11-01。只查询公开资源与模型文件清单；尚未下载权重、转换模型或本地评测。开发分支为 `dev/mopd-robotwin2-four-task`，从 `opd-project-main@5d1d1494` 创建。

## 当前结论

优先考察π0.5：有同一来源的单任务specialists、作者评测与推理代码，也有联合SFT初始化候选。GR00T找到社区权重，但尚未确认可直接组成四任务teachers的完整资源。公开成绩不代替本地复现。

## Teacher首选：PACE单任务SFT模型

- [模型仓库](https://huggingface.co/niejunnan25/pi05-robotwin)，固定revision `17ef545270cf26ed81169893b60f845f80ce6557`。
- [manifest](https://huggingface.co/niejunnan25/pi05-robotwin/blob/17ef545270cf26ed81169893b60f845f80ce6557/manifest.json)与HF API文件清单交叉核对：50个任务的manifest文件均在仓库清单中。旧搜索缓存显示18个，不能沿用。
- 作者记录每任务50条demo_clean、全参数SFT 30k updates、EMA权重；每任务约12.4 GB，四个约50 GB。
- JAX/Orbax OCDBT；三相机、14维双臂joint/gripper状态，预测50步。作者配置使用delta joint动作，解码输出绝对joint动作；不同teacher保留各自norm_stats。
- [作者配置](https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/pace/src/openpi_pace/configs.py)与[推理说明](https://github.com/niejunnan25/Openpi-PACE)可用于建立原生推理参考。
- PACE为执行时方法，固定horizon对照使用相同SFT权重。我们可以只使用模型，先固定执行horizon，不引入PACE算法。

[论文附录B.3](https://arxiv.org/html/2606.00537v2#A2.SS3)提供固定H=25的SR（作者协议，非本项目实测）：

| 任务 | 作者H25 SR | 选任务时的考虑 |
| --- | ---: | --- |
| adjust_bottle | 98.8% | 已有RLinf单任务配置，适合先验证接口 |
| handover_mic | 97.7% | 双臂交接候选 |
| place_container_plate | 81.7% | 放置任务候选 |
| stack_bowls_two | 88.4% | 堆叠任务候选 |
| handover_block | 33.2% | 旧候选；该checkpoint在H25下较弱 |
| lift_pot | 61.2% | 旧候选，仍可考虑双臂协同覆盖 |
| pick_dual_bottles | 45.3% | 旧候选；不能预设teacher高成功率 |

任务未冻结。根据接口可用性、操作类型、teacher能力及相对共同student的学习空间选择，不能根据MOPD收益事后换任务。执行horizon在方法比较前固定，不把预测50步等同于必须执行50步。

## 其他候选

| 资源 | 已确认内容 | 当前定位 |
| --- | --- | --- |
| [RLinf-Pi05-RoboTwin-SFT-adjust_bottle](https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle) | 公开PyTorch分片与norm_stats，已有本项目评测配置 | 单任务接口验证备选；尚未找到该来源四个π0.5 specialists |
| [Myungkyu联合SFT](https://huggingface.co/Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k) | 模型卡称50 tasks×50 clean demos、60k updates；JAX、chunk50、delta joints；有权重与中间checkpoint目录 | 共同student候选；需确认仿真版本/embodiment及本地SR，不宣称已复现成功 |
| [Motus联合SFT](https://huggingface.co/motus-robotics/pi0.5_robotwin2) | 模型卡称50 tasks，每task50 clean+500 randomized；PyTorch、chunk32，有模型与norm_stats文件 | student备选；数据量与chunk长度不同，不能直接当作匹配起点 |
| [社区GR00T N1.5](https://huggingface.co/locht131/gr00t_fa_robotwin_2hand_quat_s35k_bs32) | 有权重/config/metadata，无model card | 任务覆盖、训练协议及SR不清楚，不作为最快路径首选 |

已核对版本：RLinf `fa8df6ed103db0f5549c122f3a17c00ba6426c98`；Myungkyu `9affbed083307f662b0a24f7429bae4c22c914fe`；Motus `effc6e1a99bd4d0edf9447e8ba313def67649cfe`；社区GR00T `5327c432ae2e96d16fe8ddab7392fef287675804`。权重仅核对文件清单，尚未下载验hash。

## 接下来需要验证的边界

1. 用一个PACE teacher在作者兼容环境、固定horizon下建立原生推理参考，再验证JAX→本项目PyTorch转换的velocity/action一致性。
2. 核对RoboTwin revision、embodiment、相机、delta/absolute动作变换与task-specific归一化。teacher与student的flow坐标不能未经转换直接MSE；更不能直接给teacher替换成共同norm_stats。
3. 检查共同student候选在候选四任务的能力，确认specialists有互补性/能力优势。如果公开联合模型不匹配或已强于teachers，则考虑用同一四任务数据训练共同SFT起点；不人为选择低质量起点制造收益。
4. 单任务OPD与Refresh正确后，接四任务路由、Uniform-R2、Adaptive-Sync-R2。当前代码的完整池Refresh仍需增加π0.5支持。

这次发现显著降低重新训练四个teachers的可能成本，但没有消除模型转换、共同初始化及训练接口适配工作。


## 后续补查：PyTorch / LeRobot 与 base 初始化

用户倾向先直接从开源π0.5 base做student，不把Joint-SFT设为前置条件。这在研究设置上成立：RoboCasa采用Joint-SFT初始化以降低实验成本，RoboTwin考察不做目标任务student SFT的专家融合。不能据此声称整个流程不使用demonstrations：teachers本身经过SFT，归一化资产的来源也需要记录。同一benchmark内各方法必须共享起点；base起点是否有效由短程闭环训练验证。

### 已确认公开的PyTorch单任务模型

[VLADrop](https://github.com/s1ghhh/VLADrop)发布7个 `s1ghhh/VLADrop_pi05_RoboTwin_<task>_Baseline` 模型，API清单包含 `model.safetensors`、`metadata.pt` 和 `assets/robotwin_v1/<task>_demo_clean_repo/norm_stats.json`。模型卡标记为未裁剪full model、单任务30k更新。以下为作者模型卡报告的Easy/Hard SR（各100 episodes），不是本项目复现结果：

| task | Easy | Hard |
| --- | ---: | ---: |
| [open_microwave](https://huggingface.co/s1ghhh/VLADrop_pi05_RoboTwin_open-microwave_Baseline) | 82% | 60% |
| [move_playingcard_away](https://huggingface.co/s1ghhh/VLADrop_pi05_RoboTwin_move-playingcard-away_Baseline) | 70% | 44% |
| [click_bell](https://huggingface.co/s1ghhh/VLADrop_pi05_RoboTwin_click-bell_Baseline) | 68% | 50% |
| [beat_block_hammer](https://huggingface.co/s1ghhh/VLADrop_pi05_RoboTwin_beat-block-hammer_Baseline) | 32% | 16% |
| [lift_pot](https://huggingface.co/s1ghhh/VLADrop_pi05_RoboTwin_lift-pot_Baseline) | 20% | 18% |
| [turn_switch](https://huggingface.co/s1ghhh/VLADrop_pi05_RoboTwin_turn-switch_Baseline) | 18% | 8% |
| [place_dual_shoes](https://huggingface.co/s1ghhh/VLADrop_pi05_RoboTwin_place-dual-shoes_Baseline) | 2% | 0% |

推理示例仍写 `pi05_libero_dropped`，说明文字称assets为LIBERO，但实际列出的资产路径是对应RoboTwin任务。这存在文档不一致，需要检查作者实际配置，不能直接复制示例即认定兼容。只考虑Baseline，不混入裁剪架构。模型来源/初始权重、embodiment、内部delta或absolute动作约定需进一步核验。

### 联合模型与未验证资源

- [SidneyXie/pi05_robotwin](https://huggingface.co/SidneyXie/pi05_robotwin)，revision `e49e2ab6c11f07511573b67261bd129e88d0a416`：LeRobot/PyTorch，base为 `lerobot/pi05_base`，50k更新，14D、三相机、chunk50。文件包含完整pre/post processors。作者报告Easy32任务×100 episodes平均67.91%；adjust_bottle100%、beat_block_hammer93%、place_container_plate86%。是一个联合模型，不能复制四份称为四个specialists；若使用，适合作为单teacher接口验证或Joint-SFT备选。其MEAN_STD/absolute配置不可与OpenPI delta/quantile配置直接混用。
- [TACO权重](https://huggingface.co/rhodes-team-teleai/pi05_TACO_robotwin2_finetuned)，revision `0f000e2748bd1fcb43027d8790f81fcccaa04670`：LeRobot主策略一个，train_config数据为demo_clean_50tasks；五个 `cfns/*` 是辅助网络，不是五个VLA teachers。
- [Motus](https://huggingface.co/motus-robotics/pi0.5_robotwin2)：OpenPI/PyTorch联合模型，前文已记录。
- [RACE_Robotwin](https://huggingface.co/SeonghoonYu/RACE_Robotwin)：搜索缓存称发布六任务PyTorch集合，但本次实时模型API和README均返回401，不能计入当前可直接下载资源。
- `Hoshipu/pi05-robotwin2-clean-30k` / `random-60k`：实际文件为JAX/Orbax，不是PyTorch。
- `lixinyangzyy/pi05_robotwin`仅README，`kaiwen2/prefvla-robotwin-pi05`仅gitattributes；没有可用权重清单。

### 当前建议

接受base student作为优先验证起点。teacher方面，PyTorch格式不自动意味着适合四任务融合：可以先验证RLinf adjust_bottle或VLADrop较强Baseline的单任务闭环；四个teachers最终按来源明确、动作接口一致和本地能力选择。PACE仍有更完整的单任务资源与论文证据，不应仅为了省去一次转换而选择明显较弱或协议不清楚的teachers。LeRobot与RLinf/OpenPI的模型权重命名、前处理及velocity接口仍需适配，不是替换文件路径即可。
