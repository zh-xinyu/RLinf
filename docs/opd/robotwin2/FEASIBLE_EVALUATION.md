# RoboTwin 固定可行场景补充评测

2026-11-01。本轮训练继续使用 **unfiltered scenes + seen instructions**；已经运行的 matching unfiltered teacher/student 结果保留为 pilot 参照。新增评测采用 **固定 scripted-expert-screened scenes + unseen instructions**，不混在同一成绩表中，也不因此修改训练。

## 协议与用途

当前 pinned RoboTwin `dca9ec682688821c42944eacd7f5dd9bfb15396f` 的 `script/eval_policy.py` 默认先执行 `setup_demo → play_once`，只有 `plan_success && check_success()` 才接受该初始场景，然后重新初始化同一个 seed 给被测策略。PACE 的公开 Fixed-H25 评测同样采用专家筛选，语言 split 是 unseen。

这里的 expert 指 benchmark 的 scripted planner，不是我们融合的四个 VLA teachers。“可行”准确含义是本次 scripted expert 成功，不是对物理可解性的数学保证。筛选结果不依赖 teacher、student 的表现；policy 在固定池中的失败全部计入分母。

`toolkits/robotwin_mopd/feasible_pool.py` 默认对四任务各接受 5 个场景。候选取现有 RLinf `eval_seeds.json` 原顺序，但**每个候选都在当前 Aloha-clean 环境中重新跑 planner 检验**。因此它是采用官方筛选规则的固定小规模子集，不宣称与作者论文使用完全相同的场景或随机语言采样。

- 保存：accepted seed、成功 `play_once()` 返回的 `info` placeholder 字典、固定 unseen prompt、失败原因、筛选耗时、配置和源码哈希。
- 不保存或使用示范动作、图像或轨迹；`save_data=False`。筛选只服务于评测初始场景验收，不进入 MOPD loss 或 replay buffer。
- 四任务使用相同版本、Aloha embodiment、clean 配置、三相机和现有 400/600/400/900 horizon。teacher/base/student 必须共用同一个 pool 内容哈希。
- 5 episodes/task 仅用于快速补充验证；不作为最终论文中充分统计的成绩。

## 运行与结果完整性

激活隔离 runtime，等训练和当前评测释放 GPU 后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/feasible_pool.py \
  --output outputs/robotwin4/eval/feasible_pool_attempt1.json \
  --episodes-per-task 5
```

脚本启动前检查选中 GPU 没有计算进程且显存占用不超过 256 MiB。忙则直接退出；不会中断别人的任务。默认每任务最多检查前 150 个候选。已经存在的输出文件不会覆盖；失败时保留 partial JSON，下一次使用新 attempt 路径。

完整池有 `complete=true`。顶层 `pool_content_sha256` 是排除该字段后，对 `json.dumps(..., sort_keys=True, separators=(',', ':'))` 的 SHA256；换行/排版不影响哈希。`load_pool()` 会拒绝内容被改动、未完成、缺场景或 seed 重复的文件。`validate_pool_protocol()` 再核对 task config、horizon、RoboTwin git revision 和关键源码文件哈希。

评测必须按 pool 的顺序重置原 seed，使用缓存 prompt，且不允许发生不稳定后静默换 seed。运行时使用缓存记录的 `screening_overrides`（包括 `is_test=True`、`eval_mode=True`）；先验证原始 config，再对运行副本应用这些字段。筛选和重放共享这些标记，不改变训练默认。筛选成本保留在 `screening_audit[*].screening_seconds` 中，独立于训练 rollout 预算。

## 为什么保留成功 planner 的 episode info

已只读核对四个任务：它们的 `get_info()` 在初始场景中均提供对象和左右臂 placeholders；没有发现这四任务需要额外、仅由 `play_once` 初始化的语言字段。`handover_mic` 的成功判据依赖的左右臂属性在 `load_actors()` 就初始化；`stack_bowls_two` 的 `las_arm` 是 planner 内部变量，policy 成功判据不依赖它。

不过，initial `get_info()` 和实际成功 planner 使用的字段不能未经验证视为完全等同。新的 pool 总是使用成功 `play_once()['info']` 生成 prompt，并记录 `initial_info_matches_expert` 和 `initial_info_missing_keys` 供实际运行复核。这样不会把运动后的物体位置重新解释成初始左右臂指令。

## 当前验证范围

- 9 项 CPU 单测通过：planner/task 双成功条件、异常清理、prompt 来自成功 expert info、动作不保留、pool 哈希、partial/duplicate 拒绝、显卡占用拒绝，以及筛选与 NativeTaskSlot 固定重放共享 `is_test` 等运行标记。
- 开发时没有占用训练 GPU，也没有生成真实场景池；真实 planner 验证由后续空闲队列执行。
- 配置与代码接口已按 docs-check 核对；本文件是内部实验说明，无独立 EN/ZH Sphinx 页面。
