# 完整 flow 诊断（训练后运行）

目的：解释当前 projected velocity MSE 与任务 SR 的差距，不改变训练 loss，也不把诊断当作成功率评测。

模型生成 50×32 的 latent trajectory，当前 M3 返回前 25 个位置的 velocity，实际执行前 25×14。这个工具在 student 访问的同一个 `(observation, x_t, t)` 上比较完整 student/teacher velocity，并比较相同初始 noise 下分别积分得到的完整 endpoint。

- 默认每任务取固定评测池第一个场景、前 4 个决策观测；最多执行 3 个 chunk / 75 个 control targets，原生终止时提前停止。
- 时间索引 `0 4 9` 对应 `1.0, 0.6, 0.1`；不是全时间积分，也不能直接当训练 MSE 的估计。
- 分开报告 first25/last25 × physical14/padded18 的 MSE、RMS、cosine，以及各时间点和 endpoint 的统计。
- 保存场景 seed、固定 unseen prompt、pool hash、checkpoint、每个 noise seed、实际控制步数和样本数。
- “physical14”仍是**归一化 latent 坐标**，不是物理单位的关节误差。
- 不报告截短 episode 的成功率；不能仅凭未监督部分误差较大就宣称因果关系。

激活隔离 runtime，确认指定 GPU 已空闲后，在 robotwin4 worktree 运行：

```bash
CUDA_VISIBLE_DEVICES=0 python toolkits/robotwin_mopd/flow_probe.py \
  --task adjust_bottle \
  --checkpoint /absolute/path/to/global_step_40 \
  --evaluation-pool /absolute/path/to/evaluation_pool.json \
  --probes 4 --flow-indices 0 4 9 \
  --output /absolute/path/to/new_probe_adjust_bottle.json
```

`--checkpoint base` 可生成单独的基模参考。默认 `--episode-index 0`；指定另一个 accepted scene 可检查现象是否仅来自单场景。工具要求输出路径不存在，并在加载模型前拒绝有占用的 GPU。

当前验证：5 项 CPU 测试覆盖四象限统计、同 noise/同 student path/真实时间网格、最多三个执行 chunk，以及原生终止；Ruff 和 CLI 检查通过。尚未做真实 GPU probe；等当前训练和两套评测释放资源后再运行。训练与已排队评测没有改动。
