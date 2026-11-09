# 新四任务 R2 allocation：开发结果

状态：INTERIM；生成时间 2026-11-09T20:13:56.255492+00:00。每任务20次，训练seed0。

| 模型 / 轮次 | Open | Handover | Dump | Stack | Macro | Max gap |
|---|---:|---:|---:|---:|---:|---:|
| teachers | 80.0% | 100.0% | 90.0% | 95.0% | 91.3% | 0.0% |
| initial400 | 50.0% | 45.0% | 55.0% | 70.0% | 55.0% | 55.0% |
| uniform_round10 | 85.0% | 30.0% | 35.0% | 50.0% | 50.0% | 70.0% |
| uniform_round30 | 75.0% | 85.0% | 55.0% | 70.0% | 71.3% | 35.0% |
| sr_round10 | 85.0% | 45.0% | 60.0% | 70.0% | 65.0% | 55.0% |
| residual_round10 | 75.0% | 50.0% | 45.0% | 55.0% | 56.2% | 50.0% |

| 方法 | 已完成更新 | 已采集轮次 | 训练轨迹 | 训练 control targets | G-AUC (0–30) |
|---|---:|---:|---:|---:|---:|
| uniform | 60 | 30 | 960 | 383720 | 0.5583 |
| sr | 44 | 22 | 704 | 285863 | 待完成 |
| residual | 24 | 13 | 416 | 186525 | 待完成 |

SR校准额外成本：80轨迹 / 19335 control targets。开发评测成本另计，未并入训练交互。

## 解读边界

- Development selection data; the formal 100-scene pool is not used.
- SR and Uniform use Blackwell/micro16; MSE uses Ada/micro8. Cross-hardware differences are not causal allocation evidence.
- The MSE controller is uniform through round10. Its round10 differences cannot be attributed to allocation.
- Identical seeds do not ensure identical numerical prefixes, even on the same hardware.
- SR consumes an additional teacher calibration: 80 trajectories / 19335 control targets.
- Raw selected query counts are configured per optimizer update, not unique observations; weighted exposure is only a proxy.
- Failed/restarted attempt costs remain separately archived and are not merged into successful-run progress.
- A single seed and 20 episodes/task do not establish a robust ranking.
