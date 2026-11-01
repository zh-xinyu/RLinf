# RoboTwin π0.5 pilot 的训练精度

2026-11-01：建议正式 pilot 使用 **FP32 trainable master weights + BF16 AMP compute**，在实际 8 卡 smoke 通过后启用。不要通过放大 LR 来补偿 BF16 参数更新被舍入的问题。

## 原因与实现边界

- 当前 OpenPI loader 默认把大部分 backbone 参数转为 BF16；FSDP 不会自动创建额外的 FP32 master copy，普通 AdamW 的 moment dtype 也跟随参数。
- LR `2.5e-5` 小于许多 BF16 权重的半个量化间隔；warmup 时更新更小。例如约 `0.02` 的权重，单次 `2.5e-5` 更新可被舍入为零。并非所有参数都不更新，但这是可以避免的优化限制。
- 原 OpenPI PyTorch 训练脚本默认也是 BF16 参数直接 AdamW，并允许 `pytorch_training_precision=float32`；这不等于 BF16 master 是本项目短程蒸馏的最佳选择。
- 新增 `actor.cast_trainable_params_to_fp32`，默认 `false`。只在 actor 的 `model_provider_func` 内、FSDP wrapping 和 optimizer 创建之前，对 `requires_grad` 浮点参数做 FP32 转换。保留 Parameter 对象和正常 tied-weight 引用；冻结参数、buffers 不变。teacher/rollout 不经过这个 provider。
- 该转换保留 loader 已加载的数值，不恢复此前已发生的低精度舍入。它的目的在于让接下来的小更新能够累计；并不改变模型结构、loss 或 LR。

## 推荐配置组合

以下是配置约定；本次工具改动不自动修改或重启运行中的实验。

```yaml
actor:
  cast_trainable_params_to_fp32: true
  model:
    # 检查实际 master 与 Adam moments，而不是只依靠配置名。
    require_fp32_trainable_params: true
    precision: null  # 保留 OpenPI 的现有加载约定。
  fsdp_config:
    mixed_precision:
      param_dtype: null
      reduce_dtype: null
      buffer_dtype: null
    amp_autocast:
      enabled: true
      precision: bf16
    grad_scaler:
      enabled: false
```

暂不同时开启 FSDP mixed-precision 和 AMP。OpenPI 有显式 FP32 action projection 输入和 normalization 路径；AMP 按运算选择 compute dtype，避免直接把所有 FSDP 子模块参数都转为 BF16 后产生类型不匹配。FP32 masters 会增加参数、梯度和 Adam states 的存储，以及部分通信量；是否能保持当前 microbatch，以 8 卡完整 smoke 的显存峰值为准。

注意：现有 actor `amp_context` 也包裹 teacher velocity scoring 和 flow refresh。新开关不会转换 teacher 权重，但启用 AMP 可改变其部分运算的计算精度。teacher 的闭环评测与低精度评分一致性仍需实际 smoke/校准确认，不声称逐位一致。

## 已验证与待验证

- `tests/unit_tests/test_actor_master_precision.py`：3 项 CPU 测试通过。覆盖默认不启用、冻结参数/buffers 不变、共享 Parameter 保留、FP32 Adam moments，以及低 LR 更新在 FP32 master 中保留而在 BF16 中消失。
- 待确认：完整 actor update + flow refresh + teacher scoring + weight sync 的 8 GPU smoke；日志应显示所有可训练 master 和 Adam moments 为 FP32，并记录显存和耗时。
- 文档中的开关与 `fsdp_actor_worker.py`、`fsdp_model_manager.py`、FSDP 配置模板逐项核对。本文件是内部实验说明，无独立 EN/ZH Sphinx 页面。
