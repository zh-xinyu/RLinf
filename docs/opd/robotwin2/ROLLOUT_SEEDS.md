# Rollout 随机种子

正式 RoboTwin 训练显式设置 `+rollout.seed=1000000000`（Hydra override）。HF rollout worker 在模型加载前按 `rollout.seed + global_rank` 初始化 Python、NumPy 与 Torch 随机数；跨 action chunks、rollout rounds 和 student 权重同步持续使用该随机流。FrozenTeacherRolloutWorker 继承同一初始化逻辑。

未配置或设为 null 时保持旧行为。actor seed、环境 reset seed 和 rollout seed 是不同进程的控制项，应分别记录。候选训练重复 s=0/1/2：actor.seed=100003*s，rollout.seed=1000000000+100003*s，env.train.seed 与 runner.rollout_allocation.seed=42+10000*s。

这不保证不同 worker 数、batch、模型加载路径之间共享相同 noise，也不保证物理仿真逐位一致。当前 checkpoint resume 不恢复 rollout RNG；恢复训练应记录该限制。旧的未显式设种子的 pilot 保留为探索证据。

验证：`python -m pytest -q tests/unit_tests/test_huggingface_rollout_seed.py`，覆盖未配置时不改 RNG、重复性、rank 分离、参数校验和继承初始化。CPU 单元测试不替代真实训练路径验证。
