# RoboTwin 正式交接当前方案

当前执行方案以[正式交接说明](../handoffs/robotwin2_formal_sft2000_20261103/robotwin2_blackwell_4gpu_handoffs.md)为准。

- SFT2000共同起点，M0 Uniform-R1 / M1 Uniform-Refresh-R2 / M2 Adaptive-Sync-Refresh-R2 / M3 Adaptive-Sync-R1。
- T32、B256、默认micro16（smoke后四组统一）、LR1e-5、constant、warmup4 updates、100 rounds。
- **所有保存点round10/20/40/60/80/100，以及起点和teacher，均100 episodes/task**；固定cohort2 formal100。
- 首轮seed0，四组完成后再设计多seed；原50/task、只测5/10/20、自动准备两个seeds的方案已被替代。
- 新分支从opd-project-main继承现有RoboTwin实现；当前Blackwell部署验证待完成，不混入旧DEV或formal50成绩。

已有实验结果与选择依据见[预算测试记录](STRONG_SFT_BUDGET_TEST_20261103.md)。
