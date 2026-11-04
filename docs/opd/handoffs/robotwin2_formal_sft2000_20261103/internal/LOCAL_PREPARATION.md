# 负责人本机交接记录

- 新worktree：`outputs/worktrees/robotwin_formal`，分支`experiment/robotwin2-mopd`。
- 原开发worktree：`outputs/worktrees/robotwin4`；未修改其代码/未清理其未提交改动。
- 本机machine JSON：`outputs/robotwin4/handoff_20261103/machine.action00.json`。
- 已按交付布局准备资源目录：`outputs/robotwin4/handoff_20261103/resources/`。当前是指向已验证原文件的符号链接，不额外占用74.45GB。传输时必须解引用，不能把指向本机路径的链接交给合作者。
- 精确文件清单和SHA256：新分支的`toolkits/robotwin_mopd/formal/resources.json`（19文件）。

在workspace根目录可以使用：

```bash
# 将 TARGET 替换为合作者指定的传输目的地；这里只给命令，没有发起网络传输。
rsync -aL --info=progress2 outputs/robotwin4/handoff_20261103/resources/ TARGET/
```

也可打包实际文件，但会额外占用约75GB磁盘：

```bash
tar -chf /absolute/path/robotwin-formal-resources.tar \
  -C outputs/robotwin4/handoff_20261103/resources .
```

不包含SFT optimizer；SFT累计2000步权重内容保持不变。teacher原始JAX数据无需再次传输/转换。模型来源和许可继续适用。

代码通过GitHub分支`experiment/robotwin2-mopd`交付，尚未创建正式release tag。合作者应记录交付commit并先在目标节点完成validate，再运行正式实验。模型资源仍按上述方式单独传输，不包含在Git提交中；未给合作者发送消息或传输模型。

本次不自动提交Slurm，也不启动正式seed0训练；目标机Blackwell验证通过后按交接指南执行。当前action00八卡均已有工作，完整四卡VLA训练smoke暂不能启动；已完成两张卡上的受限闭环smoke和极小四卡FSDP检查，详见[交接指南第6节](../robotwin2_blackwell_4gpu_handoffs.md#6-已完成的验证与适用范围)。

新候选runtime实际安装于`/tmp/robotwin-blackwell-candidate-20261103`，仅用于本机验证、未替换原环境；临时目录可能被系统清理。正式目标机按README安装到持久磁盘。

## 2026-11-04 Hugging Face 交付已完成

74.45 GB 资源已发布到 `theo-zhou/vla-mopd-checkpoints` 的 `robotwin2/pi05/`，revision `ebf161f8743f80c9eb3fbb5a25886d172144d571`。19 文件本地和远端 SHA256 均验证通过。之前未传输的状态现已更新；原文件路径和 symlink staging 仍保留用于溯源。下载命令见 [交接指南第3节](../robotwin2_blackwell_4gpu_handoffs.md#31-下载已转换-checkpoints)。本次仅上传用户指定 HF 仓库并更新本地交接 ZIP，没有向合作者发送消息，也未修改训练代码。

当前交付形式：直接发送 `robotwin2_blackwell_4gpu_handoffs.md` 与 `checkpoint_results.csv`，不再使用 ZIP。

四组修正：正式方法为 M0/M1/M2/M3，结果表18行、72个task jobs、7,200 episodes；新增 M3 为 Adaptive-Sync R1。

当前配方已改为100 rounds/B256、默认micro16（目标机smoke选择后四组统一），保存round10/20/40/60/80/100；26行、104个task jobs、10,400 episodes。原20轮/B128测试保留为历史依据，不覆盖。
