# 负责人本机交接记录

- 新worktree：`outputs/worktrees/robotwin_formal`，分支`dev/robotwin2-formal-handoff`。
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

代码尚未push/创建公开tag，合作者不能直接clone本地branch。先完成部署验证，再固定交接commit，通过仓库分支或git bundle交付。未给合作者发送任何消息/文件。

本次不自动提交Slurm，也不启动正式seed0训练；目标机Blackwell验证通过后按README执行。当前action00八卡均已有工作，完整四卡VLA训练smoke暂不能启动；已完成两张卡上的受限闭环smoke和极小四卡FSDP检查，详见VALIDATION.md。

新候选runtime实际安装于`/tmp/robotwin-blackwell-candidate-20261103`，仅用于本机验证、未替换原环境；临时目录可能被系统清理。正式目标机按README安装到持久磁盘。
