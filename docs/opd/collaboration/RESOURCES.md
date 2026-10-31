# 实验资源与跨机器传输

软件由 Git + [固定 profile](../../../OPD_ENVIRONMENT.md) 管理；模型与仿真资产单独准备。
[checkpoints.json](checkpoints.json) 固定所有实际加载的模型文件的 SHA256 与大小。

## 三位 teachers

官方仓库：[robocasa/robocasa365_checkpoints](https://huggingface.co/robocasa/robocasa365_checkpoints)。
固定 revision `c484448aba1a9b60a04c9b0ca117241518ea69f3`。
下载脚本只取三个 teacher 的推理文件，不下载官方 optimizer 或其他模型：

```bash
python -m toolkits.robocasa_mopd.resources download-teachers \
  --destination /absolute/path/teachers
export OPD_MODEL_ROOT=/absolute/path/teachers/gr00t_n1-5/foundation_model_learning
```

目录为 `target_posttraining/{atomic_seen,composite_seen,composite_unseen}/checkpoint-60000`。
三位 teachers 必须各用自己的 `config.json` 和 `experiment_cfg/metadata.json`；不能用
通用 NVIDIA GR00T checkpoint 替代，也不能共用一份归一化 metadata。
已逐文件确认本地 teacher 与上述 HF revision 相同（权重使用 HF LFS SHA256）。

## C10：单独传输即可

C10 是三套任务10%示范联合 SFT 得到的 `checkpoint-60000`，不是 HF 上的三位 specialists。
本轮四组只需要初始模型，不恢复 C10 的 optimizer/scheduler/RNG。
完整保留以下目录结构：

```text
checkpoint-60000/
  config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  experiment_cfg/metadata.json
```

可以通过 Google Drive、服务器间 rsync/scp 或 Hugging Face 分发；下载后以
`checkpoints.json` 校验，而不是根据目录名称判断。同一个 C10 用于全部正式 seeds。
一次性交接用 Drive 足够；反复部署建议 HF 固定 revision，且仍做内容校验。
若希望仅向合作者开放，可使用双方有访问权限的私有 HF **organization** model repo；
个人私有 repo 不等价于 Drive 的指定成员分享，不要共享个人访问 token。
本仓库不上传 C10，不保存链接访问凭据；实际下载地址由负责人另行提供。

## 绑定共享 teacher 校准

[teacher_reference.portable.json](teacher_reference.portable.json) 是现有20episodes/task
校准的可移植副本。统计、训练协议和原始 provenance 保留；checkpoint 路径暂为相对标识。
在本机校验模型内容后生成路径正确的副本：

```bash
export OPD_STUDENT_CHECKPOINT=/absolute/path/C10/checkpoint-60000
python -m toolkits.robocasa_mopd.resources prepare \
  --output /absolute/path/inputs/teacher_reference.json
export OPD_TEACHER_REFERENCE=/absolute/path/inputs/teacher_reference.json
```

脚本读取 `OPD_MODEL_ROOT`，验证四个模型所有清单文件后才绑定路径；已存在输出不覆盖。
不要手工把另一组 teacher 路径写入旧reference。校准共1,000episodes、1,299,775环境控制步，
单独计入方法成本，不能当最终独立测评。M0/M1不使用校准数值，M2/M3共享它。

## RoboCasa 仿真资产

固定软件 profile 不包含厨房/物体/纹理等大文件。在固定 RoboCasa checkout 中运行：

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

或复制负责人提供的同一资产快照，并按上游 macros 配置实际路径。安装器不会自动
绑定现有服务器的共享盘。记录资产来源/版本；复用已有快照时双方校验文件清单。
目前 Git 没有完整资产哈希清单，不能据此声称逐字节锁定了全部仿真输入。
有缺失资产报错时修复资源，不跳过任务。新机器的训练与评测 smoke 是必需验收项。
