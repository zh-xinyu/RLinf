# 固定输入 flow 诊断：joint step20（2026-11-01）

**结论：在同一输入上，joint20 明确学到了其他三个 teacher 的局部 velocity，但原 place 能力对应的函数发生明显漂移。不能将本轮解释为“优化根本没有作用”，也不能将训练加权 loss 较平直接归因于所有任务的低 t 都不下降。**

协议：每任务仅1个 reset 后初始观测，实际执行0个 control targets；place-SFT reference 产生固定 noise 与全部10个 `x_t`，reference/joint20 和对应 task teacher 使用相同观测、prompt、norm、`x_t`、t。比较使用 BF16 AMP；以下 physical14 仍是归一化 latent 坐标，不是物理关节误差。

## 同输入、全50位置的 physical14 误差

| Task | Plain MSE：reference→joint20 | 变化 | M3 weighted：reference→joint20 | 变化 |
|---|---:|---:|---:|---:|
| adjust_bottle | 1.9625 → 0.9657 | -50.8% | 1.0000 → 0.3513 | -64.9% |
| handover_mic | 1.5353 → 0.7153 | -53.4% | 0.9886 → 0.3786 | -61.7% |
| place_container_plate | 0.0000 → 0.3902 | 从零增加 | 0.0000 → 0.5184 | 从零增加 |
| stack_bowls_two | 1.5202 → 0.6332 | -58.3% | 0.8908 → 0.4681 | -47.5% |

四任务等权均值：plain physical14 MSE 1.2545→0.6761，weighted 0.7198→0.4291。这只是4个固定初始观察的描述性均值，不是训练集或benchmark估计。

## 时间维度：低 t 并非统一失效

| t | adjust：前→后 | handover：前→后 | place：前→后 | stack：前→后 |
|---|---:|---:|---:|---:|
| 1.0 | 0.714→0.457 | 0.418→0.276 | 0.000→0.058 | 0.388→0.159 |
| 0.9 | 0.878→0.561 | 0.514→0.342 | 0.000→0.072 | 0.479→0.196 |
| 0.8 | 1.108→0.705 | 0.651→0.426 | 0.000→0.088 | 0.608→0.245 |
| 0.7 | 1.427→0.895 | 0.851→0.535 | 0.000→0.105 | 0.795→0.318 |
| 0.6 | 1.904→1.174 | 1.143→0.671 | 0.000→0.129 | 1.080→0.429 |
| 0.5 | 2.641→1.567 | 1.611→0.854 | 0.000→0.172 | 1.543→0.615 |
| 0.4 | 3.678→1.960 | 2.364→1.124 | 0.000→0.259 | 2.353→0.938 |
| 0.3 | 3.746→1.480 | 3.465→1.474 | 0.000→0.431 | 3.843→1.544 |
| 0.2 | 2.292→0.647 | 2.918→0.985 | 0.000→0.907 | 3.430→0.928 |
| 0.1 | 1.236→0.210 | 1.418→0.466 | 0.000→1.680 | 0.682→0.960 |

- adjust 与 handover 在全部10个时间点均下降；handover 的固定输入 weighted MSE 下降约62%，与变化的 on-policy 训练日志上升不同。两者观察/flow分布、采样范围不同，不能当作日志矛盾。
- place 原始 reference 与 teacher 同 checkpoint，全部误差严格为0；joint20 在全部时间点出现偏差。`t=0.1` alone 占其 physical14 weighted 残差 73.8%，最后两点共 91.5%。这比仅看总MSE更明确地指向原任务近动作端的函数漂移。
- stack 的前9个时间点下降，但 `t=0.1` 从0.682升至0.960；该点占 joint20 weighted 残差 46.7%。总目标改善仍可掩盖个别关键时间点恶化。

## Horizon 与 padded 坐标

| Task | first25 physical MSE：前→后 | last25 physical MSE：前→后 | joint20 padded占总平方误差 |
|---|---:|---:|---:|
| adjust_bottle | 2.1996→0.8693 | 1.7254→1.0621 | 0.58% |
| handover_mic | 1.4663→0.6355 | 1.6042→0.7951 | 0.37% |
| place_container_plate | 0.0000→0.4121 | 0.0000→0.3683 | 0.70% |
| stack_bowls_two | 1.5733→0.6419 | 1.4671→0.6245 | 0.23% |

误差主体位于 physical14，不能把现象归结为 padded18 占主导。joint20 的后25误差并非所有任务都高于前25；当前已采用full50监督，且reference/path与早期probe不同，不能将两轮表格当作受控证明。全32维均值会被较小的 padded 误差稀释，分析保留两种口径。

## 核实日志中的 adjust_bottle norm 提示

`openpi/__init__.py:42–45` 先用字典首项构建默认 data config，所以每次模型加载都会打印 adjust_bottle；`:163–178` 随后逐task读取独立 stats 并安装 task wrappers。`obs_processor` 按 task_groups 生成 normalization ID，input/output transforms按该ID分派。

已额外做CPU数值复核：使用真实 transform实现和四个stats文件，不加载模型权重，将 `.reference.pt` 中原始观测重新归一化。四任务正确stats的结果与缓存 normalized_state **逐元素完全一致**，ID为0/1/2/3。强制其他任务错误使用adjust stats时，handover/place/stack的最大状态误差分别为 **1.7053 / 1.7857 / 1.5769**。因此这些probe并没有都落到adjust的state normalizer。output路由也经源码核对；这里的独立数值测试验证的是input state，不宣称重新验证了所有action transform。文件hash、task ID和数值结果保存在 `normalization_audit.json`。

## 推进含义与限制

- 当前信号支持优先 joint SFT 初始化/更充分优化，并关注原任务保留；尚无证据应立即切换到 uncertainty/OOD truncation。
- 这是同输入函数拟合与漂移证据，不是梯度冲突测量，不证明遗忘只有一个原因，也不推翻冻结评测仍未取得新任务SR的结果。
- 每任务只有一个初始观察；10个时间点/50个位置不是独立episode样本，没有覆盖后续失败或恢复状态。所有 `x_t` 来自固定place reference，而非joint20自身flow。
- 后续 checkpoint 必须通过 `--cache-input` 复用原 `.reference.pt`，保持输入与AMP一致；可扩展少量后期状态，但不能将重新采样的观察误称为同输入对照。

原始证据（根workspace）：`outputs/robotwin4/analysis/fixed_reference_joint20_20261101/`。逐task JSON包含checkpoint路径、seed/prompt、noise/observation/x_t hash与dtype；`results_summary.json`保留本页数值及逐bin明细。`SUMMARY.md`与本页同内容。
