# PID Score 与 Bootstrap 误差条说明

## 论文 Figure 8 的格式

从论文 PDF 原图确认的样式：

- 无填充阶梯直方图：`histtype="step"`。
- score 横轴范围固定为 `0-1`。
- 纵轴使用对数坐标，标题为 `Arbitrary Scale`。
- 真实 muon-flavor CC：浅红色，约为 RGB `(255,76,76)`，即 `#FF4C4C`。
- 真实 electron-flavor CC：绿色，约为 RGB `(76,166,76)`，即 `#4CA64C`。
- 真实 NC：蓝色，约为 RGB `(76,76,255)`，即 `#4C4CFF`。
- 三类均为实线阶梯轮廓，不填充、不使用点标记。
- 白色背景、黑色粗边框、朝内刻度、无网格。
- 原论文排版为 `2行 x 2列`：上排 DeepSphere、下排 PointNet++；左列 mu-like score、右列 e-like score。

论文图中的纵轴实际表现为事件计数的对数尺度。完全模仿论文时使用同一测试集直接画计数。可额外输出面积归一化的密度图用于跨样本规模比较，但不能替代论文风格图。

本项目有四个模型，绘制：

1. 每个模型各一组左右两图。
2. 一张 `4行 x 2列` 总图，四行依次为六特征 DeepSphere，以及使用 `decon_npevst`、`decon_waveform`、WFSampling 的 CNN+DeepSphere。
3. 所有子图固定相同 score bins、颜色、线宽和横轴范围。

## 可见能量分箱 AUC

把测试事件按真值 `visE` 分箱。在每个能量 bin 中计算：

```text
AUC_mu = AUC(mu-like vs 其他两类, mu-like score)
AUC_e  = AUC(e-like  vs 其他两类, e-like score)
AUC_NC = AUC(NC-like vs 其他两类, NC-like score)
macro AUC = (AUC_mu + AUC_e + AUC_NC) / 3
```

论文 Figure 10 中的 total AUC 是三个 one-vs-rest AUC 的算术平均。点放在能量 bin 中心，水平误差条表示 bin 的左右边界。

## Bootstrap 误差条计算

Bootstrap 通过对测试样本重复进行有放回抽样，估计有限统计量造成的 AUC 波动。对每个能量 bin 独立执行：

1. 取出该 bin 的全部测试事件。
2. 在各真实类别内部有放回抽样，并保持三类原有样本数，即分层 bootstrap。
3. 使用抽到事件已经保存的预测 score，重新计算三个逐类 AUC 和 macro AUC；不重新训练模型。
4. 重复 `B` 次，建议 `B=2000`，计算量受限时至少 `B=1000`。
5. 图上中心点使用原始未重采样测试数据的 AUC。
6. 论文只说明误差来自 bootstrap，没有明确给出重复次数和区间定义。为了复刻 Figure 10 的对称误差棒，本项目使用 bootstrap AUC 的样本标准差：

```text
sigma_AUC = std(AUC_bootstrap, ddof=1)
yerr = sigma_AUC
```

同时保存 bootstrap 分布的 16%、50%、84% 分位数。如果分布明显不对称，可额外输出分位数区间图，但论文风格主图仍画对称的 `±sigma_AUC`。

必须固定随机种子。同一能量 bin 内，四个模型使用相同的 bootstrap 抽样索引，以便进行配对比较。若某个 bin 的某类事件过少，或者无法同时包含正负样本，则该 AUC 标记为无效，不能用 0、0.5 或插值代替。

## 论文 Figure 10 的格式

从 PDF 原图确认的样式：

- 白色背景和浅灰色主网格。
- x 轴为 `Visible Energy (GeV)`，范围约 `0-15 GeV`。
- y 轴为 `AUC`，显示范围 `0.80-1.00`，主刻度间隔约 `0.04`。
- PointNet++：红色实心圆点和红色误差棒。
- DeepSphere：蓝色空心方块和蓝色误差棒。
- 两组点在 x 方向轻微错位，防止重叠。
- 水平误差条表示能量 bin 半宽；竖直误差条表示 bootstrap AUC 不确定度。
- 相邻能量点之间不连线。
- 图例位于左下角且无边框。

为最大限度模仿论文，本项目主结果画三张两模型对比图：

- 六特征 DeepSphere vs `decon_npevst` CNN+DeepSphere。
- 六特征 DeepSphere vs `decon_waveform` CNN+DeepSphere。
- 六特征 DeepSphere vs WFSampling CNN+DeepSphere。

每张图均保持：

- 六特征 DeepSphere：蓝色空心方块。
- 当前比较的 CNN+DeepSphere：红色实心圆点。

另输出一张四模型汇总图。由于汇总图需要为其余模型增加颜色和点型，因此它不属于论文双模型格式的完全复刻。

参考论文中 DeepSphere 和 PointNet++ 的 score 分布图，为以下四个模型配置使用同一测试集绘图：

1. 仅6种 `elec_fea` 的 DeepSphere。
2. 6种 `elec_fea + decon_npevst` 的 CNN+DeepSphere。
3. 6种 `elec_fea + decon_waveform` 的 CNN+DeepSphere。
4. 6种 `elec_fea + WFSampling waveform` 的 CNN+DeepSphere。
5. 6种 `elec_fea + waveform` 的 CNN+DeepSphere。

### 2.1 Score 分布图

每个模型至少绘制：

- `mu-like score` 分布。
- `e-like score` 分布。

每张图中分别画出真实 `numu`、`nue`、`nc` 事例的归一化分布，便于观察三类事例的分离能力。可补充 `NC-like score` 分布。

绘图要求：

- 横轴为模型输出概率，范围固定为 `[0, 1]`。
- 三类使用固定颜色、相同 bin 和相同归一化方式。
- 标明模型输入、测试事例数和类别。
- 四个模型分别保存图片，并再生成一张统一排版的对比图。
- 绘图数据来自保存的 `y_true`、`y_prob`，不重新随机划分数据。

图形格式模仿论文 Figure 8：

- 使用无填充阶梯直方图和对数 y 轴。
- `numu` 使用浅红色 `#FF4C4C`，`nue` 使用绿色 `#4CA64C`，`nc` 使用蓝色 `#4C4CFF`。
- score 范围、bin、线宽、字体和坐标范围在四个模型间保持一致。
- 白色背景、黑色边框、朝内刻度、无网格，y 轴标注 `Arbitrary Scale`。
- 每个模型输出左右两图，并生成一张 `4行 x 2列` 汇总图：左列 mu-like score，右列 e-like score。
- 论文风格主图使用原始测试事件计数；可另画归一化密度图，但不能替代主图。

### 2.2 按可见能量分箱

从标签中的 `visE` 字段读取可见能量。先检查三类 `visE` 的单位和范围，再确定统一分箱；可参考论文的 `0.5-15 GeV` 范围，实际以当前数据为准。

对5个模型使用完全相同的能量分箱，至少绘制：

- macro/total one-vs-rest AUC 随 `visE` 的变化。
- `numu`、`nue`、`nc` 各自 one-vs-rest AUC 随 `visE` 的变化。

建议同时输出各能量 bin 的 accuracy、各类效率和事例数表。样本足够时使用 bootstrap 计算 AUC 误差条；若某个 bin 中某类样本不足或无法计算 AUC，标记为无效，不填入虚假数值。


Bootstrap 和绘图严格按照 `docs/SCORE_AND_BOOTSTRAP.md` 执行：

- 每个能量 bin 内按真实类别进行分层、有放回抽样。
- 建议重复 2000 次，至少 1000 次，并固定随机种子。
- 中心值使用原始测试集 AUC，竖直误差棒使用 bootstrap AUC 的样本标准差。
- 四个模型在同一 bin 使用相同抽样索引。
- 水平误差棒表示能量 bin 半宽。

为完全模仿论文 Figure 10，分别生成5张双模型对比图：

每张双模型图均使用：

- DeepSphere：蓝色空心方块和蓝色误差棒。
- CNN+DeepSphere：红色实心圆点和红色误差棒。
- 白色背景、浅灰网格、左下角无边框图例，不连接相邻点。
- x 轴 `Visible Energy (GeV)`；若数据覆盖论文范围，使用 `0-15 GeV`。
- y 轴 `AUC`；默认复刻论文的 `0.80-1.00`，若结果超出范围，另存自适应范围图，不能裁掉数据。
- 两模型的点在 x 方向做轻微、对称的偏移，避免误差棒重叠。

另外可以生成五模型汇总图，但必须与论文格式双模型图分开保存。
