# 论文中与 JUNO 大气中微子 PID 直接相关的内容

论文：*Neutrino type identification for atmospheric neutrinos in a large homogeneous liquid scintillation detector*，arXiv:2503.21353v2（2025-06-13）。以下是面向本项目的中文提取和说明，不是逐句全文翻译。

## 1. 论文解决的问题

论文研究大型均匀液体闪烁体探测器中的大气中微子类型鉴别，分成两个阶段：

1. 3-label 味型鉴别：`mu-like`、`e-like`、`NC-like`。
2. 2-label 中微子/反中微子统计鉴别：分别做 `nu_mu vs anti-nu_mu` 和 `nu_e vs anti-nu_e`。

本项目当前对应第一阶段：

- `numu` -> mu-like，包含 muon-flavor CC 拓扑。
- `nue` -> e-like，包含 electron-flavor CC 拓扑。
- `nc` -> NC-like。

论文指出，液闪探测器中 Cherenkov 光远弱于闪烁光，不能直接照搬水 Cherenk夫探测器的环形锐度鉴别。可用信息来自带电轻子和强子在液闪中的径迹/簇射拓扑，这些拓扑会改变各 PMT 的光子到达时间分布和波形形状。

## 2. 为什么波形能够区分三类事件

在 `nu_mu/anti-nu_mu-CC` 事件中，主相互作用产生的 muon 通常形成较长径迹；电子味 CC 中的 electron 形成较短的电磁簇射；NC 事件没有可见的出射中微子，只留下强子部分。带电强子、pi0 等还会形成不同簇射或径迹。

PMT 看到的闪烁光是径迹或簇射上许多位置发光的叠加，因此 PE 随时间的分布 `nPE(t)` 由事件拓扑决定。论文用相同动能 1 GeV、相同顶点和方向的 muon/electron 比较，显示不同 PMT 角度下两者的 PE 时间分布形状明显不同。该差异最终反映到 PMT 波形中，所以 prompt trigger 的逐 PMT 波形可以用于三分类。

这也是本项目比较以下输入的物理依据：

- 六个反卷积电子学特征：压缩后的显式波形描述。
- `decon_npevst`：更直接的 PE 时间结构。
- `decon_waveform`：降噪、反卷积后的连续波形。
- WFSampling：保留主峰上升沿与后续峰谷的稀疏变长表示。

## 3. 论文用于 3-label 的输入特征

论文使用 20 英寸 PMT 的 prompt-trigger 波形，采样率 1 GHz，触发窗约 1000 ns。提取的特征包括：

- FHT：最早光子击中时间。
- total charge：总电荷。
- waveform first 4 ns slope：前 4 ns 斜率。
- first 4 ns charge / total charge：前 4 ns 电荷占比。
- maximum-bin time and amplitude：最大 bin 的时间与幅度。
- 其他更细致特征：中位时间，以及波形的 mean、std、skewness、kurtosis 等矩。

提取前先使用反卷积和降噪算法提高特征质量。本项目 `elec_fea` 中的六项恰好覆盖论文列出的核心特征：FHT、nPE/charge、4 ns 电荷占比、峰值、峰时、4 ns 斜率。因此六特征 DeepSphere 应作为第一条可解释基线。

## 4. 论文中的模型与输出

所有 20 英寸 PMT 特征组成球面图像型数据。论文比较：

- DeepSphere：直接处理球面数据。
- PointNet++：把 PMT 输入视为三维点云。

3-label 模型主体基本保持原有架构，只把最后一层激活改成 softmax，输出三个分数，分别代表事件属于 mu-like、e-like、NC-like 的概率。三个概率和为 1，默认取最大分数对应的类别。

因此把现有方向重建脚本改成 PID 时，关键不是重写 DeepSphere 主体，而是：

- 输入通道改为六特征或“波形 CNN embedding + 六特征”。
- 方向真值改为整数类别。
- 方向损失和单位向量输出改为三分类交叉熵和 softmax。
- 输出概率、混淆矩阵与逐类 ROC/AUC。

## 5. 论文的数据设计

论文使用五个原始类别：`nu_mu-CC`、`anti-nu_mu-CC`、`nu_e-CC`、`anti-nu_e-CC`、NC，并在 3-label 阶段合并成 mu-like、e-like、NC-like。

论文准备了三套独立样本：

- GENIE flat sample：各原始类别约 7 万事例，可见能量分布大致平坦、类别统计接近，用于训练并避免模型从能谱差异中取巧。
- GENIE Honda-flux sample：更现实的 JUNO 站点大气中微子通量，总计约 9.5 万事例，用于性能评价。
- NuWro sample：统计量与 Honda-flux 接近，用于检查事件生成器依赖。

选择的可见能量范围是 0.5 至 15 GeV。探测器模拟包括 Geant4 和暗噪声、TTS、电荷涨落、电子学基线涨落、单光电子响应等效应。论文为简化只使用 20 英寸 PMT。

本项目当前三类目录不一定等价于论文的 flat/Honda/NuWro 独立样本。服务器 agent 必须审计各类能量和事件数分布。若三类能谱显著不同，模型可能学习能量先验而不是拓扑；至少应按 `visE` 分箱报告性能，并在文档中标明这一限制。

## 6. 论文的评价方式

论文用 ROC 和 AUC 避免固定分数阈值带来的依赖。多分类采用 one-vs-rest：对每个类别分别计算“该类 vs 其余类”的 ROC/AUC，再对各类 AUC 做算术平均，得到 total/macro AUC。论文还按可见能量分箱比较 total AUC，并用 bootstrap 给出统计不确定度。

本项目至少应输出：

- overall accuracy。
- 三类混淆矩阵。
- 每类 precision、recall、F1。
- 每类 one-vs-rest ROC/AUC 与 macro AUC。
- 按 `visE` 分箱的三类效率或 macro AUC；样本允许时使用 bootstrap 误差。

仅报告训练/验证 loss 不足以说明 PID 性能。

## 7. 与 2-label 和中子信息的边界

论文的中微子/反中微子鉴别还利用：

- 可见强子能量占比 `yvis = Ehad,vis/Evis`。
- 中子多重数。
- 延迟中子俘获顶点的空间分布。

中子主要在约 200 微秒后被氢俘获并释放 2.2 MeV gamma；论文选择 2 至 2.7 MeV、延迟 10 微秒至 1 毫秒的候选触发。JUNO 对中子俘获的标记效率可超过 90%。论文给出点云/DGCNN和延迟触发合并两种策略。

这些内容属于后续 `nu vs anti-nu` 任务，不应混入当前 prompt waveform 的三分类标签。当前先把 mu-like/e-like/NC-like 做稳，再考虑延迟触发和中子信息。

## 8. 对当前实现的直接建议

1. 先训练六特征 DeepSphere，作为论文方法的最直接复现基线。
2. 三类采用相同文件级划分；所有波形实验复用同一 manifest。
3. 对波形表示做三组独立消融：`decon_npevst+6 features`、`decon_waveform+6 features`、`WFSampling+6 features`。
4. WFSampling 不能只保存幅度序列而丢弃关键点时间；变长填充时至少保留 time 和 amplitude，并记录有效长度。
5. 除总指标外按 `visE` 分箱，检查模型是否只利用类别间能谱差异。
6. 检查不同生成器、制作批次或目录是否与类别完全绑定；若绑定，测试结果可能包含 domain shortcut，应在独立制作样本上复核。

论文没有规定本项目现有 NPZ 的具体键名、六特征的数值归一化、WFSampling 阈值或当前三类目录的标签编码。这些内容必须以服务器实际数据和生成脚本为准，不能从论文推断。
