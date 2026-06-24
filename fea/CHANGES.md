# train_fea.py 改动说明

本文件记录从原始方向重建脚本到三分类 PID 脚本的所有改动，以及每项改动的原因。

---

## 原始脚本状态（初始版本）

原始 `train_fea.py` 是一个**方向重建**脚本，主要特征：
- 单一数据源：只读 numu 数据（`/disk_pool1/wangjb/nu_mu_waveform/`）
- 输入特征：4 个（fht、nperatio4、npe、slope4），shape `(n, 17612, 4)`
- 标签：方向角转三维向量 `y_3d = [sin θ cos φ, sin θ sin φ, cos θ]`
- 损失函数：`direction_loss`（欧氏距离或余弦相似度）
- 输出层：归一化后的三维单位向量
- GPU：硬编码 `CUDA_VISIBLE_DEVICES="0"` 在脚本内
- 文件范围：硬编码 `range(3000, 5000)`
- 路径：多处硬编码单一类别路径

---

## 改动一：任务目标改为三分类 PID

**原因**：从方向重建改为 numu/nue/nc 三分类粒子鉴别（PID）。

具体变化：
1. 删除 `direction_loss`、余弦相似度损失和方向角转三维向量逻辑
2. 删除输出层的 `L2Normalize`（`tf.math.l2_normalize`）
3. 最后一层 `Dense(3)` 输出的语义从"三维方向向量"改为"三类 logit"
4. 损失函数改为 `SparseCategoricalCrossentropy(from_logits=True)`
5. 评价指标从 `direction_loss` 改为 `accuracy`
6. 输出增加混淆矩阵、逐类 ROC/AUC、precision/recall/F1（`pid_lib/metrics.py`）
7. 标签从 `y[:,2:4]`（theta/phi）改为目录来源标签 `{numu:0, nue:1, nc:2}`

---

## 改动二：输入特征从 4 个扩展到 6 个

**原因**：EXECUTION_PLAN.md 要求使用全部六种电子学特征，覆盖论文中描述的核心特征。

具体变化：
1. 新增 `x_peak_pmt`（peak amplitude）和 `x_peaktime_pmt`（peak time）两个特征
2. `feature_num` 从 4 改为 6，feature_num 对应 `[fht, npe, nperatio4, peak, peaktime, slope4]`
3. DeepSphere 模型输入 shape 从 `(None, 12288, 4)` 改为 `(None, 12288, 6)`

---

## 改动三：数据来源从单类改为三类

**原因**：PID 任务需要同时加载 numu、nue、nc 三个来源的数据，且标签由目录决定。

具体变化：
1. 删除硬编码的单一路径 `/disk_pool1/wangjb/nu_mu_waveform/`
2. 引入 `pid_lib/config.py` 中的 `CLASS_ROOTS`（三类路径配置）
3. 数据加载通过 `pid_lib/data_io.py` 的 `discover_valid_entries()` 和 `load_aligned_file()` 实现，自动过滤 0 字节损坏文件
4. 按文件编号做 80%/10%/10% 分层划分，固定随机种子（`pid_lib/splits.py`）
5. 划分结果保存为 `manifest_train/val/test.json`，重复运行可复用

---

## 改动四：无击中 PMT 的哨兵值处理

**原因**：三类数据中无击中 PMT 的哨兵值不同（nue 用负值 -5 到 -1，nc/numu 用 0），必须统一处理，否则模型会学到人为的格式差异而不是物理差异。

具体变化：
1. 新增 `pid_lib/data_io.py` 的 `mark_no_hit_features()`：`fht<=0` 或 `peaktime>=1008` 的 PMT 替换为 NaN
2. 新增 `pid_lib/normalize.py` 的 `apply_norm(..., fill_nan=-3.0)`：归一化后用 -3.0 填充 NaN，该值远低于正常特征范围，使模型能区分"无击中"和物理值
3. **注意**：最初版本用 `fill_nan=0.0`（与归一化后均值重叠），后改为 `-3.0`

---

## 改动五：归一化方式改为逐通道 z-score（仅训练集）

**原因**：原始脚本对 FHT 使用 `x1[x1>1024]=1024; x1[x1<=0]=1024` 的截断方式，不适合多类统计；且归一化应只用训练集统计量，避免测试集泄露。

具体变化：
1. `pid_lib/normalize.py` 的 `estimate_norm_stats()` 在训练集上计算各特征的均值和标准差（忽略 NaN 哨兵值）
2. 统计量保存到 `outputs/*/norm_stats.json`，测试时直接加载
3. 类别映射保存到 `class_mapping.json`

---

## 改动六：GPU 选择和命令行参数化

**原因**：AGENTS.md 明确要求不能把 GPU 编号、文件范围、路径硬编码在脚本中。

具体变化：
1. 删除脚本内所有硬编码的 `CUDA_VISIBLE_DEVICES="0"`
2. 新增 `--gpu` 参数，`CUDA_VISIBLE_DEVICES` 在训练前由参数设置
3. 新增 `--output-dir`（必填，要求语义化命名如 `fea6`）
4. 新增 `--epochs`、`--batch-size`、`--seed`、`--max-files-per-class`
5. 新增 `--smoke-test` 快速验证流程（3 文件/类，3 轮）

---

## 改动七：训练数据生成器的类别平衡策略（关键 Bug 修复）

**原因**：这是造成混淆矩阵全预测为单一类别的根本原因，经过三次迭代才最终修复。

**Bug 根因分析**：
- **第一版（顺序生成器 + shuffle buffer 4096）**：`train_entries` 按 numu→nue→nc 顺序排列，42k 事例远超 4096 的 buffer，导致模型在训练后期只看到 nc 数据 → 模型偏向预测 nc
- **第二版（`sample_from_datasets` 等权采样）**：每轮随机交叉采样，val_loss 剧烈震荡（4.87→14.35→2.61→16.99），因为 val_ds 也用了随机采样，导致无法稳定监控收敛
- **第三版（固定种子 flat 生成器 + 大 buffer shuffle）**：全局 shuffle 所有条目（seed=42），单一 tf.data 流，8192 buffer 在 pipeline 内混合 → val_loss 稳定，val_acc 从 Epoch 1 的 68% 持续提升至 76%+

最终方案：
1. `make_flat_generator`：用固定种子 shuffle 所有 train_entries，保证每 epoch 相同的文件顺序（生成器级别）
2. `build_dataset`：`shuffle=True` 时使用 `buffer_size=8192` 在 pipeline 内进一步混合（batch 级别随机）
3. val/test 用 `shuffle=False` 确定性顺序，使每 epoch 的 val_loss 可比

---

## 改动八：去除 class_weight

**原因**：equal-weight 采样已通过大 buffer shuffle 保证三类在训练中近似均匀出现，再叠加 `class_weight` 会对 numu（事例数最少）双重加权，反而造成训练不稳定。

具体变化：
- `class_weight=None`，不传给 `model.fit`
- 三类平衡由数据 pipeline 的 shuffle 策略保证

---

## 改动九：学习率调整

**原因**：原始脚本 lr=0.002 对方向重建任务合适，但对分类任务偏高，导致训练初期梯度震荡。

具体变化：
- `--learning-rate` 默认值从 0.002 改为 0.0005
- lr 衰减系数从每轮 ×0.99 改为 ×0.995（更平缓）

---

## 改动十：新增训练输出文件

相比原始脚本只保存权重和 loss 曲线，新脚本还保存：
- `confusion_matrix.png`：测试集混淆矩阵
- `roc_ovr.png`：三类 one-vs-rest ROC 曲线
- `test_metrics.json`：accuracy、macro AUC、每类 P/R/F1
- `y_true.npy`、`y_prob.npy`、`y_pred.npy`：测试集预测结果
- `class_mapping.json`：`{numu:0, nue:1, nc:2}`
- `loss_log.txt`：每轮训练/验证 loss 和 accuracy
- `manifest_train/val/test.json`：数据划分文件，可复用
- `norm_stats.json`：归一化统计量

