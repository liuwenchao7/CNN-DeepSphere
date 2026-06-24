# train_cnn_ds_pid.py 相对 train_200_fht.py 与各实验差异说明

更新时间：**2026-06-19**

主训练脚本：`train_cnn_ds_pid.py`（项目根目录）。存档参考：`WFSampling/train_200_fht.py`（方向重建时期 CNN+DeepSphere，**不再用于全量训练**）。

---

## 一、与 `train_200_fht.py` 的核心区别

### 1. 任务与标签

| 项目 | train_200_fht.py | train_cnn_ds_pid.py |
|------|------------------|---------------------|
| 任务 | 方向重建（回归） | 三分类 PID（numu/nue/nc） |
| 标签 | 方向三维向量 | 稀疏整数 0/1/2（`from_logits=True`） |
| 损失 | 方向余弦损失 | `SparseCategoricalCrossentropy` |
| 数据类 | 仅 numu、硬编码路径 | 三类，`pid_lib/splits.py` manifest |

### 2. 静态特征

| 项目 | train_200_fht.py | train_cnn_ds_pid.py |
|------|------------------|---------------------|
| elec 维数 | 4（fht, nperatio4, npe, slope4） | **6**（fht, npe, nperatio4, peak, peaktime, slope4） |
| 无击中 | 未统一处理 | `mark_no_hit_features` + 归一化后 `fill_nan=-3` |

### 4. 模型结构（CNN → DeepSphere）

| 项目 | train_200_fht.py | train_cnn_ds_pid.py |
|------|------------------|---------------------|
| CNN 输出维 | `cnn_out_dim=4` | **6**（与 elec 维对齐后拼接） |
| elec 进 CNN | 静态特征与波形 **分开** scatter 合并 | **6 维 elec 在时间维前拼接**（`group_input`），再进逐 PMT CNN |
| DeepSphere 输入维 | `4 + 4 = 8`（CNN+elec 各 4） | **12**（6 CNN + 6 elec 沿时间轴拼接，无单独 `elec_proj3`） |
| 逐 PMT CNN | `Lambda` 内循环，`pmt_batch=500` | `PMTBatchCNNLayer` 自定义层，`--pmt-batch-size`（默认 200，FHT 用 150） |
| DeepSphere 深度 | 较浅（Fout 最大 48） | 更深（Fout 最大 48，层数更多，与 fea 路线对齐） |

### 5. 数据 pipeline（flat generator 与 shuffle）

这是与原始脚本差异最大、也最易踩坑的部分。

#### train_200_fht.py 的做法

```text
gen_entries(entries)  →  按文件顺序遍历 (entry, event_idx)
data_generator      →  每次 yield 时 load_event（磁盘 I/O）
train_ds.shuffle(2048).batch(...)
```

- **无全局 event 列表**：顺序由 manifest 文件顺序决定。
- **shuffle buffer=2048**：在 `tf.data` 里缓存 2048 个 **已加载的样本** 再打乱；buffer 填满前 GPU 空等。
- 验证集：同一 generator 逻辑，**无 shuffle**。

#### train_cnn_ds_pid.py 的做法（2026-06-19 后）

```text
build_event_index()     →  启动时一次构建 (entry, event_idx) 列表；训练集可 balance
make_indexed_generator  →  每 epoch 按 seed+epoch_no 打乱 index，再逐条 load_event
build_dataset           →  shuffle_buffer=0（默认）→ 直接 batch；可选小 buffer
```

**flat generator 的意义**：

- Keras 需要 **(inputs, label)** 流式数据集；generator 把「多文件、多 event」展平为 **单条样本流**。
- 与「按文件 batch」不同，每条样本对应 **一个物理事例** + 其 **17612 PMT** 张量。

**shuffle buffer 的意义**：

- `tf.data.shuffle(N)` 在管道内维护 **N 个已解码样本** 的滑动窗口，训练前必须先填满（日志：`Filling up shuffle buffer`）。
- 对 **磁盘重 I/O**（每样本读 npz/npy + 17612 PMT CNN）而言，N=4096 时 **每 epoch 填充可达数小时**（decon_npevst 实测 ~24 h/ep 的主因）。
- **修复**：默认 `shuffle_buffer=0`，改为 **Python 侧打乱 event index**（每 epoch 一次），语义上等价于「epoch 级 shuffle」，无预取 4096 条的开销。

**与 train_fea.py 的类比**：

- `fea/train_fea.py` 同样用 flat generator + 大 shuffle buffer（8192），但 **无逐 PMT CNN**，单样本更轻，buffer 填充可接受。
- CNN+DeepSphere **默认不再使用大 buffer**。

#### 验证集

| 项目 | 旧 train_cnn_ds_pid | 当前 |
|------|---------------------|------|
| val balance | 曾 `balance_events=True` | **仅训练 balance**；val/test 固定 index |
| val shuffle | 无 | 无 |

避免验证子采样与加载失败导致 **val_loss 剧烈震荡**（FHT 实验已修复）。

### 6. 训练超参

| 项目 | train_200_fht.py | train_cnn_ds_pid.py |
|------|------------------|---------------------|
| learning_rate | 0.002 | **5e-4** |
| batch_size | 16 | 全量常用 **16–32** |
| class_weight | 有 `compute_class_weights` | **不用**；靠训练集 event balance |
| GPU | 硬编码 | `--gpu N`（不覆盖已有 `CUDA_VISIBLE_DEVICES`） |

---

## 二、各 CNN+DeepSphere 实验之间的区别

公共部分：**相同 manifest**、相同 6 elec 特征、相同 DeepSphere 主干、相同三类平衡策略（仅训练集）。

| 实验目录 | waveform_source | 波形形状 (N_PMT,T,C) | 加载方式 | 预处理依赖 | 相对耗时 |
|----------|-----------------|----------------------|----------|------------|----------|
| `fea6+decon_npevst` | decon_npevst | (17612, ~43, **2**) | 稀疏 hit → 固定点数 time+charge | 无（源 npy） | 中（shuffle 修复后 ~1–3 h/ep） |
| `fea6+wfsampling` | wfsampling | (17612, ~43, **2**) | WFS `.npy` 关键点 | WFSampling raw | ~3 h/ep |
| `fea6+wfs_decon_waveform` | wfs_decon_waveform | (17612, ~43, **2**) | WFS decon 目录 | WFSampling decon | 未训 |
| `fea6+decon_waveform` | decon_waveform | (17612, ~1007, **1**) | 全长度 decon 波形 | 无 | 慢（长序列） |
| `fea6+waveform` | waveform | (17612, ~1008, **1**) | raw 反相 16384-ADC | 无 | 慢 |
| `fea6+decon_waveform_fht` | decon_waveform_fht | (17612, **300**, **1**) | FHT 窗口 npz | `decon_wav_fht+-` | ~2.3 h/ep |
| `fea6+waveform_fht` | waveform_fht | (17612, **300**, **1**) | FHT 窗口 npz，**不再 invert** | `wav_fht+-`（已 16384-ADC） | 修复后重训 |

**通道含义**：

- `C=2`：归一化时间 + 电荷/幅度（decon_npevst、WFS）。
- `C=1`：ADC 或 decon 幅值；`waveform_fht` 预处理已反相，训练 **invert=False**。

**batch / pmt_batch 惯例**：

- 稀疏点（npevst / wfs）：batch 16–32，`pmt_batch_size=200`。
- FHT 300 点：batch 16，`pmt_batch_size=150`（显存）。

---

## 三、2026-06 重要修复摘要

1. **decon_npevst shuffle**：`shuffle_buffer` 默认 0，消除每 epoch 数小时 buffer 填充。
2. **FHT val_loss 震荡**：val 不 balance；固定 event index；FHT npz `.copy()` 缓存；`waveform_fht` 取消双重 invert。
3. **WFS decon**：`WFSampling.py` 支持 ragged `decon_waveform`；三类 decon WFS 已跑完。
4. **GPU**：使用 `--gpu N` 指定物理卡，勿与 `env CUDA_VISIBLE_DEVICES` 混用。

