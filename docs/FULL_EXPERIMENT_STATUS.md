# 全量实验状态说明

更新时间：**2026-06-21**（CST）

本文档汇总 CNN+DeepSphere 各实验的 **训练参数**、**样本数**、**预处理完成度**、**输入形状**、**每 epoch 耗时**（有实测则记录）。数据划分与归一化复用 `outputs/fea6/` 的 manifest 与 `norm_stats.json`。

---

## 1. 公共配置

| 项目 | 值 |
|------|-----|
| 脚本 | `train_cnn_ds_pid.py`（根目录） |
| 数据划分（文件） | train **3734** / val **466** / test **468** |
| 类别平衡（训练） | `cap_per_class = min(三类事件数)`，仅 **训练集** balance |
| 验证/测试 | 全事件，不 balance；generator 会 **静默跳过** 波形加载失败的事例 |
| `EarlyStopping` | `patience=15`，监控 `val_loss` |
| `epochs` 上限 | 150 |
| `learning_rate` | 5e-4（每 epoch ×0.995） |
| `fill_nan` | -3.0 |
| `shuffle` | 默认 `shuffle_buffer=0`：每 epoch 在 Python 侧打乱 event index |

### 1.1 平衡后训练事件数

| Split | 原始三类合计 | cap_per_class | 平衡后总事件数 |
|-------|-------------|---------------|----------------|
| train | {11439, 15722, 15435} | 11439 | **34,317** |
| val | {1487, 1964, 1875} | — | **5,326**（未截断） |
| test（elec 索引） | {1443, 1947, 1953} | — | **5,343**（未截断） |

训练 steps/epoch ≈ `34317 / batch_size`（向下取整）。

---

## 2. 数据预处理状态

### 2.1 WFSampling v1（raw waveform）

| 类 | 源文件数 | 已生成 `.npy` | 状态 |
|----|---------|---------------|------|
| numu | 1145 | 1144 | 完成（1 个源文件损坏 fid=3221） |
| nue | 1795 | 1795 | **完成** |
| nc | 1491 | 1486 | 完成 |

根目录：`/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling/{cls}/`

### 2.2 WFSampling v1（decon_waveform）

| 类 | 目标 | 已生成 | 状态 |
|----|------|--------|------|
| numu | 1302 | 1302 | **完成** |
| nue | 1712 | 1712 | **完成** |
| nc | 1428 | 1428 | **完成** |

根目录：`/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling_decon_waveform/{cls}/`

### 2.3 WFSampling v2（2026-06-22 更新参数并重启）

| 种类 | 输出目录 | 参数 | 状态 |
|------|----------|------|------|
| raw waveform | `WFS_wav_v2/{cls}/` | thr_fht=**0.2**, global_thr=**15** | **运行中**（`--force` 重跑） |
| decon_waveform | `WFS_decon_wav_v2/{cls}/` | thr_fht=**0.2**, global_thr=**1** | **运行中** |

脚本：`WFSampling/WFSampling_v2.py`。v1 数据在 `WFS_wav_v1/`。

### 2.4 FHT 窗口（fht−20, width=300）

| 种类 | numu | nue | nc | 预处理 summary |
|------|------|-----|-----|----------------|
| decon (`decon_wav_fht+-/wav`) | 1299 | 1705 | 1400 | ok=4369, skip=35, fail=264 |
| raw (`wav_fht+-/wav`) | 1144 | 1793 | 1483 | ok=4394, skip=26, fail=248 |

失败原因主要为 **`no_src`**（manifest 有 file_id 但源 npz 不存在）。

---

## 3. 各训练实验状态

### 3.1 已完成基线

| 实验 | 脚本 | batch | 输入 | 结果 | 每 epoch |
|------|------|-------|------|------|----------|
| **fea6** | `fea/train_fea.py` | 16 | `(17612, 6)` → DS `(12288, 6)` | epoch 70 早停，test acc **0.759** | **~3.4 min**（总 ~4 h） |

### 3.2 CNN+DeepSphere 全量

| 实验目录 | batch | pmt_batch | waveform 形状| 状态 | 最佳 val_loss | 每 epoch（实测） |
|----------|--------|-------|-----------|---------------|---------|----------|------|---------------|----------------|
| `fea6+decon_npevst` | 32 | 200 | `(17612, 43, 2)*` | **已暂停**（~110 s/step，~33 h/ep） | 0.7921（Ep2 ckpt） | — |
| `fea6+wfs_decon_wav_v2` | 16 | 200 | `(17612, 37, 2)*` | **训练中** Ep1+ | — | 预计 **~3 h/ep**（参考 wfs_wav） |
| `fea6+wfs_wav` | 16 | 200 | `(17612, 61, 2)*`| **完成** Ep47 早停 | 0.5625（Ep32） | **~3.0 h**（总 140 h / 47 ep） |
| `fea6+decon_wav_fht` | 16 | 150 | `(17612, 300, 1)` | **已暂停**（~23 h/ep） | 0.6228 | — |
| `fea6+wfs_wav_v2` | 16 | 200 | `(17612, 92, 2)*` | **训练中** Ep1+ | — | 预计 **~3 h/ep** |
| `fea6+wav_fht` | 16 | 150 | `(17612, 300, 1)` | invert 修复后 Ep2+ | 0.8008（新 Ep1） | 新跑 Ep1–2 **~23 h** |
| `fea6+wfs_decon_waveform` | — | — | — | — | — | **未启动**（v1 预处理已完成） | — | — |

\* `max_points` 由 `estimate_waveform_params` 统计；decon_npevst / wfsampling 约 **43** 点 × 2 通道。

**DeepSphere 输入维 12**：`6(CNN 逐 PMT) + 6(elec 静态拼在时间轴前)`。

### 3.3 `fea6+wfs_wav` 测试结果（2026-06-21）

| 指标 | 值 |
|------|-----|
| test accuracy | **0.759** |
| macro AUC | **0.915** |
| 有效测试事例 | **4137**（generator 跳过无 WFS 的事例；elec 索引共 5343） |
| 结果图 | `audit/plots/wfs_wav/` |

### 3.4 当前后台进程（`logs/pid`）

| 任务 | PID | GPU | 说明 |
|------|-----|-----|------|
| `train_wfs_decon_wav_v2` | 79790 | 2 | `outputs/fea6+wfs_decon_wav_v2` |
| `train_wfs_wav_v2` | 120462 | 3 | `outputs/fea6+wfs_wav_v2` |

---

## 4. 已知问题

1. **FHT val_loss 震荡**（已修复）：`waveform_fht` 双重 invert、验证集 balance、checkpoint 污染；修复后重训。
2. **decon_npevst shuffle 极慢**（已修复）：`tf.data` buffer=4096 预取；改为 `shuffle_buffer=0`。
3. **测试 visE 对齐**：CNN+DS 测试时 generator 跳过波形失败事例，`y_true` 行数可能小于 manifest test 事件数；能量分箱 AUC 需 `tools/export_test_vise.py` 按相同跳过逻辑导出 `vise_test.npy`。
