# 执行记录：实验耗时统计、预处理失败原因、FHT 可视化与 val_loss 修复（2026-06-19）

## 各实验运行时间与每 epoch 耗时

基于训练日志首尾时间戳与 `loss_log.txt` epoch 数估算（CNN+DeepSphere 全量，batch=16，pmt_batch=150–200）：

| 实验 | 已完成 epoch | 总墙钟时间 | 约每 epoch | 备注 |
|------|-------------|-----------|-----------|------|
| fea6（6特征 DeepSphere） | 70（早停） | ~4.0 h | ~3.4 min | 无逐 PMT CNN |
| fea6+decon_npevst | 3（已停） | ~72.8 h | ~24.3 h | shuffle buffer 填充极慢 |
| fea6+wfsampling | 34+（运行中） | ~103 h | ~3.0 h | PID 61278 |
| fea6+decon_waveform_fht | 28（已停）→ 续训 | ~63.3 h / 28 | ~2.3 h | 修复后 PID 16374 续训 |
| fea6+waveform_fht | 5（旧）→ 重训 | 旧跑 ~6 h/ep | — | invert 修复后 PID 16478 全新日志 |

FHT 路每 epoch 约 **2096** train steps（平衡后 ~33536 events / batch 16）。

## 数据预处理完成度与失败原因

### FHT 窗口（`preprocess_pid_fht_window.py`）

| 种类 | ok | skip | fail | 失败原因 |
|------|-----|------|------|----------|
| decon_waveform | 4369 | 35 | 264 | 全部 **`no_src`**：manifest 中有 file_id，但源 `decon_waveform_*.npz` 不存在 |
| waveform | 4394 | 26 | 248 | **`no_src`** 247 + **`bad_src`** 1（npz 损坏/无法读取） |

`skip`：输出文件已存在。`no_fht`：本次日志中未出现。

### WFSampling raw

| 类 | 文件数 | 状态 |
|----|--------|------|
| numu / nue / nc | 1144 / 1795 / 1486 | 完成 |

### WFSampling decon_waveform

| 类 | 文件数 | 状态 |
|----|--------|------|
| numu | 1302/1302 | 完成 |
| nc | 1428/1428 | 完成 |
| nue | ~1676/1712 | 进行中（PID 71187） |

此前 decon WFS 全失败原因为 `WFSampling.py` 未支持 ragged `waveform_offsets`，已修复。

## FHT 可视化

新增 `tools/visualize_pid_data.py --fht-only`，输出至 `audit/viz/`：

- `fht_numu_f2000_e0_p16194.png`
- `fht_nue_f2500_e0_p3081.png`
- `fht_nc_f0_e0_p7803.png`

## val_loss 剧烈震荡原因与修复

### 原因

1. **`waveform_fht` 双重取反**：预处理已做 `16384-ADC`，训练又 `invert=True`，波形极性错误。
2. **验证集 `balance_events=True`**：验证子采样与训练不一致，指标不稳定。
3. **每 epoch 重建 generator + 静默跳过加载失败**：有效 val 样本数波动 → val_loss 偶发飙升（如 1.7、7.8），val_acc 跌至 ~0.45。

### 代码修改（`train_cnn_ds_pid.py`）

- `waveform_fht`：`invert=False`
- `build_event_index`：启动时固定 event 列表；**仅训练集 balance**
- `_load_compact_npz_arrays`：`.copy()` 避免缓存隐患
- `tools/visualize_pid_data.py`：FHT 对比图

### 重启训练

- `fea6+waveform_fht`：GPU2，PID **16478**（新日志，invert 修复）
- `fea6+decon_waveform_fht`：GPU3，PID **16374**（从 checkpoint 续训）
- smoke 验证：`outputs/fea6+waveform_fht_smoke_fix` 3 epoch val_loss 单调下降
