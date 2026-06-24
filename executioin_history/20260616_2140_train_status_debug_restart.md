# 执行记录：全量训练状态检查、失败任务 debug 与重启（2026-06-16 21:40）

## 任务

检查全量训练运行情况；失败任务 debug 并重启；检查数据预处理完成度。

## 训练状态（检查时点）

### 正常运行（未重启）

| 任务 | PID | 进度 | 备注 |
|------|-----|------|------|
| `fea6+decon_npevst` | 13980 | Epoch 2/150 进行中 | Epoch1 val_loss=0.8715 |
| `fea6+wfsampling` | 61278 | Epoch 11/150 | Epoch10 val_loss=0.6190, val_acc=0.7400 |

两任务因历史 `setup_tf` 覆盖 `CUDA_VISIBLE_DEVICES`，实际均占用物理 GPU0（与 `env CUDA_VISIBLE_DEVICES=2/3` 启动方式冲突）。为保留进度未迁移 GPU。

### 已失败 / 已重启

| 任务 | 原问题 | 处理 |
|------|--------|------|
| `fea6+waveform_fht` | `NpzFile.zip` 缓存 bug；旧进程 13819 已停 | 修复 `_load_compact_npz_arrays`；从 epoch1 checkpoint 续训，GPU1，PID 71493 |
| `fea6+decon_waveform_fht` | GPU0 OOM + TypeSpec 错误，从未成功训练 | 全新启动 GPU2，PID 71185 |

## 代码修复

### `train_cnn_ds_pid.py`

1. **`setup_tf`**：若环境已设置 `CUDA_VISIBLE_DEVICES` 则不再覆盖；后续启动请直接用 `--gpu N` 指定物理卡。
2. **`_load_compact_npz_arrays`**（此前已改）：FHT npz 载入为纯数组 dict，避免 LRU 缓存 `NpzFile` 导致 `AttributeError: zip`。

### `WFSampling/WFSampling.py`（此前已改）

`_load_waveform_events` 支持 decon_waveform 的 ragged `waveform_offsets` 布局。

## 数据预处理状态

| 数据 | numu | nue | nc | 状态 |
|------|------|-----|-----|------|
| WFSampling (raw waveform) | 1144 | 1795 | 1486 | 完成 |
| WFSampling_decon_waveform | 0 | 0 | 0 | 此前全失败；已重跑（PID 71186–71188） |
| FHT decon (`decon_wav_fht+-/wav`) | 1299 | 1705 | 1400 | 完成 |
| FHT raw (`wav_fht+-/wav`) | 1144 | 1793 | 1483 | 完成 |

WFS decon 单文件体积大，首批输出需等待数分钟；载入 smoke test（file 2000）已通过。

## 新增 / 更新文件

- `logs/pid`：更新运行中任务 PID
- `logs/train_cnn_ds_waveform_fht_full.log`：追加重启日志
- `logs/train_cnn_ds_decon_waveform_fht_full.log`：重新写入
- `logs/wfs_decon_waveform_{numu,nue,nc}.log`：重新写入

## 后续

1. WFS decon 三类完成后启动 `fea6+wfs_decon_waveform` 全量训练（建议 `--gpu` 指定空闲卡）。
2. `fea6+decon_npevst` / `fea6+wfsampling` 若需迁到独立 GPU，需在合适 epoch 检查点后续训并改 `--gpu`。
3. EXECUTION_PLAN 后续：score 分布图、visE 分箱 AUC。
