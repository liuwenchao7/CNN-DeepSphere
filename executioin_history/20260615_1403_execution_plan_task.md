# 执行记录：EXECUTION_PLAN 本次任务（2026-06-15）

时间：2026-06-15 14:03

## 任务一：修复 decon_npevst 训练并重启

### 问题诊断

1. **数据载入未真正使用缓存**：`make_sample_loader` 虽将 `decon_npevst` 文件放入 LRU 缓存，但随后仍调用 `load_decon_npevst()` 再次 `np.load()` 读盘，与 `get_stacked_data`（整文件载入、按 event 索引）不一致。
2. **GPU OOM**：全量训练在 Epoch 1 中途崩溃（`ResourceExhaustedError`，CNN 子批 `batch*500=8000` 过大）。

### 修复

**`pid_lib/waveform_io.py`**
- 新增 `parse_decon_npevst_hits()`、`decon_npevst_event_from_file()`：从预加载文件数组按 event 索引解析。
- `load_decon_npevst()` 改为先载入整文件再调用 event 解析函数。

**`train_cnn_ds_pid.py`**
- `decon_npevst` 分支改为：`file_arr = cache[...]` → `decon_npevst_event_from_file(file_arr, event_idx, ...)`，不再重复读盘。
- 新增 `--pmt-batch-size`（默认 **200**，原硬编码 500），降低 GPU 峰值显存。
- `PMTBatchCNNLayer` / `pmt_batch_cnn` 使用可配置 PMT 子批大小。

### 验证与启动

- Smoke：`outputs/fea6+decon_npevst_smoke_v3`（batch=16, pmt_batch=200, 1 epoch）通过。
- 全量训练已重启：
  - 日志：`logs/train_cnn_ds_decon_npevst_full.log`
  - PID：**61377**，GPU 2，`batch=16`，`pmt_batch=200`

---

## 任务二：WFS 数据 get_stacked_data 式载入并训练

### 背景

WFSampling 全量处理已完成（nue 1795 ok，nc 1486 ok + 5 error，numu 已完成）。

### 修复

**`pid_lib/waveform_io.py`**
- 新增 `load_wfsampling_file()`：整文件载入（`.npy` 对象字典或 `.npz`）。
- 新增 `wfsampling_event_from_file()`：从缓存按 event 索引解析 `(N_PMT, max_points, 2)`。
- `load_wfsampling()` 改为调用上述两函数。

**`train_cnn_ds_pid.py`**
- `wfsampling` 分支增加文件级 LRU 缓存，与 `decon_npevst` 同模式。

### 验证与启动

- Smoke：`outputs/fea6+wfsampling_smoke_v1`（timesteps=61, channels=2）通过。
- 全量训练已启动：
  - 日志：`logs/train_cnn_ds_wfsampling_full.log`
  - PID：**61278**，GPU 3，`batch=16`，`pmt_batch=200`
  - 输出目录：`outputs/fea6+wfsampling`

---

## logs 更新

- 删除已结束的 `wfs_nue_full.log`、`wfs_nc_full.log`
- 新增 `train_cnn_ds_wfsampling_full.log`
- 更新 `logs/pid`（4 个运行中任务）

## 修改文件列表

| 文件 | 变更 |
|------|------|
| `pid_lib/waveform_io.py` | decon_npevst / wfsampling 文件级载入与 event 解析 |
| `train_cnn_ds_pid.py` | 缓存修复、wfs 缓存、`--pmt-batch-size` |
| `logs/pid` | 更新 PID |
| `README.md` | logs 目录说明 |
| `outputs/fea6+decon_npevst_smoke_v3/` | smoke 输出（新增） |
| `outputs/fea6+wfsampling_smoke_v1/` | smoke 输出（新增） |

## 待办（EXECUTION_PLAN 后续）

- FHT 预处理完成后启动 `decon_waveform_fht` / `waveform_fht` 训练
- 5 模型 score 分布图与 visE 分箱 AUC 图（`docs/SCORE_AND_BOOTSTRAP.md`）
