# 2026-06-29 修复 decon_waveform h_thr 并重启预处理与训练

## 问题背景

WFS_decon_wav_v2 预处理（`WFSampling_v2.py`）使用了 `h_thr=50`，但 decon_waveform 的 AC 值经过
`ac = (signal - 1000) / 100` 缩放后量级约为 0–10，h_thr=50 导致关键点确认条件永远无法触发，
所有 PMT 的关键点实际为空，等效于只用 fea6 特征训练（已完成的错误训练 acc=0.757，与 fea6 基线相近）。

正确值：`h_thr=0.5`（对应 decon AC 量级）。

## 修改文件

### `WFSampling/WFSampling_v2.py`

1. **新增 `default_h_thr(waveform_kind)` 函数**（位于 `default_global_thr()` 之后）：
   ```python
   def default_h_thr(waveform_kind):
       return 0.5 if waveform_kind == "decon_waveform" else 50.0
   ```

2. **`--h-thr` CLI 参数**：`default=None`（原为 `0.5`），运行时按类型解析：
   ```python
   h_thr = args.h_thr if args.h_thr is not None else default_h_thr(args.waveform_kind)
   ```

3. **`sampling_params.json` 改为处理完成后写入**（原为处理前写入），新增字段：
   - `results`：各状态文件计数（ok/skipped/missing/error）
   - `n_files_processed`：ok + skipped 合计
   - `completed_at`：ISO 8601 时间戳

4. **新增 `import datetime`** 至顶部 import 区。

5. **启动日志输出新增 h_thr/global_thr 值**，便于核查参数：
   ```
   [WFSampling_v2] 1302 decon_waveform files class=numu h_thr=0.5 global_thr=1.0 -> ...
   ```

## 操作记录

### 1. 清理旧训练输出（错误数据训练结果）

删除 `outputs/fea6+wfs_decon_wav_v2/` 下：
`checkpoint`, `checkpoint_pid.*`, `loss_log.txt`, `confusion_matrix.png`,
`detail.txt`, `learning_curve.png`, `roc_ovr.png`, `test_metrics.json`,
`y_pred.npy`, `y_prob.npy`, `y_true.npy`（保留 `class_mapping.json`）

### 2. 重新运行 WFS_decon_wav_v2 预处理

```bash
# 三类均使用 h_thr=0.5（自动默认），--force 强制覆盖
nohup python3 -u WFSampling/WFSampling_v2.py \
  --class numu --waveform-kind decon_waveform --force --workers 8 \
  > logs/wfs_decon_wav_v2_numu.log 2>&1 &   # PID 122638
nohup python3 -u WFSampling/WFSampling_v2.py \
  --class nue --waveform-kind decon_waveform --force --workers 8 \
  > logs/wfs_decon_wav_v2_nue.log 2>&1 &    # PID 123041
nohup python3 -u WFSampling/WFSampling_v2.py \
  --class nc --waveform-kind decon_waveform --force --workers 8 \
  > logs/wfs_decon_wav_v2_nc.log 2>&1 &     # PID 123659
```

输出目录：`/disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2/{numu,nue,nc}/`

### 3. 等待预处理完成后自动启动训练

写入 `logs/wait_and_train_decon.sh`（PID 124141），等待三类预处理 PID 结束后自动执行：
```bash
env CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn_ds_pid.py \
  --output-dir outputs/fea6+wfs_decon_wav_v2 \
  --waveform-source wfs_decon_waveform \
  --wfs-decon-root /disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2 \
  --manifest-dir outputs/fea6 --gpu 2 \
  --batch-size 16 --pmt-batch-size 200 --shuffle-buffer 0
```

训练日志：`logs/train_cnn_ds_wfs_decon_wav_v2_full.log`

### 4. wfs_wav_v2 训练（继续，无干预）

PID 120536，GPU 3，当前 Epoch 7，val_loss 0.6083，val_acc 0.7447，正常运行。
训练结束后自动生成 `test_metrics.json`、`confusion_matrix.png`、`roc_ovr.png`。

## 新增/修改文件一览

| 路径 | 操作 | 说明 |
|------|------|------|
| `WFSampling/WFSampling_v2.py` | 修改 | 添加 `default_h_thr()`，修复 `--h-thr` 默认值逻辑，改进 `sampling_params.json` 写入 |
| `logs/wfs_decon_wav_v2_numu.log` | 新增 | numu 预处理日志 |
| `logs/wfs_decon_wav_v2_nue.log` | 新增 | nue 预处理日志 |
| `logs/wfs_decon_wav_v2_nc.log` | 新增 | nc 预处理日志 |
| `logs/wait_and_train_decon.sh` | 新增 | 等待预处理完成后自动启动训练的脚本 |
| `logs/wait_and_train_decon.log` | 新增 | 等待脚本运行日志 |
| `logs/pid` | 修改 | 更新当前运行任务 PID |
| `outputs/fea6+wfs_decon_wav_v2/` | 清理 | 删除错误数据产生的 checkpoint 和评估结果 |
| `executioin_history/20260629_fix_decon_h_thr_and_rerun.md` | 新增 | 本执行记录 |
