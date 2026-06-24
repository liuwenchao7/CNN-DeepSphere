# 执行记录：调大 batch_size 并重启 CNN+DS 全量训练

**时间**：2026-06-14  
**原因**：用户反馈 `decon_npevst` 输入远小于长波形但 epoch 耗时差距不大（~16h vs ~24h），要求调大 `batch_size` 并重启。

## 慢速原因分析

1. **共享固定开销**：三路实验 DeepSphere 输入均为 `(12288, 9)`，GCN 计算量相同。
2. **逐 PMT CNN 全扫**：每个样本对 **17612 个 PMT** 分块跑 CNN（`stride=500`），`decon_npevst` 虽只有 43 点，但仍要遍历全部 PMT。
3. **CPU/数据管线瓶颈**：`tf.data` 使用 Python generator 逐事件从磁盘读 `elec_fea` + 波形；`nvidia-smi` 显示 GPU 利用率长期接近 0%，说明主要时间花在 I/O 与 CPU 预处理，而非 GPU 算力。
4. **内存压力**：长波形 `batch=8` 时单进程 RSS ~290GB，三路并行占满 755GB 内存，进一步拖慢调度。

因此输入形状缩小 **23×** 不会带来同比例加速。

## 操作

1. 停止旧进程（PID 61190 / 61020 / 61019）。
2. 清除旧 checkpoint 与 loss_log（保留输出目录结构）。
3. 以更大 batch 重启（`nohup`）：

| 实验 | GPU | 旧 batch | 新 batch | 新 PID | 预估 steps/epoch |
|------|-----|----------|----------|--------|-----------------|
| `fea6+decon_npevst` | 2 | 8 | **32** | 92032 | 1073 |
| `fea6+decon_waveform` | 3 | 8 | **10** | 92033 | 3432 |
| `fea6+waveform` | 0 | 8 | **10** | 92034 | 3432 |

长波形仅调到 10（非 32），因三路并行时单进程约 360GB，再大易 OOM；`decon_npevst` 内存占用小，可大幅提到 32。

## 日志

- `logs/train_cnn_ds_decon_npevst_full.log`
- `logs/train_cnn_ds_decon_waveform_full.log`
- `logs/train_cnn_ds_waveform_full.log`
