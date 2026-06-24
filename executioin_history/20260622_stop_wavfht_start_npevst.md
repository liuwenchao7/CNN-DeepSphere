# 执行记录：停止 wav_fht / 启动 decon_npevst / 清理 WFS_wav_v2（2026-06-22）

## 1. 停止 `fea6+wav_fht` 训练

- 终止 PID **17441**（`waveform_fht`，GPU2）
- 从 `logs/pid` 移除 `train_waveform_fht`

## 2. 启动 `fea6+decon_npevst` 训练

- 原进程 33828 已不在运行（Ep1 保存 checkpoint 时崩溃）
- 在 **GPU2** 重新启动：
  ```bash
  train_cnn_ds_pid.py --output-dir outputs/fea6+decon_npevst \
    --waveform-source decon_npevst --manifest-dir outputs/fea6 \
    --gpu 2 --batch-size 32 --pmt-batch-size 200 --shuffle-buffer 0
  ```
- 新 PID：**127925**（从 Ep2 checkpoint 续训，best val_loss=0.7921）
- 日志：`logs/train_cnn_ds_decon_npevst_full.log`

## 3. 清理 `WFS_wav_v2` 旧参数数据

- 判定标准：修改时间早于 **2026-06-22 16:45**（thr_fht=0.2 全量重跑开始前）
- 删除 **754** 个 `wfsampling_*.npy`，释放约 **27.9 GB**
- 保留新参数下已生成的文件（numu 77 / nue 73 / nc 63 等）
- 后台 `wfs_wav_v2_*` 预处理（`--force`）继续运行，会补全缺失文件

## 当前 GPU 分配

| GPU | 任务 |
|-----|------|
| 2 | `fea6+decon_npevst` |
| 3 | `fea6+decon_wav_fht` |
