# 执行记录：暂停 decon_wav_fht / 启动 wfs_wav_v2 训练（2026-06-24）

## 1. 暂停 `fea6+decon_wav_fht`

- 已停止 PID **16374**（GPU3，`decon_waveform_fht`）
- 原因：FHT 窗口训练每 epoch **~23 h**（page cache 冷启动 + 300 时间步 CNN），过慢

## 2. `WFS_wav_v2` 格式转换

- v2 文件需转为 deepsphere 可读的 list 格式（同 decon 处理）
- `tools/convert_wfs_npy_compat.py` 增加快速 skip 逻辑
- **4425** 个文件已兼容（numu 1144 / nue 1795 / nc 1486）

## 3. `fea6+wfs_wav_v2` smoke + 全量

**Smoke**（`outputs/fea6+wfs_wav_v2_smoke`）：
- timesteps=**92**, channels=2
- test acc **0.476**, macro AUC **0.692**

**全量训练**（GPU3）：
```bash
--output-dir outputs/fea6+wfs_wav_v2 \
--waveform-source wfsampling \
--wfsampling-root /disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v2 \
--gpu 3 --batch-size 16 --pmt-batch-size 200 --shuffle-buffer 0
```
- PID：**120462**（python 子进程见 `ps`）
- 日志：`logs/train_cnn_ds_wfs_wav_v2_full.log`
- 预计 **~3 h/epoch**（参考 fea6+wfs_wav v1）

## 当前 GPU

| GPU | 任务 |
|-----|------|
| 2 | `fea6+wfs_decon_wav_v2`（79790，续训中） |
| 3 | `fea6+wfs_wav_v2`（新启动） |
