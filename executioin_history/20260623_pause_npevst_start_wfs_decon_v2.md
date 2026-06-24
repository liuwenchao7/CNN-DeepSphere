# 执行记录：暂停 decon_npevst / 启动 wfs_decon_wav_v2 训练（2026-06-23）

## 1. 暂停 `fea6+decon_npevst` 并检查

- 已停止 PID **127925**（GPU2）
- **实测比预期更慢**：当前 run 在 Ep1 step **728/1073** 时约 **110 s/step** → 每 epoch **~33 h**（此前估计 8–29 h）
- **根因**（与 260622 分析一致）：每个样本在 Python 中执行 `parse_decon_npevst_hits` 的 `for pid in np.unique(pmt)` 循环（最多 17612 次），CPU 成为瓶颈，GPU 长期空闲
- 另：上次 Ep1 结束时曾因 **磁盘满** 保存 checkpoint 失败；当前磁盘约 1.1 TB 可用

**结论**：decon_npevst 全量训练暂停，待后续考虑 WFS 替代或向量化优化后再启。

## 2. WFS v2 预处理完成确认

| 目录 | numu | nue | nc |
|------|------|-----|-----|
| `WFS_wav_v2` | 1144 | 1795 | 1486 |
| `WFS_decon_wav_v2` | 1302 | 1712 | 1428 |

## 3. numpy 兼容性修复

WFS v2 由 `cnn_deepsphere`（numpy 2.x）生成，**deepsphere**（numpy 1.x）无法直接 `np.load`。

- 新增 `tools/convert_wfs_npy_compat.py`：读 numpy2 → 转 list 格式 → deepsphere 保存
- 已对 `WFS_decon_wav_v2` **4442** 个文件完成转换
- `WFSampling_v2.py` 今后直接保存 list 格式，避免重复问题
- `waveform_io.py` 支持 list 型 offsets/time/adc

## 4. `fea6+wfs_decon_wav_v2` smoke + 全量训练

**Smoke**（`outputs/fea6+wfs_decon_wav_v2_smoke`）：
- timesteps=**37**, channels=2
- 1 epoch ~**83 s**（2 steps，小样本）
- test acc 0.38（smoke 样本过少，仅验证流程）

**全量训练已启动**：
```bash
train_cnn_ds_pid.py --output-dir outputs/fea6+wfs_decon_wav_v2 \
  --waveform-source wfs_decon_waveform \
  --wfs-decon-root /disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2 \
  --gpu 2 --batch-size 16 --pmt-batch-size 200 --shuffle-buffer 0
```
- PID：**79754**
- 日志：`logs/train_cnn_ds_wfs_decon_wav_v2_full.log`
- 参考 `fea6+wfs_wav`：预计 **~3 h/epoch**

## 修改文件

| 文件 | 说明 |
|------|------|
| `tools/convert_wfs_npy_compat.py` | numpy2→deepsphere 格式转换 |
| `WFSampling/WFSampling_v2.py` | 保存 list 兼容格式 |
| `pid_lib/waveform_io.py` | list 格式加载 |
| `train_cnn_ds_pid.py` | 默认 wfs 根目录改为 v2 |
