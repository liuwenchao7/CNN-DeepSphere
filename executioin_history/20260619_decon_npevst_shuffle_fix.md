# 执行记录：decon_npevst shuffle 修复与文档更新（2026-06-19）

## shuffle 极慢原因

`tf.data.shuffle(buffer_size=4096, reshuffle_each_iteration=True)` 每 epoch 需预取 4096 个样本；decon_npevst 每样本磁盘读 `elec_fea` + `decon_npevst.npy`，填充 buffer 数小时，表现为日志 `Filling up shuffle buffer: 4xxx of 4096`，实测 ~24 h/epoch。

## 修复（`train_cnn_ds_pid.py`）

- 新增 `--shuffle-buffer`（默认 **0**）
- `make_indexed_generator`：每 epoch 用 `seed+epoch_no` 打乱 event index
- `shuffle_buffer>0` 时可选小 buffer，且 `reshuffle_each_iteration=False`

## 续训

- PID **33828**，GPU1，`batch=32`，`--shuffle-buffer 0`，从 Ep2 checkpoint 续训
- 修复后首 epoch 直接 **1073 steps**（无 buffer 等待）

## 文档

- 更新 `docs/FULL_EXPERIMENT_STATUS.md`
- 重写 `CHANGES.md`（相对 train_200_fht 与各实验差异）
