# 执行记录：EXECUTION_PLAN 本次任务（进行中）

时间：2026-06-14 20:43+

## 已执行

1. 停止当前 3 路 PID 训练进程（decon_npevst / decon_waveform / waveform）。

2. 重构 `train_cnn_ds_pid.py`（按方向脚本骨干）：
   - 去掉 `fea_proj = Dense(3)`；
   - 恢复 `PMTBatchCNNLayer`（`tf.keras.layers.Layer`）并把 `cnn_model_0/1` 作为成员追踪；
   - PMT 端 CNN 改为：
     - `Conv1D(8) -> MaxPool -> Conv1D(16) -> MaxPool -> Conv1D(16) -> Flatten -> Dense(128) -> Dense(feature_num)`；
   - 静态特征与 CNN 输出直接拼接，不再额外投影；
   - 设定 `static_feats=6`、`feature_num=6`，每 PMT 最终 12 维；
   - DeepSphere 输入变为 `(None, 12288, 12)`，输出 `(None, 3)`；
   - 支持在入口把 `(waveform, elec_features)` 合并为 `group_input`（静态特征置于时间维前部）。

3. `decon_npevst` 载入方式优化（参考 `get_stacked_data` 思路）：
   - 对文件级 `decon_npevst` 与 `elec_fea` 做 LRU 缓存，减少重复磁盘读取；
   - 使用 `batch_size=16` 启动训练。

4. 增加 FHT 窗口预处理脚本：
   - 新增 `tools/preprocess_pid_fht_window.py`；
   - 按 `(fht-20, fht+280)` 截取 300ns 窗口；
   - 输出到：
     - `/disk_pool1/liuwc/data/cnn+ds/pid/decon_wav_fht+-/wav`
     - `/disk_pool1/liuwc/data/cnn+ds/pid/wav_fht+-/wav`
   - 训练脚本新增 `decon_waveform_fht` / `waveform_fht` 两种 source（预处理中，待跑）。

5. 路径兼容：
   - 新增 `WFSampling/train_cnn_ds_pid.py` 包装器，兼容旧启动命令路径。

## 验证结果（smoke）

`decon_npevst` smoke (`batch=16`, 1 epoch) 通过，日志显示：

- `pmt_to_pix` shape: `(None, 12288, 12)`；
- 模型输出 shape: `(None, 3)`；
- `trainable_variables` 中包含两套 PMT CNN 的 `conv1d*` / `dense*` 权重。

## 当前后台任务

1. `decon_npevst` 全量训练（新结构，batch=16）已启动：
   - `logs/train_cnn_ds_decon_npevst_full.log`

2. FHT 窗口预处理已启动（CPU 多进程）：
   - `logs/preprocess_decon_waveform_fht.log`
   - `logs/preprocess_waveform_fht.log`

## 观察

- 目前 GPU 利用率仍偏低（接近 0%），说明瓶颈仍以 CPU/IO 为主；
- 已通过缓存和 batch 提升做第一轮优化，待窗口预处理数据落地后再用 `*_fht` source 进一步提速并复测 GPU 利用率。
