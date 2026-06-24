# 执行记录：EXECUTION_PLAN 本次任务（2026-06-15）

## 任务一：decon_waveform WFS + wfs_decon_waveform 训练入口

### 代码
- `WFSampling/WFSampling.py`：新增 `--waveform-kind {waveform, decon_waveform}`；decon 不做 `16384-ADC` 反转；输出默认至 `WFSampling_decon_waveform/`；修复 `decon_waveform_{id}.npz` 的 file_id 解析。
- `train_cnn_ds_pid.py`：新增波形源 `wfs_decon_waveform` 与 `--wfs-decon-root`；载入方式与 `wfsampling` 相同（文件级缓存 + event 索引）。
- `pid_lib/config.py`：新增 `WFSAMPLING_DECON_WAVEFORM_ROOT`。

### 后台
三类 decon_waveform WFSampling 已启动（各 4 workers）：
- `logs/wfs_decon_waveform_{numu,nue,nc}.log`
- 输出：`/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling_decon_waveform/`

全量训练 `fea6+wfs_decon_waveform` 待 WFS 产物覆盖足够文件后启动。

---

## 任务二：停止 decon_npevst 并以更大 batch 重启

- 已停止旧进程（PID 61377，batch=16）。
- Smoke：`batch=32, pmt_batch=200` 通过。
- 已重启全量训练：
  - 输出：`outputs/fea6+decon_npevst`
  - 日志：`logs/train_cnn_ds_decon_npevst_full.log`
  - PID：**13980**，GPU 2，`batch=32`

---

## 任务三：FHT 截取数据测试与全量训练

- FHT 预处理已完成（decon ~4369 ok，wav ~4394 ok）。
- `train_cnn_ds_pid.py`：FHT npz 改为文件级 LRU 缓存（`_event_from_compact_z`）。
- Smoke：`decon_waveform_fht` / `waveform_fht`，`batch=16, pmt_batch=150` 均通过。
- 已启动全量训练：

| 实验 | 日志 | PID | GPU | batch |
|------|------|-----|-----|-------|
| fea6+decon_waveform_fht | `train_cnn_ds_decon_waveform_fht_full.log` | 13818 | 0 | 16 |
| fea6+waveform_fht | `train_cnn_ds_waveform_fht_full.log` | 13819 | 1 | 16 |

---

## 任务四：方向重建 fea4 项目 Agent 说明

已写入 **`docs/DIR_RECON_AGENT_BRIEF.md`**，包含：
- 5 组 fea4+波形实验定义
- 数据路径、特征列表、波形来源
- 参考脚本（`train_200_fht.py` / `train_cnn_ds_pid.py`）
- get_stacked_data 载入要求与环境约定

---

## 当前并行训练（保留）

| 任务 | PID |
|------|-----|
| fea6+wfsampling | 61278 |
| fea6+decon_npevst (batch=32) | 13980 |
| fea6+decon_waveform_fht | 13818 |
| fea6+waveform_fht | 13819 |

## 修改文件

- `WFSampling/WFSampling.py`
- `train_cnn_ds_pid.py`
- `pid_lib/config.py`
- `docs/DIR_RECON_AGENT_BRIEF.md`
- `logs/pid`
