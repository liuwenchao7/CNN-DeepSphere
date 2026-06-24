# 执行记录：全量实验状态信息汇总

**时间**：2026-06-14 16:18  
**类型**：信息收集（无代码改动、无新训练启动）

## 任务

应用户要求：

1. 监控 WFSampling 全量处理是否完成；
2. 汇总当前进行中的全量实验的脚本输入形状、模型输入形状、每 epoch 步数、耗时估算；
3. 写入说明文档。

## 调查结果

### WFSampling

- **numu**：已完成（1144 ok，1 error：`fid=3221`）
- **nue**：1368/1795（76%），进行中，ETA ~16h
- **nc**：1198/1491（80%），进行中，5 个源文件错误，ETA ~13h

### CNN+DeepSphere 全量训练（3 路并行）

| 实验 | GPU | Epoch | steps/epoch | ~h/epoch |
|------|-----|-------|-------------|----------|
| fea6+decon_npevst | 2 | 3/150 | 4290 | 16.2 |
| fea6+decon_waveform | 3 | 2/150 | 4191 | 24.3 |
| fea6+waveform | 0 | 2/150 | 4030 | 24.3 |

- `fea6+wfsampling`：未启动，预估形状 `(17612,63,2)`

## 新增/更新文件

| 文件 | 说明 |
|------|------|
| `docs/FULL_EXPERIMENT_STATUS.md` | **新建**：全量实验状态、形状、步数、耗时说明 |
| `executioin_history/20260614_1618_full_experiment_status.md` | 本执行记录 |
| `README.md` | 更新：指向新说明文档，修正 fea6 状态为已完成 |

## 未执行操作

- 未修改训练脚本或模型结构
- 未启动 `fea6+wfsampling` 训练
- 未重跑失败的 WFSampling 源文件
