# 2026-06-12 15:40 后续执行记录

## 用户要求

- 执行 `docs/EXECUTION_PLAN.md` 新任务。  
- 确认 smoke 训练状态，并准备开启全量训练。  

## 1) smoke 结果核查

已检查三组 smoke 日志并确认都正常跑完：

- `fea6+decon_npevst`：`[done] Test accuracy: 0.2500  Macro AUC: nan`
- `fea6+decon_waveform`：`[done] Test accuracy: 0.2778  Macro AUC: 0.4433`
- `fea6+waveform`：`[done] Test accuracy: 0.4167  Macro AUC: 0.6215`

说明：`decon_npevst` smoke 的 `Macro AUC=nan` 来自极小样本下某类在测试中缺少正/负样本，不代表全量训练会失败。

## 2) 全量训练准备与启动

为避免继承 smoke 的小样本 checkpoint，先清理并重建输出目录：

- `outputs/fea6+decon_npevst`
- `outputs/fea6+decon_waveform`
- `outputs/fea6+waveform`

随后复用 `outputs/fea6` 的正式数据划分（manifest）启动全量训练：

- `logs/train_cnn_ds_decon_npevst_full.log`，PID 文件：`logs/train_cnn_ds_decon_npevst_full.pid`，GPU 2
- `logs/train_cnn_ds_decon_waveform_full.log`，PID 文件：`logs/train_cnn_ds_decon_waveform_full.pid`，GPU 3
- `logs/train_cnn_ds_waveform_full.log`，PID 文件：`logs/train_cnn_ds_waveform_full.pid`，GPU 0

当前三组均已进入 `Epoch 1/150`。

## 3) 执行计划“新绘图任务”代码准备

已升级 `tools/plot_pid_scores_energy.py` 以匹配新增要求与 `docs/SCORE_AND_BOOTSTRAP.md`：

- score 图改为论文风格：
  - `histtype="step"`、log y 轴、`Arbitrary Scale`、固定颜色（`#FF4C4C/#4CA64C/#4C4CFF`）
  - 输出 PNG + PDF
- 增加 `4x2` 总览图（当前脚本按模型数自动生成 `N x 2`）
- visE 分箱 AUC 图改为“点 + 误差棒，不连线”
- 误差条改为分层 bootstrap 标准差（默认 `--bootstrap 1000`，可配置）
- 增加水平误差棒（能量 bin 半宽）
- 保存样式与 bootstrap 元数据：`plot_style_and_bootstrap.json`

并完成一次脚本验证运行（仅 `outputs/fea6`）。

## 4) 当前未启动项

- `fea6+wfsampling` 全量训练尚未启动。  
  原因：新版 `WFSampling.py` 的 `.npy` 全量重建任务仍在后台进行中；为保证该实验使用一致的新版输入，待覆盖度满足后启动。  
