# `train_fea.py` 可调参数清单

以下参数会直接影响三分类效果与训练稳定性：

- `--learning-rate`：Adam 初始学习率（默认 `5e-4`）。
- `--lr-decay`：每轮结束的学习率乘法衰减系数（默认 `0.995`）。
- `--batch-size`：批大小（默认 `16`）。
- `--early-stop-patience`：EarlyStopping 容忍轮数（默认 `15`）。
- `--shuffle-buffer`：`tf.data` 混洗缓冲区（默认 `8192`）。
- `--fill-nan`：无击中 PMT 的归一化填充值（默认 `-3.0`）。

辅助参数：

- `--gpu`：使用的 GPU 编号。
- `--smoke-test` / `--smoke-epochs`：快速冒烟测试开关和轮数。
- `--output-dir`：输出目录（建议按输入方案语义命名）。

## 本轮优化尝试配置（已执行）

目标：在稳定收敛前提下提升 `val_acc` 与 macro AUC。

- `learning-rate=3e-4`
- `lr-decay=0.997`
- `batch-size=24`
- `early-stop-patience=20`
- `shuffle-buffer=16384`
- `fill-nan=-3.0`
- `seed=42`

输出目录：`outputs/fea6_tune_v1`
日志文件：`logs/train_fea6_tune_v1.log`
