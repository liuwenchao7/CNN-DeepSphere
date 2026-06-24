# JUNO 大气中微子三分类 PID 项目

> **GitHub:** https://github.com/liuwenchao7/CNN-DeepSphere  
> English tutorial-style documentation: see [`readme.md`](readme.md) (format参考 [CNN-PointNet](https://github.com/mrheng9/CNN-PointNet)).

本仓库基于 DeepSphere 和 CNN+DeepSphere，对 JUNO 大气中微子事例进行粒子鉴别（PID），完成 `numu / nue / nc` 三分类任务。

参考论文：*Neutrino type identification for atmospheric neutrinos in a large homogeneous liquid scintillation detector*（arXiv:2503.21353）

---

## 目录结构

```
pid/
├── README.md               ← 本文件
├── AGENTS.md               ← 项目背景、数据路径、执行入口（背景知识）
├── docs/
│   ├── EXECUTION_PLAN.md   ← 任务列表和验收要求（执行方案）
│   ├── DATA_AND_VISUALIZATION.md ← 数据格式说明、特征含义、图解读
│   ├── FULL_EXPERIMENT_STATUS.md ← 全量实验进度、输入形状、步数与耗时
│   └── PAPER_PID_NOTES.md  ← 论文方法摘要（三分类方法、特征、评价）
│
├── pid_lib/                ← 项目共享库（被所有训练脚本导入）
│   ├── __init__.py
│   ├── config.py           ← 类别映射、数据路径、特征名等常量
│   ├── data_io.py          ← 数据发现、加载、y/y_pmt 对齐、无击中处理
│   ├── splits.py           ← 按文件编号的分层 train/val/test 划分
│   ├── normalize.py        ← 逐通道 z-score 归一化（仅训练集统计）
│   ├── metrics.py          ← 混淆矩阵、ROC/AUC、F1 等评价指标
│   └── waveform_io.py      ← 三种波形格式的读取和投影函数
│
├── fea/                    ← DeepSphere 6-特征 PID 基线
│   ├── train_fea.py        ← 训练脚本（当前主要训练脚本）
│   └── CHANGES.md          ← 从原始方向重建脚本到当前 PID 脚本的改动说明
│
├── WFSampling/             ← 波形关键点提取和 CNN+DeepSphere 训练
│   ├── WFSampling.py       ← 波形关键点提取脚本（v1，适配三类 compact NPZ）
│   ├── WFSampling_v2.py    ← WFSampling v2（对应 WFSampling_v2.cc）
│   ├── WFSampling.cc       ← 原始 C++ 参考实现（只读，不修改）
│   ├── show.py             ← WFSampling 结果可视化（3 子图：raw/AC/关键点）
│   ├── train_200_fht.py    ← 原始方向重建脚本（存档，不再使用）
│   ├── train_cnn_ds_pid.py ← CNN+DeepSphere PID 脚本（重写后的新版本）
│   └── CHANGES.md          ← train_cnn_ds_pid.py 改动说明
│
├── tools/                  ← 辅助工具脚本
│   ├── audit_pid_data.py   ← 数据审计脚本（只读，输出到 audit/）
│   ├── visualize_pid_data.py ← 可视化脚本（det_fea + 3 种波形对比）
│   ├── visualize_pid_event_suite.py ← 跨模态事例对照（det_fea/fht/wfs/waveform）
│   ├── plot_pid_scores_energy.py ← PID score 分布与 visE 分箱 AUC 图
│   ├── export_test_vise.py ← 导出与测试预测对齐的 visE
│   └── preprocess_pid_fht_window.py ← 按 FHT 截取波形窗口（fht-20, fht+280）
│
├── audit/                  ← 数据审计和可视化结果
│   ├── audit_pid_data.json ← 三类数据的完整审计报告（JSON）
│   ├── audit_pid_data.txt  ← 审计报告文本摘要
│   └── viz/                ← 可视化图片
│       ├── det_fea_{cls}_f{id}_e{evt}.png     ← det_fea 时间-nPE 图（含单 PMT 子图）
│       ├── waveform_{cls}_f{id}_e{evt}_p{pmt}.png ← 三种波形对比图
│       └── show_{cls}_f{id}_e{evt}_p{pmt}.png ← WFSampling 关键点图
│
├── outputs/                ← 训练输出目录（命名规则：输入内容描述）
│   ├── fea6/               ← 6 特征 DeepSphere PID 基线（已完成，epoch 70 早停）
│   │   ├── checkpoint_pid  ← 最优 val_loss 的模型权重
│   │   ├── loss_log.txt    ← 每轮 train/val loss 和 accuracy
│   │   ├── learning_curve.png
│   │   ├── confusion_matrix.png  ← 测试集混淆矩阵
│   │   ├── roc_ovr.png           ← 三类 one-vs-rest ROC 曲线
│   │   ├── test_metrics.json     ← accuracy/AUC/F1 等指标
│   │   ├── y_true/pred/prob.npy  ← 测试集预测结果
│   │   ├── class_mapping.json    ← {numu:0, nue:1, nc:2}
│   │   ├── norm_stats.json       ← 归一化统计量（仅训练集）
│   │   └── manifest_train/val/test.json ← 数据划分（可复用）
│   ├── fea6_smoke/         ← smoke test 输出（验证流程用）
│   ├── fea6+waveform/      ← fea6 + 原始 waveform 的 CNN+DeepSphere 任务
│   ├── fea6+decon_npevst/  ← fea6 + decon_npevst 的 CNN+DeepSphere 任务
│   ├── fea6+decon_waveform/← fea6 + decon_waveform 的 CNN+DeepSphere 任务
│   └── fea6+wfs_wav/       ← fea6 + WFSampling v1 全量训练（已完成，acc 0.759）
│
└── logs/                   ← 后台任务日志（仅保留当前运行任务）
    ├── pid                   ← 所有后台任务主进程 PID（key=value，一行一个）
    ├── train_cnn_ds_decon_npevst_full.log
    ├── train_cnn_ds_wfsampling_full.log
    ├── train_cnn_ds_decon_waveform_fht_full.log
    ├── train_cnn_ds_waveform_fht_full.log
    ├── wfs_decon_waveform_{numu,nue,nc}.log
    └── preprocess_*.log      ← 已完成可删除
```

---

## 数据路径

| 类别 | 标签 | 路径 |
|------|------|------|
| numu (mu-like CC) | 0 | `/disk_pool1/weijsh/waveform/npy` |
| nue (e-like CC) | 1 | `/disk_pool1/liuxy/nu_e/npy` |
| nc (NC-like) | 2 | `/disk_pool1/liuxk/Muon/J24/nc/npy` |

WFSampling v1 输出：`/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling/{numu,nue,nc}/`

WFSampling v1 输出：`/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v1/{numu,nue,nc}/`

WFSampling v2 输出：
- raw：`/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v2/{numu,nue,nc}/`（thr_fht=0.2, global_thr=15）
- decon：`/disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2/{numu,nue,nc}/`（thr_fht=0.2, global_thr=1）

- 当前主输出格式：`wfsampling_*.npy`（对象字典，包含每个 event 的 `time/adc/offsets`）
- 兼容旧格式：`wfsampling_*.npz`（旧版压缩 CSR 结构）

---

## 快速上手

### 查看训练进度
```bash
cat /disk_pool1/liuwc/cursor_workspace/pid/outputs/fea6/loss_log.txt
tail -f /disk_pool1/liuwc/cursor_workspace/pid/logs/train_cnn_ds_decon_npevst_full.log
cat /disk_pool1/liuwc/cursor_workspace/pid/logs/pid
```

### 运行数据审计
```bash
cd /disk_pool1/liuwc/cursor_workspace/pid
python3 tools/audit_pid_data.py --output-dir audit/
```

### 可视化某一事例
```bash
python3 tools/visualize_pid_data.py --class numu --file-id 2000 --event 0 --output-dir audit/viz
python3 WFSampling/show.py --class nc --file-id 0 --event 0 --pmt 0 --output-dir audit/viz
```

### 查看全量实验进度与形状说明
```bash
cat /disk_pool1/liuwc/cursor_workspace/pid/outputs/FULL_EXPERIMENT_STATUS.md
```

### 绘制 fea6+wfs_wav 结果图（需 `cnn_deepsphere` 环境）
```bash
/disk_pool1/liuwc/anaconda3/envs/cnn_deepsphere/bin/python3 tools/plot_pid_scores_energy.py \
  --model-dir outputs/fea6 --model-label fea6 \
  --model-dir outputs/fea6+wfs_wav --model-label fea6+wfs_wav \
  --output-dir audit/plots/wfs_wav
```

### 跨模态事例可视化（numu/nue/nc 各 1 事例）
```bash
python3 tools/visualize_pid_event_suite.py --seed 42 --output-dir audit/viz/event_suite
```

### 运行 WFSampling v2 预处理
```bash
nohup /disk_pool1/liuwc/anaconda3/envs/cnn_deepsphere/bin/python3 -u WFSampling/WFSampling_v2.py \
  --class numu --workers 8 > logs/wfs_v2_numu.log 2>&1 &
```

### 生成 FHT 窗口波形（300ns）
```bash
python3 tools/preprocess_pid_fht_window.py --kind decon_waveform --classes numu nue nc \
  --start -20 --width 300 --workers 6 \
  --out-root /disk_pool1/liuwc/data/cnn+ds/pid/decon_wav_fht+-/wav
```

### 训练 6 特征 DeepSphere（smoke test）
```bash
source /disk_pool1/liuwc/anaconda3/etc/profile.d/conda.sh && conda activate deepsphere
env CUDA_VISIBLE_DEVICES=0 python3 fea/train_fea.py \
  --output-dir outputs/fea6_smoke --gpu 0 --smoke-test
```

### 后台启动全量训练（命名规则示例）
```bash
nohup env CUDA_VISIBLE_DEVICES=0 python3 fea/train_fea.py \
  --output-dir outputs/fea6 --gpu 0 \
  > logs/train_fea6.log 2>&1 &
```

---

## 关键说明

1. **标签来自目录，不来自 y 文件**：`y[:,4]` 是 PDG 粒子码（±12/±14），不是三分类标签。nc 目录中 y[:,4] 包含 ±12 和 ±14 属正常现象（NC 事例只记录了粒子类型）。

2. **无击中 PMT 处理**：`fht<=0` 或 `peaktime>=1008` 的 PMT 视为无击中，归一化后填 `fill_nan=-3.0`，不是填 0。nue 数据约 61% PMT 无击中，nc/numu 约 0%。

3. **训练 pipeline**：使用固定种子全局 shuffle + 大 buffer（8192）确保类别混合，避免顺序偏差导致全预测单一类的 Bug（详见 `fea/CHANGES.md`）。

4. **conda 环境**：训练使用 `deepsphere`（Python 3.6 + TF 2.6.2 + deepsphere）。

5. **GPU 选择**：`train_cnn_ds_pid.py` 通过 `--gpu N` 指定物理 GPU。CNN+DeepSphere 默认 `--shuffle-buffer 0`（每 epoch 打乱 event index，避免 decon_npevst 等大 buffer 预取极慢）。当前运行任务见 `logs/pid`。
