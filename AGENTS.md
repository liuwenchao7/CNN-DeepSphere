# JUNO 大气中微子 PID 项目

## 项目背景

本项目使用 DeepSphere 和 CNN+DeepSphere 对 JUNO 大气中微子事例进行粒子鉴别（PID）。参考论文为：

`Neutrino type identification for atmospheric neutrinos in a large homogeneous liquid scintillation detector`

当前先完成三分类：

- `numu`：mu-like
- `nue`：e-like
- `nc`：NC-like

## 数据位置

- `numu`: `/disk_pool1/weijsh/waveform/npy`
- `nue`: `/disk_pool1/liuxy/nu_e/npy`
- `nc`: `/disk_pool1/liuxk/Muon/J24/nc/npy`

每类数据包含：

- `det_fea/`
- `elec_fea/`
- `waveform/`
- `y/`

`elec_fea` 中使用以下六种逐 PMT 特征：

```text
x_fht_pmt
x_npe_pmt
x_nperatio4_pmt
x_peak_pmt
x_peaktime_pmt
x_slope4_pmt
```

## 执行要求
执行时保留现有模型主体。修改前检查已有代码和终端历史；训练前检查 GPU 占用，使用空闲 GPU，并按用户习惯通过 `nohup` 后台运行。

不要删除原始数据或覆盖无关修改。发现文件格式、标签含义或三种波形组合方式与文档不一致时，先依据实际数据核实，并在结果中说明。

每次执行完任务后保存一个执行文档至/disk_pool1/liuwc/cursor_workspace/pid/executioin_history下，详细记录进行了哪些更改，新增了哪些文件夹和文件，分别是什么