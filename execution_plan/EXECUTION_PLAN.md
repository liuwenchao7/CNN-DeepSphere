# JUNO 三分类 PID 执行方案

## 目标

完成 `numu / nue / nc` 三分类 PID，并比较：

1. 六种 `elec_fea` 特征的 DeepSphere 模型。
2. 波形与六种特征联合输入的 CNN+DeepSphere 模型。

标签：`numu=0`、`nue=1`、`nc=2`。

## 1.运行 `train_cnn_ds_pid.py`

decon_npevst
WFSamplin(wfs) 后的 waveform(wav)
wfs后的decon_wav
decon_wav_fht+-
waveform_fht+-
```

将它们作为5组独立实验，分别与六种特征组合，便于比较。


## 2.绘制 score 分布和可见能量分箱图


