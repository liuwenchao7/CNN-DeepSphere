# JUNO 大气中微子 PID 数据与可视化说明

> 参考：`EXECUTION_PLAN.md` Task 5。本文件包含 README 中文翻译、数据结构、特征含义、可视化图解读。

---

## 一、原始数据说明（README.txt 中文翻译）

原文路径：`/disk_pool1/wangjb/nu_mu_waveform/README.txt`

### 四个子目录含义

| 目录 | 内容 |
|------|------|
| `det_fea/` | 探测器模拟（detsim）输出的时间和电荷，文件格式为 NPZ |
| `elec_fea/` | 经反卷积处理后得到的逐 PMT 电子学特征 |
| `waveform/` | 原始 FADC 波形及反卷积波形，格式为 NPZ |
| `y/` | 真实标签，包含运动学量和粒子信息 |

### y 文件列定义

`y_*.npy`，shape `(n_evt, 17)`，列含义：

| 列索引 | 字段名 | 说明 |
|--------|--------|------|
| 0 | file_num | 文件编号 |
| 1 | evt | 事例编号 |
| 2 | theta | 极角（rad） |
| 3 | phi | 方位角（rad） |
| 4 | pid | 粒子类型（PDG 码：±12=nu_e, ±14=nu_mu） |
| 5 | energy | 总能量（MeV） |
| 6 | depE | 沉积能量（MeV） |
| 7 | visE | 可见能量（MeV） |
| 8–16 | 顶点/动量 | 初末顶点坐标和初始动量分量（mm/MeV/c） |

**重要注意**：`y[:,4]`（pid）是中微子的 PDG 粒子类型码（±12/±14），**不是**三分类的训练标签（0/1/2）。三分类标签由**数据来源目录**确定：
- `numu` 目录 → label 0（mu-like CC）
- `nue` 目录 → label 1（e-like CC）
- `nc` 目录 → label 2（NC-like）

nc 目录中的事例在 y[:,4] 中会出现 ±12 和 ±14，因为 NC 反应类型在 y 文件里只保存粒子类型，不保存反应类型。

`y_pmt_*.npy`，shape `(n_evt, 15)`，比 y_*.npy 少两列（depE, visE），其余字段相同，用于与 `elec_fea` 对齐。

---

## 二、数据根目录与文件格式

| 类别 | 标签 | 路径 | 文件编号范围 |
|------|------|------|-------------|
| numu | 0 | `/disk_pool1/weijsh/waveform/npy` | 2000–3499 |
| nue | 1 | `/disk_pool1/liuxy/nu_e/npy` | 2500–5000 |
| nc | 2 | `/disk_pool1/liuxk/Muon/J24/nc/npy` | 0–1499 |


---

## 三、六种电子学特征（`elec_fea/`）

文件模式：`x_*_pmt_{file_id}.npy`，shape `(n_evt, 17612)`，每行一个事例，每列一个 PMT。

| 特征名 | 文件前缀 | 含义 |
|--------|---------|------|
| fht | `x_fht_pmt_*` | First Hit Time：该 PMT 最早光子到达时间（ns） |
| npe | `x_npe_pmt_*` | 该 PMT 接收的光电子数 |
| nperatio4 | `x_nperatio4_pmt_*` | 前 4 ns 电荷占总电荷之比 |
| peak | `x_peak_pmt_*` | 波形最大幅度值（ADC 单位） |
| peaktime | `x_peaktime_pmt_*` | 波形最大幅度时刻（ns） |
| slope4 | `x_slope4_pmt_*` | 前 4 ns 波形斜率 |

### 无击中（no-hit PMT）含义与处理

**"无击中"是指单个 PMT 没有接收到光子，不是指整个事例。** 探测器共有 17612 个 PMT，每个事例只有部分 PMT 会实际记录到光信号。未接收到光子的 PMT 在 `elec_fea` 数组中存储的是无效的哨兵值，需要与真实物理值区分开。

| 特征 | 无击中判定条件 | 数据中的实际表现 |
|------|--------------|----------------|
| fht | `<= 0` | nue 数据中无击中 PMT 的 fht 为 -5 到 -1（负值）；nc/numu 数据中无击中 PMT 的 fht 为 0 |
| peaktime | `>= 1008` | 时间窗末尾哨兵值 |

**处理流程**：`mark_no_hit_features` 将哨兵值替换为 `NaN`，然后 `apply_norm` 归一化后用 `fill_nan=-3.0` 填充（该值远低于正常归一化特征的分布范围，使模型能清晰区分"无击中"和物理值）。

**三类数据的无击中比例差异**（来自数据审计）：
- nc / numu：约 0% 无击中 PMT（几乎所有 PMT 都有信号，对应 1-15 GeV 强子簇射）
- nue：约 61% PMT 无击中（电磁簇射较集中，激发 PMT 数量少）

这一差异是三类事例在 fht 特征上最直接的可区分信息之一。

---

## 四、波形文件格式（`waveform/`）

### 4.1 原始波形 `waveform_*.npz`

```
键名          shape                  说明
event_offsets (n_events+1,) int64    每个事例在 copyNo/waveform 维的起止索引
copyNo        (n_hit,)     uint16   命中 PMT 的编号（0–17611）
gain          (n_hit,)     uint8    增益标志
waveform      (n_hit, 1008) uint16  原始 ADC 采样值（14-bit FADC）
n_samples     scalar       int32    每 PMT 的采样点数（通常 1008）
n_events      scalar       int32    该文件中的事例数
```

读法：`event_offsets[i]` 到 `event_offsets[i+1]` 之间的 `copyNo` 和 `waveform` 行对应事例 i 的命中 PMT。

物理信号 = 16384 − ADC（14-bit 翻转）。

### 4.2 反卷积波形 `decon_waveform_*.npz`

与原始波形格式相同，但增加了 `waveform_offsets` 键（变长波形），列顺序同上。

### 4.3 反卷积 nPE 时间结构 `decon_npevst_*.npy`

格式：`np.load(..., allow_pickle=True)`，返回 object 数组，每个元素是一个事例的命中信息。

元素 shape `(n_hits, 3)`，列为 `[npe, time, pmt_id]`（float64）。

### 4.4 det_fea `time_npe_*.npz`

```
键名           shape
event_offsets  (n_events+1,) int64    事例分界
pmt_offsets    (sum_pmt+1,)  int64    各事例内 PMT 分界
pmtID          (sum_pmt,)    int32    命中 PMT 的编号
time           (sum_hits,)   float32  光子击中时间（ns）
npe            (sum_hits,)   float32  光电子数
```

### 4.5 WFSampling 结果（`/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling/{numu,nue,nc}/`）

文件格式：`wfsampling_{file_id}.npz`，CSR 压缩结构：

```
键名           说明
time_values    所有事例、所有 PMT 关键点时间，展平为 1D float32
adc_values     对应幅度值，展平为 1D float32
pmt_offsets    长度 = n_events×17612+1，pmt_offsets[e*17612+p] 为
               event e, PMT p 的关键点在 time_values 中的起始索引
event_offsets  长度 = n_events+1，event_offsets[e] = e*17612
file_id        文件编号
n_events       事例数
n_pmt          PMT 数（17612）
```

读法示例：
```python
z = np.load("wfsampling_0.npz")
n_pmt = int(z["n_pmt"])
base = int(z["event_offsets"][event_idx]) * n_pmt + pmt_id
s = int(z["pmt_offsets"][base])
e = int(z["pmt_offsets"][base + 1])
t_pts = z["time_values"][s:e]
a_pts = z["adc_values"][s:e]
```

---

## 五、审计结果（2026-06-11）

| 类 | 标签 | 有效文件 | 有效事例 |
|----|------|----------|----------|
| numu | 0 | 1392 | 14,369 |
| nue | 1 | 1793 | 19,633 |
| nc | 2 | 1485 | 19,263 |
| **合计** | | | **53,265** |

FHT 无击中比例：nue ~61%，nc/numu ~0%（nc/numu 数据几乎所有 PMT 都有信号）。

---

## 六、可视化图解读

图片路径：`/disk_pool1/liuwc/cursor_workspace/pid/audit/viz/`

### `det_fea_{cls}_f{id}_e{evt}.png`

散点图：横轴为光子到达时间（ns），纵轴为 nPE。不同颜色代表不同命中 PMT。图中 PMT 数量和时间分布反映了粒子径迹/簇射形状。

- numu：muon 径迹较长，PMT 击中时间分布较宽且连续
- nue：electron 簇射较短，时间分布较集中
- nc：无可见 CC 轻子，主要为强子信号，时间分布弥散

### `waveform_{cls}_f{id}_e{evt}_p{pmt}.png`

三列对比图（同一事例、同一 PMT）：

1. **raw waveform**：原始 ADC 波形（已翻转），展示 14-bit FADC 的实际采样；
2. **decon_waveform**：经反卷积降噪后的波形，基线更平坦，主峰更清晰；
3. **decon_npevst**：以 PE 棒状图展示，横轴时间，纵轴 nPE，是最紧凑的物理信号表示。

### `show_{cls}_f{id}_e{evt}_p{pmt}.png`（WFSampling 结果）

三列图：

1. **Raw waveform**：原始翻转 ADC；
2. **AC waveform**：扣除基线、截断负值后的信号，蓝色虚线为全局阈值（global_thr=15 ADC）；
3. **Key points**：提取的峰谷特征点（红点），标注关键点数量。

关键点提取规则：
- 先扣除基线（寻找前 100 ns 平坦段的均值）
- 超过 global_thr 触发脉冲检测
- 识别超过 h_thr 的峰高变化为峰/谷
- 记录起始谷、峰前上升沿（rise_step 间隔采样）、峰顶和下一谷
