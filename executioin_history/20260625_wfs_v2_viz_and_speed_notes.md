# 2026-06-25 WFS v2 对比可视化

## 新增

- `tools/visualize_wfs_v2_compare.py`：raw vs decon WFS v2 对比图
- 输出目录 `audit/viz/wfs_v2_compare/`

## 抽样事例

| 类 | file | event | PMT | wav 点数 | decon 点数 |
|----|------|-------|-----|----------|------------|
| numu | 2000 | 0 | 16747 | 42 | 21 |
| nue | 2574 | 11 | 14796 | 30 | 5 |
| nc | 569 | 3 | 14436 | 15 | 17 |

每事例两张图：`density_*`（PMT 关键点数分布）、`compare_*`（2×3 波形+WFS 叠加）。统计见 `summary.json`。
