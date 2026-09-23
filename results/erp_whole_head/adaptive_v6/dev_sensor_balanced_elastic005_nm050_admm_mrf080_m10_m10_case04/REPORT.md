# case04 固定低 lambda × 弱 ridge 交互开发诊断

唯一设置为 `noise_multiplier=0.5`、`ridge_fraction=0.05`、`mrf_strength=0.8`。这是预先固定的单次交互诊断，不是参数网格；未运行 comparators，未读取 calibration/validation。

| 设置 | 假设 | 收敛 H0/H1 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) | confirmation | partial |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | H0 | 1/1 | 0.7230 | 0.9541 | 0.9869/0.5000 | 7.2814/NA | NA/NA | -0.0027 | 0.8067 |
| baseline | H1 | 1/1 | 0.7329 | 0.7231 | 0.7136/1.0000 | 7.2814/21.4847 | NA/0.0000 | -0.0027 | 0.8067 |
| interaction | H0 | 1/1 | 0.5056 | 0.7677 | 0.7937/0.5000 | 25.2741/NA | 30.0354/NA | 0.0204 | 0.1051 |
| interaction | H1 | 1/1 | 0.6712 | 0.8030 | 0.7979/0.9333 | 13.2784/22.5172 | NA/20.0000 | 0.0204 | 0.1051 |

## 配对结论

- H1 An_cal AUC：`0.732913 → 0.671209`，明显下降。
- H1 An_auc：`0.723120 → 0.803046`，但不足以抵消 parcel AUC 与定位误差恶化。
- confirmation：`-0.002717 → 0.020372`；partial：`0.806714 → 0.105135`。
- 表层 AUC `0.7136 → 0.7979`，深层 AUC `1.0000 → 0.9333`。
- 表层 SD `7.2814 → 13.2784 mm`；深层 SD `21.4847 → 22.5172 mm`。
- 深层 DLE `0 → 20.0000 mm`，真实深源峰已偏离，故拒绝该交互设置。

这是单例开发诊断，confirmation/partial 未校准，不能作为盲检阈值或验证结论。按约定本诊断后停止。
