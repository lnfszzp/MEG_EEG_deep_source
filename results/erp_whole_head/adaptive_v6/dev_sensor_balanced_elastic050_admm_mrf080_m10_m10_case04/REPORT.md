# case04 固定 elastic-net 0.5 开发诊断

只比较预先固定的 `ridge_fraction=0.5` 与同一病例 ridge0 基线；这不是参数网格。未运行 comparators，未读取 calibration/validation。

| 设置 | 假设 | 收敛 H0/H1 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) | confirmation | partial |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge0 | H0 | 1/1 | 0.7230 | 0.9541 | 0.9869/0.5000 | 7.2814/NA | NA/NA | -0.0027 | 0.8067 |
| ridge0 | H1 | 1/1 | 0.7329 | 0.7231 | 0.7136/1.0000 | 7.2814/21.4847 | NA/0.0000 | -0.0027 | 0.8067 |
| ridge0.5 | H0 | 1/1 | 0.5523 | 0.8234 | 0.8515/0.5000 | 7.2814/NA | NA/NA | 0.0048 | 0.8067 |
| ridge0.5 | H1 | 1/1 | 0.7328 | 0.7223 | 0.7136/1.0000 | 7.2814/16.2624 | NA/0.0000 | 0.0048 | 0.8067 |

## 配对结论

- H1 An_cal AUC：`0.732913 → 0.732762`（-0.000152）。
- H1 An_auc：`0.723120 → 0.722279`（-0.000842）。
- confirmation：`-0.002717 → 0.004807`；partial 保持 `0.806714`。
- 深层 SD 改善 `21.4847 → 16.2624 mm`，深层 DLE 保持 `0 mm`；表层 DLE 仍不可计算。
- H0 AUC 明显下降，H1 AUC 也未改善，因此不采用 `ridge_fraction=0.5`。

这是单例开发诊断，confirmation/partial 未校准，不能作为盲检阈值或验证结论。按预先约定不再尝试第三个 ridge。
