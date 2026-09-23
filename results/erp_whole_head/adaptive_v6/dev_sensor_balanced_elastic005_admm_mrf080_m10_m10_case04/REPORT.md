# case04 固定 elastic-net 开发诊断

只比较预先固定的 `ridge_fraction=0.05` 与同一病例 ridge0 基线；未做网格，未读取 calibration/validation。

| 设置 | 假设 | 收敛 H0/H1 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) | prediction | partial |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ridge0 | H0_surface_only | 1/1 | 0.7230 | 0.9541 | 0.9869/0.5000 | 7.2814/NA | NA/NA | -0.0027 | 0.8067 |
| ridge0 | H1_surface_deep | 1/1 | 0.7329 | 0.7231 | 0.7136/1.0000 | 7.2814/21.4847 | NA/0.0000 | -0.0027 | 0.8067 |
| ridge0.05 | H0_surface_only | 1/1 | 0.7168 | 0.9533 | 0.9861/0.5000 | 7.2814/NA | NA/NA | 0.0139 | 0.8067 |
| ridge0.05 | H1_surface_deep | 1/1 | 0.7329 | 0.7224 | 0.7136/1.0000 | 7.2814/18.0600 | NA/0.0000 | 0.0139 | 0.8067 |

## 配对结论

- H1 An_cal AUC 变化：+0.000000
- H1 An_auc 变化：-0.000733
- confirmation prediction 变化：+0.016626
- partial confirmation 变化：+0.000000

这是单例开发诊断，confirmation/partial 均未校准，不能作为盲检阈值或验证结论。
