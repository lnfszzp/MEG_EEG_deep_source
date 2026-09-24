# 四组SNR无量纲盲门控（balanced20开发集）

运行时只使用train20拟合和独立confirmation20评分；真值仅在决定后计算本表。U=(L0−L1)/max(L0−Enoise, 0.05·Enoise)。

| EEG/MEG SNR | 情况 | U | 盲判 | 后验正确 |
|---|---|---:|---|---|
| -10/-10 | surface_only | 0.02417 | H0 | 是 |
| -10/-10 | surface_only | 0.03163 | H0 | 是 |
| -10/-10 | deep_only | 0.28196 | H1 | 是 |
| -10/-10 | deep_plus_surface | 0.14305 | H1 | 是 |
| -10/-10 | deep_plus_two_surface | 0.14462 | H1 | 是 |
| -10/+20 | surface_only | 0.10101 | H0 | 是 |
| -10/+20 | surface_only | 0.00717 | H0 | 是 |
| -10/+20 | deep_only | 0.98639 | H1 | 是 |
| -10/+20 | deep_plus_surface | 0.64529 | H1 | 是 |
| -10/+20 | deep_plus_two_surface | 0.38718 | H1 | 是 |
| +20/-10 | surface_only | 0.06396 | H0 | 是 |
| +20/-10 | surface_only | 0.04176 | H0 | 是 |
| +20/-10 | deep_only | 0.86431 | H1 | 是 |
| +20/-10 | deep_plus_surface | 0.61453 | H1 | 是 |
| +20/-10 | deep_plus_two_surface | 0.57227 | H1 | 是 |
| +5/+5 | surface_only | 0.02649 | H0 | 是 |
| +5/+5 | surface_only | 0.00320 | H0 | 是 |
| +5/+5 | deep_only | 0.84154 | H1 | 是 |
| +5/+5 | deep_plus_surface | 0.43263 | H1 | 是 |
| +5/+5 | deep_plus_two_surface | 0.35966 | H1 | 是 |

20/20严格收敛；TP/FN/TN/FP=12/0/8/0；门控AUC=1.000。
纯表层最大U=0.10101；深源最小U=0.14305；开发间隔=0.04204。
0.12看过本开发集标签，只能用于后续开发闭环；正式结论必须重新冻结同分布校准/验证，当前accepted=false。
