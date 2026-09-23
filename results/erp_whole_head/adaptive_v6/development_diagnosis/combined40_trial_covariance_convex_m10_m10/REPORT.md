# 同链路 20 与 40 试次定位诊断

两次拟合使用当前同一代码、同一 trial-covariance 白化器。case01 的 H0 与 case03 的 H1 是按开发真值选取，只用于判断定位上限，不是可部署 presence 门控。

| case | 试次数 | oracle family | 局部 AUC | An_auc | 表层 SD/DLE mm | 深层 DLE mm | 深层检出 |
|---:|---|---|---:|---:|---:|---:|---:|
| 01 | train20 | surface-only | 0.562 | 0.945 | 7.28/nan | — | 0 |
| 01 | combined40 | surface-only | 0.676 | 0.959 | 6.23/8.59 | — | 0 |
| 03 | train20 | full-ungated | 0.738 | 0.964 | 8.14/11.82 | 0.00 | 1 |
| 03 | combined40 | full-ungated | 0.738 | 0.967 | 8.14/11.82 | 0.00 | 1 |

局部 0.90 探索目标：未达到。
case01 的未门控 H1 原始深源误报：1。
没有冻结 presence 决策，因此本实验固定 accepted=false。
