# v5 多轮 MM 四病例探针结果（conjunctive）

预声明分离条件：`max(H0) < min(H1)`；margin = 0.001633。
八个拟合全部达到多轮固定点：True；探针通过：True。

| 病例 | 真值组 | conjunctive score | 事后方法 | AUC | 表层DLE mm | 深峰距离 mm |
|---:|---|---:|---|---:|---:|---:|
| 00 | H0 | -0.031734 | v6-surface-only | 0.500 | — | — |
| 02 | H1 | 0.018629 | v6-full-ungated | 1.000 | — | 0.00 |
| 04 | H1 | 0.001749 | v6-full-ungated | 0.659 | — | 30.00 |
| 10 | H0 | 0.000116 | v6-surface-only | 0.999 | 3.98 | — |

这是有标签开发探针，不是正式校准或 validation。
