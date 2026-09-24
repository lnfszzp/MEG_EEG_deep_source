# 层级浅深盲定位开发结果

一级用开发集冻结的 (3×EEG+MEG)/4 held-out 改善判深源；H1 内用双模态 LOO 改善之和定位深源，固定深源后再拟合表层。

| 病例 | 场景 | EEG/MEG SNR | gate | 盲选 | An_auc | requested AUC | 表层DLE/逐源最大 mm | 精确patch命中 | 深峰距离 mm |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 00 | surface_only | -10/-10 | 0.004102 | H0 combined40 MRF-innovation dipole | 1.000 | 1.000 | 0.92/0.92 | 1/1 | — |
| 01 | surface_only | -10/-10 | -0.024896 | H0 combined40 MRF-innovation dipole | 0.958 | 0.748 | 7.57/12.10 | 1/2 | — |
| 02 | deep_only | -10/-10 | 0.021986 | H1 deep-only combined40 refit | 1.000 | 1.000 | —/— | — | 0.00 |
| 03 | deep_plus_surface | -10/-10 | 0.015811 | H1 deep+surface combined40 refit | 0.982 | 0.768 | 7.21/7.21 | 0/1 | 0.00 |
| 04 | deep_plus_two_surface | -10/-10 | 0.014947 | H1 deep+surface combined40 refit | 0.986 | 0.782 | 9.69/9.84 | 0/2 | 0.00 |
| 05 | surface_only | -10/+20 | -0.380129 | H0 combined40 MRF-innovation dipole | 1.000 | 1.000 | 0.92/0.92 | 1/1 | — |
| 06 | surface_only | -10/+20 | -0.029088 | H0 combined40 MRF-innovation dipole | 0.999 | 0.999 | 4.73/6.40 | 2/2 | — |
| 07 | deep_only | -10/+20 | 0.333184 | H1 deep-only combined40 refit | 1.000 | 1.000 | —/— | — | 0.00 |
| 08 | deep_plus_surface | -10/+20 | 0.183024 | H1 deep+surface combined40 refit | 0.992 | 0.975 | 7.39/7.39 | 1/1 | 0.00 |
| 09 | deep_plus_two_surface | -10/+20 | 0.117158 | H1 deep+surface combined40 refit | 0.983 | 0.958 | 9.69/9.84 | 1/2 | 0.00 |
| 10 | surface_only | +20/-10 | 0.004184 | H0 combined40 MRF-innovation dipole | 1.000 | 1.000 | 0.92/0.92 | 1/1 | — |
| 11 | surface_only | +20/-10 | -0.224322 | H0 combined40 MRF-innovation dipole | 0.999 | 0.999 | 6.08/6.40 | 2/2 | — |
| 12 | deep_only | +20/-10 | 0.601932 | H1 deep-only combined40 refit | 1.000 | 1.000 | —/— | — | 0.00 |
| 13 | deep_plus_surface | +20/-10 | 0.441095 | H1 deep+surface combined40 refit | 1.000 | 1.000 | 0.99/0.99 | 1/1 | 0.00 |
| 14 | deep_plus_two_surface | +20/-10 | 0.074128 | H1 deep+surface combined40 refit | 0.992 | 0.964 | 6.48/9.55 | 1/2 | 0.00 |

Development screen passed：`True`。
一级路径错误病例：`无`。
缺失深源距离病例：`无`。

纯深源表层误报病例：`无`。
补充警告——精确真值 patch 未命中病例：`['erp-v6-localization-v5-development-01-eeg-10-meg-10', 'erp-v6-localization-v5-development-03-eeg-10-meg-10', 'erp-v6-localization-v5-development-04-eeg-10-meg-10', 'erp-v6-localization-v5-development-09-eeg-10-meg+20', 'erp-v6-localization-v5-development-14-eeg+20-meg-10']`。
补充 requested AUC 最低值（不替代 An_auc 验收）：`0.7481666666666666`。

这是看过失败模式后的 development 候选，不是正式校准或 validation。
