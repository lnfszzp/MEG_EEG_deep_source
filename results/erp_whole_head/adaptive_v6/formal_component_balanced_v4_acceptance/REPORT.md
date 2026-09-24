# 完整盲定位正式验收

结论：**未通过**。本报告只读取冻结后生成的结果；真值只用于最终评价，七个对比方法不参与调参或通过判定。

## 冻结绑定

- execution lock SHA256：`05c8edd1130ef22f0af8938d94cdc8c25fb133385adcdb5c440f5cb7b594df83`
- frozen calibration SHA256：`7dc2b57e97acaab71de4ea684ca9ae48ff731a15a7e6e6b67203fff044d8caf4`
- validation manifest SHA256：`68a405ace52d0c04a193c37198304682a0440a0007fed0433fe3a43e686d9aa7`
- 决策：train20拟合H0/H1，独立check20门控，决定后combined40定位。

## 硬性验收

| 条件 | 结果 |
|---|---:|
| source_acceptance_passed | FAIL |
| h0_false_positives_each_snr_equal_zero | PASS |
| localized_deep_detections_at_least_10_of_12 | PASS |
| all_selected_families_converged | PASS |
| all_local_AUC_at_least_0p90 | FAIL |
| maximum_surface_DLE_at_most_15mm | FAIL |
| maximum_surface_SD_at_most_20mm | PASS |
| maximum_deep_DLE_at_most_10mm | FAIL |

| EEG/MEG SNR | H0误报/4 | 深源定位检出/4 | 最低AUC | 最大表层DLE/SD mm | 最大深层DLE mm |
|---|---:|---:|---:|---:|---:|
| -10/-10 | 0/4 | 3/4 | 0.725 | 14.39/16.11 | 14.14 |
| -10/+20 | 0/4 | 4/4 | 0.902 | 11.90/14.51 | 0.00 |
| +20/-10 | 0/4 | 4/4 | 0.790 | 6.93/3.03 | 0.00 |

## 方法对比（不改变验收）

| 方法 | 数据 | 平均/最低AUC | 平均/最大表层DLE mm | 深源检出/12 | H0误报/12 |
|---|---|---:|---:|---:|---:|
| MNE | train20 | 0.756/0.230 | 31.98/109.65 | 4/12 | 7/12 |
| MNE | combined40 | 0.787/0.390 | 32.97/109.65 | 5/12 | 8/12 |
| dSPM | train20 | 0.795/0.010 | 21.62/55.69 | 6/12 | 10/12 |
| dSPM | combined40 | 0.868/0.627 | 20.30/36.00 | 8/12 | 11/12 |
| sLORETA | train20 | 0.761/0.282 | 39.46/98.62 | 8/12 | 11/12 |
| sLORETA | combined40 | 0.790/0.398 | 35.72/98.62 | 10/12 | 11/12 |
| eLORETA | train20 | 0.823/0.020 | 23.00/109.65 | 7/12 | 11/12 |
| eLORETA | combined40 | 0.851/0.190 | 12.93/58.90 | 9/12 | 11/12 |
| LCMV | train20 | 0.650/0.210 | 67.49/109.66 | 1/12 | 0/12 |
| LCMV | combined40 | 0.660/0.160 | 64.40/133.90 | 2/12 | 0/12 |
| Dipole fitting (grid) | train20 | 0.621/0.499 | 10.88/32.96 | 7/12 | 0/12 |
| Dipole fitting (grid) | combined40 | 0.666/0.499 | 8.80/32.96 | 9/12 | 0/12 |
| RAP-MUSIC | train20 | 0.628/0.499 | 10.05/20.37 | 8/12 | 0/12 |
| RAP-MUSIC | combined40 | 0.674/0.500 | 7.47/13.56 | 10/12 | 0/12 |
| OASTER-ERP-v6 | train20/check20 gate + combined40 localization | 0.955/0.725 | 7.35/14.39 | 11/12 | 0/12 |

对比方法的深源检出沿用0.14分数阈值和10 mm半径；OASTER使用冻结的独立确认门控，因此表中已明确分别标注，不能把两者当作同一决策器。
