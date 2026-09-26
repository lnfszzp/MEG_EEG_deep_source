# v2盲可靠性融合开发

只读取69号current15、retired113和hard6开发分数；没有读取validation，也没有运行逆解。

## 冻结选择规则

current overall and each of three SNR-cell An_auc >= 0.9; current H1 positive 9/9; retired113 An_auc >= 0.98; hard6 An_auc == 1 and H1_min-H0_max > 0

通过候选：adaptive_balance_p0p5, adaptive_balance_p1, adaptive_balance_p2。
选择：`adaptive_balance_p1`。
选择理由固定为：`b`本身就是可靠性平衡量，`p=1`直接使用它，不再添加额外幂。

## 候选结果

| 候选 | current | 三格最小 | H1正分 | retired113 | hard6 | hard间隔 | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| per_modality_fixed_3_to_1_e05 | 0.9259 | 0.6667 | 9/9 | 0.9909 | 1.0000 | 0.0101421 | 0 |
| per_modality_fixed_3_to_1_e0625 | 0.9444 | 0.8333 | 9/9 | 0.9903 | 1.0000 | 0.000870259 | 0 |
| per_modality_fixed_3_to_1_e075 | 0.9815 | 1.0000 | 9/9 | 0.9900 | 0.8889 | -0.00511756 | 0 |
| per_modality_fixed_3_to_1_e0875 | 1.0000 | 1.0000 | 9/9 | 0.9875 | 0.8889 | -0.00860752 | 0 |
| per_modality_fixed_3_to_1_e10 | 1.0000 | 1.0000 | 9/9 | 0.9834 | 0.7778 | -0.0106172 | 0 |
| sqrt_weight_eeg_0p5 | 0.9259 | 0.8333 | 8/9 | 0.9919 | 1.0000 | 0.0125997 | 0 |
| sqrt_weight_eeg_0p6 | 0.9074 | 0.8333 | 8/9 | 0.9915 | 1.0000 | 0.0116167 | 0 |
| sqrt_weight_eeg_0p666667 | 0.9444 | 0.8333 | 8/9 | 0.9909 | 1.0000 | 0.0109613 | 0 |
| sqrt_weight_eeg_0p75 | 0.9259 | 0.6667 | 9/9 | 0.9909 | 1.0000 | 0.0101421 | 0 |
| adaptive_balance_p0p5 | 0.9630 | 1.0000 | 9/9 | 0.9919 | 1.0000 | 0.0101902 | 1 |
| adaptive_balance_p1 | 0.9630 | 1.0000 | 9/9 | 0.9919 | 1.0000 | 0.0102373 | 1 |
| adaptive_balance_p2 | 0.9630 | 1.0000 | 9/9 | 0.9922 | 1.0000 | 0.0103288 | 1 |
| reference_global_sqrt_fixed_3_to_1 | 0.9259 | 0.6667 | 8/9 | 0.9925 | 1.0000 | 0.0100825 | 0 |
| reference_snr_blind_consensus | 0.8889 | 0.6667 | 6/9 | 0.9890 | 0.7778 | -0.0243761 | 0 |

## 限制

三组数据均已参与开发判断；这些结果只用于冻结下一轮新校准方案，不能作为独立验证或论文性能声明。
