# hierarchical-v2 正式 validation

最终结论：`FAIL`。

| 病例 | 场景 | SNR EEG/MEG | family | An_auc | surface AUC | surface DLE/component/SD mm | deep mm | H0 deep FP | deep-only surface FP |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | surface_only | -10/-10 | H0 combined40 MRF-innovation dipole | 1.000 | 1.000 | 2.34/2.34/0.00 | — | 0 | 0 |
| 1 | surface_only | -10/-10 | H0 combined40 MRF-innovation dipole | 1.000 | 1.000 | 2.15/2.15/0.00 | — | 0 | 0 |
| 2 | surface_only | -10/-10 | H0 combined40 MRF-innovation dipole | 0.999 | 0.999 | 7.44/10.59/6.34 | — | 0 | 0 |
| 3 | surface_only | -10/-10 | H0 combined40 MRF-innovation dipole | 0.992 | 0.992 | 7.44/10.28/5.89 | — | 0 | 0 |
| 4 | deep_only | -10/-10 | H1 deep-only combined40 refit | 1.000 | — | — | 0.00 | 0 | 0 |
| 5 | deep_plus_surface | -10/-10 | H1 deep+surface combined40 refit | 0.987 | 0.985 | 2.99/2.99/3.63 | 0.00 | 0 | 0 |
| 6 | deep_plus_two_surface | -10/-10 | H1 deep-only combined40 refit | 0.517 | 0.500 | — | 0.00 | 0 | 0 |
| 7 | surface_only | -10/+20 | H0 combined40 MRF-innovation dipole | 0.999 | 0.999 | 5.36/5.36/0.00 | — | 0 | 0 |
| 8 | surface_only | -10/+20 | H0 combined40 MRF-innovation dipole | 0.992 | 0.992 | 9.90/9.90/7.11 | — | 0 | 0 |
| 9 | surface_only | -10/+20 | H0 combined40 MRF-innovation dipole | 0.990 | 0.990 | 9.08/13.76/3.24 | — | 0 | 0 |
| 10 | surface_only | -10/+20 | H0 combined40 MRF-innovation dipole | 0.999 | 0.999 | 2.08/2.18/0.00 | — | 0 | 0 |
| 11 | deep_only | -10/+20 | H1 deep-only combined40 refit | 1.000 | — | — | 0.00 | 0 | 0 |
| 12 | deep_plus_surface | -10/+20 | H1 deep+surface combined40 refit | 1.000 | 1.000 | 2.16/2.16/0.00 | 0.00 | 0 | 0 |
| 13 | deep_plus_two_surface | -10/+20 | H0 combined40 MRF-innovation dipole | 0.970 | 0.998 | 6.11/9.56/6.17 | — | 0 | 0 |
| 14 | surface_only | +20/-10 | H0 combined40 MRF-innovation dipole | 0.941 | 0.941 | 23.28/23.28/26.51 | — | 0 | 0 |
| 15 | surface_only | +20/-10 | H1 deep+surface combined40 refit | 0.999 | 0.999 | 10.22/10.22/7.81 | — | 1 | 0 |
| 16 | surface_only | +20/-10 | H0 combined40 MRF-innovation dipole | 0.999 | 0.999 | 4.41/4.82/0.00 | — | 0 | 0 |
| 17 | surface_only | +20/-10 | H0 combined40 MRF-innovation dipole | 0.999 | 0.999 | — | — | 0 | 0 |
| 18 | deep_only | +20/-10 | H1 deep-only combined40 refit | 1.000 | — | — | 0.00 | 0 | 0 |
| 19 | deep_plus_surface | +20/-10 | H1 deep+surface combined40 refit | 1.000 | 1.000 | 3.33/3.33/0.00 | 0.00 | 0 | 0 |
| 20 | deep_plus_two_surface | +20/-10 | H1 deep+surface combined40 refit | 1.000 | 1.000 | 2.92/3.61/0.00 | 0.00 | 0 | 0 |

## 硬门

- complete_finite_metrics: `False`
- global_An_auc: `False`
- surface_An_auc: `False`
- surface_DLE: `False`
- surface_component_DLE: `False`
- surface_SD: `False`
- deep_distance: `True`
- h0_deep_false_positive_per_snr: `False`
- h1_case_count: `True`
- h1_detected_total: `True`
- h1_detected_per_snr: `True`
- deep_only_surface_false_positive: `True`
- all_selection_and_final_solvers_converged: `True`
- H0 每SNR深源误报数：`{'eeg-10_meg-10': 0, 'eeg-10_meg+20': 0, 'eeg+20_meg-10': 1}`
- H1 总检出：`8/9`
- H1 每SNR检出数：`{'eeg-10_meg-10': 3, 'eeg-10_meg+20': 2, 'eeg+20_meg-10': 3}`

## 补充指标（不替代硬门）

- family 判断不一致病例：`['erp-v6-hierarchical-v1-validation-13-eeg-10-meg+20', 'erp-v6-hierarchical-v1-validation-15-eeg+20-meg-10']`
- requested parcel AUC 最低值：`0.4622222222222222`
- 精确真值 patch 未命中病例：`['erp-v6-hierarchical-v1-validation-02-eeg-10-meg-10', 'erp-v6-hierarchical-v1-validation-03-eeg-10-meg-10', 'erp-v6-hierarchical-v1-validation-05-eeg-10-meg-10', 'erp-v6-hierarchical-v1-validation-06-eeg-10-meg-10', 'erp-v6-hierarchical-v1-validation-08-eeg-10-meg+20', 'erp-v6-hierarchical-v1-validation-09-eeg-10-meg+20', 'erp-v6-hierarchical-v1-validation-14-eeg+20-meg-10', 'erp-v6-hierarchical-v1-validation-17-eeg+20-meg-10']`
