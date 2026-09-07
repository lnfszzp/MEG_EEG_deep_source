# OASTER deep-rescue EBIC development experiment

This run used simulated development cases only; no strict-blind input or result was read.

## Rule

From the global primary reduced residual $R$, recover the observable fitted-design column space $D$ by SVD. For each deep gain $g_d$, form $h_d=(I-P_D)g_d$ and $\Delta_d=\|h_d^T R\|_2^2/\|h_d\|_2^2$. The best candidate is accepted when

$$N\log(\mathrm{RSS}_{new}/\mathrm{RSS}_{old})+r\log N+2\log p_d+\tau<0,$$

where $r$ is the observed temporal rank, $p_d$ is the observed deep-candidate count, and $\tau\in\{0,4,8\}$. At most one deep source is added. Its residual least-squares reduced time course is expanded through the observed temporal basis, then the unchanged 5% spectral evidence is fused.

The global selector does not expose its design columns, so SVD of its fitted reduced sensor signal is an explicit observational approximation to that design. No truth, SNR label, or fixed true source count enters reconstruction.

## Sample and runtime

- Source cases: 980 = 49 EEG×MEG SNR pairs × 4 scenarios × 5 fixed development configurations.
- Metric rows: 3920 (4 methods on identical observations).
- Wall time: 1856.9 s.

## Key results

| method | pair_macro_auc_tie_corrected | worst_pair_macro_auc_tie_corrected | pairs_auc_tie_corrected_ge_0_90 | pair_macro_auc | rmse | surface_sd_mm_penalized | surface_dle_mm_penalized | deep_sd_mm_penalized | deep_dle_mm_penalized | deep_sensitivity | deep_specificity | deep_balanced_accuracy | rescue_acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OASTER_baseline | 0.966997 | 0.954883 | 49 | 0.878135 | 0.820454 | 3.923108 | 9.382521 | 8.548211 | 8.331582 | 0.580952 | 1.000000 | 0.790476 | 0.000000 |
| OASTER_deep_rescue_tau0 | 0.970543 | 0.958094 | 49 | 0.908017 | 0.777257 | 3.897057 | 9.382521 | 5.280826 | 4.998098 | 0.760544 | 0.979592 | 0.870068 | 0.178571 |
| OASTER_deep_rescue_tau4 | 0.970326 | 0.958089 | 49 | 0.906439 | 0.778972 | 3.900427 | 9.382521 | 5.419725 | 5.160381 | 0.752381 | 0.983673 | 0.868027 | 0.164286 |
| OASTER_deep_rescue_tau8 | 0.970023 | 0.958089 | 49 | 0.904017 | 0.780692 | 3.900660 | 9.382521 | 5.702393 | 5.453597 | 0.736054 | 0.983673 | 0.859864 | 0.143878 |

AUC pair columns are scenario-macro means over 49 pairs; spatial penalties replace a missing expected layer localization by the head bounding-box diagonal. Raw requested AUC and corrected tied-rank An_auc are both retained.
