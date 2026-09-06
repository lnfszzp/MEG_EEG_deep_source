# OASTER deep-rescue EBIC development experiment

This run used simulated development cases only; no strict-blind input or result was read.

## Rule

From the global primary reduced residual $R$, recover the observable fitted-design column space $D$ by SVD. For each of the 15 deep gains $g_d$, form $h_d=(I-P_D)g_d$ and $\Delta_d=\|h_d^T R\|_2^2/\|h_d\|_2^2$. The best candidate is accepted when

$$N\log(\mathrm{RSS}_{new}/\mathrm{RSS}_{old})+r\log N+2\log 15+\tau<0,$$

where $r$ is the observed temporal rank and $\tau\in\{0,2,4\}$. At most one deep source is added. Its residual least-squares reduced time course is expanded through the observed temporal basis, then the unchanged 5% spectral evidence is fused.

The global selector does not expose its design columns, so SVD of its fitted reduced sensor signal is an explicit observational approximation to that design. No truth, SNR label, or fixed true source count enters reconstruction.

## Sample and runtime

- Source cases: 392 = 49 EEG×MEG SNR pairs × 4 scenarios × 2 fixed development configurations.
- Metric rows: 1568 (4 methods on identical observations).
- Wall time: 228.0 s.

## Key results

| method | pair_macro_auc_tie_corrected | worst_pair_macro_auc_tie_corrected | pairs_auc_tie_corrected_ge_0_90 | pair_macro_auc | rmse | surface_sd_mm_penalized | surface_dle_mm_penalized | deep_sd_mm_penalized | deep_dle_mm_penalized | deep_sensitivity | deep_specificity | deep_balanced_accuracy | rescue_acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OASTER_baseline | 0.987492 | 0.978502 | 49 | 0.962050 | 0.610418 | 3.205216 | 9.001707 | 1.771878 | 1.563746 | 0.942177 | 1.000000 | 0.971088 | 0.000000 |
| OASTER_deep_rescue_tau0 | 0.988408 | 0.979279 | 49 | 0.971076 | 0.592028 | 3.193065 | 9.001707 | 1.151523 | 0.803006 | 0.965986 | 0.969388 | 0.967687 | 0.084184 |
| OASTER_deep_rescue_tau2 | 0.988409 | 0.979279 | 49 | 0.971079 | 0.591696 | 3.193074 | 9.001707 | 1.094426 | 0.754904 | 0.969388 | 0.969388 | 0.969388 | 0.073980 |
| OASTER_deep_rescue_tau4 | 0.988375 | 0.979279 | 49 | 0.970849 | 0.591803 | 3.193330 | 9.001707 | 1.201464 | 0.754904 | 0.965986 | 0.989796 | 0.977891 | 0.066327 |

AUC pair columns are scenario-macro means over 49 pairs; spatial penalties replace a missing expected layer localization by the head bounding-box diagonal. Raw requested AUC and corrected tied-rank An_auc are both retained.
