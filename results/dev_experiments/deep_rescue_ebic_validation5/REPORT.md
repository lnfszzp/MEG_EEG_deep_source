# OASTER deep-rescue EBIC development experiment

This run used simulated development cases only; no strict-blind input or result was read.

## Rule

From the global primary reduced residual $R$, recover the observable fitted-design column space $D$ by SVD. For each of the 15 deep gains $g_d$, form $h_d=(I-P_D)g_d$ and $\Delta_d=\|h_d^T R\|_2^2/\|h_d\|_2^2$. The best candidate is accepted when

$$N\log(\mathrm{RSS}_{new}/\mathrm{RSS}_{old})+r\log N+2\log 15+\tau<0,$$

where $r$ is the observed temporal rank and $\tau\in\{0,2,4\}$. At most one deep source is added. Its residual least-squares reduced time course is expanded through the observed temporal basis, then the unchanged 5% spectral evidence is fused.

The global selector does not expose its design columns, so SVD of its fitted reduced sensor signal is an explicit observational approximation to that design. No truth, SNR label, or fixed true source count enters reconstruction.

## Sample and runtime

- Source cases: 980 = 49 EEG×MEG SNR pairs × 4 scenarios × 5 fixed development configurations.
- Metric rows: 3920 (4 methods on identical observations).
- Wall time: 559.3 s.

## Key results

| method | pair_macro_auc_tie_corrected | worst_pair_macro_auc_tie_corrected | pairs_auc_tie_corrected_ge_0_90 | pair_macro_auc | rmse | surface_sd_mm_penalized | surface_dle_mm_penalized | deep_sd_mm_penalized | deep_dle_mm_penalized | deep_sensitivity | deep_specificity | deep_balanced_accuracy | rescue_acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OASTER_baseline | 0.974974 | 0.956545 | 49 | 0.934927 | 0.783931 | 3.105882 | 7.898508 | 3.587094 | 3.038223 | 0.823129 | 0.995918 | 0.909524 | 0.000000 |
| OASTER_deep_rescue_tau0 | 0.976345 | 0.961673 | 49 | 0.945011 | 0.766324 | 3.099850 | 7.898508 | 2.461426 | 1.932080 | 0.895238 | 0.959184 | 0.927211 | 0.135714 |
| OASTER_deep_rescue_tau2 | 0.976346 | 0.961673 | 49 | 0.944988 | 0.766123 | 3.099854 | 7.898508 | 2.414885 | 1.888052 | 0.896599 | 0.959184 | 0.927891 | 0.125510 |
| OASTER_deep_rescue_tau4 | 0.976323 | 0.961673 | 49 | 0.945049 | 0.766258 | 3.099956 | 7.898508 | 2.490191 | 1.893688 | 0.892517 | 0.967347 | 0.929932 | 0.120408 |

AUC pair columns are scenario-macro means over 49 pairs; spatial penalties replace a missing expected layer localization by the head bounding-box diagonal. Raw requested AUC and corrected tied-rank An_auc are both retained.
