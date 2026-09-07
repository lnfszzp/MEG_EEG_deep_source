# OASTER deep-rescue EBIC development experiment

This run used simulated development cases only; no strict-blind input or result was read.

## Rule

From the global primary reduced residual $R$, recover the observable fitted-design column space $D$ by SVD. For each deep gain $g_d$, form $h_d=(I-P_D)g_d$ and $\Delta_d=\|h_d^T R\|_2^2/\|h_d\|_2^2$. The best candidate is accepted when

$$N\log(\mathrm{RSS}_{new}/\mathrm{RSS}_{old})+r\log N+2\log p_d+\tau<0,$$

where $r$ is the observed temporal rank, $p_d$ is the observed deep-candidate count, and $\tau\in\{0,4,8\}$. At most one deep source is added. Its residual least-squares reduced time course is expanded through the observed temporal basis, then the unchanged 5% spectral evidence is fused.

The global selector does not expose its design columns, so SVD of its fitted reduced sensor signal is an explicit observational approximation to that design. No truth, SNR label, or fixed true source count enters reconstruction.

## Sample and runtime

- Source cases: 196 = 49 EEG×MEG SNR pairs × 4 scenarios × 1 fixed development configurations.
- Metric rows: 784 (4 methods on identical observations).
- Wall time: 472.1 s.

## Key results

| method | pair_macro_auc_tie_corrected | worst_pair_macro_auc_tie_corrected | pairs_auc_tie_corrected_ge_0_90 | pair_macro_auc | rmse | surface_sd_mm_penalized | surface_dle_mm_penalized | deep_sd_mm_penalized | deep_dle_mm_penalized | deep_sensitivity | deep_specificity | deep_balanced_accuracy | rescue_acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OASTER_baseline | 0.982573 | 0.959642 | 49 | 0.902427 | 0.602872 | 5.483003 | 6.332606 | 9.196566 | 8.643168 | 0.571429 | 1.000000 | 0.785714 | 0.000000 |
| OASTER_deep_rescue_tau0 | 0.987998 | 0.967607 | 49 | 0.948161 | 0.559674 | 5.446767 | 6.332606 | 3.836419 | 3.575618 | 0.829932 | 0.979592 | 0.904762 | 0.209184 |
| OASTER_deep_rescue_tau4 | 0.987647 | 0.967607 | 49 | 0.945541 | 0.561442 | 5.453815 | 6.332606 | 4.203253 | 3.952307 | 0.816327 | 0.979592 | 0.897959 | 0.198980 |
| OASTER_deep_rescue_tau8 | 0.987393 | 0.967607 | 49 | 0.943902 | 0.563409 | 5.454414 | 6.332606 | 4.489554 | 4.263633 | 0.802721 | 0.979592 | 0.891156 | 0.183673 |

AUC pair columns are scenario-macro means over 49 pairs; spatial penalties replace a missing expected layer localization by the head bounding-box diagonal. Raw requested AUC and corrected tied-rank An_auc are both retained.
