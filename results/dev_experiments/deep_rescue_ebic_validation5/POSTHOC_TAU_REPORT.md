# Deep-rescue conservative-τ post-hoc check

This check only reclassified already-computed development rows. It did not rerun reconstruction and did not read strict-blind inputs or results. For each case, a value `τ > 4` accepts the stored rescue exactly when

$$\Delta_{\tau=4}+(\tau-4)<0.$$

Otherwise, the paired baseline row from the identical observation is used.

## Cohorts

- Full development expansion: 980 cases = 49 SNR pairs × 4 scenarios × 5 configurations.
- Unseen subset: 588 cases = 49 SNR pairs × 4 scenarios × 3 configurations. The first and last configuration of each scenario used by the two-case screen were excluded.

## Full 980 cases

| Variant | corrected AUC | worst corrected | raw AUC | worst raw | RMSE | surface pSD / pDLE (mm) | deep pSD / pDLE (mm) | sensitivity | specificity | BA | accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.974974 | 0.956545 | 0.934927 | 0.883494 | 0.783931 | 3.105882 / 7.898508 | 3.587094 / 3.038223 | 0.823129 | 0.995918 | 0.909524 | 0.000000 |
| τ=6 | 0.976256 | 0.961673 | 0.944562 | 0.913597 | 0.766693 | 3.099956 / 7.898508 | 2.531600 / 1.926534 | 0.889796 | 0.975510 | 0.932653 | 0.113265 |
| τ=8 | 0.976243 | 0.961673 | 0.944486 | 0.913597 | 0.767177 | 3.099956 / 7.898508 | 2.542970 / 1.945775 | 0.888435 | 0.975510 | 0.931973 | 0.109184 |
| τ=10 | 0.976203 | 0.961673 | 0.944267 | 0.911847 | 0.767563 | 3.099956 / 7.898508 | 2.533464 / 1.957046 | 0.887075 | 0.979592 | 0.933333 | 0.101020 |
| τ=12 | 0.976185 | 0.961673 | 0.944011 | 0.911847 | 0.767923 | 3.099940 / 7.898508 | 2.540575 / 1.970652 | 0.887075 | 0.979592 | 0.933333 | 0.100000 |
| τ=16 | 0.976131 | 0.961673 | 0.943712 | 0.911847 | 0.768450 | 3.101519 / 7.898508 | 2.552660 / 1.989893 | 0.885714 | 0.983673 | 0.934694 | 0.091837 |

## Unseen 588 cases

| Variant | corrected AUC | worst corrected | raw AUC | worst raw | RMSE | surface pSD / pDLE (mm) | deep pSD / pDLE (mm) | sensitivity | specificity | BA | accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.966629 | 0.939617 | 0.916844 | 0.823481 | 0.899606 | 3.039659 / 7.163042 | 4.797238 / 4.021207 | 0.743764 | 0.993197 | 0.868481 | 0.000000 |
| τ=6 | 0.968176 | 0.948163 | 0.927038 | 0.864899 | 0.883288 | 3.037707 / 7.163042 | 3.418357 / 2.707621 | 0.839002 | 0.965986 | 0.902494 | 0.146259 |
| τ=8 | 0.968177 | 0.948163 | 0.927030 | 0.864899 | 0.883229 | 3.037707 / 7.163042 | 3.401786 / 2.707621 | 0.839002 | 0.965986 | 0.902494 | 0.141156 |
| τ=10 | 0.968110 | 0.948163 | 0.926665 | 0.861982 | 0.883963 | 3.037707 / 7.163042 | 3.433565 / 2.726406 | 0.836735 | 0.972789 | 0.904762 | 0.130952 |
| τ=12 | 0.968110 | 0.948163 | 0.926665 | 0.861982 | 0.883963 | 3.037707 / 7.163042 | 3.433565 / 2.726406 | 0.836735 | 0.972789 | 0.904762 | 0.130952 |
| τ=16 | 0.968060 | 0.948163 | 0.926593 | 0.861982 | 0.884003 | 3.037835 / 7.163042 | 3.438189 / 2.735798 | 0.834467 | 0.979592 | 0.907029 | 0.119048 |

All 49 unseen SNR pairs remained above 0.90 corrected AUC. The worst corrected pair was EEG 20 dB / MEG -10 dB; the worst raw pair after rescue was EEG 10 dB / MEG -10 dB.

## Decision

Reject formal integration. Every conservative value improved or preserved the mean AUC, RMSE, penalized spatial metrics, and BA, but none met the predeclared unseen-specificity loss limit of one percentage point. Even τ=16 reduced specificity from 0.993197 to 0.979592 (1.36 percentage points). `candidates/oaster_rebuilt.py` remains unchanged.
