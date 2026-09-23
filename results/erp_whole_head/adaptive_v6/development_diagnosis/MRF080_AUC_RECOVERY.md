# Development checkpoint: restore SISSES spatial propagation to 0.8

Scope: the five previously seen 5/5 dB development cases only. Frozen
calibration and validation cases were not read. H0 and H1 were fitted from the
training half; confirmation data were not used by the inverse solver.

The recovered SISSES source uses `tau=0.8` (`main_sisses.m`), whereas the
current v6 development default had drifted to `mrf_strength=0.5`. Re-running
the unchanged trial-covariance ADMM solver with only this value restored gave:

| Case | Scenario | H1 `An_cal_AUC` | H1 `An_auc` | Deep AUC | Deep DLE (mm) | H0/H1 converged |
|---|---|---:|---:|---:|---:|---|
| 00 | surface only | 0.9963 | 0.9972 | — | — | yes/yes |
| 01 | two surface | 0.9919 | 0.9969 | — | — | yes/yes |
| 02 | deep only | 1.0000 | 1.0000 | 1.0000 | 0.0 | yes/yes |
| 03 | deep + surface | 0.9925 | 0.9959 | 0.8667 | 10.0 | yes/yes |
| 04 | deep + two surface | 0.9984 | 0.9973 | 1.0000 | 0.0 | yes/yes |

Mean H1 `An_cal_AUC` was 0.9958 (minimum 0.9919), compared with 0.7486 for
the 0.5 development run. Mean H1 `An_auc` was 0.9975. All ten inverse fits
converged. The two pure-surface H0 confirmation losses decreased by 6.0% and
11.2% relative to the 0.5 run, so the range gain is not a post-hoc display
diffusion artifact.

This checkpoint establishes spatial localization, not deep-presence
acceptance. Independent partial-regression scores at 0.8 were `-0.4309` and
`-0.2993` for the two pure-surface cases, but `1.6103`, `-0.3695`, and `3.2558`
for the three deep-positive cases. The negative and positive ranges overlap,
so 0.8 must not replace the 0.5 detector. The minimal candidate is a two-stage
screen/refit: retain the independently confirmed 0.5 fit for deep-presence
screening and use 0.8 only for the final spatial estimate. The -10/-10 dB
slice must still pass before freezing this candidate or consuming
calibration/validation.

The pointwise surface-null alternative was rejected: blind selection improved
confirmation loss, but the two H1 `An_cal_AUC` values were only 0.9079 and
0.5893. Continuous graph Tikhonov was also rejected because confirmation loss
worsened by about 17%.
