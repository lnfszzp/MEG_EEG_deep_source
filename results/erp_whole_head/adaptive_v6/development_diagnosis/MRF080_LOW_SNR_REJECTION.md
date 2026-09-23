# Development rejection: MRF 0.8 at EEG/MEG -10/-10 dB

Scope: five previously seen development cases. Calibration and validation were
not read. This is the same trial-covariance ADMM fit used in the 5/5 dB
checkpoint, with only `mrf_strength=0.8` fixed in advance.

The candidate is rejected as a universal low-SNR inverse. One of ten H0/H1
fits did not converge. The ungated H1 `An_cal_AUC` values were 0.4942, 0.7393,
1.0000, 0.6012, and 0.7329. The two pure-surface H0 values were 0.9417 and
0.7167, so even correct deep rejection would not meet the 0.90 range-AUC gate
for the two-surface case. Mixed-case deep DLE values were 14.14 mm, and deep
internal AUC fell to 0.4333 and 0.8000.

Two truth-free information-reuse checks were also rejected:

- Bidirectional 20/20 cross-fit presence scoring: all 20 fits converged, but
  the two negative score sums overlapped the three positive score sums
  (`max(null)=-0.1992`, `min(deep)=-1.7027`).
- Post-decision combined-40 localization with the same spatial prior: pure
  surface DLE improved to 12.64 and 8.43 mm, but the two-surface H0
  `An_cal_AUC` remained 0.7150.

Thus the 5/5 dB AUC recovery is real but cannot be extrapolated to -10/-10 dB.
No threshold fitted to these labels and no frozen calibration/validation case
was used. The next justified check is an adaptive ERP temporal basis during
post-decision localization, not another spatial-parameter grid.
