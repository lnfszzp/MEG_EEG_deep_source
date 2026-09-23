# Early stop: MRF 0.95, EEG/MEG -10/-10 dB

Development only; calibration and validation were not read.

The first pure-surface case completed with a non-converged H1 fit. Its H1
`An_cal_AUC` was 0.492222 and `An_auc` was 0.949069, with a legacy amplitude
false positive. This already failed the mandatory convergence gate, so the
remaining four cases were not run.

The earlier post-fit diffusion diagnostic is therefore not a valid substitute
for a refit: the actual MRF 0.95 objective changed the selected support and did
not reproduce the diagnostic AUC.
