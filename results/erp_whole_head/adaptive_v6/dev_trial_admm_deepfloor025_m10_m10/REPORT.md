# Development ablation: deep reweight floor 0.25, EEG/MEG -10/-10 dB

All five H0/H1 fits converged. Compared with the default low-SNR run, true
deep peaks within 10 mm improved from 0/3 to 2/3 and mean deep `An_auc`
increased from 0.722222 to 0.811111. The remaining mixed single-surface case
was still 14.142 mm from the true deep point. Mean `An_cal_AUC` remained low at
0.689624, and the independent partial-confirmation scores did not separate
pure-surface and deep-positive cases. This is a useful localization mechanism
but does not pass the complete acceptance gate.
