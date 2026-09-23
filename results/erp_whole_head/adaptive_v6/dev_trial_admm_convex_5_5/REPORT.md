# Development ablation: one convex ADMM round, EEG/MEG 5/5 dB

All five H0/H1 fits converged. Removing the non-convex reweighting increased
mean `An_cal_AUC` from the default 0.748631 to 0.831059, but deep localization
fell from 3/3 to 1/3 within 10 mm and mean deep `An_auc` fell to 0.911111.
Independent partial-confirmation scores still separated the two pure-surface
and three deep-positive development cases. This ablation is rejected because
the AUC gain does not compensate for the localization failure.
