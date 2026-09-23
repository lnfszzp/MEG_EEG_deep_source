# Early stop: MRF 0.95, EEG/MEG 5/5 dB

Development only; calibration and validation were not read.

The first case had not completed after about 31 minutes. The same fixed
candidate had already failed the -10/-10 dB convergence gate, so this run was
stopped without reporting an AUC result. `metadata.json` is intentionally left
with `complete=false`.
