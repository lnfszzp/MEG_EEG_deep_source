# Frozen v5 false-deep diagnosis (development only)

This reads all 12 saved non-Gaussian shape results. It does **not** rerun an inverse, change an estimator, select a threshold, or edit v5 results. The final 20-case pilot did not save source arrays, so these quantitative time-course checks apply to the supplementary shape set, not the pilot. Truth is only used for descriptive correlations, never to construct an inverse or decision rule.

Run the procedural `deep_leakage_saved_arrays.py` in this directory. It checks the orthogonal energy decomposition and reproduces all archived v5 deep scores from the float32 arrays within `1e-7` absolute/relative tolerance.

## Findings

| Saved case (EEG/MEG dB) | True deep source | Original deep score | Active RMS score without baseline subtraction | In-DCT RMS score | Deep-peak baseline/active power | Deep-peak energy outside DCT |
|---|---|---:|---:|---:|---:|---:|
| postcentral uniform patch, 5/5 | No | 0.620 | 0.673 | 0.572 | 16.27% | 30.36% |
| postcentral uniform patch, -10/20 | No | 1.000 | 1.000 | 1.000 | 0.48% | 1.61% |
| superior frontal wide patch, 5/5 | No | 1.000 | 1.000 | 1.000 | 11.46% | 29.24% |
| lateral occipital thin path, 5/5 | No | 0.272 | 0.291 | 0.232 | 13.06% | 37.26% |
| superior parietal wide path + deep, 5/5 | Yes | 0.663 | 0.700 | 0.601 | 10.50% | 26.77% |

The score is `max(deep amplitude) / max(all amplitude)`, hence lies in `[0,1]`. **Seven of the nine pure-surface cases have score exactly 1.** No threshold at or below 1 can remove those seven false positives (the existing rule uses `>=`); a threshold above 1 rejects every source. Adjusting this score's threshold cannot solve the problem.

Baseline subtraction is not creating the false positives: it slightly reduces the deep/global score in the 5/5 examples. Their deep estimates already have substantially more energy in the fitted active window than in baseline. Selection and time-course reconstruction reuse that active window, so this difference is not an independent significance test.

## Is the pseudoinverse time-course extension the cause?

Let `Q` be the frozen v5 ten-row orthonormal DCT basis on the active window. Decompose the saved source estimate exactly as

`X_active = X_active Q.T Q + X_active (I - Q.T Q)`.

The two terms have orthogonal temporal energy. The outside-DCT term measures content reintroduced by v5's `pinv(response) @ weighted_data` extension. Deep-peak outside-DCT energy ranges from 1.61% to 48.14% in the pure-surface cases. Thus the extension is a material issue in many cases, but **not the only source of the false positives**: all nine pure-surface cases still have in-DCT RMS deep/global scores at least 0.232, exceeding the historical 0.14 criterion even before baseline adjustment.

The in-DCT RMS column is a geometric diagnostic, **not a proposed estimator**. An actual replacement must treat active and baseline comparably; simply zeroing baseline while projecting the active estimate would create misleading detection evidence.

The clearest systematic signal-leakage example is postcentral -10/20: the false deep waveform correlates 0.983 with the true cortical waveform (0.991 after the DCT projection), with only 0.48% baseline/active power and 1.61% out-of-basis energy. This false deep source carries cortical evoked signal, not merely reinstated temporal noise. In contrast, postcentral 5/5 has cortical correlation only 0.009, showing that the failure also includes fitted noise components. A single mechanism is not sufficient to explain all cases.

## Consequence for the next candidate

The current separate-layer baseline regularization makes deep penalties much cheaper: in the pure-surface 5/5 cases the scalar deep lambda is 26–28% of surface lambda before sensitivity weights. This corrects a previous suppression problem but does not account for cortical-to-deep model ambiguity. Current-amplitude ratio is not evidence that a deep model is needed.

The minimal useful check for a new candidate is a **data-only comparison of the cortical-only and cortical-plus-deep models on independent prediction evidence**, retaining localization errors and reporting both false positives and true positives on frozen cases. Pure baseline nulls alone do not represent cortical leakage under the null of “no deep source.” Reproducible cortical misattribution can survive a naive split, so source-model identifiability and penalty balance still need attention, not just a new scalar gate. No correction is claimed or implemented by this diagnosis.

Files: the script and `deep_leakage_saved_arrays.json` contain all 12 exact records, source peak indices, layer penalties, response condition numbers, convergence flags, energy decompositions, and diagnostic correlations. Original arrays remain under `results/erp_whole_head/adaptive_v5/shape_final/`.
