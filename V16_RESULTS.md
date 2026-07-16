# V16 frozen benchmark

## What a deep miss means

The frozen benchmark separates three failure modes for a case that contains a true thalamic source:

1. **Rescue rejected:** the residual deep candidate does not pass the EEG/MEG evidence gate.
2. **No metric support:** a candidate is accepted, but no deep point remains above the common 10% full-head energy support used by SD/DLE.
3. **Detection miss:** the final deep score is below 0.10 or the strongest deep peak is more than 10 mm from the true deep point.

These counts overlap. V15 had 147 rejected rescues, 163 cases without deep metric support, and 177 detection false negatives among 702 true-deep test cases.

## Method

`V16-evidence-rescue` keeps the V15 candidate space and makes three changes:

1. The original strong gate remains: both EEG and MEG excess residual reduction must be at least 0.02. A development-calibrated joint path also accepts a candidate when EEG + MEG excess is at least 0.06 and neither modality is negative.
2. Surface components and an accepted deep point are jointly amplitude-refit with non-negative least squares on the response window. Their spatial support and waveforms are not replaced.
3. Each surface component is scored by its leave-one-component-out residual drop in the response window minus the same drop in baseline. Component peak energy is reweighted by the squared relative evidence score.

V16 does not add source vertices, connect cortical and deep graphs, use a truth-dependent source count, or tune on the frozen test set. Surface graph operations use only `VertConn[:n_surf, :n_surf]`; all 15 thalamic points remain graph-isolated from cortex.

## Protocol

- Development: 185 cases, used once to choose joint excess `0.06`, modality floor `0.0`, and evidence power `2.0`.
- Frozen test: 1,110 unseen cases at 0, 10, and 20 dB.
- Source grid: 7,498 cortical points plus 15 thalamic points.
- AUC: user `An_cal_AUC(..., threshold=0.01)` with a block-diagonal surface/deep graph.
- RMSE, SD, and DLE: user Python functions with the common 10% full-head energy support.
- Missing layer localization: also reported with the frozen 234.029 mm penalty.

The test output contains exactly 3,330 unique `(case_id, method)` rows for V14, V15, and V16, with zero errors. V14 and V15 reproduce the prior frozen results exactly.

## Frozen test results

| Metric | V15 | V16 | Change |
|---|---:|---:|---:|
| AUC | 0.8212 | **0.8258** | +0.0046 |
| RMSE | 0.8244 | **0.7701** | -0.0543 |
| Surface SD, conditional | 12.70 mm | **4.45 mm** | -8.25 mm |
| Surface SD, miss-penalized | 17.72 mm | **11.25 mm** | -6.47 mm |
| Surface DLE, conditional | 19.60 mm | **13.33 mm** | -6.27 mm |
| Surface DLE, miss-penalized | 24.37 mm | **19.80 mm** | -4.57 mm |
| Surface valid cases | **1002/1020** | 997/1020 | -5 |
| Deep SD/DLE, conditional | **2.41 mm** | 2.46 mm | +0.05 mm |
| Deep SD/DLE, miss-penalized | 42.44 mm | **40.09 mm** | -2.35 mm |
| Deep valid cases | 539/702 | **552/702** | +13 |
| Deep sensitivity, micro | 0.7479 | **0.7692** | +0.0214 |
| Deep specificity, micro | **0.9583** | 0.9387 | -0.0196 |
| Deep F1, micro | 0.8441 | **0.8524** | +0.0084 |
| Deep BA, scenario/SNR macro | **0.8803** | 0.8762 | -0.0041 |

V16 reduces deep rescue rejections from 147 to 121, deep cases without metric support from 163 to 150, and detection false negatives from 177 to 162. The tradeoff is that deep false positives in the 408 surface-only cases increase from 17 to 25.

## Remaining failures

| Scenario / SNR | V16 valid deep | V16 conditional deep DLE | V16 penalized deep DLE |
|---|---:|---:|---:|
| Deep only / 0, 10, 20 dB | 90/90 | 0.00 mm | 0.00 mm |
| Deep + surface / 0 dB | 55/136 | 7.26 mm | 142.32 mm |
| Deep + surface / 10 dB | 123/136 | 0.88 mm | 23.17 mm |
| Deep + surface / 20 dB | 133/136 | 0.30 mm | 5.46 mm |
| Deep + two surface / 0 dB | 23/68 | 9.07 mm | 157.94 mm |
| Deep + two surface / 10 dB | 64/68 | 3.12 mm | 16.70 mm |
| Deep + two surface / 20 dB | 64/68 | 1.51 mm | 15.19 mm |

The dominant unresolved problem is therefore not isolated deep localization. It is deep-source separation from stronger cortical sources at 0 dB. Surface localization also remains weak for two-source cases: V16 conditional surface DLE is 30.60 mm at 0 dB, 23.09 mm at 10 dB, and 13.33 mm at 20 dB for `deep_plus_two_surface`.

## Position among baselines

After adding V16 to the 12 frozen benchmark rows, its descriptive ranks are:

- RMSE: **1/13**.
- Deep F1: **1/13**.
- Deep scenario/SNR macro balanced accuracy: 2/13, behind LCMV.
- Surface miss-penalized SD: 3/13, behind RAP-MUSIC and grid dipole fitting.
- Surface miss-penalized DLE: 6/13.
- Deep miss-penalized DLE: 5/13; LCMV remains much better.
- AUC: 8/13; SISSES remains much better.

V16 is a substantial surface compactness and waveform-reconstruction improvement over V15, but it does not solve low-SNR mixed deep localization and is not a universal winner.

## Artifacts

- `results/v16_frozen/dev_scores.csv`
- `results/v16_frozen/dev_summary.csv`
- `results/v16_frozen/dev_by_cell.csv`
- `results/v16_frozen/dev_metadata.json`
- `results/v16_frozen/test_scores.csv`
- `results/v16_frozen/test_summary.csv`
- `results/v16_frozen/test_by_cell.csv`
- `results/v16_frozen/test_metadata.json`
