# V15 frozen benchmark

## Method

`V15-support-rescue` makes two scoped changes to V14:

1. It removes the global `threshold_mask(v11, 0.50)` output gate and uses V11's broad `0.10` surface range. This retains weak surface components that would otherwise be suppressed by a stronger component's global peak.
2. It scans all 15 deep points with a whitened matched filter whose score is active-window projection power minus baseline projection power. A deep point is retained only when both EEG and MEG show at least `0.02` excess residual reduction.

Surface and deep sources share the EEG/MEG data fit but never share graph edges. Surface regularization uses only `VertConn[:n_surf, :n_surf]`; deep points are handled independently.

The attempted full-cortex residual-peak expansion was rejected on the development subset: its top four residual patches were within 10 mm of only 8.3% of true surface groups. V15 therefore does not claim to rescue a surface component absent from both SISSES and V11's broad range.

## Protocol

- Frozen development set: 185 cases.
- Unseen frozen test set: 1,110 cases: 408 surface-only, 90 deep-only, 408 deep plus surface, and 204 deep plus two surface.
- SNR: 0, 10, and 20 dB.
- Common source grid: 7,498 cortical points plus 15 thalamic points.
- AUC: user `An_cal_AUC(..., threshold=0.01)`.
- AUC graph: block diagonal. Surface-to-deep and deep-to-surface edge counts are both zero.
- RMSE, SD, and DLE: the user's Python functions; source support for SD/DLE uses the frozen 10% full-head energy rule.
- Missing layer-localization values are also reported with the frozen 234.029 mm bounding-box penalty.

V14 was rerun in the same process from the same V11 solve, so every V14/V15 row is paired. V14's RMSE and spatial metrics reproduce the previous frozen report. Its AUC changes from the stale report because the invalid AUC-only surface/deep cross edges were removed; all comparisons below use the corrected block graph for both methods.

## Test results

| Metric | V14 | V15 | Direction |
|---|---:|---:|---:|
| AUC | 0.6543 | **0.8212** | +0.1669 |
| RMSE | 0.9931 | **0.8244** | -0.1688 |
| Surface SD, conditional | **10.65 mm** | 12.70 mm | +2.04 mm |
| Surface SD, miss-penalized | 58.67 mm | **17.72 mm** | -40.95 mm |
| Surface DLE, conditional | 22.00 mm | **19.60 mm** | -2.40 mm |
| Surface DLE, miss-penalized | 66.96 mm | **24.37 mm** | -42.59 mm |
| Surface valid cases | 819/1020 | **1002/1020** | +183 |
| Deep SD/DLE, conditional | 3.44 mm | **2.41 mm** | -1.03 mm |
| Deep SD/DLE, miss-penalized | 78.09 mm | **42.44 mm** | -35.65 mm |
| Deep valid cases | 405/702 | **539/702** | +134 |
| Deep sensitivity, scenario/SNR macro | 0.6707 | **0.8023** | +0.1316 |
| Deep specificity, scenario/SNR macro | **0.9853** | 0.9583 | -0.0270 |
| Deep balanced accuracy, scenario/SNR macro | 0.8280 | **0.8803** | +0.0523 |
| Deep F1, micro | 0.7323 | **0.8441** | +0.1117 |

## Position among baselines

V15 is not the best method on every endpoint.

- Best RMSE: V15 `0.824`; the next values are V14 `0.993`, RAP-MUSIC `1.133`, and SISSES `1.147`.
- Best deep F1: V15 `0.844`, narrowly above LCMV `0.841`.
- Deep macro balanced accuracy: V15 `0.880`, narrowly below LCMV `0.883`.
- AUC: V15 `0.821` remains below SISSES `0.970`, SISSY `0.968`, FAST-IRES `0.947`, and several distributed baselines.
- Surface miss-penalized SD: V15 `17.72 mm` is behind RAP-MUSIC `4.32 mm` and grid dipole fitting `4.89 mm`.
- Deep miss-penalized DLE: V15 `42.44 mm` remains far behind LCMV `3.71 mm`, FAST-IRES `15.80 mm`, SISSES `28.25 mm`, and SISSY `29.11 mm`.

The remaining failure mode is low-SNR mixed localization. At 0 dB, V15's miss-penalized deep DLE is `165.77 mm` for deep plus surface and `164.35 mm` for deep plus two surface. At 20 dB these values fall to `8.82 mm` and `8.30 mm`. V15 is therefore a strong waveform-reconstruction and balanced deep-detection method, not a universal spatial-localization winner.

## Artifacts

- [Development summary](results/v15_frozen/dev_summary.csv)
- [Development scenario/SNR table](results/v15_frozen/dev_by_cell.csv)
- [Development per-case scores](results/v15_frozen/dev_scores.csv)
- [Test summary](results/v15_frozen/test_summary.csv)
- [Test scenario/SNR table](results/v15_frozen/test_by_cell.csv)
- [Test per-case paired scores](results/v15_frozen/test_scores.csv)
- [Test scoring metadata](results/v15_frozen/test_metadata.json)
