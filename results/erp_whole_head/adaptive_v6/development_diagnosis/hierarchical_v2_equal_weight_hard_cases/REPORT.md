# Hierarchical-v2 equal-weight hard-case probe

## Scope and evidential status

This is a **development-only, posthoc diagnostic**. It reruns exactly six deliberately selected retired hierarchical-v1 cases: pure-surface H0 suffixes `001`, `033`, and `054`, and pure-deep H1 suffixes `000`, `005`, and `027`. The target validation set was not read. Truth labels and SNR labels were used to select and describe these hard cases, but none of the reported score formulas uses truth.

The primary null/full fits use weight 1 for every already-whitened channel and an outer-iteration ceiling of 80; all other runner settings are unchanged. All twelve null/full solves converged, the largest observed outer count was 18, and the largest relative stationarity gap was `9.995823578479392e-4`. Therefore these differences are not explained by exhausting the 80-iteration ceiling.

Accidental attempts at H0 suffixes `000`, `032`, and interrupted `053` are excluded. `rows.csv` contains only the six requested suffixes.

## Exact score definitions

For modality `m`, let held-out whitened confirmation data be `Y_m` with shape `C_m x T`. Subtract each channel's baseline mean to obtain `Yc_m`. Let `B` be the fixed smooth active-window DCT basis with shape `K x T`, orthonormal on the active samples and zero elsewhere. Here EEG/MEG channel counts are 58/99, `T=601`, baseline count `B0=97`, active count 61, and `K=10`.

Define:

```text
R_m  = Yc_m B^T
P0_m = G_m X0 B^T
P1_m = G_m X1 B^T
d_m  = ||R_m-P0_m||_F^2 - ||R_m-P1_m||_F^2
v_mi = sum_{t in baseline} Yc_m[i,t]^2 / (B0-1)
u    = sum_{t in active} B[:,t]
N_m  = sum_i(v_mi) * [K + ||u||_2^2/B0]
g_m  = d_m/N_m
q_m  = max(||R_m||_F^2/N_m - 1, 0)
```

No retired training-derived channel weight is applied during held-out scoring. The predictions do, of course, come from the equal-weight null/full fits.

The current SNR-blind consensus diagnostic is:

```text
[min(g_EEG,g_MEG) + sqrt(max(g_EEG,0)*max(g_MEG,0))]
----------------------------------------------------------------
               [1 + max(q_EEG,q_MEG)]^0.25
```

The simpler energy-normalized fixed-weight candidate is:

```text
       0.75*g_EEG + 0.25*g_MEG
------------------------------------------------
       sqrt(1 + max(q_EEG,q_MEG))
```

For the studentized diagnostic, set `Delta_m=P1_m-P0_m`, shape `C_m x K`, and use

```text
V_m = 4 * sum_i v_mi * [||Delta_m[i,:]||_2^2
                         + (Delta_m[i,:] dot u)^2/B0]
z_m = d_m / sqrt(V_m)
z   = 0.75*z_EEG + 0.25*z_MEG
```

The second term retains the temporal covariance introduced by subtracting a common baseline mean. Cross-channel covariance is approximated as diagonal after whitening. The factor 4 is not tuned: the stochastic part of a squared-loss difference is twice the noise inner product with `Delta_m`, and variance squares that coefficient. If both `Delta_m` and `d_m` are zero, `z_m` is defined as zero. Neither the former training `channel_weights` nor any truth field enters this calculation.

## Results

| Role | Exact suffix | SNR EEG/MEG | Old weights EEG/MEG | Equal-weight gains EEG/MEG | Consensus | Energy-fixed | Studentized 3:1 | Null/full outer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 001 | -10/-10 | 1/.6587 | .01323/.01174 | .023363 | .011987 | 1.4819 | 5/4 |
| H0 | 033 | -10/+20 | .0510/1 | .001620/-.005422 | -.001665 | -.000013 | .3810 | 14/8 |
| H0 | 054 | +20/-10 | 1/.02188 | .116658/.000218 | .000912 | .002635 | 5.0379 | 10/14 |
| H1 | 000 | -10/-10 | 1/0 | .01755/.05118 | .045867 | .024187 | 2.9373 | 2/5 |
| H1 | 005 | -10/-10 | 1/0 | .02032/.02900 | .044176 | .022069 | 2.4567 | 18/7 |
| H1 | 027 | -10/+20 | .03825/1 | -.002881/.823371 | -.001013 | .025178 | 3.5871 | 12/2 |

Equal weights rescue both H1 cases whose retired fit had deleted MEG completely (`000` and `005`). This supports a narrowly scoped fix: the hierarchical-v2 primary fit should not reuse active-excess weights after whitening. It does **not** solve the whole gate problem: suffix `054` retains a large high-SNR EEG-only raw gain (`0.11666`).

Strict cross-modal consensus suppresses that H0 leak, but it also rejects H1 suffix `027` because the weak EEG gain is slightly negative. In these six hard cases its H0 maximum is `0.0233630`, H1 minimum is `-0.0010131`, and only two of three H1 scores are positive.

The energy-normalized 3:1 candidate is the only tested statistic that separates all six here: H0 maximum `0.01198697`, H1 minimum `0.02206950`, a ratio of about `1.84`. This is a promising mechanism, not an acceptance result. The square-root response normalization was inspected on the same retired labeled pool and these selected cases, so it now needs a fresh development calibration before the untouched validation is opened.

Studentization alone is rejected for this gate: H0 suffix `054` scores `5.0379`, above the H1 minimum `2.4567`. Its noise scaling does not remove the structured, high-SNR single-modality surface leakage.

## Development contamination boundary

The consensus formula and its exponent were selected after examining 113 retired-v1 labeled cases. On that same non-independent pool it had `An_auc=0.9890` and detected 54/56 H1 cases at the second-largest H0 score. The energy-normalized fixed score was also inspected on that pool; it had `An_auc=0.99248` and detected 55/56 H1 cases at the second-largest H0 score. Those numbers are useful for choosing what to test next, but cannot support a paper claim, a frozen threshold, or validation acceptance.

The minimal next experiment is therefore: freeze equal primary-fit weights plus the energy-normalized fixed score, generate a fresh truth-balanced development/calibration cohort with the same SNR cells, calibrate the upper-tail H0 decision without looking at validation, and require both no-deep specificity and pure-deep sensitivity before any validation run.
