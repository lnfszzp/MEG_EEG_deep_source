# Strict-blind 49-pair runner audit

## Frozen contract

- Manifest: `results/strict_blind/manifest.json`
- SHA-256: `3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76`
- Design: 49 ordered EEG/MEG SNR pairs × 185 configurations = 9,065 unique cases.
- Every pair contains the same 185 configuration IDs. The manifest contains explicit source configuration, seed, EEG SNR and MEG SNR.
- Detection threshold remains fixed at `0.1`. Neither runner calibrates or selects a threshold from strict-blind results.

## Available methods

| Method | Existing adapter | Input |
|---|---|---|
| MNE, dSPM, sLORETA, eLORETA | `benchmark/methods.py::minimum_norm_family` | joint EEG+MEG |
| LCMV | `benchmark/methods.py::lcmv` | joint EEG+MEG |
| Dipole fitting (grid) | `benchmark/methods.py::dipole_fit` | joint EEG+MEG |
| RAP-MUSIC | `benchmark/methods.py::rap_music` | joint EEG+MEG |
| OASTER | `run_oaster.py::_score_case` | joint EEG+MEG |
| SISSES | `benchmark/matlab_adapters.m` | joint EEG+MEG |
| ConvDip | `benchmark/train_convdip.py` | EEG-only and cortex-only; deep output unsupported, so excluded from the joint/deep strict-blind ranking |

No algorithm was reimplemented in the matrix runners.

## Smoke result

The first strict-blind case (`strict-blind-00000-surface_only-eeg-10-meg-10`) was run only.

- Python7 + OASTER: 8/8 successful rows.
- SISSES: MATLAB output and metric row successful.
- Re-running Python/OASTER took 1.34 s and skipped the completed pair part.
- Re-running SISSES reported `skip strict_00000_00001` and skipped its completed score row.
- `EEG`, `MEG`, `Gain_EEG` and `Gain_MEG` in the SISSES MATLAB chunk were each exactly array-equal to the arrays generated from the same manifest row for Python/OASTER.
- Array byte hashes were respectively `cee3bc45…3f26`, `3a749c73…5aed`, `68222ce6…76c8`, and `eaaaa3c6…54fb`.
- All estimates were scored by `benchmark/metrics.py::evaluate_estimate` through the existing `run_frozen_v15._row`/benchmark adapter.
- Relevant tests: `2 passed` (`benchmark/test_methods.py`, `test_run_oaster.py`).

## Commands

Python7 + OASTER smoke:

```powershell
python run_snr_comparators_matrix.py --manifest results\strict_blind\manifest.json --output results\strict_blind\smoke_python_oaster --limit-pairs 1 --limit-cases 1 --workers 1
```

Python7 + OASTER full matrix (not started):

```powershell
python run_snr_comparators_matrix.py --manifest results\strict_blind\manifest.json --output results\strict_blind\python_oaster --workers 4
```

It atomically replaces one CSV per SNR pair. Re-running skips every valid 185-case pair. `--start-pair`, `--limit-pairs` and `--limit-cases` support serial shards and smoke runs. A final unbounded invocation skips completed parts and assembles `rows.csv` only after all 49 parts validate.

SISSES smoke:

```powershell
python run_strict_blind_sisses.py prepare --manifest results\strict_blind\manifest.json --output results\strict_blind\smoke_sisses --limit-pairs 1 --limit-cases 1 --chunk-size 1
python run_strict_blind_sisses.py matlab --output results\strict_blind\smoke_sisses --workers 1
python run_strict_blind_sisses.py score --manifest results\strict_blind\manifest.json --output results\strict_blind\smoke_sisses --limit-pairs 1 --limit-cases 1 --workers 1
```

SISSES full matrix (not started):

```powershell
python run_strict_blind_sisses.py prepare --manifest results\strict_blind\manifest.json --output results\strict_blind\sisses --chunk-size 20
python run_strict_blind_sisses.py matlab --output results\strict_blind\sisses --workers 4
python run_strict_blind_sisses.py score --manifest results\strict_blind\manifest.json --output results\strict_blind\sisses --workers 4
```

Preparation validates and skips existing chunks. MATLAB skips complete chunks; an interrupted partial chunk is rerun. Scoring checkpoints atomically after each case.

## Measured cost

Historical runs of these exact adapters give the following per-case means: minimum-norm family 1.789 s (one solve produces four methods), LCMV 0.141 s, dipole fitting 0.038 s, RAP-MUSIC 0.145 s, and OASTER 1.144 s. Across 9,065 cases this is about 8.2 cumulative CPU-hours, with an ideal four-worker lower bound near 2.1 wall-hours before initialization and I/O overhead.

SISSES historically averaged 107.6 s/case (median 55.5 s; 95th percentile 312.4 s). The strict-blind one-case smoke took about 109 s inside the MATLAB stage. A full matrix projects to roughly 271 cumulative MATLAB-hours, or an optimistic 68 wall-hours with four independent one-thread MATLAB workers.

The SISSES smoke raw file was 34.9 MB. Existing 20-case chunks indicate a full run requires approximately 316 GB of raw SISSES results plus about 9 GB of compressed MATLAB inputs. Storage, rather than Python computation, is the main operational constraint.
