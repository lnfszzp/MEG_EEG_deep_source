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
| OASTER | `candidates/oaster_rebuilt.py::reconstruct` | archived joint EEG+MEG |
| SISSES | preserved `D:\oaster_strict_blind_sisses\scores.csv` | archived joint EEG+MEG |
| ConvDip | no recovered runnable adapter | EEG-only and cortex-only; retraining code was not recovered and deep output is unsupported, so it is excluded from the joint/deep strict-blind ranking |

The maintained strict runners read the archived observations directly. The
historical reconstructed runners named below were replaced by thin aliases;
they no longer depend on the unrecovered `run_oaster.py`,
`run_oaster_snr_matrix.py`, or `run_snr_robustness.py` modules.

## Smoke result

The first immutable 20-case chunk was used for smoke checks only.

- OASTER: 20/20 successful rows.
- Seven Python comparators: 140/140 successful rows.
- Both runners consumed archived `F_EEG`/`F_MEG`; observations were not regenerated.
- The frozen manifest, embedded case IDs, format version, and all 454 input chunk
  boundaries passed the read-only archive verifier.
- Re-running a valid part skips reconstruction after revalidating its metadata.

## Commands

OASTER smoke (one archived 20-case chunk):

```powershell
python run_strict_oaster.py --input-root D:\oaster_strict_blind_sisses\matlab_input --data-root $env:SOURCE_DATA_ROOT --output results\strict_blind\oaster_rebuilt_smoke --limit-chunks 1 --workers 1
```

OASTER full matrix:

```powershell
python run_strict_oaster.py --input-root D:\oaster_strict_blind_sisses\matlab_input --data-root $env:SOURCE_DATA_ROOT --output results\strict_blind\oaster_rebuilt --workers 4
```

Seven Python comparators use the same options through
`run_strict_comparators.py`. Both runners atomically checkpoint one CSV per
immutable input chunk; `--start-chunk` and `--limit-chunks` support shards and
smoke runs. A final unbounded invocation validates all chunks before assembly.

Verify the preserved SISSES inputs without scanning the 315 GB output tree:

```powershell
python run_strict_blind_sisses.py verify --archive D:\oaster_strict_blind_sisses
```

Validate and summarize the preserved SISSES score table:

```powershell
python run_strict_blind_sisses.py summarize --archive D:\oaster_strict_blind_sisses --data-root $env:SOURCE_DATA_ROOT --output results\strict_blind\sisses_preserved
```

This entry point never regenerates observations, invokes MATLAB, or writes
inside the archive. Re-running SISSES would require an external licensed
checkout plus a separately reviewed adapter, which are not bundled.

## Measured cost

Historical runs of these exact adapters give the following per-case means: minimum-norm family 1.789 s (one solve produces four methods), LCMV 0.141 s, dipole fitting 0.038 s, RAP-MUSIC 0.145 s, and OASTER 1.144 s. Across 9,065 cases this is about 8.2 cumulative CPU-hours, with an ideal four-worker lower bound near 2.1 wall-hours before initialization and I/O overhead.

SISSES historically averaged 107.6 s/case (median 55.5 s; 95th percentile 312.4 s). The strict-blind one-case smoke took about 109 s inside the MATLAB stage. A full matrix projects to roughly 271 cumulative MATLAB-hours, or an optimistic 68 wall-hours with four independent one-thread MATLAB workers.

The SISSES smoke raw file was 34.9 MB. Existing 20-case chunks indicate a full run requires approximately 316 GB of raw SISSES results plus about 9 GB of compressed MATLAB inputs. Storage, rather than Python computation, is the main operational constraint.
