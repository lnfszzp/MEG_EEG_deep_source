"""Observable EEG/MEG quality bins for a frozen development gate pilot."""
from __future__ import annotations

import numpy as np


def observable_quality(data_a, data_b, baseline_mask, temporal_basis, modality_sizes):
    """Bin independent half-means without source truth, SNR or inverse fits.

    Cross-half smooth response energy suppresses independent noise in expectation.
    The denominator includes baseline-mean uncertainty, as in the existing gate.
    This is an observed quality proxy, not the simulator's single-trial SNR.
    """
    data_a, data_b = np.asarray(data_a, float), np.asarray(data_b, float)
    baseline = np.asarray(baseline_mask, bool)
    basis = np.asarray(temporal_basis, float)
    sizes = np.asarray(modality_sizes)
    if (data_a.ndim != 2 or data_b.shape != data_a.shape
            or baseline.shape != (data_a.shape[1],) or baseline.sum() < 4
            or basis.ndim != 2 or basis.shape[1] != data_a.shape[1]
            or not len(basis) or sizes.shape != (2,)
            or sizes.dtype.kind not in "iu" or np.any(sizes <= 0)
            or sizes.sum() != data_a.shape[0]
            or not all(np.isfinite(x).all() for x in (data_a, data_b, basis))):
        raise ValueError("require finite paired observations, baseline, basis and two EEG/MEG block sizes")
    if (np.any(basis[:, baseline] != 0)
            or not np.allclose(basis @ basis.T, np.eye(len(basis)), rtol=1e-10, atol=1e-10)):
        raise ValueError("temporal basis must be orthonormal and zero on baseline")
    centered_a = data_a - data_a[:, baseline].mean(axis=1, keepdims=True)
    centered_b = data_b - data_b[:, baseline].mean(axis=1, keepdims=True)
    response_a, response_b = centered_a @ basis.T, centered_b @ basis.T
    mean_correction = float(np.sum(basis.sum(axis=1) ** 2) / baseline.sum())
    noise_factor = len(basis) + mean_correction
    q, cross, noise_a, noise_b = [], [], [], []
    start = 0
    for size in sizes:
        block = slice(start, start + int(size))
        cross.append(float(np.sum(response_a[block] * response_b[block])))
        noise_a.append(float(np.sum(centered_a[block][:, baseline] ** 2)
                             / (baseline.sum() - 1) * noise_factor))
        noise_b.append(float(np.sum(centered_b[block][:, baseline] ** 2)
                             / (baseline.sum() - 1) * noise_factor))
        if (not np.isfinite([cross[-1], noise_a[-1], noise_b[-1]]).all()
                or noise_a[-1] <= 0 or noise_b[-1] <= 0):
            raise ValueError("every modality half needs finite positive baseline noise energy")
        q.append(max(cross[-1] / (np.sqrt(noise_a[-1]) * np.sqrt(noise_b[-1])), 0.))
        start += int(size)
    if not np.isfinite(q).all():
        raise ValueError("observable quality is nonfinite")
    # shortcut: 10 dB bins cover the three extreme development conditions only; validate other regimes separately.
    observed_bin = "".join("H" if value >= 10. else "L" for value in q)
    return dict(q_eeg=q[0], q_meg=q[1],
                q_eeg_db=float(10 * np.log10(q[0])) if q[0] > 0 else None,
                q_meg_db=float(10 * np.log10(q[1])) if q[1] > 0 else None,
                observable_bin=observed_bin, threshold_q=10.,
                cross_energy=cross, expected_noise_half_a=noise_a,
                expected_noise_half_b=noise_b,
                baseline_mean_mode_correction=mean_correction,
                modality_order=["EEG", "MEG"], truth_used=False,
                rule="q=max(cross-half smooth energy / geometric-mean baseline noise expectation,0); H iff q>=10")
