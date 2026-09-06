"""Forensic reconstruction of the observation-only OASTER core.

The EBIC loop and spectral-evidence helpers below are recovered exact fragments.
Only ``_temporal_basis`` is reconstructed from its surviving callers and the
recorded active-vs-baseline singular-value edge formula.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import welch

import protected_multilayer as protected


METHOD = "OASTER"
LONG_NAME = "Observation-Adaptive Spatiotemporal Evidence Reconstruction"
NOISE_SAMPLES = 200
SURFACE_SCALES_MM = (0.0, 4.0, 7.0)
RIDGE_FRACTION = 0.03
SPECTRAL_FRACTION = 0.05


# Recovered exact from the later protected_multilayer.py transcript fragment.
def _standardized_active_projection_scores(
    data: np.ndarray,
    leadfield: np.ndarray,
    *,
    noise_samples: int = NOISE_SAMPLES,
    eps: float = 1e-12,
) -> np.ndarray:
    """Score every leadfield by its active-minus-baseline projection z statistic."""
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    if data.ndim != 2 or leadfield.ndim != 2 or data.shape[0] != leadfield.shape[0]:
        raise ValueError("data and leadfield must share the channel axis")
    if not 0 < noise_samples < data.shape[1]:
        raise ValueError("noise_samples must split baseline and active samples")
    projection = (leadfield.T @ data) / np.maximum(
        np.linalg.norm(leadfield, axis=0), eps
    )[:, None]
    baseline = np.mean(projection[:, :noise_samples] ** 2, axis=1)
    active = np.mean(projection[:, noise_samples:] ** 2, axis=1)
    standard_error = np.sqrt(
        2.0
        * baseline**2
        * (1.0 / noise_samples + 1.0 / (data.shape[1] - noise_samples))
    )
    return np.maximum(active - baseline, 0.0) / np.maximum(standard_error, eps)


# Recovered exact from the later protected_multilayer.py transcript fragment.
def _adaptive_active_spectral_filter(
    data: np.ndarray,
    *,
    noise_samples: int = NOISE_SAMPLES,
    segment_samples: int = 128,
    eps: float = 1e-12,
) -> np.ndarray:
    """Suppress frequencies without pooled active-over-baseline power evidence."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not 0 < noise_samples < data.shape[1]:
        raise ValueError("data must be channels x time with baseline and active samples")
    size = min(int(segment_samples), noise_samples, data.shape[1] - noise_samples)
    if size < 8:
        raise ValueError("each window needs at least eight samples")
    overlap = size // 2
    frequencies, baseline_psd = welch(
        data[:, :noise_samples], nperseg=size, noverlap=overlap, axis=1
    )
    _, active_psd = welch(
        data[:, noise_samples:], nperseg=size, noverlap=overlap, axis=1
    )
    baseline_power = np.mean(baseline_psd, axis=0)
    active_power = np.mean(active_psd, axis=0)
    kernel = np.ones(3) / 3.0
    baseline_power = np.convolve(baseline_power, kernel, mode="same")
    active_power = np.convolve(active_power, kernel, mode="same")
    ratio = active_power / np.maximum(baseline_power, eps)
    null_ratio = float(np.median(ratio[1:])) if ratio.size > 1 else float(ratio[0])
    gain = np.clip(1.0 - null_ratio / np.maximum(ratio, eps), 0.0, 1.0)
    gain[0] = 0.0
    result = np.zeros_like(data)
    for window in (slice(0, noise_samples), slice(noise_samples, None)):
        block = data[:, window]
        fft_frequency = np.fft.rfftfreq(block.shape[1])
        interpolated = np.interp(fft_frequency, frequencies, gain)
        spectrum = np.fft.rfft(block - block.mean(axis=1, keepdims=True), axis=1)
        result[:, window] = np.fft.irfft(
            spectrum * interpolated[None, :], n=block.shape[1], axis=1
        )
    return result


# Recovered exact from the later protected_multilayer.py transcript fragment.
def _multiscale_spectral_evidence_source(
    data: np.ndarray,
    leadfield: np.ndarray,
    n_surf: int,
    surface_kernel: sparse.spmatrix,
    *,
    noise_samples: int = NOISE_SAMPLES,
) -> np.ndarray:
    """Reconstruct data-driven spectral evidence on one physical surface scale."""
    filtered = _adaptive_active_spectral_filter(data, noise_samples=noise_samples)
    surface_gain = np.asarray(leadfield[:, :n_surf] @ surface_kernel)
    template_gain = np.column_stack((surface_gain, leadfield[:, n_surf:]))
    scores = _standardized_active_projection_scores(
        filtered, template_gain, noise_samples=noise_samples
    )
    coefficients = (template_gain.T @ filtered) / np.maximum(
        np.sum(template_gain**2, axis=0), 1e-30
    )[:, None]
    active_norm = np.linalg.norm(coefficients[:, noise_samples:], axis=1)
    coefficients *= scores[:, None] / np.maximum(active_norm, 1e-30)[:, None]
    return np.vstack((surface_kernel @ coefficients[:n_surf], coefficients[n_surf:]))


# Recovered exact from the later protected_multilayer.py transcript fragment.
def _add_scaled_evidence(
    primary: np.ndarray,
    evidence: np.ndarray,
    fraction: float,
    *,
    noise_samples: int = NOISE_SAMPLES,
) -> np.ndarray:
    """Add secondary evidence at a fraction of the primary active peak."""
    primary = np.asarray(primary, dtype=float)
    evidence = np.asarray(evidence, dtype=float)
    if primary.shape != evidence.shape or primary.ndim != 2:
        raise ValueError("primary and evidence must have the same source-matrix shape")
    if not 0.0 <= fraction <= 1.0 or not 0 < noise_samples < primary.shape[1]:
        raise ValueError("fraction and noise_samples are outside their valid ranges")
    primary_peak = float(
        np.linalg.norm(primary[:, noise_samples:], axis=1).max(initial=0.0)
    )
    evidence_peak = float(
        np.linalg.norm(evidence[:, noise_samples:], axis=1).max(initial=0.0)
    )
    if evidence_peak <= 0:
        return primary.copy()
    if primary_peak <= 0:
        return evidence.copy()
    return primary + fraction * primary_peak / evidence_peak * evidence


def _temporal_basis(
    data: np.ndarray, noise_samples: int = NOISE_SAMPLES
) -> np.ndarray:
    """Select active-window temporal modes above the observed baseline noise edge.

    Reconstructed: surviving experiments record this exact edge calculation but
    the original function body was not present in the recovered transcript.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not 0 < noise_samples < data.shape[1]:
        raise ValueError("data must be channels x time with baseline and active samples")
    spectral_filter = getattr(
        protected, "adaptive_active_spectral_filter", _adaptive_active_spectral_filter
    )
    filtered = spectral_filter(data, noise_samples=noise_samples)
    baseline = filtered[:, :noise_samples].copy()
    active = filtered[:, noise_samples:].copy()
    baseline -= baseline.mean(axis=1, keepdims=True)
    active -= active.mean(axis=1, keepdims=True)
    edge = float(np.linalg.svd(baseline, compute_uv=False)[0])
    edge *= (np.sqrt(data.shape[0]) + np.sqrt(active.shape[1])) / (
        np.sqrt(data.shape[0]) + np.sqrt(baseline.shape[1])
    )
    _, singular, right = np.linalg.svd(active, full_matrices=False)
    rank = int(np.sum(singular > edge))
    basis = np.zeros((rank, data.shape[1]))
    basis[:, noise_samples:] = right[:rank]
    return basis


def _ebic_templates(
    data: np.ndarray,
    leadfield: np.ndarray,
    n_surf: int,
    kernels,
    basis: np.ndarray,
    ridge_fraction: float = RIDGE_FRACTION,
) -> tuple[np.ndarray, int]:
    """Greedily select 0/4/7-mm surface or deep templates until EBIC stops."""
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if data.ndim != 2 or leadfield.ndim != 2 or data.shape[0] != leadfield.shape[0]:
        raise ValueError("data and leadfield must share the channel axis")
    if basis.ndim != 2 or basis.shape[1] != data.shape[1]:
        raise ValueError("basis must be temporal-rank x time")
    if not 0 < n_surf <= leadfield.shape[1]:
        raise ValueError("n_surf is outside the leadfield source axis")

    result = np.zeros((leadfield.shape[1], data.shape[1]))
    if basis.size == 0:
        return result, 0
    reduced = data @ basis.T
    surface = [
        (kernel, np.asarray(leadfield[:, :n_surf] @ kernel))
        for _scale, kernel in kernels
    ]
    candidates = [gain for _kernel, gain in surface] + [leadfield[:, n_surf:]]
    norms = [np.maximum(np.sum(gain**2, axis=0), 1e-30) for gain in candidates]
    selected: list[tuple[int, int]] = []
    columns: list[np.ndarray] = []
    residual = reduced.copy()
    n_obs = reduced.size
    universe = sum(gain.shape[1] for gain in candidates)
    score = n_obs * np.log(np.sum(residual**2) / n_obs + np.finfo(float).eps)

    while len(columns) < min(reduced.shape):
        best = None
        for family, gain in enumerate(candidates):
            drops = np.sum((gain.T @ residual) ** 2, axis=1) / norms[family]
            for old_family, old_index in selected:
                old = candidates[old_family][:, old_index]
                correlation = np.abs(gain.T @ old) / np.sqrt(
                    norms[family] * max(float(old @ old), 1e-30)
                )
                drops[correlation >= 0.98] = -np.inf
            index = int(np.argmax(drops))
            candidate = (float(drops[index]), family, index)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or not np.isfinite(best[0]):
            break
        _drop, family, index = best
        trial = np.column_stack((*columns, candidates[family][:, index]))
        coefficients = np.linalg.lstsq(trial, reduced, rcond=None)[0]
        trial_residual = reduced - trial @ coefficients
        count = len(columns) + 1
        trial_score = (
            n_obs
            * np.log(np.sum(trial_residual**2) / n_obs + np.finfo(float).eps)
            + count * basis.shape[0] * np.log(n_obs)
            + 2.0 * count * np.log(universe)
        )
        if trial_score >= score:
            break
        score = trial_score
        selected.append((family, index))
        columns.append(candidates[family][:, index])
        residual = trial_residual

    if not columns:
        return result, 0
    design = np.column_stack(columns)
    gram = design.T @ design
    ridge = max(float(ridge_fraction) * np.trace(gram) / len(columns), 1e-12)
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(len(columns)), design.T @ reduced
    ) @ basis
    for row, (family, index) in enumerate(selected):
        if family < len(surface):
            spatial = surface[family][0].getcol(index).toarray().ravel()
            result[:n_surf] += spatial[:, None] * coefficients[row]
        else:
            result[n_surf + index] += coefficients[row]
    return result, len(selected)


def reconstruct(
    eeg_data: np.ndarray,
    meg_data: np.ndarray,
    gain_eeg: np.ndarray,
    gain_meg: np.ndarray,
    n_surf: int,
    kernels,
) -> tuple[np.ndarray, dict]:
    """Reconstruct sources using only EEG/MEG observations and forward operators."""
    kernels = tuple(kernels)
    kernel_by_scale = dict(kernels)
    if 4.0 not in kernel_by_scale:
        raise ValueError("kernels must include the recovered 4-mm evidence scale")
    data, leadfield = protected.whitened_joint_system(
        {"F": eeg_data, "Gain": gain_eeg},
        {"F": meg_data, "Gain": gain_meg},
    )
    basis = _temporal_basis(data)
    primary, count = _ebic_templates(data, leadfield, n_surf, kernels, basis)
    spectral_fn = getattr(
        protected,
        "multiscale_spectral_evidence_source",
        _multiscale_spectral_evidence_source,
    )
    add_fn = getattr(protected, "add_scaled_evidence", _add_scaled_evidence)
    spectral = spectral_fn(data, leadfield, n_surf, kernel_by_scale[4.0])
    estimate = add_fn(primary, spectral, SPECTRAL_FRACTION)
    return estimate, {
        "temporal_rank": basis.shape[0],
        "selected_templates": count,
    }
