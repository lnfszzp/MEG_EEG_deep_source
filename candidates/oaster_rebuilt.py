"""Forensic reconstruction of the observation-only OASTER core.

The EBIC loop and spectral-evidence helpers below are recovered exact fragments.
Only ``_temporal_basis`` is reconstructed from its surviving callers and the
recorded active-vs-baseline singular-value edge formula.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

import protected_multilayer as protected


METHOD = "OASTER"
LONG_NAME = "Observation-Adaptive Spatiotemporal Evidence Reconstruction"
NOISE_SAMPLES = 200
SURFACE_SCALES_MM = (0.0, 4.0, 7.0)
RIDGE_FRACTION = 0.03
SPECTRAL_FRACTION = 0.05
DEEP_RESCUE_TAU = 0.0
MAX_DEEP_RESCUES = 1
ERP_LAMBDA2 = 1.0 / 9.0
ERP_DEPTH = 0.8
ERP_TEMPORAL_SCALES = (0.0, 0.10, 0.25)
ERP_NULL_QUANTILE = 0.99
ERP_MAX_TEMPORAL_RANK = 3
ERP_MAX_TEMPLATES = 6
ERP_DEEP_EBIC_DELTA = -6.0


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


def _column_space(matrix: np.ndarray) -> np.ndarray:
    """Return a stable orthonormal basis for a fitted sensor-space design."""
    matrix = np.asarray(matrix, dtype=float)
    if not matrix.size or not np.any(matrix):
        return np.zeros((matrix.shape[0], 0))
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
    return left[:, singular > tolerance]


def deep_rescue_trial(
    data: np.ndarray,
    leadfield: np.ndarray,
    primary: np.ndarray,
    n_surf: int,
    basis: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Propose one observation-only deep source using conditional EBIC.

    ``_ebic_templates`` does not expose its selected design. Its fitted reduced
    sensor signal has the same observable column space when selected temporal
    coefficients are full rank, so that space is recovered by SVD and used as
    the conditioning design. This is the method's only approximation.
    """
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    primary = np.asarray(primary, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if data.ndim != 2 or leadfield.ndim != 2 or data.shape[0] != leadfield.shape[0]:
        raise ValueError("data and leadfield must share the channel axis")
    if primary.shape != (leadfield.shape[1], data.shape[1]):
        raise ValueError("primary must be sources x time")
    if basis.ndim != 2 or basis.shape[1] != data.shape[1]:
        raise ValueError("basis must be temporal-rank x time")
    if not 0 < n_surf < leadfield.shape[1]:
        raise ValueError("the leadfield must contain surface and deep candidates")

    rescue = np.zeros_like(primary)
    universe = int(leadfield.shape[1] - n_surf)
    diagnostics = {
        "accepted_without_tau": False,
        "deep_local": -1,
        "ebic_delta_without_tau": math.inf,
        "universe": universe,
        "recovered_design_rank": 0,
    }
    if basis.size == 0:
        return rescue, diagnostics

    reduced = data @ basis.T
    fitted = leadfield @ (primary @ basis.T)
    design = _column_space(fitted)
    diagnostics["recovered_design_rank"] = int(design.shape[1])
    residual = reduced - fitted
    deep_gain = leadfield[:, n_surf:]
    if design.shape[1]:
        residual = residual - design @ (design.T @ residual)
        residualized_gain = deep_gain - design @ (design.T @ deep_gain)
    else:
        residualized_gain = deep_gain

    norms = np.sum(residualized_gain**2, axis=0)
    valid = norms > np.finfo(float).eps
    rss_old = float(np.sum(residual**2))
    if not np.any(valid) or rss_old <= np.finfo(float).eps:
        return rescue, diagnostics
    drops = np.full(universe, -np.inf)
    drops[valid] = (
        np.sum((residualized_gain[:, valid].T @ residual) ** 2, axis=1)
        / norms[valid]
    )
    index = int(np.argmax(drops))
    drop = min(float(drops[index]), rss_old)
    n_obs = int(reduced.size)
    rank = int(basis.shape[0])
    rss_new = max(rss_old - drop, np.finfo(float).tiny)
    delta = (
        n_obs * math.log(rss_new / rss_old)
        + rank * math.log(n_obs)
        + 2.0 * math.log(universe)
    )
    coefficients = (residualized_gain[:, index].T @ residual) / norms[index]
    rescue[n_surf + index] = coefficients @ basis
    diagnostics.update(
        accepted_without_tau=bool(delta < 0.0),
        deep_local=index,
        ebic_delta_without_tau=float(delta),
    )
    return rescue, diagnostics


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
    candidates = [gain for _kernel, gain in surface]
    if n_surf < leadfield.shape[1]:
        candidates.append(leadfield[:, n_surf:])
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


def reconstruct_from_whitened(
    data: np.ndarray,
    leadfield: np.ndarray,
    n_surf: int,
    kernels,
) -> tuple[np.ndarray, dict]:
    """Run OASTER on an already whitened sensor system."""
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    if data.ndim != 2 or leadfield.ndim != 2 or data.shape[0] != leadfield.shape[0]:
        raise ValueError("data and leadfield must share the channel axis")
    kernels = tuple(kernels)
    kernel_by_scale = dict(kernels)
    if 4.0 not in kernel_by_scale:
        raise ValueError("kernels must include the recovered 4-mm evidence scale")
    basis = _temporal_basis(data)
    primary, count = _ebic_templates(data, leadfield, n_surf, kernels, basis)
    if n_surf < leadfield.shape[1]:
        rescue, rescue_diagnostics = deep_rescue_trial(
            data, leadfield, primary, n_surf, basis
        )
    else:
        rescue = np.zeros_like(primary)
        rescue_diagnostics = {
            "accepted_without_tau": False,
            "deep_local": -1,
            "ebic_delta_without_tau": math.inf,
            "universe": 0,
            "recovered_design_rank": 0,
        }
    rescue_delta = (
        float(rescue_diagnostics["ebic_delta_without_tau"]) + DEEP_RESCUE_TAU
    )
    rescue_accepted = rescue_delta < 0.0
    if rescue_accepted:
        primary = primary + rescue
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
        "deep_rescue_tau": DEEP_RESCUE_TAU,
        "deep_rescue_accepted": rescue_accepted,
        "deep_rescue_deep_local": rescue_diagnostics["deep_local"],
        "deep_rescue_ebic_delta": rescue_delta,
        "deep_rescue_universe": rescue_diagnostics["universe"],
        "deep_rescue_design_rank": rescue_diagnostics["recovered_design_rank"],
    }


def reconstruct_evoked_from_whitened(
    data: np.ndarray,
    leadfield: np.ndarray,
    *,
    lambda2: float = ERP_LAMBDA2,
    depth: float = ERP_DEPTH,
) -> tuple[np.ndarray, dict]:
    """Localize phase-locked evoked data without spectral windowing.

    This is the ERP branch: it keeps the signed time course, uses a
    depth-weighted minimum-norm inverse, and standardizes every source by its
    propagated whitened-noise standard deviation.  Its mathematical backbone
    is dSPM-like and is an auditable baseline, not a claimed novel substitute
    for dSPM.  The caller must whiten the data and leadfield from single-trial
    baseline samples or a noise covariance.  When ``data`` is an average of
    several trials, the returned scale is relative noise-normalized amplitude,
    not a z score, unless the whitener already includes the averaging factor.
    """
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    if data.ndim != 2 or leadfield.ndim != 2 or data.shape[0] != leadfield.shape[0]:
        raise ValueError("data and leadfield must share the channel axis")
    if not data.shape[0] or not data.shape[1] or not leadfield.shape[1]:
        raise ValueError("data and leadfield must be non-empty")
    if not np.isfinite(data).all() or not np.isfinite(leadfield).all():
        raise ValueError("data and leadfield must contain only finite values")
    if not np.isfinite(lambda2) or lambda2 <= 0.0:
        raise ValueError("lambda2 must be positive and finite")
    if not np.isfinite(depth) or not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be between zero and one")

    column_power = np.sum(leadfield**2, axis=0)
    valid = column_power > 0.0
    if not np.any(valid):
        raise ValueError("leadfield has no observable source")
    source_variance = np.zeros_like(column_power)
    source_variance[valid] = column_power[valid] ** (-depth)
    if not np.isfinite(source_variance[valid]).all():
        raise ValueError("leadfield sensitivity is too ill-conditioned for depth weighting")
    depth_prior_dynamic_range = float(
        source_variance[valid].max() / source_variance[valid].min()
    )
    sensor_covariance = (leadfield * source_variance) @ leadfield.T
    sensor_rank = int(np.linalg.matrix_rank(sensor_covariance, hermitian=True))
    covariance_trace = float(np.trace(sensor_covariance))
    if sensor_rank == 0 or covariance_trace <= 0.0:
        raise ValueError("leadfield has zero sensor-space rank")

    source_variance *= sensor_rank / covariance_trace
    sensor_covariance = (leadfield * source_variance) @ leadfield.T
    regularized = sensor_covariance + float(lambda2) * np.eye(data.shape[0])
    inverse = (source_variance[:, None] * leadfield.T) @ np.linalg.solve(
        regularized, np.eye(data.shape[0])
    )
    source = inverse @ data
    noise_normalization = np.linalg.norm(inverse, axis=1)
    standardized = np.zeros_like(source)
    np.divide(
        source,
        noise_normalization[:, None],
        out=standardized,
        where=noise_normalization[:, None] > 0.0,
    )
    return standardized, {
        "mode": "signed_time_domain_noise_normalized_minimum_norm",
        "lambda2": float(lambda2),
        "depth": float(depth),
        "sensor_rank": sensor_rank,
        "valid_sources": int(np.sum(valid)),
        "depth_prior_dynamic_range": depth_prior_dynamic_range,
    }


def _evoked_temporal_basis(
    data: np.ndarray,
    baseline: np.ndarray,
    active: np.ndarray,
    *,
    temporal_scales: tuple[float, ...] = ERP_TEMPORAL_SCALES,
    null_quantile: float = ERP_NULL_QUANTILE,
    max_rank: int = ERP_MAX_TEMPORAL_RANK,
) -> tuple[np.ndarray, dict]:
    """Select signed ERP modes against same-width prestimulus noise windows."""
    data = np.asarray(data, dtype=float)
    baseline = np.asarray(baseline, dtype=bool).ravel()
    active = np.asarray(active, dtype=bool).ravel()
    if data.ndim != 2 or baseline.size != data.shape[1] or active.size != data.shape[1]:
        raise ValueError("data and ERP masks must share the time axis")
    baseline_index = np.flatnonzero(baseline)
    active_index = np.flatnonzero(active)
    if not baseline_index.size or not active_index.size:
        raise ValueError("baseline and active masks must both be non-empty")
    if np.any(active & baseline):
        raise ValueError("baseline and active masks must not overlap")
    if np.any(np.diff(baseline_index) != 1) or np.any(np.diff(active_index) != 1):
        raise ValueError("each ERP mask must describe one contiguous window")
    width = int(active_index.size)
    if width > baseline_index.size:
        raise ValueError("the ERP baseline must be at least as long as the active window")
    if not 0.0 < float(null_quantile) < 1.0 or int(max_rank) < 1:
        raise ValueError("null_quantile and max_rank are outside their valid ranges")

    centered = data - data[:, baseline].mean(axis=1, keepdims=True)
    baseline_data = centered[:, baseline]
    active_data = centered[:, active]
    candidates: list[tuple[float, np.ndarray]] = []
    edges = []
    for fraction in temporal_scales:
        sigma = max(0.0, float(fraction)) * width
        response = (
            active_data
            if sigma == 0.0
            else gaussian_filter1d(active_data, sigma=sigma, axis=1, mode="nearest")
        )
        singular, right = np.linalg.svd(response, full_matrices=False)[1:]
        null = []
        for start in range(baseline_data.shape[1] - width + 1):
            block = baseline_data[:, start : start + width]
            if sigma > 0.0:
                block = gaussian_filter1d(block, sigma=sigma, axis=1, mode="nearest")
            null.append(float(np.linalg.svd(block, compute_uv=False)[0]))
        edge = float(np.quantile(null, null_quantile))
        edges.append(edge)
        for value, vector in zip(singular[:max_rank], right[:max_rank]):
            if value > edge:
                candidates.append((float(value / max(edge, np.finfo(float).eps)), vector))

    kept: list[np.ndarray] = []
    for _strength, vector in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(abs(float(vector @ old)) < 0.95 for old in kept):
            kept.append(vector)
        if len(kept) == int(max_rank):
            break
    basis = np.zeros((len(kept), data.shape[1]))
    if kept:
        orthogonal, _ = np.linalg.qr(np.asarray(kept).T, mode="reduced")
        basis[:, active] = orthogonal.T
    return basis, {
        "active_samples": width,
        "temporal_rank": int(basis.shape[0]),
        "null_edges": edges,
    }


def _evoked_ebic_surface_templates(
    reduced: np.ndarray,
    surface_gains: tuple[np.ndarray, ...],
    *,
    universe: int,
    max_templates: int = ERP_MAX_TEMPLATES,
    max_correlation: float = 0.98,
    require_one: bool = True,
) -> tuple[list[tuple[int, int]], np.ndarray, float, list[float]]:
    """Select surface templates by conditional residual drops and EBIC."""
    reduced = np.asarray(reduced, dtype=float)
    if reduced.ndim != 2 or not surface_gains:
        raise ValueError("reduced data and surface gains must be non-empty matrices")
    norms = tuple(
        np.maximum(np.sum(gain**2, axis=0), np.finfo(float).eps)
        for gain in surface_gains
    )
    selected: list[tuple[int, int]] = []
    score_deltas: list[float] = []
    columns: list[np.ndarray] = []
    residual = reduced.copy()
    n_obs = int(reduced.size)
    rank = int(reduced.shape[1])
    score = n_obs * math.log(
        float(np.sum(residual**2)) / n_obs + np.finfo(float).eps
    )

    while len(columns) < int(max_templates):
        if columns:
            design = np.column_stack(columns)
            orthogonal = np.linalg.qr(design, mode="reduced")[0]
        else:
            orthogonal = np.empty((reduced.shape[0], 0))
        best = None
        for family, gain in enumerate(surface_gains):
            residualized = gain - orthogonal @ (orthogonal.T @ gain)
            residualized_norm = np.sum(residualized**2, axis=0)
            drops = np.full(gain.shape[1], -np.inf)
            valid = residualized_norm > np.finfo(float).eps
            drops[valid] = (
                np.sum((residualized[:, valid].T @ residual) ** 2, axis=1)
                / residualized_norm[valid]
            )
            for old_family, old_index in selected:
                old = surface_gains[old_family][:, old_index]
                correlation = np.abs(gain.T @ old) / np.sqrt(
                    norms[family] * max(float(old @ old), np.finfo(float).eps)
                )
                drops[correlation >= max_correlation] = -np.inf
            index = int(np.argmax(drops))
            candidate = (float(drops[index]), family, index)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or not np.isfinite(best[0]):
            break
        _drop, family, index = best
        trial = np.column_stack((*columns, surface_gains[family][:, index]))
        coefficients = np.linalg.lstsq(trial, reduced, rcond=None)[0]
        trial_residual = reduced - trial @ coefficients
        count = len(columns) + 1
        trial_score = (
            n_obs
            * math.log(float(np.sum(trial_residual**2)) / n_obs + np.finfo(float).eps)
            + count * rank * math.log(n_obs)
            + 2.0 * count * math.log(universe)
        )
        score_delta = float(trial_score - score)
        forced_single = bool(score_delta >= 0.0 and not columns and require_one)
        if trial_score >= score and not forced_single:
            break
        score = trial_score
        selected.append((family, index))
        score_deltas.append(score_delta)
        columns.append(surface_gains[family][:, index])
        residual = trial_residual
        if forced_single:
            break

    design = (
        np.column_stack(columns)
        if columns
        else np.empty((reduced.shape[0], 0))
    )
    return selected, design, float(score), score_deltas


def reconstruct_evoked_oaster_from_whitened(
    data: np.ndarray,
    leadfield: np.ndarray,
    n_surf: int,
    kernels,
    *,
    baseline: np.ndarray,
    active_windows,
    window_channel_weights=None,
    ridge_fraction: float = RIDGE_FRACTION,
    max_templates: int = ERP_MAX_TEMPLATES,
) -> tuple[np.ndarray, dict]:
    """Run the sparse multiscale OASTER core on phase-locked evoked responses.

    Each preregistered ERP window gets its own observation-selected temporal
    basis and spatial EBIC search.  The union of accepted templates is then
    refitted to the complete baseline-centered epoch, preserving waveform sign
    and a real prestimulus residual instead of forcing the baseline to zero.
    """
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    baseline = np.asarray(baseline, dtype=bool).ravel()
    if data.ndim != 2 or leadfield.ndim != 2 or data.shape[0] != leadfield.shape[0]:
        raise ValueError("data and leadfield must share the channel axis")
    if baseline.size != data.shape[1] or not baseline.any():
        raise ValueError("baseline must be a non-empty mask on the data time axis")
    if not np.isfinite(data).all() or not np.isfinite(leadfield).all():
        raise ValueError("data and leadfield must contain only finite values")
    if not 0 < int(n_surf) <= leadfield.shape[1]:
        raise ValueError("n_surf is outside the leadfield source axis")
    if not np.isfinite(ridge_fraction) or ridge_fraction < 0.0:
        raise ValueError("ridge_fraction must be finite and non-negative")
    kernels = tuple(kernels)
    if not kernels:
        raise ValueError("at least one surface kernel is required")
    for _scale, kernel in kernels:
        if kernel.shape != (n_surf, n_surf):
            raise ValueError("every surface kernel must match n_surf")

    windows = tuple(np.asarray(window, dtype=bool).ravel() for window in active_windows)
    if not windows or any(window.size != data.shape[1] for window in windows):
        raise ValueError("active_windows must contain masks on the data time axis")
    occupied = np.zeros(data.shape[1], dtype=bool)
    for window in windows:
        if not window.any() or np.any(window & baseline) or np.any(window & occupied):
            raise ValueError("active windows must be non-empty, disjoint, and outside baseline")
        occupied |= window

    if window_channel_weights is None:
        channel_weights = tuple(np.ones(data.shape[0]) for _window in windows)
    else:
        channel_weights = tuple(
            np.asarray(weights, dtype=float).ravel() for weights in window_channel_weights
        )
        if len(channel_weights) != len(windows):
            raise ValueError("window_channel_weights must match active_windows")
        if any(
            weights.size != data.shape[0]
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
            for weights in channel_weights
        ):
            raise ValueError("each channel-weight vector must be finite, non-negative, and non-zero")

    centered = data - data[:, baseline].mean(axis=1, keepdims=True)
    surface_gains = tuple(
        np.asarray(leadfield[:, :n_surf] @ kernel) for _scale, kernel in kernels
    )
    universe = int(sum(gain.shape[1] for gain in surface_gains))
    universe += int(leadfield.shape[1] - n_surf)
    selected_by_window: list[list[tuple[int, int]]] = []
    window_diagnostics = []

    for window_number, (window, weights) in enumerate(zip(windows, channel_weights)):
        weighted_data = centered * weights[:, None]
        weighted_leadfield = leadfield * weights[:, None]
        weighted_surface_gains = tuple(
            gain * weights[:, None] for gain in surface_gains
        )
        basis, basis_diagnostics = _evoked_temporal_basis(
            weighted_data, baseline, window
        )
        reduced = weighted_data @ basis.T
        selected, design, score, score_deltas = _evoked_ebic_surface_templates(
            reduced,
            weighted_surface_gains,
            universe=universe,
            max_templates=max_templates,
        ) if basis.size else ([], np.empty((data.shape[0], 0)), math.inf, [])
        deep_local = -1
        deep_delta = math.inf
        if basis.size and n_surf < leadfield.shape[1]:
            if design.shape[1]:
                coefficients = np.linalg.lstsq(design, reduced, rcond=None)[0]
                residual = reduced - design @ coefficients
                orthogonal = np.linalg.qr(design, mode="reduced")[0]
            else:
                residual = reduced.copy()
                orthogonal = np.empty((data.shape[0], 0))
            deep_gain = weighted_leadfield[:, n_surf:]
            residualized = deep_gain - orthogonal @ (orthogonal.T @ deep_gain)
            norms = np.sum(residualized**2, axis=0)
            valid = norms > np.finfo(float).eps
            if np.any(valid):
                drops = np.full(deep_gain.shape[1], -np.inf)
                drops[valid] = (
                    np.sum((residualized[:, valid].T @ residual) ** 2, axis=1)
                    / norms[valid]
                )
                deep_local = int(np.argmax(drops))
                trial = np.column_stack((design, deep_gain[:, deep_local]))
                trial_coefficients = np.linalg.lstsq(trial, reduced, rcond=None)[0]
                trial_residual = reduced - trial @ trial_coefficients
                count = len(selected) + 1
                n_obs = int(reduced.size)
                trial_score = (
                    n_obs
                    * math.log(float(np.sum(trial_residual**2)) / n_obs + np.finfo(float).eps)
                    + count * basis.shape[0] * math.log(n_obs)
                    + 2.0 * count * math.log(universe)
                )
                deep_delta = float(trial_score - score)
                if deep_delta < ERP_DEEP_EBIC_DELTA:
                    selected.append((len(kernels), deep_local))
        selected_by_window.append(selected)
        window_diagnostics.append({
            "window": window_number,
            **basis_diagnostics,
            "channel_weight_min": float(weights.min()),
            "channel_weight_max": float(weights.max()),
            "weighted_channels": int(np.sum(weights > 0.0)),
            "selected_surface_templates": int(sum(family < len(kernels) for family, _ in selected)),
            "surface_ebic_deltas": score_deltas,
            "selected_templates": [
                {
                    "layer": "surface" if family < len(kernels) else "deep",
                    "scale_mm": float(kernels[family][0]) if family < len(kernels) else None,
                    "index": int(index),
                }
                for family, index in selected
            ],
            "deep_candidate_local": deep_local,
            "deep_ebic_delta": deep_delta,
            "deep_accepted": bool(selected and selected[-1] == (len(kernels), deep_local)),
        })

    estimate = np.zeros((leadfield.shape[1], data.shape[1]))
    baseline_estimates = []
    selected_total = 0
    selected_surface_total = 0
    selected_deep_total = 0
    for window, selected in zip(windows, selected_by_window):
        kept: list[tuple[int, int]] = []
        kept_gains: list[np.ndarray] = []
        for family, index in selected:
            gain = (
                surface_gains[family][:, index]
                if family < len(kernels)
                else leadfield[:, n_surf + index]
            )
            norm = max(float(np.linalg.norm(gain)), np.finfo(float).eps)
            if all(
                abs(float(gain @ old))
                < 0.98 * norm * max(float(np.linalg.norm(old)), np.finfo(float).eps)
                for old in kept_gains
            ):
                kept.append((family, index))
                kept_gains.append(gain)
        if not kept:
            continue
        selected_total += len(kept)
        selected_surface_total += sum(family < len(kernels) for family, _ in kept)
        selected_deep_total += sum(family == len(kernels) for family, _ in kept)
        design = np.column_stack(kept_gains)
        norms = np.linalg.norm(design, axis=0)
        normalized = design / norms
        gram = normalized.T @ normalized
        ridge = max(
            float(ridge_fraction) * np.trace(gram) / len(kept),
            np.finfo(float).eps,
        )
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(len(kept)), normalized.T @ centered
        )
        coefficients /= norms[:, None]
        window_estimate = np.zeros_like(estimate)
        for row, (family, index) in enumerate(kept):
            if family < len(kernels):
                spatial = kernels[family][1].getcol(index).toarray().ravel()
                window_estimate[:n_surf] += spatial[:, None] * coefficients[row]
            else:
                window_estimate[n_surf + index] += coefficients[row]
        estimate[:, window] = window_estimate[:, window]
        baseline_estimates.append(window_estimate[:, baseline])
    if baseline_estimates:
        estimate[:, baseline] = np.mean(baseline_estimates, axis=0)

    return estimate, {
        "mode": "signed_multiscale_erp_ebic",
        "surface_scales_mm": [float(scale) for scale, _kernel in kernels],
        "selected_templates": int(selected_total),
        "selected_surface_templates": int(selected_surface_total),
        "selected_deep_templates": int(selected_deep_total),
        "ridge_fraction": float(ridge_fraction),
        "max_templates_per_window": int(max_templates),
        "windows": window_diagnostics,
    }


def reconstruct(
    eeg_data: np.ndarray,
    meg_data: np.ndarray,
    gain_eeg: np.ndarray,
    gain_meg: np.ndarray,
    n_surf: int,
    kernels,
) -> tuple[np.ndarray, dict]:
    """Reconstruct sources using only EEG/MEG observations and forward operators."""
    data, leadfield = protected.whitened_joint_system(
        {"F": eeg_data, "Gain": gain_eeg},
        {"F": meg_data, "Gain": gain_meg},
    )
    return reconstruct_from_whitened(data, leadfield, n_surf, kernels)
