from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def whiten(data: np.ndarray, gain: np.ndarray, noise_samples: int = 200):
    data = np.asarray(data, dtype=float)
    gain = np.asarray(gain, dtype=float)
    noise = data[:, :noise_samples] - data[:, :noise_samples].mean(axis=1, keepdims=True)
    covariance = noise @ noise.T / max(1, noise.shape[1] - 1)
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    keep = values > values.max() * 1e-8
    whitener = (vectors[:, keep] / np.sqrt(values[keep])).T
    return whitener @ data, whitener @ gain


def joint_whiten(eeg: np.ndarray, meg: np.ndarray, gain_eeg: np.ndarray, gain_meg: np.ndarray):
    eeg_white, eeg_gain = whiten(eeg, gain_eeg)
    meg_white, meg_gain = whiten(meg, gain_meg)
    return np.vstack([eeg_white, meg_white]), np.vstack([eeg_gain, meg_gain])


def _solve_spd(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return cho_solve(cho_factor(matrix, check_finite=False), rhs, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix, rcond=1e-12) @ rhs


def _minimum_norm_kernel(
    gain: np.ndarray, lambda2: float, depth: float, depth_limit: float
) -> np.ndarray:
    column_power = np.maximum(np.sum(gain**2, axis=0), np.finfo(float).eps)
    inverse_power = 1.0 / column_power
    ordered = np.sort(inverse_power)
    cap_ratio = float(depth_limit) ** 2
    beyond = np.flatnonzero(ordered > cap_ratio * ordered[0])
    cap = ordered[beyond[0]] if beyond.size else ordered[-1]
    covariance = np.minimum(inverse_power / cap, 1.0) ** float(depth)
    sensor_covariance = (gain * covariance) @ gain.T
    scale = np.trace(sensor_covariance) / max(1, np.linalg.matrix_rank(gain))
    covariance /= max(float(scale), np.finfo(float).eps)
    sensor_covariance = (gain * covariance) @ gain.T
    inverse = _solve_spd(
        sensor_covariance + float(lambda2) * np.eye(gain.shape[0]),
        np.eye(gain.shape[0]),
    )
    return covariance[:, None] * (gain.T @ inverse)


def _eloreta_kernel(
    gain: np.ndarray,
    lambda2: float,
    *,
    max_iter: int = 20,
    tolerance: float = 1e-6,
) -> np.ndarray:
    rank = max(1, np.linalg.matrix_rank(gain))
    weights = np.ones(gain.shape[1])
    for _ in range(max_iter):
        covariance = (gain * weights) @ gain.T
        normalizer = np.trace(covariance) / rank
        weights /= max(float(normalizer), np.finfo(float).eps)
        covariance /= max(float(normalizer), np.finfo(float).eps)
        inverse = _solve_spd(
            covariance + float(lambda2) * np.eye(gain.shape[0]),
            np.eye(gain.shape[0]),
        )
        updated = 1.0 / np.sqrt(
            np.maximum(np.sum(gain * (inverse @ gain), axis=0), np.finfo(float).eps)
        )
        updated_covariance = (gain * updated) @ gain.T
        updated /= max(
            float(np.trace(updated_covariance) / rank), np.finfo(float).eps
        )
        delta = np.linalg.norm(updated - weights) / max(
            np.linalg.norm(weights), np.finfo(float).eps
        )
        weights = updated
        if delta < tolerance:
            break
    covariance = (gain * weights) @ gain.T
    scale = np.trace(covariance) / rank
    weights /= max(float(scale), np.finfo(float).eps)
    covariance = (gain * weights) @ gain.T
    inverse = _solve_spd(
        covariance + float(lambda2) * np.eye(gain.shape[0]),
        np.eye(gain.shape[0]),
    )
    return weights[:, None] * (gain.T @ inverse)


def minimum_norm_family(
    data: np.ndarray,
    gain: np.ndarray,
    *,
    lambda2: float = 1.0 / 9.0,
    depth: float = 0.8,
    depth_limit: float = 10.0,
) -> dict[str, np.ndarray]:
    """Fixed-orientation MNE, dSPM, sLORETA and eLORETA on whitened data."""
    kernel = _minimum_norm_kernel(gain, lambda2, depth, depth_limit)
    mne = kernel @ data
    dspm_norm = np.sqrt(np.sum(kernel**2, axis=1))
    sloreta_norm = np.sqrt(np.abs(np.sum(kernel * gain.T, axis=1)))
    eloreta = _eloreta_kernel(gain, lambda2) @ data
    return {
        "MNE": mne,
        "dSPM": mne / np.maximum(dspm_norm[:, None], np.finfo(float).eps),
        "sLORETA": mne / np.maximum(sloreta_norm[:, None], np.finfo(float).eps),
        "eLORETA": eloreta,
    }


def lcmv(
    data: np.ndarray,
    gain: np.ndarray,
    active: np.ndarray,
    *,
    reg: float = 0.05,
) -> np.ndarray:
    active_data = data[:, np.asarray(active, dtype=int)]
    covariance = active_data @ active_data.T / max(1, active_data.shape[1] - 1)
    covariance += float(reg) * np.trace(covariance) / covariance.shape[0] * np.eye(
        covariance.shape[0]
    )
    inverse_gain = _solve_spd(covariance, gain)
    denominator = np.sum(gain * inverse_gain, axis=0)
    weights = inverse_gain.T / np.maximum(denominator[:, None], np.finfo(float).eps)
    return weights @ data


def _temporal_basis(data: np.ndarray, active: np.ndarray, max_rank: int = 12):
    _, singular, right = np.linalg.svd(data[:, active], full_matrices=False)
    keep = singular**2 >= np.mean(singular**2)
    rank = max(1, min(int(keep.sum()), int(max_rank)))
    basis_active = right[:rank]
    basis = np.zeros((rank, data.shape[1]))
    basis[:, active] = basis_active
    return data @ basis.T, basis


def dipole_fit(
    data: np.ndarray,
    gain: np.ndarray,
    active: np.ndarray,
    *,
    max_dipoles: int = 3,
) -> np.ndarray:
    """Grid-projected 1..max_dipoles ECD fit; BIC chooses source count."""
    compressed, basis = _temporal_basis(data, np.asarray(active, dtype=int))
    residual = compressed.copy()
    selected: list[int] = []
    candidates = []
    gain_power = np.sum(gain**2, axis=0)
    n_obs = compressed.size
    for _ in range(int(max_dipoles)):
        score = np.sum((gain.T @ residual) ** 2, axis=1) / np.maximum(
            gain_power, np.finfo(float).eps
        )
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
        coefficients = np.linalg.lstsq(gain[:, selected], compressed, rcond=None)[0]
        residual = compressed - gain[:, selected] @ coefficients
        rss = float(np.linalg.norm(residual, "fro") ** 2)
        bic = n_obs * np.log(rss / n_obs + np.finfo(float).eps) + (
            len(selected) * compressed.shape[1]
        ) * np.log(n_obs)
        candidates.append((bic, selected.copy(), coefficients.copy()))
    _, selected, coefficients = min(candidates, key=lambda item: item[0])
    estimate = np.zeros((gain.shape[1], data.shape[1]))
    estimate[selected] = coefficients @ basis
    return estimate


def rap_music(
    data: np.ndarray,
    gain: np.ndarray,
    active: np.ndarray,
    *,
    max_sources: int = 3,
) -> np.ndarray:
    """Minimal fixed-grid RAP-MUSIC with BIC source-count selection."""
    compressed, basis = _temporal_basis(data, np.asarray(active, dtype=int))
    signal_basis = np.linalg.svd(compressed, full_matrices=False)[0]
    residual_gain = gain.copy()
    selected: list[int] = []
    candidates = []
    n_obs = compressed.size
    for _ in range(int(max_sources)):
        normalized = residual_gain / np.maximum(
            np.linalg.norm(residual_gain, axis=0, keepdims=True), np.finfo(float).eps
        )
        score = np.sum((signal_basis.T @ normalized) ** 2, axis=0)
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
        coefficients = np.linalg.lstsq(gain[:, selected], compressed, rcond=None)[0]
        residual = compressed - gain[:, selected] @ coefficients
        rss = float(np.linalg.norm(residual, "fro") ** 2)
        bic = n_obs * np.log(rss / n_obs + np.finfo(float).eps) + (
            len(selected) * compressed.shape[1]
        ) * np.log(n_obs)
        candidates.append((bic, selected.copy(), coefficients.copy()))
        projector = np.eye(gain.shape[0]) - gain[:, selected] @ np.linalg.pinv(
            gain[:, selected]
        )
        residual_gain = projector @ gain
        signal_basis = np.linalg.svd(projector @ compressed, full_matrices=False)[0]
    _, selected, coefficients = min(candidates, key=lambda item: item[0])
    estimate = np.zeros((gain.shape[1], data.shape[1]))
    estimate[selected] = coefficients @ basis
    return estimate
