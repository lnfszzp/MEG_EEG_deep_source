"""Development-only held-out evidence for an already fitted deep candidate."""
from __future__ import annotations

import numpy as np

from benchmark.metrics import source_amplitude
from candidates.oaster_balanced import _smooth_temporal_basis
from candidates.oaster_predictive import _validate_observations


def score_partial_deep_presence(confirmation, gain, null_estimate, full_estimate,
                                n_surf, *, baseline, active, channel_weights,
                                support_energy_fraction=.01):
    """Test one training-selected deep lead field beyond H0 cortical support.

    Source support and the deep candidate come only from the training fits.
    Confirmation coefficients are nuisance/test-statistic fits and are never
    returned as a source estimate. Pure-surface calibration must set any final
    decision threshold; this development score has no reference cutoff.
    """
    confirmation, gain, baseline, active, weights = _validate_observations(
        confirmation, gain, baseline, active, channel_weights)
    if baseline.sum() <= 3:
        raise ValueError("at least four baseline samples are required for finite-baseline centering")
    null_estimate = np.asarray(null_estimate, float)
    full_estimate = np.asarray(full_estimate, float)
    expected = (gain.shape[1], confirmation.shape[1])
    if (not isinstance(n_surf, (int, np.integer)) or not 0 < n_surf < gain.shape[1]
            or null_estimate.shape != expected or full_estimate.shape != expected
            or not np.isfinite(null_estimate).all() or not np.isfinite(full_estimate).all()
            or not np.isfinite(support_energy_fraction)
            or not 0 < support_energy_fraction < 1):
        raise ValueError("finite full-grid training fits and 0 < support_energy_fraction < 1 required")

    null_amplitude = source_amplitude(null_estimate, np.flatnonzero(active), baseline)
    surface_peak = float(null_amplitude[:n_surf].max(initial=0.))
    surface_support = (np.flatnonzero(null_amplitude[:n_surf] >=
        np.sqrt(support_energy_fraction) * surface_peak)
        if surface_peak > 0 else np.empty(0, dtype=int))
    full_amplitude = source_amplitude(full_estimate, np.flatnonzero(active), baseline)
    deep_peak = float(full_amplitude[n_surf:].max(initial=0.))
    common = dict(mode="held_out_nested_partial_regression",
        score="(B-3)/(B-1) * partial improvement / plug-in baseline expectation - 1",
        gaussian_centering="mean zero for an independent fixed direction under temporally iid Gaussian noise",
        calibration_note="ranking statistic only until independent pure-surface calibration",
        support_energy_fraction=float(support_energy_fraction),
        support_note="wide nuisance support; 1% energy equals 10% amplitude and is not the metrics 10% energy support",
        surface_support=surface_support.tolist(), surface_support_count=int(surface_support.size),
        truth_used=False, reference_threshold=None)
    if deep_peak <= 0:
        return -1., dict(common, candidate_available=False, identifiable=False,
            nuisance_rank=None, selected_deep_index=None, selected_deep_local_index=None,
            conditional_gain_fraction=None, reason="full training fit has no nonzero deep candidate",
            confirmation_refitted_for_localization=False,
            confirmation_used_only_for_nested_test_statistic=False)
    deep_index = n_surf + int(np.argmax(full_amplitude[n_surf:]))

    weighted_gain = weights[:, None] * gain
    nuisance = weighted_gain[:, surface_support]
    u, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    tolerance = np.finfo(float).eps * max(nuisance.shape) * singular.max(initial=0.)
    nuisance_rank = int(np.count_nonzero(singular > tolerance))
    nuisance_basis = u[:, :nuisance_rank]
    deep_gain = weighted_gain[:, deep_index]
    conditional_gain = deep_gain - nuisance_basis @ (nuisance_basis.T @ deep_gain)
    conditional_norm = float(np.linalg.norm(conditional_gain))
    if conditional_norm <= np.finfo(float).eps * max(float(np.linalg.norm(deep_gain)), 1.):
        return -1., dict(common, candidate_available=True, identifiable=False,
            nuisance_rank=nuisance_rank, selected_deep_index=deep_index,
            selected_deep_local_index=int(deep_index - n_surf), conditional_gain_fraction=0.,
            reason="training-selected deep gain is contained in the cortical nuisance span",
            confirmation_refitted_for_localization=False,
            confirmation_used_only_for_nested_test_statistic=False)
    direction = conditional_gain / conditional_norm

    basis, basis_info = _smooth_temporal_basis(confirmation, baseline, active)
    centered = weights[:, None] * (
        confirmation - confirmation[:, baseline].mean(axis=1, keepdims=True))
    response = centered @ basis.T
    null_residual = response - nuisance_basis @ (nuisance_basis.T @ response)
    fitted_deep = direction @ null_residual
    improvement = float(np.sum(fitted_deep ** 2))
    null_loss = float(np.sum(null_residual ** 2))
    full_loss = float(null_loss - improvement)

    projected_baseline = direction @ centered[:, baseline]
    projected_variance = float(np.sum(projected_baseline ** 2) / (baseline.sum() - 1))
    mean_correction = float(np.sum(basis[:, active].sum(axis=1) ** 2) / baseline.sum())
    expected_noise = projected_variance * (len(basis) + mean_correction)
    if not np.isfinite(expected_noise) or expected_noise <= 0:
        raise ValueError("confirmation baseline has no finite conditional noise variance")
    # E[1 / S^2] = (B-1) / ((B-3) sigma^2) for Gaussian baseline
    # variance with B-1 degrees of freedom. This removes that finite-B bias.
    finite_baseline_correction = (baseline.sum() - 3) / (baseline.sum() - 1)
    score = finite_baseline_correction * improvement / expected_noise - 1.
    return float(score), dict(**basis_info, **common,
        finite_baseline_correction=float(finite_baseline_correction),
        candidate_available=True, identifiable=True,
        nuisance_rank=nuisance_rank, selected_deep_index=deep_index,
        selected_deep_local_index=int(deep_index - n_surf),
        conditional_gain_fraction=conditional_norm / float(np.linalg.norm(deep_gain)),
        null_loss=null_loss, full_loss=full_loss, loss_improvement=improvement,
        projected_baseline_variance=projected_variance,
        expected_null_improvement=expected_noise,
        baseline_mean_mode_correction=mean_correction,
        confirmation_refitted_for_localization=False,
        confirmation_used_only_for_nested_test_statistic=True,
        reason=None)
