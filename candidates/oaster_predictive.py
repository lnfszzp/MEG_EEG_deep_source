"""Experimental ERP deep-presence evidence from independent confirmation data.

Training fits a complete surface-only model and a joint model. Prediction gain
is an empirical score, not proof of a deep source: repeatable cortical fitting
errors can also improve it. Independent surface-only calibration must therefore
cover that leakage. No per-SNR or real-data error guarantee is implied.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from candidates.oaster_balanced import (
    _smooth_temporal_basis,
    reconstruct_evoked_oaster_v5_from_whitened,
)


def _validate_observations(data, gain, baseline, active, channel_weights):
    data, gain = np.asarray(data, float), np.asarray(gain, float)
    baseline, active = np.asarray(baseline), np.asarray(active)
    weights = np.asarray(channel_weights, float)
    if (data.ndim != 2 or gain.ndim != 2 or not data.size or not gain.size
            or data.shape[0] != gain.shape[0]
            or baseline.dtype != np.bool_ or active.dtype != np.bool_
            or baseline.shape != (data.shape[1],) or active.shape != baseline.shape
            or baseline.sum() < 2 or not active.any() or np.any(baseline & active)
            or active.sum() > baseline.sum()
            or any(np.any(np.diff(np.flatnonzero(mask)) != 1) for mask in (baseline, active))
            or weights.shape != (data.shape[0],) or np.any(weights < 0)
            or not np.any(weights > 0)
            or not all(np.isfinite(value).all() for value in (data, gain, weights))):
        raise ValueError("require finite sensor matrices, nonnegative nonzero weights, and disjoint continuous boolean masks")
    return data, gain, baseline, active, weights


def fit_predictive_models(training, gain, n_surf, *, adjacency, baseline, active,
                          channel_weights, solver_settings=None):
    """Fit H0 (all cortex) and H1 (cortex + deep) using training data only.

    Return complete source matrices ``(null, full, diagnostics)``. When deep
    evidence is rejected, use ``null``; zeroing just the deep rows of ``full``
    would retain the surface bias of the rejected joint decomposition.
    """
    training, gain, baseline, active, weights = _validate_observations(
        training, gain, baseline, active, channel_weights)
    if not isinstance(n_surf, (int, np.integer)) or not 0 < n_surf < gain.shape[1]:
        raise ValueError("n_surf must split nonempty cortical and deep grids")
    graph = sparse.csr_matrix(adjacency)
    if graph.shape != (gain.shape[1], gain.shape[1]) or not np.isfinite(graph.data).all():
        raise ValueError("adjacency must match the full source grid")
    # Development alternatives; freeze the selected objective before calibration.
    overrides = dict(solver_settings or {})
    solver_kind = overrides.pop("solver_kind", "irls")
    if solver_kind not in {"irls", "admm"}:
        raise ValueError("solver_kind must be irls or admm")
    if overrides.pop("temporal_mode", "smooth") != "smooth":
        raise ValueError("predictive models require the fixed smooth temporal basis")
    structural = {name: overrides.pop(name, value) for name, value in
                  dict(noise_multiplier=1., edge_fraction=.5,
                       mrf_strength=.5, calibration="layer",
                       surface_reweight_floor=0., deep_reweight_floor=0.,
                       ridge_fraction=0.).items()}
    settings = (dict(max_iter=100, tolerance=1e-5, epsilon_fraction=1.0,
                     smoothing_fraction=.1) if solver_kind == "irls" else
                dict(outer_iterations=20, max_iter=2000, tolerance=.001,
                     outer_tolerance=.01))
    settings.update(overrides)
    options = dict(baseline=baseline, active_windows=(active,),
                   window_channel_weights=(weights,), require_one=False,
                   temporal_mode="smooth", solver_kind=solver_kind,
                   **structural, **settings)
    cortical, null_info = reconstruct_evoked_oaster_v5_from_whitened(
        training, gain[:, :n_surf], n_surf,
        adjacency=graph[:n_surf, :n_surf], **options)
    full, full_info = reconstruct_evoked_oaster_v5_from_whitened(
        training, gain, n_surf, adjacency=graph, **options)
    null = np.zeros_like(full)
    null[:n_surf] = cortical
    return null, full, dict(mode="independent_confirmation_joint_vs_surface",
        solver_kind=solver_kind, structural_settings=structural,
        solver_settings=settings, null_model=null_info, full_model=full_info,
        confirmation_used_for_fit=False, spatial_templates_used=False)


def score_predictive_models(confirmation, gain, null_estimate, full_estimate, *,
                            baseline, active, channel_weights):
    """Return signed held-out loss improvement in the fixed smooth ERP basis.

    Confirmation is never refit. Its baseline alone supplies the scalar noise
    normalization. This finite-baseline estimate assumes temporal white noise;
    independent null calibration, not a Gaussian reference law, sets decisions.
    """
    confirmation, gain, baseline, active, weights = _validate_observations(
        confirmation, gain, baseline, active, channel_weights)
    null_estimate, full_estimate = np.asarray(null_estimate, float), np.asarray(full_estimate, float)
    expected_shape = (gain.shape[1], confirmation.shape[1])
    if any(value.shape != expected_shape or not np.isfinite(value).all()
           for value in (null_estimate, full_estimate)):
        raise ValueError("both training source estimates must match the complete grid and epoch")
    basis, basis_info = _smooth_temporal_basis(confirmation, baseline, active)
    centered = confirmation - confirmation[:, baseline].mean(axis=1, keepdims=True)
    weighted = weights[:, None] * centered
    response = weighted @ basis.T
    weighted_gain = weights[:, None] * gain
    null_prediction = weighted_gain @ (null_estimate @ basis.T)
    full_prediction = weighted_gain @ (full_estimate @ basis.T)
    null_loss = float(np.sum((response - null_prediction) ** 2))
    full_loss = float(np.sum((response - full_prediction) ** 2))
    variance_sum = float(np.sum(weighted[:, baseline] ** 2) / (baseline.sum() - 1))
    # Subtracting the independent baseline mean adds variance to the DC mode.
    mean_correction = float(np.sum(basis[:, active].sum(axis=1) ** 2) / baseline.sum())
    expected_noise = variance_sum * (len(basis) + mean_correction)
    if not np.isfinite(expected_noise) or expected_noise <= 0:
        raise ValueError("confirmation baseline must have positive finite weighted noise variance")
    score = (null_loss - full_loss) / expected_noise
    excess_floor = .05 * expected_noise
    excess_denominator = max(null_loss - expected_noise, excess_floor)
    excess_fraction = (null_loss - full_loss) / excess_denominator
    if not np.isfinite(score) or not np.isfinite(excess_fraction):
        raise ValueError("predictive score is nonfinite")
    return float(score), dict(**basis_info, null_loss=null_loss, full_loss=full_loss,
        loss_improvement=null_loss - full_loss, expected_response_noise_energy=expected_noise,
        null_excess_loss=null_loss - expected_noise,
        excess_noise_floor_fraction=.05, excess_normalization_denominator=excess_denominator,
        excess_fraction_score=float(excess_fraction),
        excess_normalization="(null_loss-full_loss)/max(null_loss-expected_noise, 0.05*expected_noise)",
        baseline_variance_sum=variance_sum, baseline_mean_mode_correction=mean_correction,
        normalization="confirmation baseline sample variance; temporal-white expectation",
        confirmation_refitted=False, score_clipped=False)


def conformal_decision(score, null_scores, alpha=.05):
    """Upper-tail rank calibration, conservative for ties, with positive gain.

    Null cases must be independent of development and exchangeable with the
    target pure-surface distribution. Freeze scoring before this calibration;
    tuning on calibration or evaluation labels invalidates that interpretation.
    """
    score, alpha = float(score), float(alpha)
    null_scores = np.asarray(null_scores, float)
    if (not np.isfinite(score) or not np.isfinite(alpha) or not 0 < alpha < 1
            or null_scores.ndim != 1 or len(null_scores) < 19
            or not np.isfinite(null_scores).all()):
        raise ValueError("require finite score, 0 < alpha < 1 and at least 19 finite independent null scores")
    p_value = float((1 + np.count_nonzero(null_scores >= score)) / (len(null_scores) + 1))
    return dict(p_value=p_value, deep_present=bool(p_value <= alpha and score > 0),
                alpha=alpha, n_calibration=int(len(null_scores)), score=score,
                minimum_p_value=1. / (len(null_scores) + 1), positive_gain_required=True)
