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


def snr_blind_consensus_score(modality_scores, modality_response_excess_ratios):
    """Fuse EEG/MEG gains after blind active-to-baseline energy normalization."""
    scores = np.asarray(modality_scores, float)
    response_excess = np.asarray(modality_response_excess_ratios, float)
    if (scores.shape != (2,) or response_excess.shape != (2,)
            or not np.isfinite(scores).all() or not np.isfinite(response_excess).all()
            or np.any(response_excess < 0)):
        raise ValueError("SNR-blind consensus requires two finite EEG/MEG scores and nonnegative excess ratios")
    positive = np.maximum(scores, 0.)
    consensus = float(scores.min() + np.sqrt(positive[0]) * np.sqrt(positive[1]))
    score = consensus / (1. + float(response_excess.max())) ** .25
    if not np.isfinite(score):
        raise ValueError("SNR-blind consensus score is nonfinite")
    return float(score)


def snr_blind_reliability_score(modality_scores, modality_response_excess_ratios):
    """Fuse energy-normalized EEG/MEG gains with blind reliability balance."""
    scores = np.asarray(modality_scores, float)
    response_excess = np.asarray(modality_response_excess_ratios, float)
    if (scores.shape != (2,) or response_excess.shape != (2,)
            or not np.isfinite(scores).all() or not np.isfinite(response_excess).all()
            or np.any(response_excess < 0)):
        raise ValueError("SNR-blind reliability requires two finite EEG/MEG scores and nonnegative excess ratios")
    response_scale = 1. + response_excess
    normalized = scores / np.sqrt(response_scale)
    balance = float(response_scale.min() / response_scale.max())
    eeg_weight = .5 + .25 * balance
    score = float(eeg_weight * normalized[0] + (1. - eeg_weight) * normalized[1])
    if not np.isfinite(score):
        raise ValueError("SNR-blind reliability score is nonfinite")
    return score, float(eeg_weight), balance


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
                       ridge_fraction=0., deep_alias_penalty=False).items()}
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
                            baseline, active, channel_weights, modality_sizes=None):
    """Return signed held-out loss improvement in the fixed smooth ERP basis.

    Confirmation is never refit. Its baseline alone supplies the scalar noise
    normalization. This finite-baseline estimate assumes temporal white noise;
    independent null calibration, not a Gaussian reference law, sets decisions.
    When modality sizes are supplied, each whitened block is also scored before
    applying training-derived modality weights and their minimum is reported.
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

    raw_sizes = np.array([confirmation.shape[0]]) if modality_sizes is None else np.asarray(modality_sizes)
    if (raw_sizes.ndim != 1 or not raw_sizes.size or raw_sizes.dtype.kind not in "iu"
            or np.any(raw_sizes <= 0) or raw_sizes.sum() != confirmation.shape[0]):
        raise ValueError("modality_sizes must be positive integer blocks covering all channels")
    source_modes = null_estimate @ basis.T, full_estimate @ basis.T
    modality_scores, modality_null_losses, modality_full_losses, modality_noise = [], [], [], []
    modality_response_energy, modality_response_excess = [], []
    start = 0
    for size in raw_sizes:
        block = slice(start, start + int(size))
        raw_response = centered[block] @ basis.T
        raw_gain = gain[block]
        block_null = float(np.sum((raw_response - raw_gain @ source_modes[0]) ** 2))
        block_full = float(np.sum((raw_response - raw_gain @ source_modes[1]) ** 2))
        block_variance = float(np.sum(centered[block, baseline] ** 2) / (baseline.sum() - 1))
        block_noise = block_variance * (len(basis) + mean_correction)
        if not np.isfinite(block_noise) or block_noise <= 0:
            raise ValueError("every modality must have positive finite confirmation baseline variance")
        modality_null_losses.append(block_null)
        modality_full_losses.append(block_full)
        modality_noise.append(block_noise)
        modality_scores.append((block_null - block_full) / block_noise)
        response_energy = float(np.sum(raw_response ** 2))
        modality_response_energy.append(response_energy)
        modality_response_excess.append(max(response_energy / block_noise - 1., 0.))
        start += int(size)
    conjunctive_score = float(min(modality_scores))
    snr_blind_score = (snr_blind_consensus_score(modality_scores, modality_response_excess)
                       if len(modality_scores) == 2 else None)
    reliability_score, reliability_eeg_weight, reliability_balance = (
        snr_blind_reliability_score(modality_scores, modality_response_excess)
        if len(modality_scores) == 2 else (None, None, None))
    return float(score), dict(**basis_info, null_loss=null_loss, full_loss=full_loss,
        loss_improvement=null_loss - full_loss, expected_response_noise_energy=expected_noise,
        null_excess_loss=null_loss - expected_noise,
        excess_noise_floor_fraction=.05, excess_normalization_denominator=excess_denominator,
        excess_fraction_score=float(excess_fraction),
        excess_normalization="(null_loss-full_loss)/max(null_loss-expected_noise, 0.05*expected_noise)",
        baseline_variance_sum=variance_sum, baseline_mean_mode_correction=mean_correction,
        normalization="confirmation baseline sample variance; temporal-white expectation",
        modality_sizes=raw_sizes.astype(int).tolist(),
        modality_null_losses=modality_null_losses, modality_full_losses=modality_full_losses,
        modality_expected_response_noise_energy=modality_noise,
        modality_response_energy=modality_response_energy,
        modality_response_excess_ratios=modality_response_excess,
        modality_noise_scores=modality_scores, conjunctive_modality_score=conjunctive_score,
        conjunctive_rule="min modality held-out improvement / modality baseline-noise expectation",
        snr_blind_consensus_score=snr_blind_score,
        snr_blind_consensus_rule=(
            "[min(g_EEG,g_MEG)+sqrt(max(g_EEG,0)*max(g_MEG,0))] / "
            "[1+max(active_response_energy/noise_expectation-1,0)]^0.25"
            if snr_blind_score is not None else None),
        snr_blind_reliability_score=reliability_score,
        snr_blind_reliability_eeg_weight=reliability_eeg_weight,
        snr_blind_reliability_balance=reliability_balance,
        snr_blind_reliability_rule=(
            "h_m=g_m/sqrt(1+q_m); b=min(1+q_m)/max(1+q_m); "
            "w_EEG=0.5+0.25*b; score=w_EEG*h_EEG+(1-w_EEG)*h_MEG"
            if reliability_score is not None else None),
        modality_scoring_weights="none_after_whitening",
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
