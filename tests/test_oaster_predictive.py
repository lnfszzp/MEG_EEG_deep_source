"""Small independent-prediction and calibration checks; no full-grid runs."""
import inspect

import numpy as np
import pytest
from scipy import sparse

from candidates import oaster_predictive as inverse


def test_fit_uses_only_training_and_keeps_complete_null(monkeypatch):
    rng = np.random.default_rng(16)
    training = rng.normal(size=(3, 30))
    baseline = np.arange(30) < 15
    active = (np.arange(30) >= 20) & (np.arange(30) < 25)
    calls = []

    def capture(data, gain, n_surf, **kwargs):
        calls.append((data.copy(), gain.shape, n_surf, kwargs))
        return np.ones((gain.shape[1], data.shape[1])) * gain.shape[1], {"converged": True}

    monkeypatch.setattr(inverse, "reconstruct_evoked_oaster_v5_from_whitened", capture)
    null, full, info = inverse.fit_predictive_models(
        training, np.eye(3), 2, adjacency=sparse.eye(3), baseline=baseline,
        active=active, channel_weights=np.ones(3))
    assert len(calls) == 2 and all(np.array_equal(call[0], training) for call in calls)
    assert [call[1] for call in calls] == [(3, 2), (3, 3)]
    assert [call[3]["adjacency"].shape for call in calls] == [(2, 2), (3, 3)]
    assert np.all(null[:2] == 2) and not null[2:].any() and np.all(full == 3)
    assert calls[0][3]["epsilon_fraction"] == 1 and calls[0][3]["solver_kind"] == "irls"
    assert calls[0][3]["surface_reweight_floor"] == 0
    assert calls[0][3]["deep_reweight_floor"] == 0
    assert calls[0][3]["ridge_fraction"] == 0
    assert calls[0][3]["max_iter"] == 100 and calls[0][3]["smoothing_fraction"] == .1
    assert not info["confirmation_used_for_fit"]
    assert "confirmation" not in inspect.signature(inverse.fit_predictive_models).parameters
    for function in (inverse.fit_predictive_models, inverse.score_predictive_models, inverse.conformal_decision):
        assert not {"truth", "labels", "groups", "deep_index"} & set(inspect.signature(function).parameters)


def test_admm_defaults_and_structural_overrides_are_forwarded_without_mutation(monkeypatch):
    data = np.ones((3, 30))
    baseline = np.arange(30) < 15
    active = (np.arange(30) >= 20) & (np.arange(30) < 25)
    calls = []

    def capture(data, gain, n_surf, **kwargs):
        calls.append(kwargs)
        return np.zeros((gain.shape[1], data.shape[1])), {}

    monkeypatch.setattr(inverse, "reconstruct_evoked_oaster_v5_from_whitened", capture)
    settings = dict(solver_kind="admm", noise_multiplier=2., edge_fraction=.25,
                    mrf_strength=.3, calibration="global",
                    surface_reweight_floor=.2, deep_reweight_floor=.4,
                    ridge_fraction=.15)
    original = settings.copy()
    _, _, diagnostics = inverse.fit_predictive_models(
        data, np.eye(3), 2, adjacency=sparse.eye(3), baseline=baseline,
        active=active, channel_weights=np.ones(3), solver_settings=settings)
    assert settings == original and len(calls) == 2
    for options in calls:
        assert options["solver_kind"] == "admm" and options["temporal_mode"] == "smooth"
        assert options["outer_iterations"] == 20 and options["max_iter"] == 2000
        assert options["tolerance"] == .001 and options["outer_tolerance"] == .01
        assert options["noise_multiplier"] == 2 and options["edge_fraction"] == .25
        assert options["mrf_strength"] == .3 and options["calibration"] == "global"
        assert options["surface_reweight_floor"] == .2
        assert options["deep_reweight_floor"] == .4
        assert options["ridge_fraction"] == .15
        assert "smoothing_fraction" not in options and "epsilon_fraction" not in options
    assert diagnostics["solver_kind"] == "admm"
    assert diagnostics["structural_settings"]["ridge_fraction"] == .15
    for malformed in ({"solver_kind": "unknown"}, {"temporal_mode": "v4"}):
        with pytest.raises(ValueError):
            inverse.fit_predictive_models(data, np.eye(3), 2, adjacency=sparse.eye(3),
                baseline=baseline, active=active, channel_weights=np.ones(3), solver_settings=malformed)


def test_prediction_sign_basis_rotation_and_baseline_only_scale(monkeypatch):
    rng = np.random.default_rng(17)
    baseline = np.arange(200) < 90
    active = (np.arange(200) >= 110) & (np.arange(200) < 171)
    confirmation = rng.normal(size=(3, 200))
    gain = rng.normal(size=(3, 5))
    centered = confirmation - confirmation[:, baseline].mean(axis=1, keepdims=True)
    full = np.linalg.pinv(gain) @ centered
    null = np.zeros_like(full)
    options = dict(baseline=baseline, active=active, channel_weights=np.array([1., .7, 2.]))
    score, info = inverse.score_predictive_models(confirmation, gain, null, full, **options)
    _, modality_info = inverse.score_predictive_models(
        confirmation, gain, null, full, modality_sizes=(1, 2), **options)
    _, reweighted_modality_info = inverse.score_predictive_models(
        confirmation, gain, null, full, baseline=baseline, active=active,
        channel_weights=np.array([3., .1, .2]), modality_sizes=(1, 2))
    opposite, _ = inverse.score_predictive_models(confirmation, gain, full, null, **options)
    tied, _ = inverse.score_predictive_models(confirmation, gain, full, full, **options)
    assert score > 0 and opposite == -score and tied == 0
    expected_denominator = max(
        info["null_loss"] - info["expected_response_noise_energy"],
        .05 * info["expected_response_noise_energy"])
    assert info["excess_normalization_denominator"] == pytest.approx(expected_denominator)
    assert info["excess_fraction_score"] == pytest.approx(
        info["loss_improvement"] / expected_denominator)
    assert np.isclose(info["baseline_mean_mode_correction"], active.sum() / baseline.sum())
    assert len(modality_info["modality_noise_scores"]) == 2
    assert modality_info["conjunctive_modality_score"] == min(
        modality_info["modality_noise_scores"])
    positive_scores = np.maximum(modality_info["modality_noise_scores"], 0)
    consensus = (min(modality_info["modality_noise_scores"])
                 + np.sqrt(positive_scores[0] * positive_scores[1]))
    assert modality_info["snr_blind_consensus_score"] == pytest.approx(
        consensus / (1 + max(modality_info["modality_response_excess_ratios"])) ** .25)
    response_scale = 1 + np.asarray(modality_info["modality_response_excess_ratios"])
    balance = response_scale.min() / response_scale.max()
    eeg_weight = .5 + .25 * balance
    normalized = np.asarray(modality_info["modality_noise_scores"]) / np.sqrt(response_scale)
    assert modality_info["snr_blind_reliability_score"] == pytest.approx(
        eeg_weight * normalized[0] + (1 - eeg_weight) * normalized[1])
    assert modality_info["snr_blind_reliability_eeg_weight"] == pytest.approx(eeg_weight)
    assert modality_info["snr_blind_reliability_balance"] == pytest.approx(balance)
    assert modality_info["modality_scoring_weights"] == "none_after_whitening"
    assert np.allclose(modality_info["modality_noise_scores"],
                       reweighted_modality_info["modality_noise_scores"])
    basis = inverse._smooth_temporal_basis(confirmation, baseline, active)[0]
    source_modes = (null @ basis.T, full @ basis.T)
    for index, block in enumerate((slice(0, 1), slice(1, 3))):
        response = centered[block] @ basis.T
        manual_null = np.sum((response - gain[block] @ source_modes[0]) ** 2)
        manual_full = np.sum((response - gain[block] @ source_modes[1]) ** 2)
        manual_noise = (np.sum(centered[block, baseline] ** 2) / (baseline.sum() - 1)
                        * (len(basis) + active.sum() / baseline.sum()))
        assert modality_info["modality_null_losses"][index] == pytest.approx(manual_null)
        assert modality_info["modality_full_losses"][index] == pytest.approx(manual_full)
        assert modality_info["modality_expected_response_noise_energy"][index] == pytest.approx(manual_noise)
        manual_response_energy = np.sum(response ** 2)
        assert modality_info["modality_response_energy"][index] == pytest.approx(
            manual_response_energy)
        assert modality_info["modality_response_excess_ratios"][index] == pytest.approx(
            max(manual_response_energy / manual_noise - 1, 0))
        assert modality_info["modality_noise_scores"][index] == pytest.approx(
            (manual_null - manual_full) / manual_noise)
    changed = confirmation.copy()
    changed[:, active] *= 8
    _, changed_info = inverse.score_predictive_models(changed, gain, null, full, **options)
    assert changed_info["expected_response_noise_energy"] == info["expected_response_noise_energy"]
    original_basis = inverse._smooth_temporal_basis
    rotation = np.linalg.qr(rng.normal(size=(10, 10)))[0]

    def rotated_basis(*args):
        basis, metadata = original_basis(*args)
        return rotation @ basis, metadata

    monkeypatch.setattr(inverse, "_smooth_temporal_basis", rotated_basis)
    rotated, rotated_info = inverse.score_predictive_models(confirmation, gain, null, full, **options)
    assert np.isclose(rotated, score, rtol=1e-12)
    assert np.isclose(rotated_info["expected_response_noise_energy"], info["expected_response_noise_energy"])


def test_conformal_ties_negative_gain_and_resolution():
    assert inverse.conformal_decision(1., np.zeros(19))["deep_present"]
    assert inverse.conformal_decision(1., np.zeros(19))["p_value"] == .05
    tied = inverse.conformal_decision(1., np.ones(19))
    assert tied["p_value"] == 1 and not tied["deep_present"]
    negative = inverse.conformal_decision(-1., np.full(19, -2.))
    assert negative["p_value"] == .05 and not negative["deep_present"]
    pooled = np.arange(57, dtype=float)
    assert not inverse.conformal_decision(55., pooled)["deep_present"]
    assert inverse.conformal_decision(np.nextafter(55., np.inf), pooled)["deep_present"]
    for null in ([], np.zeros(18), [np.nan] * 19, np.zeros((19, 1))):
        with pytest.raises(ValueError):
            inverse.conformal_decision(1., null)
    with pytest.raises(ValueError):
        inverse.conformal_decision(1., np.zeros(19), alpha=0.)


def test_post_gate_location_sums_leave_one_out_modality_evidence():
    n_times = 30
    baseline = np.arange(n_times) < 12
    active = (np.arange(n_times) >= 15) & (np.arange(n_times) < 20)
    basis = inverse._smooth_temporal_basis(np.zeros((2, n_times)), baseline, active)[0]
    gain = np.array([[.1, .4, 1.], [2., .4, 1.]])
    full = np.zeros((3, n_times))
    full[:2] = basis[0]
    confirmation = np.zeros((2, n_times))
    confirmation[:, baseline] = np.array([[1., -1.] * 6, [1., -1.] * 6])
    confirmation += (gain[:, 0] + .75 * gain[:, 1])[:, None] * basis[0]
    modality_scores = []
    for source in range(3):
        without = full.copy()
        without[source] = 0
        _, info = inverse.score_predictive_models(
            confirmation, gain, without, full, baseline=baseline, active=active,
            channel_weights=np.ones(2), modality_sizes=(1, 1))
        modality_scores.append(info["modality_noise_scores"])
    modality_scores = np.asarray(modality_scores)
    assert np.argmax(modality_scores.min(axis=1)) == 1
    assert np.argmax(modality_scores.sum(axis=1)) == 0
    assert not np.linalg.norm(full[2] @ basis.T) and not modality_scores[2].any()


def test_small_real_inverse_accepts_surface_only_adjacency_and_returns_diagnostics():
    rng = np.random.default_rng(18)
    data = rng.normal(size=(4, 80))
    gain = rng.normal(size=(4, 6))
    baseline = np.arange(80) < 35
    active = (np.arange(80) >= 45) & (np.arange(80) < 66)
    adjacency = sparse.diags((np.ones(5), np.ones(5)), (-1, 1), shape=(6, 6))
    null, full, diagnostics = inverse.fit_predictive_models(
        data, gain, 4, adjacency=adjacency, baseline=baseline, active=active,
        channel_weights=np.ones(4), solver_settings={"max_iter": 3})
    assert null.shape == full.shape == (6, 80)
    assert not null[4:].any() and np.isfinite(full).all()
    for name in ("null_model", "full_model"):
        solver = diagnostics[name]["windows"][0]["solver"]
        assert solver["iterations"] <= 3 and isinstance(solver["converged"], bool)


def test_malformed_observations_are_rejected_before_inverse():
    baseline = np.arange(30) < 15
    active = (np.arange(30) >= 20) & (np.arange(30) < 25)
    options = dict(adjacency=sparse.eye(3), baseline=baseline, active=active, channel_weights=np.ones(3))
    for data, gain, n_surf in ((np.zeros(30), np.eye(3), 2),
                                (np.zeros((3, 30)), np.eye(2), 1),
                                (np.full((3, 30), np.nan), np.eye(3), 2),
                                (np.zeros((3, 30)), np.eye(3), 3)):
        with pytest.raises(ValueError):
            inverse.fit_predictive_models(data, gain, n_surf, **options)
    with pytest.raises(ValueError):
        inverse.fit_predictive_models(np.zeros((3, 30)), np.eye(3), 2,
                                     **{**options, "active": baseline})
    with pytest.raises(ValueError):
        inverse.fit_predictive_models(np.zeros((3, 30)), np.eye(3), 2,
                                     **{**options, "channel_weights": np.zeros(3)})
    with pytest.raises(ValueError):
        inverse.score_predictive_models(np.zeros((3, 30)), np.eye(3), np.zeros((3, 30)),
            np.zeros((3, 30)), baseline=baseline, active=active, channel_weights=np.ones(3))
    finite_confirmation = np.arange(90, dtype=float).reshape(3, 30)
    for sizes in ((), (1, 1), (1., 2.), (1, -1, 3)):
        with pytest.raises(ValueError):
            inverse.score_predictive_models(
                finite_confirmation, np.eye(3), np.zeros((3, 30)), np.zeros((3, 30)),
                baseline=baseline, active=active, channel_weights=np.ones(3),
                modality_sizes=sizes)
    for scores, excess in (([1.], [0.]), ([1., np.nan], [0., 0.]),
                           ([1., 2.], [0., -1.])):
        with pytest.raises(ValueError):
            inverse.snr_blind_reliability_score(scores, excess)
