import numpy as np
import pytest

from candidates.oaster_partial_confirmation import (
    score_low_rank_deep_scan, score_partial_deep_presence)


def test_low_rank_scan_survives_full_rank_surface_support():
    rng = np.random.default_rng(11)
    n_sensors, n_surf, n_times = 6, 6, 80
    baseline = np.zeros(n_times, bool)
    baseline[:40] = True
    active = np.zeros(n_times, bool)
    active[45:75] = True
    surface_gain = np.eye(n_sensors)
    deep_gain = np.c_[np.array([1., -1., 0., 0., 0., 0.]) / np.sqrt(2),
                      np.array([0., 0., 1., -1., 0., 0.]) / np.sqrt(2)]
    gain = np.c_[surface_gain, deep_gain]
    null = np.zeros((gain.shape[1], n_times))
    null[:n_surf, active] = 1.
    full = null.copy()
    full[n_surf, active] = 1.
    confirmation = .05 * rng.normal(size=(n_sensors, n_times))
    confirmation[:, active] += np.ones((n_sensors, 1)) + deep_gain[:, :1]

    scores, info = score_low_rank_deep_scan(
        confirmation, gain, null, n_surf, baseline=baseline, active=active,
        channel_weights=np.ones(n_sensors))
    old_score, old_info = score_partial_deep_presence(
        confirmation, gain, null, full, n_surf, baseline=baseline,
        active=active, channel_weights=np.ones(n_sensors))

    assert np.linalg.matrix_rank(surface_gain) == n_sensors
    assert np.linalg.norm(surface_gain @ np.linalg.lstsq(
        surface_gain, deep_gain[:, 0], rcond=None)[0] - deep_gain[:, 0]) < 1e-12
    assert old_info["surface_support"] == list(range(n_surf))
    assert old_info["nuisance_rank"] == n_sensors
    assert old_score == -1 and not old_info["identifiable"]
    assert scores.shape == (2,) and np.isfinite(scores).all()
    assert scores[0] > 10 and scores[0] > scores[1]
    assert info["nuisance_rank"] == 1
    assert info["nuisance_rank"] <= info["nuisance_rank_bound"]
    assert info["selected_deep_local_index"] == 0
    assert info["truth_used"] is False and info["reference_threshold"] is None
    assert info["finite_baseline_correction"] == pytest.approx(37 / 39)

    noise_scores, noise_info = score_low_rank_deep_scan(
        .05 * rng.normal(size=confirmation.shape), gain, null, n_surf,
        baseline=baseline, active=active, channel_weights=np.ones(n_sensors))
    assert np.isfinite(noise_scores).all()
    assert noise_info["nuisance_rank"] == info["nuisance_rank"]
    assert noise_info["conditional_gain_fraction"] == pytest.approx(
        info["conditional_gain_fraction"])


def test_low_rank_scan_rejects_exact_and_near_numerical_aliases():
    rng = np.random.default_rng(12)
    n_sensors, n_surf, n_times = 6, 6, 80
    baseline = np.zeros(n_times, bool)
    baseline[:40] = True
    active = np.zeros(n_times, bool)
    active[45:75] = True
    surface_gain = np.eye(n_sensors)
    exact = surface_gain[:, 0] + surface_gain[:, 1]
    near = exact + np.finfo(float).eps * surface_gain[:, 2]
    gain = np.c_[surface_gain, exact, near]
    null = np.zeros((gain.shape[1], n_times))
    null[0, active] = 1.
    null[1, active] = np.linspace(-1., 1., active.sum())

    scores, info = score_low_rank_deep_scan(
        .05 * rng.normal(size=(n_sensors, n_times)), gain, null, n_surf,
        baseline=baseline, active=active, channel_weights=np.ones(n_sensors))

    assert info["nuisance_rank"] == 2
    assert scores.tolist() == [-1., -1.]
    assert info["identifiable"] == [False, False]
    assert info["identifiable_count"] == 0
    assert info["selected_deep_local_index"] is None
    assert info["selected_deep_index"] is None
    assert info["conditional_gain_fraction_threshold"] == pytest.approx(
        np.finfo(float).eps * max(n_sensors, info["temporal_rank"]))
    assert (max(info["conditional_gain_fraction"])
            <= info["conditional_gain_fraction_threshold"])
    assert info["conditional_gain_fraction"][1] > 0
    assert "nuisance SVD tolerance" in info["identifiability_rule"]


def test_partial_score_uses_training_support_and_held_out_deep_direction():
    rng = np.random.default_rng(7)
    n_times = 80
    baseline = np.zeros(n_times, bool)
    baseline[:40] = True
    active = np.zeros(n_times, bool)
    active[45:75] = True
    gain = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
    null = np.zeros((3, n_times))
    full = np.zeros_like(null)
    null[0, active] = 1.
    null[1, active] = .05
    full[:] = null
    full[2, active] = 1.
    confirmation = .1 * rng.normal(size=(3, n_times))
    confirmation[0, active] += 1.
    confirmation[:, active] += gain[:, 2, None]

    score, detail = score_partial_deep_presence(
        confirmation, gain, null, full, 2, baseline=baseline, active=active,
        channel_weights=np.ones(3))

    assert score > 10
    assert detail["surface_support"] == [0]
    assert detail["selected_deep_index"] == 2
    assert detail["truth_used"] is False
    assert detail["reference_threshold"] is None
    assert detail["support_energy_fraction"] == .01
    assert detail["finite_baseline_correction"] == pytest.approx(37 / 39)

    noise_only = .1 * rng.normal(size=confirmation.shape)
    noise_score, noise_detail = score_partial_deep_presence(
        noise_only, gain, null, full, 2, baseline=baseline, active=active,
        channel_weights=np.ones(3))
    assert np.isfinite(noise_score)
    assert noise_detail["surface_support"] == detail["surface_support"]
    assert noise_detail["selected_deep_index"] == detail["selected_deep_index"]


def test_partial_score_validates_training_fits():
    baseline = np.r_[np.ones(20, bool), np.zeros(30, bool)]
    active = np.r_[np.zeros(25, bool), np.ones(20, bool), np.zeros(5, bool)]
    with pytest.raises(ValueError):
        score_partial_deep_presence(
            np.zeros((2, 50)), np.ones((2, 3)), np.zeros((3, 50)),
            np.zeros((3, 50)), 3, baseline=baseline, active=active,
            channel_weights=np.ones(2))


def test_partial_score_handles_empty_training_support_and_candidate():
    rng = np.random.default_rng(8)
    baseline = np.r_[np.ones(30, bool), np.zeros(40, bool)]
    active = np.r_[np.zeros(35, bool), np.ones(30, bool), np.zeros(5, bool)]
    gain = np.eye(3)
    confirmation = rng.normal(size=(3, 70))
    empty = np.zeros((3, 70))
    surface = empty.copy()
    surface[0, active] = 1.
    deep = surface.copy()
    deep[2, active] = 1.
    options = dict(baseline=baseline, active=active, channel_weights=np.ones(3))

    score, detail = score_partial_deep_presence(confirmation, gain, empty, deep, 2, **options)
    assert np.isfinite(score) and detail["surface_support"] == []
    assert detail["candidate_available"] and detail["identifiable"]

    score, detail = score_partial_deep_presence(confirmation, gain, surface, surface, 2, **options)
    assert score == -1 and detail["selected_deep_index"] is None
    assert not detail["candidate_available"] and not detail["identifiable"]

    aliased_gain = np.array([[1., 0., 1.], [0., 1., 0.], [0., 0., 0.]])
    score, detail = score_partial_deep_presence(
        confirmation, aliased_gain, surface, deep, 2, **options)
    assert score == -1 and detail["candidate_available"] and not detail["identifiable"]

    short_baseline = np.r_[np.ones(3, bool), np.zeros(7, bool)]
    short_active = np.r_[np.zeros(5, bool), np.ones(3, bool), np.zeros(2, bool)]
    with pytest.raises(ValueError, match="at least four baseline"):
        score_partial_deep_presence(
            confirmation[:, :10], gain, deep[:, :10], deep[:, :10], 2,
            baseline=short_baseline, active=short_active, channel_weights=np.ones(3))
