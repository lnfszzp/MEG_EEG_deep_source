"""The cortical noise maximum must not set a deep-source penalty in v5."""
import numpy as np
from scipy import sparse

from candidates import oaster_balanced as inverse


def test_smooth_basis_is_orthonormal_and_not_selected_by_signal_amplitude():
    baseline = np.arange(200) < 90
    active = (np.arange(200) >= 110) & (np.arange(200) < 171)
    rng = np.random.default_rng(10)
    basis, info = inverse._smooth_temporal_basis(rng.normal(size=(4, 200)), baseline, active)
    changed, _ = inverse._smooth_temporal_basis(1000 * rng.normal(size=(4, 200)), baseline, active)
    assert np.array_equal(basis, changed)
    assert info["temporal_rank"] == 10
    assert np.allclose(basis @ basis.T, np.eye(10), atol=1e-14)
    assert not basis[:, ~active].any()


def test_layer_calibration_and_global_ablation(monkeypatch):
    rng = np.random.default_rng(27)
    data = rng.normal(size=(3, 30))
    data[0] *= 20
    baseline = np.arange(30) < 15
    active = (np.arange(30) >= 20) & (np.arange(30) < 25)
    basis = np.zeros((1, 30))
    basis[0, active] = np.array([1., 2., 3., 2., 1.]) / np.sqrt(19)
    monkeypatch.setattr(inverse, "_evoked_temporal_basis", lambda *a: (basis, {"temporal_rank": 1}))
    penalties = []

    def capture(response, gain, incidence, **kwargs):
        penalties.append(kwargs["source_penalty"].copy())
        return np.zeros((gain.shape[1], response.shape[1])), {"converged": True}

    monkeypatch.setattr(inverse, "solve_reweighted_graph_v5", capture)
    settings = dict(adjacency=sparse.eye(3), baseline=baseline,
                    active_windows=(active,), mrf_strength=0, temporal_mode="v4")
    _, local = inverse.reconstruct_evoked_oaster_v5_from_whitened(data, np.eye(3), 2, **settings)
    _, global_fit = inverse.reconstruct_evoked_oaster_v5_from_whitened(
        data, np.eye(3), 2, calibration="global", **settings)
    assert np.array_equal(penalties[0][:2], penalties[1][:2])
    assert penalties[0][-1] < penalties[1][-1] / 5
    changed = data.copy()
    changed[0, baseline] *= 3
    inverse.reconstruct_evoked_oaster_v5_from_whitened(changed, np.eye(3), 2, **settings)
    assert penalties[2][-1] == penalties[0][-1]
    assert local["calibration"] == "layer" and global_fit["calibration"] == "global"


def test_no_temporal_evidence_returns_exact_zero_without_solver(monkeypatch):
    baseline = np.arange(30) < 15
    active = (np.arange(30) >= 20) & (np.arange(30) < 25)
    monkeypatch.setattr(inverse, "_evoked_temporal_basis", lambda *a: (np.empty((0, 30)), {"temporal_rank": 0}))

    def forbidden(*args, **kwargs):
        raise AssertionError("no mode must not trigger a spatial solve")

    monkeypatch.setattr(inverse, "solve_reweighted_graph_v5", forbidden)
    source, diagnostic = inverse.reconstruct_evoked_oaster_v5_from_whitened(
        np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
        baseline=baseline, active_windows=(active,), temporal_mode="v4")
    assert not source.any()
    assert diagnostic["windows"][0]["solver"] is None


def test_reweight_floors_are_applied_to_their_admm_layers(monkeypatch):
    baseline = np.arange(30) < 15
    active = (np.arange(30) >= 20) & (np.arange(30) < 25)
    captured = {}

    def capture(response, gain, incidence, **kwargs):
        captured.update(kwargs)
        return np.zeros((gain.shape[1], response.shape[1])), {"converged": True}

    monkeypatch.setattr(inverse, "solve_reweighted_graph_v5", capture)
    source, info = inverse.reconstruct_evoked_oaster_v5_from_whitened(
        np.arange(90, dtype=float).reshape(3, 30), np.eye(3), 2,
        adjacency=sparse.eye(3), baseline=baseline, active_windows=(active,),
        mrf_strength=0, surface_reweight_floor=.2, deep_reweight_floor=.4,
        ridge_fraction=.25, edge_penalty_mode="elementwise",
        surface_penalty_multiplier=np.array([.5, .75]))
    assert source.shape == (3, 30)
    assert np.array_equal(captured["amplitude_weight_floor"], [.2, .2, .4])
    assert info["surface_reweight_floor"] == .2
    assert info["deep_reweight_floor"] == .4
    assert captured["ridge_penalty"] == .25
    assert captured["edge_penalty_mode"] == "elementwise"
    assert info["edge_penalty_mode"] == "elementwise"
    assert info["surface_penalty_multiplier_count"] == 2
    assert info["surface_penalty_multiplier_range"] == [.5, .75]
    assert np.isclose(captured["source_penalty"][0] / captured["source_penalty"][1], 2 / 3)
    assert info["ridge_fraction"] == .25
    assert info["windows"][0]["ridge_scale"] == 1.
    for name in ("surface_reweight_floor", "deep_reweight_floor"):
        for invalid in (-.1, 1.1):
            with np.testing.assert_raises(ValueError):
                inverse.reconstruct_evoked_oaster_v5_from_whitened(
                    np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
                    baseline=baseline, active_windows=(active,), **{name: invalid})
    for invalid in (-.1, np.nan):
        with np.testing.assert_raises(ValueError):
            inverse.reconstruct_evoked_oaster_v5_from_whitened(
                np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
                baseline=baseline, active_windows=(active,), ridge_fraction=invalid)
    with np.testing.assert_raises(ValueError):
        inverse.reconstruct_evoked_oaster_v5_from_whitened(
            np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
            baseline=baseline, active_windows=(active,), ridge_fraction=.1,
            solver_kind="irls")
    with np.testing.assert_raises(ValueError):
        inverse.reconstruct_evoked_oaster_v5_from_whitened(
            np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
            baseline=baseline, active_windows=(active,),
            edge_penalty_mode="elementwise", solver_kind="irls")
    with np.testing.assert_raises(ValueError):
        inverse.reconstruct_evoked_oaster_v5_from_whitened(
            np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
            baseline=baseline, active_windows=(active,), edge_penalty_mode="bad")
    with np.testing.assert_raises(ValueError):
        inverse.reconstruct_evoked_oaster_v5_from_whitened(
            np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
            baseline=baseline, active_windows=(active,),
            surface_penalty_multiplier=[1.])


def test_surface_elementwise_mode_passes_cortical_mask_and_rejects_irls(monkeypatch):
    baseline = np.arange(30) < 15
    active = (np.arange(30) >= 20) & (np.arange(30) < 25)
    captured = {}

    def capture(response, gain, incidence, **kwargs):
        captured["response"] = response.copy()
        captured.update(kwargs)
        return np.zeros((gain.shape[1], response.shape[1])), {"converged": True}

    monkeypatch.setattr(inverse, "solve_reweighted_graph_v5", capture)
    _, mixed = inverse.reconstruct_evoked_oaster_v5_from_whitened(
        np.arange(90, dtype=float).reshape(3, 30), np.eye(3), 2,
        adjacency=sparse.eye(3), baseline=baseline, active_windows=(active,),
        mrf_strength=0, source_penalty_mode="surface_elementwise")
    mixed_solver = captured.copy()
    captured.clear()
    inverse.reconstruct_evoked_oaster_v5_from_whitened(
        np.arange(90, dtype=float).reshape(3, 30), np.eye(3), 2,
        adjacency=sparse.eye(3), baseline=baseline, active_windows=(active,),
        mrf_strength=0)
    assert mixed_solver["source_penalty_mode"] == "surface_elementwise"
    assert np.array_equal(mixed_solver["elementwise_source_mask"], [True, True, False])
    assert np.isclose(mixed_solver["source_penalty"][-1], captured["source_penalty"][-1])
    assert np.all(mixed_solver["source_penalty"][:2] <= captured["source_penalty"][:2])
    assert np.isclose(mixed_solver["edge_penalty"], captured["edge_penalty"])
    assert mixed["source_penalty_mode"] == "surface_elementwise"
    gram = mixed_solver["response"].T @ mixed_solver["response"]
    assert np.allclose(gram, np.diag(np.diag(gram)), atol=1e-9)
    assert mixed["windows"][0]["temporal_rotation"] == \
        "training_sensor_svd_with_blockwise_null"
    with np.testing.assert_raises(ValueError):
        inverse.reconstruct_evoked_oaster_v5_from_whitened(
            np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
            baseline=baseline, active_windows=(active,), solver_kind="irls",
            source_penalty_mode="surface_elementwise")
