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


def test_deep_reweight_floor_is_applied_only_to_deep_admm_rows(monkeypatch):
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
        mrf_strength=0, deep_reweight_floor=.4)
    assert source.shape == (3, 30)
    assert np.array_equal(captured["amplitude_weight_floor"], [0., 0., .4])
    assert info["deep_reweight_floor"] == .4
    for invalid in (-.1, 1.1):
        with np.testing.assert_raises(ValueError):
            inverse.reconstruct_evoked_oaster_v5_from_whitened(
                np.zeros((3, 30)), np.eye(3), 2, adjacency=sparse.eye(3),
                baseline=baseline, active_windows=(active,), deep_reweight_floor=invalid)
