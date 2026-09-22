"""Numerical checks for the full-grid, signed adaptive graph inverse."""
import numpy as np
import pytest
from scipy import sparse

from algorithms.spatial_fused_fusion import _incidence_matrix
from candidates.oaster_adaptive import solve_reweighted_graph
from candidates import oaster_adaptive


def test_convex_identity_has_exact_group_soft_threshold_solution():
    data = np.array([[3., 4.], [0., 0.], [-2., 0.]])
    estimate, diagnostics = solve_reweighted_graph(
        data, np.eye(3), sparse.csr_matrix((0, 3)), source_penalty=1.,
        edge_penalty=0., outer_iterations=1, max_iter=2000, tolerance=1e-9)
    assert np.allclose(estimate, [[2.4, 3.2], [0., 0.], [-1., 0.]], atol=1e-7)
    assert diagnostics["converged"]


@pytest.mark.parametrize("initial_rho", [1e-4, 1e4])
def test_residual_balancing_changes_rho_without_changing_convex_solution(initial_rho):
    estimate, diagnostics = solve_reweighted_graph(
        np.array([[3., 4.], [0., 0.], [-2., 0.]]), np.eye(3),
        sparse.csr_matrix((0, 3)), source_penalty=1., edge_penalty=0.,
        outer_iterations=1, max_iter=2000, tolerance=1e-8, rho=initial_rho)
    assert np.allclose(estimate, [[2.4, 3.2], [0., 0.], [-1., 0.]], atol=1e-6)
    assert diagnostics["converged"]
    assert diagnostics["rho_updates"] > 0
    assert diagnostics["rho_final"] > initial_rho if initial_rho < 1 else diagnostics["rho_final"] < initial_rho


def test_graph_fuses_patch_interior_and_keeps_a_boundary():
    incidence = _incidence_matrix(4, np.array([0, 1, 2]), np.array([1, 2, 3]))
    estimate, diagnostics = solve_reweighted_graph(
        np.array([[3.], [3.], [0.], [0.]]), np.eye(4), incidence,
        source_penalty=.1, edge_penalty=.15, outer_iterations=1,
        max_iter=4000, tolerance=1e-8)
    assert np.allclose(estimate[:, 0], [2.825, 2.825, 0., 0.], atol=1e-6)
    assert diagnostics["converged"]


def test_reweight_is_rotation_invariant_and_log_objective_decreases():
    rng = np.random.default_rng(15)
    gain = rng.normal(size=(7, 5))
    data = rng.normal(size=(7, 3))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    incidence = _incidence_matrix(5, np.arange(4), np.arange(1, 5))
    settings = dict(source_penalty=.25, edge_penalty=.2,
                    outer_iterations=4, max_iter=4000, tolerance=1e-8)
    estimate, diagnostics = solve_reweighted_graph(data, gain, incidence, **settings)
    rotated, rotated_diagnostics = solve_reweighted_graph(data @ rotation, gain, incidence, **settings)
    assert np.allclose(rotated, estimate @ rotation, atol=1e-6)
    assert diagnostics["inner_converged"] and rotated_diagnostics["inner_converged"]
    objectives = [entry["log_objective"] for entry in diagnostics["history"]]
    assert np.all(np.diff(objectives) <= 1e-7)


def test_zero_and_subthreshold_noise_do_not_create_sources():
    empty_edges = sparse.csr_matrix((0, 4))
    zero, diagnostics = solve_reweighted_graph(
        np.zeros((4, 2)), np.eye(4), empty_edges,
        source_penalty=1., edge_penalty=0.)
    assert np.array_equal(zero, np.zeros((4, 2)))
    assert diagnostics["converged"]
    noise = np.random.default_rng(4).normal(scale=.001, size=(4, 2))
    threshold = 1.1 * np.linalg.norm(noise, axis=1).max()
    estimate, _ = solve_reweighted_graph(
        noise, np.eye(4), empty_edges, source_penalty=threshold,
        edge_penalty=0., outer_iterations=1, max_iter=2000, tolerance=1e-8)
    assert np.linalg.norm(estimate) < 1e-8


def test_iteration_limit_is_not_reported_as_convergence():
    estimate, diagnostics = solve_reweighted_graph(
        np.array([[3., 4.], [-2., 1.]]), np.eye(2),
        sparse.csr_matrix((0, 2)), source_penalty=.2, edge_penalty=0.,
        outer_iterations=1, max_iter=1, tolerance=1e-12)
    assert np.isfinite(estimate).all()
    assert not diagnostics["converged"]
    assert diagnostics["history"][0]["iterations"] == 1


def test_full_epoch_extension_preserves_optimized_coefficients_and_sign(monkeypatch):
    rng = np.random.default_rng(32)
    data = .01 * rng.normal(size=(4, 240))
    data[:2, 140:181] += 3 * np.sin(np.linspace(0, np.pi, 41))
    data[:2, 205:226] -= 2 * np.sin(np.linspace(0, np.pi, 21))
    baseline = np.arange(240) < 120
    active = (np.arange(240) >= 140) & (np.arange(240) < 181)
    graph = sparse.diags([np.ones(3), np.ones(3)], [-1, 1], shape=(4, 4))
    captured = []
    original = oaster_adaptive.solve_reweighted_graph

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(oaster_adaptive, "solve_reweighted_graph", capture)
    estimate, diagnostics = oaster_adaptive.reconstruct_evoked_oaster_v4_from_whitened(
        data, np.eye(4), 4, adjacency=graph, baseline=baseline,
        active_windows=(active,), max_iter=3000, tolerance=1e-7)
    basis, _ = oaster_adaptive._evoked_temporal_basis(data, baseline, active)
    assert np.allclose(estimate @ basis.T, captured[0][0], atol=1e-8)
    assert np.all(estimate[:2, 160] > 0)
    assert np.all(estimate[:2, 215] < 0)
    assert np.linalg.norm(estimate[:, baseline]) > 0
    assert diagnostics["graph_edges"] == 3


@pytest.mark.parametrize("bad_argument", ["source_penalty", "edge_penalty"])
def test_nonfinite_penalty_is_rejected(bad_argument):
    settings = dict(source_penalty=.2, edge_penalty=.2)
    settings[bad_argument] = np.nan
    with pytest.raises(ValueError):
        solve_reweighted_graph(np.eye(2), np.eye(2), sparse.csr_matrix((0, 2)), **settings)


@pytest.mark.parametrize("strength", [0., .5])
def test_mrf_forward_transform_and_recovered_physical_current(monkeypatch, strength):
    rng = np.random.default_rng(25)
    gain = rng.normal(size=(6, 4))
    baseline = np.arange(200) < 100
    active = (np.arange(200) >= 120) & (np.arange(200) < 161)
    source = np.zeros((4, 200))
    source[:2, active] = 3 * np.sin(np.linspace(0, np.pi, active.sum()))
    data = gain @ source + .01 * rng.normal(size=(6, 200))
    # The last node is deep; its supplied surface connection must not enter M.
    graph = sparse.diags([np.ones(3), np.ones(3)], [-1, 1], shape=(4, 4))
    transition = np.array([[0., 1., 0., 0.], [.5, 0., .5, 0.],
                           [0., 1., 0., 0.], [0., 0., 0., 0.]])
    mrf = np.eye(4) - strength * transition
    gain_scale = np.median(np.linalg.norm(gain, axis=0))
    captured = []
    original = oaster_adaptive.solve_reweighted_graph

    def capture(response, design, incidence, **settings):
        result = original(response, design, incidence, **settings)
        captured.append((design.copy(), result[0].copy()))
        return result

    monkeypatch.setattr(oaster_adaptive, "solve_reweighted_graph", capture)
    settings = dict(adjacency=graph, baseline=baseline, active_windows=(active,),
                    outer_iterations=1, max_iter=3000, tolerance=1e-8)
    estimate, diagnostics = oaster_adaptive.reconstruct_evoked_oaster_v4_from_whitened(
        data, gain, 3, mrf_strength=strength, **settings)
    transformed_gain, innovation = captured[0]
    basis, _ = oaster_adaptive._evoked_temporal_basis(data, baseline, active)
    current = np.linalg.solve(mrf, innovation) / gain_scale
    assert np.allclose(transformed_gain @ mrf, gain / gain_scale, atol=1e-10)
    assert np.allclose(gain @ current, transformed_gain @ innovation, atol=1e-10)
    assert np.allclose(estimate @ basis.T, current, atol=1e-9)
    assert np.allclose(current[3], innovation[3] / gain_scale, atol=1e-10)
    assert diagnostics["graph_edges"] == 2
    rotation = np.linalg.qr(rng.normal(size=(innovation.shape[1], innovation.shape[1])))[0]
    assert np.allclose(np.linalg.solve(mrf, innovation @ rotation),
                       np.linalg.solve(mrf, innovation) @ rotation, atol=1e-10)
    if strength == 0:
        default_estimate, _ = oaster_adaptive.reconstruct_evoked_oaster_v4_from_whitened(
            data, gain, 3, **settings)
        assert np.allclose(estimate, default_estimate, atol=1e-12)
