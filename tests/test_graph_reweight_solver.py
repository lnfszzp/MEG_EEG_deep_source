"""Numerical certificates and monotone MM, independent of source ground truth."""
import numpy as np
import pytest
from scipy import sparse

from algorithms.spatial_fused_fusion import _incidence_matrix
from candidates.graph_reweight_solver import _certificate, solve_reweighted_graph_v5


def test_dual_bound_brackets_an_analytic_group_lasso_optimum():
    data = np.array([[3., 4.], [-2., 0.]])
    optimum = np.array([[2.4, 3.2], [-1., 0.]])
    objective = .5 * np.sum((optimum - data) ** 2) + np.linalg.norm(optimum, axis=1).sum()
    graph = sparse.csr_matrix((0, 2))
    for estimate in (optimum, np.zeros_like(optimum), data, -data):
        cert = _certificate(data, np.eye(2), graph, estimate, np.zeros((0, 2)), np.ones(2), np.ones(0))
        assert cert["dual_objective"] <= objective + 1e-10
        assert cert["objective"] >= objective - 1e-10
        assert cert["source_dual_violation"] <= 1e-10
    cert = _certificate(data, np.eye(2), graph, optimum, np.zeros((0, 2)), np.ones(2), np.ones(0))
    assert cert["gap"] < 1e-10
    # A separate ADMM data-dual candidate remains valid at any primal point.
    improved = _certificate(data, np.eye(2), graph, data, np.zeros((0, 2)),
                            np.ones(2), np.ones(0), optimum - data)
    assert np.isclose(improved["dual_objective"], objective)
    assert improved["dual_candidate"] == "admm_linear_residual"


@pytest.mark.parametrize("rho", [1e-3, 1., 1e3])
def test_certified_convex_identity_solution_is_independent_of_admm_rho(rho):
    estimate, diagnostics = solve_reweighted_graph_v5(
        np.array([[3., 4.], [0., 0.], [-2., 0.]]), np.eye(3),
        sparse.csr_matrix((0, 3)), source_penalty=1., edge_penalty=0.,
        outer_iterations=1, max_iter=500, tolerance=1e-8, rho=rho)
    assert np.allclose(estimate, [[2.4, 3.2], [0., 0.], [-1., 0.]], atol=1e-5)
    assert diagnostics["converged"]
    assert diagnostics["final_stationarity_gap_relative"] <= 1e-8


def test_reweighted_steps_are_monotone_and_invariant_under_temporal_rotation():
    rng = np.random.default_rng(15)
    gain, data = rng.normal(size=(7, 5)), rng.normal(size=(7, 3))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    graph = _incidence_matrix(5, np.arange(4), np.arange(1, 5))
    settings = dict(source_penalty=.25, edge_penalty=.2, outer_iterations=5,
                    max_iter=500, tolerance=1e-7)
    estimate, diagnostics = solve_reweighted_graph_v5(data, gain, graph, **settings)
    rotated, _ = solve_reweighted_graph_v5(data @ rotation, gain, graph, **settings)
    assert np.allclose(rotated, estimate @ rotation, atol=2e-6)
    objectives = [entry["log_objective"] for entry in diagnostics["history"]]
    assert np.all(np.diff(objectives) <= 1e-10)
    for entry in diagnostics["history"]:
        assert entry["surrogate_objective"] <= entry["surrogate_reference"] + entry["surrogate_roundoff"]
        if entry["converged"]:
            assert entry["primal_dual_gap_relative"] <= settings["tolerance"]


def test_zero_signal_and_budget_exhaustion_are_distinguished():
    graph = sparse.csr_matrix((0, 2))
    zero, diagnostics = solve_reweighted_graph_v5(
        np.zeros((2, 2)), np.eye(2), graph, source_penalty=1., edge_penalty=0.)
    assert np.array_equal(zero, np.zeros((2, 2))) and diagnostics["converged"]
    estimate, diagnostics = solve_reweighted_graph_v5(
        np.array([[3., 4.], [-2., 1.]]), np.eye(2), graph,
        source_penalty=.2, edge_penalty=0., outer_iterations=1,
        max_iter=1, max_inner_retries=0, tolerance=1e-12)
    assert np.isfinite(estimate).all()
    assert not diagnostics["converged"]
    assert diagnostics["history"][0]["iterations"] == 1
