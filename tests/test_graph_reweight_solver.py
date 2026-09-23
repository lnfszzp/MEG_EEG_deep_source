"""Numerical certificates and monotone MM, independent of source ground truth."""
import json

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


def test_per_source_amplitude_floor_preserves_zero_default_and_matches_objective():
    data = np.array([[3., 4.], [-2., 1.]])
    graph = sparse.csr_matrix((0, 2))
    settings = dict(source_penalty=np.array([.2, .3]), edge_penalty=0.,
                    outer_iterations=3, max_iter=500, tolerance=1e-8)
    default, _ = solve_reweighted_graph_v5(data, np.eye(2), graph, **settings)
    explicit_zero, _ = solve_reweighted_graph_v5(
        data, np.eye(2), graph, amplitude_weight_floor=np.zeros(2), **settings)
    assert np.array_equal(default, explicit_zero)

    estimate, diagnostics = solve_reweighted_graph_v5(
        data, np.eye(2), graph, amplitude_weight_floor=np.array([0., 1.]), **settings)
    norms = np.linalg.norm(estimate, axis=1)
    epsilon = diagnostics["amplitude_epsilon"]
    expected = (.5 * np.sum((estimate - data) ** 2)
                + settings["source_penalty"][0] * epsilon * np.log1p(norms[0] / epsilon)
                + settings["source_penalty"][1] * norms[1])
    assert np.isclose(diagnostics["history"][-1]["log_objective"], expected)
    assert diagnostics["amplitude_weight_floor_range"] == [0., 1.]


def test_ridge_zero_is_exactly_the_default():
    data = np.array([[3., 4.], [-2., 1.]])
    graph = sparse.csr_matrix((0, 2))
    settings = dict(source_penalty=.2, edge_penalty=0., outer_iterations=3,
                    max_iter=500, tolerance=1e-8)
    default, default_info = solve_reweighted_graph_v5(data, np.eye(2), graph, **settings)
    explicit, explicit_info = solve_reweighted_graph_v5(
        data, np.eye(2), graph, ridge_penalty=0., **settings)
    assert np.array_equal(default, explicit)
    assert default_info == explicit_info


def test_group_edge_mode_is_exactly_the_default():
    data = np.array([[3., 4.], [-2., 1.]])
    graph = _incidence_matrix(2, np.array([0]), np.array([1]))
    settings = dict(source_penalty=.2, edge_penalty=.3, outer_iterations=3,
                    max_iter=500, tolerance=1e-8)
    default, default_info = solve_reweighted_graph_v5(data, np.eye(2), graph, **settings)
    explicit, explicit_info = solve_reweighted_graph_v5(
        data, np.eye(2), graph, edge_penalty_mode="group", **settings)
    assert np.array_equal(default, explicit)
    assert default_info == explicit_info


def test_group_source_mode_is_exactly_the_default():
    data = np.array([[3., 4.], [-2., 1.]])
    graph = sparse.csr_matrix((0, 2))
    settings = dict(source_penalty=.2, edge_penalty=0., outer_iterations=3,
                    max_iter=500, tolerance=1e-8)
    default, default_info = solve_reweighted_graph_v5(data, np.eye(2), graph, **settings)
    explicit, explicit_info = solve_reweighted_graph_v5(
        data, np.eye(2), graph, source_penalty_mode="group", **settings)
    assert np.array_equal(default, explicit)
    assert default_info == explicit_info


def test_mixed_source_penalty_matches_elastic_net_kkt_and_certificate():
    data = np.array([[3., .5], [3., 4.]])
    graph = sparse.csr_matrix((0, 2))
    mask = np.array([True, False])
    ridge = .5
    estimate, diagnostics = solve_reweighted_graph_v5(
        data, np.eye(2), graph, source_penalty=1., edge_penalty=0.,
        ridge_penalty=ridge, source_penalty_mode="surface_elementwise",
        elementwise_source_mask=mask, outer_iterations=1, max_iter=1000,
        tolerance=1e-9)
    expected = np.vstack([
        np.sign(data[0]) * np.maximum(np.abs(data[0]) - 1, 0) / (1 + ridge),
        data[1] * (1 - 1 / np.linalg.norm(data[1])) / (1 + ridge),
    ])
    assert np.allclose(estimate, expected, atol=2e-6)
    surface_kkt = estimate[0] - data[0] + ridge * estimate[0] + np.array([1., .5])
    deep_kkt = (estimate[1] - data[1] + ridge * estimate[1]
                + estimate[1] / np.linalg.norm(estimate[1]))
    assert np.linalg.norm(surface_kkt) < 2e-6
    assert np.linalg.norm(deep_kkt) < 2e-6
    certificate = _certificate(
        data, np.eye(2), graph, expected, np.zeros((0, 2)), np.ones(2),
        np.ones(0), expected - data, ridge_penalty=ridge,
        source_penalty_mode="surface_elementwise", elementwise_source_mask=mask)
    assert certificate["dual_objective"] <= certificate["objective"] + 1e-12
    assert certificate["gap"] < 1e-10
    assert diagnostics["converged"]
    assert diagnostics["source_penalty_mode"] == "surface_elementwise"
    assert np.allclose(diagnostics["amplitude_epsilon"]["surface_elementwise"],
                       np.maximum(.05 * np.abs(expected[0]), 1e-12))
    assert np.isclose(diagnostics["amplitude_epsilon"]["deep_group"],
                      .05 * np.linalg.norm(expected[1]))
    json.dumps(diagnostics)


def test_elementwise_edge_mode_matches_two_sample_fused_ridge_solution():
    data = np.array([[0., 0.], [4., 1.]])
    graph = _incidence_matrix(2, np.array([0]), np.array([1]))
    estimate, diagnostics = solve_reweighted_graph_v5(
        data, np.eye(2), graph, source_penalty=0., edge_penalty=.5,
        ridge_penalty=.5, edge_penalty_mode="elementwise",
        outer_iterations=1, max_iter=1000, tolerance=1e-9)
    expected = np.array([[1 / 3, 1 / 3], [7 / 3, 1 / 3]])
    assert np.allclose(estimate, expected, atol=2e-6)
    objective = (.5 * np.sum((estimate - data) ** 2)
                 + .25 * np.sum(estimate ** 2)
                 + .5 * np.sum(np.abs(graph @ estimate)))
    assert np.isclose(diagnostics["history"][0]["surrogate_objective"], objective)
    assert diagnostics["converged"]
    assert diagnostics["final_stationarity_gap_relative"] <= 1e-9
    assert diagnostics["edge_penalty_mode"] == "elementwise"
    expected_epsilon = np.maximum(.05 * np.abs(graph @ estimate).max(axis=0), 1e-12)
    assert np.allclose(diagnostics["edge_epsilon"], expected_epsilon)
    json.dumps(diagnostics)

    _, reweighted = solve_reweighted_graph_v5(
        data, np.eye(2), graph, source_penalty=0., edge_penalty=.5,
        ridge_penalty=.5, edge_penalty_mode="elementwise",
        outer_iterations=5, outer_tolerance=1e-6, max_iter=1000, tolerance=1e-9)
    assert reweighted["converged"] and len(reweighted["history"]) > 1
    assert np.allclose(reweighted["edge_epsilon"], expected_epsilon)
    assert np.all(np.diff([item["log_objective"] for item in reweighted["history"]]) <= 1e-10)
    json.dumps(reweighted)


@pytest.mark.parametrize("rho", [1e-3, 1., 1e3])
def test_positive_ridge_matches_group_elastic_net_kkt_and_dual_gap(rho):
    data = np.array([[3., 4.], [-2., 0.]])
    graph = sparse.csr_matrix((0, 2))
    ridge = .5
    shrink = np.maximum(1 - 1 / np.linalg.norm(data, axis=1), 0)[:, None]
    optimum = data * shrink / (1 + ridge)
    estimate, diagnostics = solve_reweighted_graph_v5(
        data, np.eye(2), graph, source_penalty=1., edge_penalty=0.,
        ridge_penalty=ridge, outer_iterations=1, max_iter=1000,
        tolerance=1e-9, rho=rho)
    assert np.allclose(estimate, optimum, atol=2e-6)
    kkt = estimate - data + ridge * estimate + estimate / np.linalg.norm(estimate, axis=1)[:, None]
    assert np.linalg.norm(kkt) < 5e-6
    certificate = _certificate(data, np.eye(2), graph, optimum, np.zeros((0, 2)),
                               np.ones(2), np.ones(0), optimum - data,
                               ridge_penalty=ridge)
    assert certificate["dual_objective"] <= certificate["objective"] + 1e-12
    assert certificate["gap"] < 1e-10
    assert diagnostics["converged"]
    assert diagnostics["final_stationarity_gap_relative"] <= 1e-9


@pytest.mark.parametrize("ridge", [-1., np.nan, np.inf, np.array([0., 1.])])
def test_invalid_ridge_is_rejected(ridge):
    with pytest.raises(ValueError):
        solve_reweighted_graph_v5(
            np.ones((2, 2)), np.eye(2), sparse.csr_matrix((0, 2)),
            source_penalty=1., edge_penalty=0., ridge_penalty=ridge)


def test_invalid_edge_penalty_mode_is_rejected():
    with pytest.raises(ValueError):
        solve_reweighted_graph_v5(
            np.ones((2, 2)), np.eye(2), sparse.csr_matrix((0, 2)),
            source_penalty=1., edge_penalty=0., edge_penalty_mode="per_mode")


@pytest.mark.parametrize("mode, mask", [
    ("per_mode", None),
    ("surface_elementwise", None),
    ("surface_elementwise", np.array([False, False])),
    ("surface_elementwise", np.array([1, 0])),
])
def test_invalid_source_penalty_mode_or_mask_is_rejected(mode, mask):
    with pytest.raises(ValueError):
        solve_reweighted_graph_v5(
            np.ones((2, 2)), np.eye(2), sparse.csr_matrix((0, 2)),
            source_penalty=1., edge_penalty=0., source_penalty_mode=mode,
            elementwise_source_mask=mask)
