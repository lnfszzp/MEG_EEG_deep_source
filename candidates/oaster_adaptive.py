"""ERP-v4: full-grid, iteratively reweighted source and anatomical-edge sparsity.

No spatial templates, candidate pruning, or truth-dependent parameters.  Graph TV
acts on signed physical-current coefficients, not on normalized sensor scores.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from algorithms.spatial_fused_fusion import _incidence_matrix, vector_soft_threshold
from candidates.oaster_rebuilt import _evoked_temporal_basis


def solve_reweighted_graph(data, gain, incidence, *, source_penalty, edge_penalty,
                           outer_iterations=6, max_iter=600, tolerance=1e-3,
                           epsilon_fraction=0.05, rho=1.0):
    """MM log-sum penalties; ADMM subproblems use sparse + sensor-space solves.

    First solve is convex. Then epsilons are frozen and weights are
    eps/(row_L2 + eps), both for current amplitude and graph differences.
    A nonconverged subproblem is reported, never silently called converged.
    """
    data, gain = np.asarray(data, float), np.asarray(gain, float)
    if data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]:
        raise ValueError("data and gain must be matrices sharing the sensor axis")
    n = gain.shape[1]
    incidence = sparse.csr_matrix(incidence)
    source_penalty = np.broadcast_to(np.asarray(source_penalty, float), (n,)).copy()
    if (data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]
            or incidence.shape[1] != n or not np.all(np.isfinite(data))
            or not np.all(np.isfinite(gain)) or not np.all(np.isfinite(incidence.data))
            or not np.all(np.isfinite(source_penalty)) or np.any(source_penalty < 0)
            or not np.isfinite(edge_penalty) or edge_penalty < 0 or rho <= 0 or outer_iterations < 1
            or max_iter < 1 or tolerance <= 0 or epsilon_fraction <= 0):
        raise ValueError("invalid graph inverse inputs or regularization")
    gain_data = gain.T @ data
    if np.all(np.linalg.norm(gain_data, axis=1) <= source_penalty):
        return np.zeros((n, data.shape[1])), dict(
            history=[dict(outer=0, iterations=0, converged=True, primal_relative=0.,
                          dual_relative=0., log_objective=float(.5 * np.sum(data ** 2)),
                          relative_change=0., active_rows=0)], converged=True,
            inner_converged=True, outer_converged=True, descent_guard_triggered=False,
            amplitude_epsilon=1e-12, edge_epsilon=1e-12,
            amplitude_weight_range=[1., 1.], edge_weight_range=[1., 1.])
    # Woodbury avoids allocating/factorizing a dense source x source matrix.
    graph_factor = splu((rho * (sparse.eye(n) + incidence.T @ incidence)).tocsc())
    q_gain = graph_factor.solve(gain.T)
    sensor_factor = linalg.cho_factor(np.eye(gain.shape[0]) + gain @ q_gain,
                                    lower=True, check_finite=False)
    x = np.zeros((n, data.shape[1]))
    z, u = x.copy(), x.copy()
    v = np.zeros((incidence.shape[0], data.shape[1]))
    h = v.copy()
    amplitude_weights = np.ones(n)
    edge_weights = np.ones(incidence.shape[0])
    eps_a = eps_e = None
    history = []
    outer_converged = outer_iterations == 1
    descent_guard = False
    for outer in range(outer_iterations):
        previous_outer = x.copy()
        converged = False
        for iteration in range(1, max_iter + 1):
            old_z, old_v = z.copy(), v.copy()
            rhs = gain_data + rho * (z - u + incidence.T @ (v - h))
            solved = graph_factor.solve(rhs)
            x = solved - q_gain @ linalg.cho_solve(
                sensor_factor, gain @ solved, check_finite=False)
            dx = incidence @ x
            z = vector_soft_threshold(x + u, source_penalty * amplitude_weights / rho)
            v = vector_soft_threshold(dx + h, edge_penalty * edge_weights / rho)
            u += x - z
            h += dx - v
            primal = float(np.hypot(np.linalg.norm(x - z), np.linalg.norm(dx - v)))
            dual = float(rho * np.linalg.norm(z - old_z + incidence.T @ (v - old_v)))
            scale = max(np.linalg.norm(x), np.linalg.norm(dx), 1e-12)
            if primal <= tolerance * scale and dual <= tolerance * scale:
                converged = True
                break
        norms = np.linalg.norm(x, axis=1)
        jumps = np.linalg.norm(incidence @ x, axis=1)
        if eps_a is None:
            eps_a = max(epsilon_fraction * float(norms.max(initial=0)), 1e-12)
            eps_e = max(epsilon_fraction * float(jumps.max(initial=0)), 1e-12)
        objective = (0.5 * np.sum((data - gain @ x) ** 2)
                     + np.sum(source_penalty * eps_a * np.log1p(norms / eps_a))
                     + edge_penalty * eps_e * np.log1p(jumps / eps_e).sum())
        # An inexact ADMM subproblem must not make the MM objective worse.
        # Keep the last accepted estimate, and expose the early stop explicitly.
        if history and objective > history[-1]["log_objective"] + 1e-9 * max(1., history[-1]["log_objective"]):
            x = previous_outer
            descent_guard = True
            break
        change = np.linalg.norm(x - previous_outer) / max(np.linalg.norm(x), 1e-12)
        history.append(dict(outer=outer, iterations=iteration, converged=converged,
                            primal_relative=primal / scale, dual_relative=dual / scale,
                            log_objective=float(objective), relative_change=float(change),
                            active_rows=int(np.count_nonzero(norms > norms.max(initial=0) * .01))))
        amplitude_weights = eps_a / (norms + eps_a)
        edge_weights = eps_e / (jumps + eps_e)
        if outer > 0 and converged and change < tolerance:
            outer_converged = True
            break
    # Keep the primal coefficient solution: thresholded splits are auxiliaries.
    return x, dict(history=history, converged=all(item["converged"] for item in history) and outer_converged and not descent_guard,
                   inner_converged=all(item["converged"] for item in history),
                   outer_converged=outer_converged, descent_guard_triggered=descent_guard,
                   amplitude_epsilon=eps_a, edge_epsilon=eps_e,
                   amplitude_weight_range=[float(amplitude_weights.min()), float(amplitude_weights.max())],
                   edge_weight_range=[float(edge_weights.min(initial=1)), float(edge_weights.max(initial=0))])


def reconstruct_evoked_oaster_v4_from_whitened(
        data, gain, n_surf, kernels=(), *, adjacency, baseline, active_windows,
        window_channel_weights=None, require_one=False, edge_fraction=0.5,
        noise_multiplier=1.0, outer_iterations=6, max_iter=600, tolerance=1e-3):
    """Same ERP temporal evidence as v3, but spatial support is learned on graph.

    Depth nodes are independent here: their fixed orientations can be opposite,
    so a scalar signed TV edge between them would be anatomically misleading.
    Output retains signed, unthresholded time courses over the complete epoch.
    """
    data, gain = np.asarray(data, float), np.asarray(gain, float)
    baseline = np.asarray(baseline, bool)
    if (data.ndim != 2 or gain.ndim != 2 or gain.shape[0] != data.shape[0]
            or baseline.shape != (data.shape[1],) or not baseline.any()
            or not 0 <= n_surf <= gain.shape[1] or require_one
            or edge_fraction < 0 or noise_multiplier <= 0
            or len(active_windows) != 1):
        raise ValueError("invalid ERP-v4 inputs; forced source detection is not supported")
    graph = sparse.csr_matrix(adjacency)
    if graph.shape != (gain.shape[1], gain.shape[1]):
        raise ValueError("anatomical adjacency must match the complete source grid")
    upper = sparse.triu(graph[:n_surf, :n_surf].maximum(graph[:n_surf, :n_surf].T), k=1).tocoo()
    incidence = _incidence_matrix(gain.shape[1], upper.row, upper.col)
    centered = data - data[:, baseline].mean(axis=1, keepdims=True)
    estimate = np.zeros((gain.shape[1], data.shape[1]))
    windows = []
    for window_index, active in enumerate(active_windows):
        active = np.asarray(active, bool)
        channel_weights = (np.ones(data.shape[0]) if window_channel_weights is None
                           else np.asarray(window_channel_weights[window_index], float))
        if channel_weights.shape != (data.shape[0],) or not np.all(np.isfinite(channel_weights)) or np.any(channel_weights < 0):
            raise ValueError("channel weights must be finite nonnegative sensor weights")
        weighted_data = channel_weights[:, None] * centered
        weighted_gain = channel_weights[:, None] * gain
        basis, info = _evoked_temporal_basis(weighted_data, baseline, active)
        if not basis.size:
            windows.append(dict(info, solver=None))
            continue
        sensitivity = np.linalg.norm(weighted_gain, axis=0)
        gain_scale = float(np.median(sensitivity[sensitivity > 0]))
        if not np.isfinite(gain_scale) or gain_scale <= 0:
            raise ValueError("lead field has no nonzero source columns")
        design = weighted_gain / gain_scale
        # Fixed sensitivity penalty; graph differences remain in physical-current
        # coordinates. One global scale, never separate surface/deep peak scaling.
        depth_weights = np.maximum(sensitivity / gain_scale, .1) ** .8
        response = weighted_data @ basis.T
        response_singular = np.linalg.svd(response, compute_uv=False)
        response_condition = float(response_singular[0] / max(response_singular[-1], np.finfo(float).tiny))
        local_basis = basis[:, active].T
        baseline_data = weighted_data[:, baseline]
        baseline_projection = design.T @ baseline_data
        width = local_basis.shape[0]
        starts = np.unique(np.linspace(0, baseline_data.shape[1] - width, 16).astype(int))
        null_maxima = [float(np.max(np.linalg.norm(
            baseline_projection[:, start:start + width] @ local_basis, axis=1) / depth_weights))
            for start in starts]
        lam = float(noise_multiplier * np.quantile(null_maxima, .99))
        degree = 2 * incidence.shape[0] / max(n_surf, 1)
        beta = lam * edge_fraction / max(degree, 1)
        coefficients, diagnostics = solve_reweighted_graph(
            response, design, incidence, source_penalty=lam * depth_weights,
            edge_penalty=beta, outer_iterations=outer_iterations, max_iter=max_iter,
            tolerance=tolerance)
        # Extend the selected sensor modes over the signed complete epoch.
        # pinv(response) @ data @ basis.T == I, so projecting the returned
        # time courses recovers EXACTLY the optimized spatial coefficients.
        # No second spatial refit, fixed template, or evidence halo is added.
        time_courses = np.linalg.pinv(response) @ weighted_data
        estimate += coefficients @ time_courses / gain_scale
        windows.append(dict(info, source_lambda=lam, edge_lambda=beta, solver=diagnostics,
                            response_condition=response_condition,
                            null_rule="99th percentile of up to 16 overlapping baseline projections; heuristic, not p-value"))
    return estimate, dict(mode="adaptive_source_and_edge_logsum_full_grid", windows=windows,
                          selected_templates=[], graph_edges=int(incidence.shape[0]),
                          deep_graph_edges=0, spatial_templates_used=False)
