"""ERP-v4: full-grid, iteratively reweighted source and anatomical-edge sparsity.

No spatial templates, candidate pruning, or truth-dependent parameters. Graph TV
acts on signed current, or on MRF current innovations when that option is enabled.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from algorithms.spatial_fused_fusion import _incidence_matrix, vector_soft_threshold
from candidates.oaster_rebuilt import _evoked_temporal_basis


def solve_reweighted_graph(data, gain, incidence, *, source_penalty, edge_penalty,
                           outer_iterations=12, max_iter=2000, tolerance=3e-4,
                           epsilon_fraction=0.05, rho=1.0, outer_tolerance=0.01,
                           adaptive_rho=True):
    """MM log-sum penalties; ADMM subproblems use sparse + sensor-space solves.

    First solve is convex. Then epsilons are frozen and weights are
    eps/(row_L2 + eps), both for current amplitude and graph differences.
    Descent is checked within the inner solve's relative numerical tolerance.
    Convergence flags describe these finite tolerances, not a global optimum.
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
            or not np.isfinite(edge_penalty) or edge_penalty < 0
            or not np.isfinite([rho, tolerance, epsilon_fraction, outer_tolerance]).all()
            or rho <= 0 or outer_iterations < 1 or max_iter < 1
            or tolerance <= 0 or epsilon_fraction <= 0 or outer_tolerance <= 0):
        raise ValueError("invalid graph inverse inputs or regularization")
    gain_data = gain.T @ data
    if np.all(np.linalg.norm(gain_data, axis=1) <= source_penalty):
        return np.zeros((n, data.shape[1])), dict(
            history=[dict(outer=0, iterations=0, converged=True, primal_relative=0.,
                          dual_relative=0., log_objective=float(.5 * np.sum(data ** 2)),
                          relative_change=0., active_rows=0, rho=float(rho), rho_updates=0,
                          objective_descent_slack=0.)], converged=True,
            inner_converged=True, outer_converged=True, descent_guard_triggered=False,
            descent_guard_step=None,
            rho_initial=float(rho), rho_final=float(rho), rho_updates=0,
            outer_tolerance=float(outer_tolerance), adaptive_rho=bool(adaptive_rho),
            objective_descent_tolerance=float(tolerance),
            objective_descent_slack_rule="tolerance * max(1, previous_log_objective)",
            convergence_note="Finite residual/change tolerances; no global-optimality claim.",
            amplitude_epsilon=1e-12, edge_epsilon=1e-12,
            amplitude_weight_range=[1., 1.], edge_weight_range=[1., 1.])
    # Factor Q=I+D'D once. Changing rho only refactorizes the small sensor system.
    # (G'G+rho Q)^-1 b = [Q^-1 b - Q^-1 G'(rho I+G Q^-1 G')^-1 G Q^-1 b]/rho.
    graph_factor = splu((sparse.eye(n) + incidence.T @ incidence).tocsc())
    q_gain = graph_factor.solve(gain.T)
    sensor_identity = np.eye(gain.shape[0])
    sensor_gram = gain @ q_gain
    sensor_gram = (sensor_gram + sensor_gram.T) * .5
    sensor_factor = linalg.cho_factor(rho * sensor_identity + sensor_gram,
                                    lower=True, check_finite=False)
    initial_rho, rho_updates = float(rho), 0
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
    descent_guard_step = None
    for outer in range(outer_iterations):
        previous_outer = x.copy()
        previous_rho_updates = rho_updates
        converged = False
        for iteration in range(1, max_iter + 1):
            old_z, old_v = z.copy(), v.copy()
            rhs = gain_data + rho * (z - u + incidence.T @ (v - h))
            solved = graph_factor.solve(rhs)
            x = (solved - q_gain @ linalg.cho_solve(
                sensor_factor, gain @ solved, check_finite=False)) / rho
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
            if adaptive_rho and iteration % 25 == 0:
                new_rho = (min(2 * rho, 1e6) if primal > 5 * dual else
                           max(rho / 2, 1e-6) if dual > 5 * primal else rho)
                if new_rho != rho:
                    # u,h are scaled duals; preserve the unscaled multipliers.
                    u *= rho / new_rho
                    h *= rho / new_rho
                    rho = new_rho
                    rho_updates += 1
                    sensor_factor = linalg.cho_factor(rho * sensor_identity + sensor_gram,
                                                    lower=True, check_finite=False)
        # The source consensus variable is the actual proximal sparse estimate.
        # Returning the unconstrained linear-system iterate instead leaves tiny
        # ADMM residuals at thousands of inactive sites and corrupts rank metrics.
        x = z.copy()
        norms = np.linalg.norm(x, axis=1)
        jumps = np.linalg.norm(incidence @ x, axis=1)
        if eps_a is None:
            eps_a = max(epsilon_fraction * float(norms.max(initial=0)), 1e-12)
            eps_e = max(epsilon_fraction * float(jumps.max(initial=0)), 1e-12)
        objective = (0.5 * np.sum((data - gain @ x) ** 2)
                     + np.sum(source_penalty * eps_a * np.log1p(norms / eps_a))
                     + edge_penalty * eps_e * np.log1p(jumps / eps_e).sum())
        # Match the objective check to the inexact ADMM solve, rather than
        # requiring machine-precision descent from finite-tolerance coefficients.
        descent_slack = (tolerance * max(1., history[-1]["log_objective"]) if history else 0.)
        if history and objective > history[-1]["log_objective"] + descent_slack:
            x = previous_outer
            descent_guard = True
            descent_guard_step = dict(outer=outer, iterations=iteration,
                previous_log_objective=history[-1]["log_objective"],
                proposed_log_objective=float(objective), allowed_increase=float(descent_slack),
                inner_converged=converged, primal_relative=primal / scale, dual_relative=dual / scale)
            break
        change = np.linalg.norm(x - previous_outer) / max(np.linalg.norm(x), 1e-12)
        history.append(dict(outer=outer, iterations=iteration, converged=converged,
                            primal_relative=primal / scale, dual_relative=dual / scale,
                            log_objective=float(objective), relative_change=float(change),
                            objective_descent_slack=float(descent_slack),
                            active_rows=int(np.count_nonzero(norms > norms.max(initial=0) * .01)),
                            rho=float(rho), rho_updates=rho_updates - previous_rho_updates))
        amplitude_weights = eps_a / (norms + eps_a)
        edge_weights = eps_e / (jumps + eps_e)
        if outer > 0 and converged and change < outer_tolerance:
            outer_converged = True
            break
    # z is the source-consensus estimate; its primal residual is logged above.
    return x, dict(history=history, converged=all(item["converged"] for item in history) and outer_converged and not descent_guard,
                   inner_converged=all(item["converged"] for item in history),
                   outer_converged=outer_converged, descent_guard_triggered=descent_guard,
                   descent_guard_step=descent_guard_step,
                   rho_initial=initial_rho, rho_final=float(rho), rho_updates=rho_updates,
                   outer_tolerance=float(outer_tolerance), adaptive_rho=bool(adaptive_rho),
                   objective_descent_tolerance=float(tolerance),
                   objective_descent_slack_rule="tolerance * max(1, previous_log_objective)",
                   convergence_note="Finite residual/change tolerances; no global-optimality claim.",
                   amplitude_epsilon=eps_a, edge_epsilon=eps_e,
                   amplitude_weight_range=[float(amplitude_weights.min()), float(amplitude_weights.max())],
                   edge_weight_range=[float(edge_weights.min(initial=1)), float(edge_weights.max(initial=0))])


def reconstruct_evoked_oaster_v4_from_whitened(
        data, gain, n_surf, kernels=(), *, adjacency, baseline, active_windows,
        window_channel_weights=None, require_one=False, edge_fraction=0.5,
        noise_multiplier=1.0, outer_iterations=12, max_iter=2000, tolerance=3e-4,
        mrf_strength=0.0, outer_tolerance=0.01, adaptive_rho=True):
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
            or len(active_windows) != 1 or not 0 <= mrf_strength < 1):
        raise ValueError("invalid ERP-v4 inputs; forced source detection is not supported")
    graph = sparse.csr_matrix(adjacency)
    if graph.shape != (gain.shape[1], gain.shape[1]):
        raise ValueError("anatomical adjacency must match the complete source grid")
    upper = sparse.triu(graph[:n_surf, :n_surf].maximum(graph[:n_surf, :n_surf].T), k=1).tocoo()
    incidence = _incidence_matrix(gain.shape[1], upper.row, upper.col)
    mrf_factor = None
    if mrf_strength:
        neighbors = sparse.csr_matrix((np.ones(2 * upper.nnz),
            (np.r_[upper.row, upper.col], np.r_[upper.col, upper.row])), shape=graph.shape)
        degree = np.asarray(neighbors.sum(axis=1)).ravel()
        transition = sparse.diags(1 / np.maximum(degree, 1)) @ neighbors
        mrf_factor = splu((sparse.eye(gain.shape[1]) - mrf_strength * transition).tocsc())
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
        # Fixed sensitivity penalty and one global current scale. With MRF the
        # following sparse solve acts on innovations rather than current itself.
        depth_weights = np.maximum(sensitivity / gain_scale, .1) ** .8
        if mrf_factor is not None:
            # SISSES-like innovation z=M x, M=I-tau P; sparse source and edge
            # weights now act on z and D z. x=M^-1 z is the physical current.
            # This is graph diffusion, not a choice among Gaussian patch radii.
            design = mrf_factor.solve(design.T, trans="T").T
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
            tolerance=tolerance, outer_tolerance=outer_tolerance, adaptive_rho=adaptive_rho)
        if mrf_factor is not None:
            coefficients = mrf_factor.solve(coefficients)
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
                          deep_graph_edges=0, spatial_templates_used=False,
                          mrf_strength=mrf_strength,
                          weighting_coordinates="physical_current" if not mrf_strength else "MRF_current_innovation")
