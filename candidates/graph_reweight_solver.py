"""Certified convex ADMM steps for the unchanged reweighted graph objective.

This is a numerical successor to v4, not a new spatial prior. A feasible dual
bound checks the returned sparse coefficients themselves, and MM accepts only
decreases of its fixed-weight convex surrogate (up to floating-point roundoff).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from algorithms.spatial_fused_fusion import vector_soft_threshold


def _project_rows(values, radii):
    norms = np.linalg.norm(values, axis=1)
    scale = np.ones(norms.shape)
    np.divide(radii, norms, out=scale, where=norms > radii)
    return values * scale[:, None]


def _certificate(data, gain, incidence, x, edge_dual, amplitude_penalty, edge_penalty, data_dual=None):
    """A feasible convex dual gives a rigorous lower bound on the optimum."""
    residual = gain @ x - data
    dx = incidence @ x
    objective = (.5 * np.sum(residual ** 2)
                 + amplitude_penalty @ np.linalg.norm(x, axis=1)
                 + edge_penalty @ np.linalg.norm(dx, axis=1))
    q = _project_rows(edge_dual, edge_penalty)
    edge_gradient = incidence.T @ q
    dual, scale, violation, dual_candidate = -np.inf, 0., 0., "primal_residual"
    candidates = [("primal_residual", residual)]
    if data_dual is not None:
        candidates.append(("admm_linear_residual", data_dual))
    for name, sensor_dual in candidates:
        norms = np.linalg.norm(-gain.T @ sensor_dual - edge_gradient, axis=1)
        nonzero = norms > 0
        candidate_scale = min(1., float(np.min(amplitude_penalty[nonzero] / norms[nonzero], initial=1.)))
        # Either sensor residual yields a valid dual after this common scaling.
        # The linear iterate often has much better dual stationarity than z.
        candidate_dual = (-.5 * candidate_scale ** 2 * np.sum(sensor_dual ** 2)
                          - candidate_scale * np.sum(data * sensor_dual))
        if candidate_dual > dual:
            dual, scale, dual_candidate = candidate_dual, candidate_scale, name
            violation = float(np.maximum(scale * norms - amplitude_penalty, 0.).max(initial=0.))
    gap = max(float(objective - dual), 0.)
    return dict(objective=float(objective), dual_objective=float(dual), gap=gap,
                gap_relative=gap / max(abs(float(objective)), abs(float(dual)), 1e-12),
                dual_scale=scale, dual_candidate=dual_candidate, source_dual_violation=violation)


def solve_reweighted_graph_v5(data, gain, incidence, *, source_penalty, edge_penalty,
                              outer_iterations=12, max_iter=2000, tolerance=3e-4,
                              epsilon_fraction=.05, rho=1., outer_tolerance=.01,
                              adaptive_rho=True, max_inner_retries=2):
    """Solve the v4 log-sum objective with verified, monotone MM updates.

    ``max_iter`` is one inner budget; an unfinished convex subproblem continues
    with unchanged weights for at most ``max_inner_retries`` additional budgets.
    Final convergence needs a primal-dual gap for the *current* MM weights, not
    just a small iterate change. Earlier approximate steps remain in history.
    """
    data, gain = np.asarray(data, float), np.asarray(gain, float)
    if data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]:
        raise ValueError("data and gain must be matrices sharing the sensor axis")
    n = gain.shape[1]
    incidence = sparse.csr_matrix(incidence)
    source_penalty = np.broadcast_to(np.asarray(source_penalty, float), (n,)).copy()
    edge_penalty = np.broadcast_to(np.asarray(edge_penalty, float), (incidence.shape[0],)).copy()
    if (not n or not data.shape[1] or incidence.shape[1] != n
            or any(not np.isfinite(v).all() for v in (data, gain, incidence.data, source_penalty, edge_penalty))
            or np.any(source_penalty < 0) or np.any(edge_penalty < 0)
            or not np.isfinite([rho, tolerance, epsilon_fraction, outer_tolerance]).all()
            or min(rho, tolerance, epsilon_fraction, outer_tolerance) <= 0
            or min(outer_iterations, max_iter) < 1 or max_inner_retries < 0):
        raise ValueError("invalid graph inverse inputs or regularization")
    gain_data = gain.T @ data
    x = np.zeros((n, data.shape[1]))
    graph_factor = splu((sparse.eye(n) + incidence.T @ incidence).tocsc())
    q_gain = graph_factor.solve(gain.T)
    gram = gain @ q_gain
    gram = .5 * (gram + gram.T)
    eye = np.eye(gain.shape[0])
    sensor_factor = linalg.cho_factor(rho * eye + gram, lower=True, check_finite=False)
    z, u = x.copy(), x.copy()
    v = np.zeros((incidence.shape[0], data.shape[1]))
    h = v.copy()
    amplitude_weights, edge_weights = np.ones(n), np.ones(incidence.shape[0])
    eps_a = eps_e = None
    history = []
    initial_rho, rho_updates = float(rho), 0
    stalled = outer_converged = False
    local_certificate = None
    linear_residual = -data.copy()

    for outer in range(outer_iterations):
        previous = x.copy()
        a, b = source_penalty * amplitude_weights, edge_penalty * edge_weights
        # Reweighting changes the dual balls. Keep valid warm duals and start
        # both consensus variables at the previously accepted physical point.
        u = _project_rows(rho * u, a) / rho
        h = _project_rows(rho * h, b) / rho
        z, v = previous.copy(), incidence @ previous
        reference = _certificate(data, gain, incidence, previous, rho * h, a, b, linear_residual)
        roundoff = 128 * np.finfo(float).eps * max(1., reference["objective"])
        previous_rho_updates = rho_updates
        inner_converged = False
        total_budget = max_iter * (max_inner_retries + 1)
        for iteration in range(1, total_budget + 1):
            old_z, old_v = z.copy(), v.copy()
            rhs = gain_data + rho * (z - u + incidence.T @ (v - h))
            solved = graph_factor.solve(rhs)
            linear_x = (solved - q_gain @ linalg.cho_solve(
                sensor_factor, gain @ solved, check_finite=False)) / rho
            dx = incidence @ linear_x
            z = vector_soft_threshold(linear_x + u, a / rho)
            v = vector_soft_threshold(dx + h, b / rho)
            u += linear_x - z
            h += dx - v
            primal = float(np.hypot(np.linalg.norm(linear_x - z), np.linalg.norm(dx - v)))
            dual = float(rho * np.linalg.norm(z - old_z + incidence.T @ (v - old_v)))
            magnitude = max(np.linalg.norm(linear_x), np.linalg.norm(dx), 1e-12)
            if iteration % 25 == 0 or iteration == total_budget:
                linear_residual = gain @ linear_x - data
                certificate = _certificate(data, gain, incidence, z, rho * h, a, b, linear_residual)
                descent = certificate["objective"] <= reference["objective"] + roundoff
                if certificate["gap_relative"] <= tolerance and descent:
                    inner_converged = True
                    break
                if adaptive_rho:
                    new_rho = (min(2 * rho, 1e6) if primal > 5 * dual else
                               max(rho / 2, 1e-6) if dual > 5 * primal else rho)
                    if new_rho != rho:
                        u *= rho / new_rho
                        h *= rho / new_rho
                        rho, rho_updates = new_rho, rho_updates + 1
                        sensor_factor = linalg.cho_factor(rho * eye + gram, lower=True, check_finite=False)

        x, step = z.copy(), 1.
        # If the sparse consensus estimate is still too inaccurate, test a
        # convex segment before discarding this complete inner solve.
        if certificate["objective"] > reference["objective"] + roundoff:
            direction = z - previous
            for exponent in range(1, 17):
                step = .5 ** exponent
                candidate = previous + step * direction
                trial = _certificate(data, gain, incidence, candidate, rho * h, a, b, linear_residual)
                if trial["objective"] <= reference["objective"] + roundoff:
                    x, certificate = candidate, trial
                    inner_converged = certificate["gap_relative"] <= tolerance
                    break
            else:
                x, step = previous, 0.
                certificate = _certificate(data, gain, incidence, previous, rho * h,
                                           a, b, linear_residual)
                inner_converged = certificate["gap_relative"] <= tolerance
                stalled = not inner_converged
        norms, jumps = np.linalg.norm(x, axis=1), np.linalg.norm(incidence @ x, axis=1)
        if eps_a is None:
            eps_a = max(epsilon_fraction * float(norms.max(initial=0)), 1e-12)
            eps_e = max(epsilon_fraction * float(jumps.max(initial=0)), 1e-12)
        log_objective = (.5 * np.sum((gain @ x - data) ** 2)
                         + np.sum(source_penalty * eps_a * np.log1p(norms / eps_a))
                         + np.sum(edge_penalty * eps_e * np.log1p(jumps / eps_e)))
        change = np.linalg.norm(x - previous) / max(np.linalg.norm(x), 1e-12)
        amplitude_weights, edge_weights = eps_a / (norms + eps_a), eps_e / (jumps + eps_e)
        local_certificate = _certificate(data, gain, incidence, x, rho * h,
            source_penalty * amplitude_weights, edge_penalty * edge_weights, linear_residual)
        history.append(dict(outer=outer, iterations=iteration, retries=(iteration - 1) // max_iter,
            converged=inner_converged, primal_relative=primal / magnitude, dual_relative=dual / magnitude,
            surrogate_reference=reference["objective"], surrogate_objective=certificate["objective"],
            surrogate_roundoff=roundoff, primal_dual_gap=certificate["gap"],
            primal_dual_gap_relative=certificate["gap_relative"], dual_scale=certificate["dual_scale"],
            dual_candidate=certificate["dual_candidate"],
            source_dual_violation=certificate["source_dual_violation"],
            log_objective=float(log_objective), local_stationarity_gap_relative=local_certificate["gap_relative"],
            relative_change=float(change), accepted_step=step, rho=float(rho),
            rho_updates=rho_updates - previous_rho_updates,
            active_rows=int(np.count_nonzero(norms > norms.max(initial=0) * .01))))
        if outer_iterations == 1:
            outer_converged = inner_converged
        elif outer > 0 and change <= outer_tolerance and local_certificate["gap_relative"] <= tolerance:
            outer_converged = True
        if outer_converged or stalled:
            break
    final_certificate = certificate if outer_iterations == 1 else local_certificate
    return x, dict(history=history, converged=outer_converged and not stalled,
        inner_converged=all(item["converged"] for item in history), outer_converged=outer_converged,
        final_stationarity_gap_relative=final_certificate["gap_relative"],
        numerical_stall=stalled, descent_guard_triggered=stalled,
        amplitude_epsilon=eps_a, edge_epsilon=eps_e,
        amplitude_weight_range=[float(amplitude_weights.min()), float(amplitude_weights.max())],
        edge_weight_range=[float(edge_weights.min(initial=1)), float(edge_weights.max(initial=0))],
        rho_initial=initial_rho, rho_final=float(rho), rho_updates=rho_updates,
        outer_tolerance=float(outer_tolerance), adaptive_rho=bool(adaptive_rho),
        convergence_note="Feasible primal-dual bound for the current convex MM surrogate; no global-optimality claim for log-sum.")
