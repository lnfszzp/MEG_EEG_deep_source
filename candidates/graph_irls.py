"""Experimental IRLS for a smooth source/edge log-sum objective.

This is a different objective from the nonsmooth ADMM candidate. Positive
smoothing radii, frozen from one ridge initialization, remove exact sparsity.
No wrapper defaults or historical algorithms are changed by this module.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu


def _objective_gradient_weights(x, data, gain, incidence, source_penalty,
                               edge_penalty, eps_a, eps_e, delta_a, delta_e):
    """Return the zero-offset objective, its analytic gradient and MM weights."""
    residual = gain @ x - data
    edges = incidence @ x
    radius_a = np.sqrt(np.sum(x * x, axis=1) + delta_a * delta_a)
    radius_e = np.sqrt(np.sum(edges * edges, axis=1) + delta_e * delta_e)
    weight_a = source_penalty * eps_a / (radius_a * (eps_a + radius_a))
    weight_e = edge_penalty * eps_e / (radius_e * (eps_e + radius_e))
    objective = (
        .5 * np.sum(residual * residual)
        + np.sum(source_penalty * eps_a * (
            np.log1p(radius_a / eps_a) - np.log1p(delta_a / eps_a)))
        + edge_penalty * eps_e * np.sum(
            np.log1p(radius_e / eps_e) - np.log1p(delta_e / eps_e))
    )
    gradient = gain.T @ residual + weight_a[:, None] * x + incidence.T @ (weight_e[:, None] * edges)
    return float(objective), gradient, weight_a, weight_e


def solve_reweighted_graph_irls(
        data, gain, incidence, *, source_penalty, edge_penalty,
        max_iter=100, tolerance=1e-5, epsilon_fraction=.05,
        smoothing_fraction=.01, ridge_fraction=.05,
        descent_tolerance=1e-10, linear_tolerance=1e-7):
    """Minimize a smooth log-sum by exact quadratic majorization steps.

    F(X) = .5 ||Y-GX||_F^2
         + sum_j lambda_j eps_a log(1 + sqrt(||X_j||^2+delta_a^2)/eps_a)
         + beta eps_e sum_e log(1 + sqrt(||(DX)_e||^2+delta_e^2)/eps_e).

    Constants at X=0 are subtracted only for reporting. Epsilons and deltas
    stay fixed throughout optimization. Since the penalties are concave in
    squared row norms, their tangents give Q=diag(w_a)+D.T diag(w_e) D.
    The exact step solves (G.T G + Q)X=G.T Y through sparse Q and sensor-space
    Woodbury. Positive source penalties guarantee Q is positive definite.
    Convergence means finite gradient/change tolerances, not global optimality.
    """
    data, gain = np.asarray(data, float), np.asarray(gain, float)
    if data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]:
        raise ValueError("data and gain must be matrices with the same sensor axis")
    if not data.size or not gain.size:
        raise ValueError("data and gain must be nonempty")
    incidence = sparse.csr_matrix(incidence, dtype=float)
    penalties = np.broadcast_to(np.asarray(source_penalty, float), (gain.shape[1],)).copy()
    parameters = (edge_penalty, tolerance, epsilon_fraction, smoothing_fraction,
                  ridge_fraction, descent_tolerance, linear_tolerance)
    if (incidence.shape[1] != gain.shape[1]
            or not all(np.isfinite(values).all() for values in (data, gain, incidence.data, penalties))
            or not np.isfinite(parameters).all() or np.any(penalties <= 0)
            or edge_penalty < 0 or any(value <= 0 for value in parameters[1:])
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
        raise ValueError("require finite inputs, positive source penalties/scales/tolerances, nonnegative edge penalty and positive max_iter")

    # One rotation-invariant ridge initialization; freeze both spatial scales.
    sensor_gram = gain @ gain.T
    ridge_scale = ridge_fraction * max(float(np.trace(sensor_gram) / gain.shape[0]), np.finfo(float).eps)
    initial = gain.T @ linalg.cho_solve(
        linalg.cho_factor(sensor_gram + ridge_scale * np.eye(gain.shape[0]), check_finite=False),
        data, check_finite=False)
    source_scale = max(float(np.linalg.norm(initial, axis=1).max(initial=0)), 1e-12)
    edge_scale = max(float(np.linalg.norm(incidence @ initial, axis=1).max(initial=0)), 1e-12)
    eps_a, eps_e = epsilon_fraction * source_scale, epsilon_fraction * edge_scale
    delta_a, delta_e = smoothing_fraction * source_scale, smoothing_fraction * edge_scale
    x = initial
    objective, gradient, weight_a, weight_e = _objective_gradient_weights(
        x, data, gain, incidence, penalties, edge_penalty, eps_a, eps_e, delta_a, delta_e)
    initial_objective = objective
    gain_data = gain.T @ data
    gradient_reference = max(float(np.linalg.norm(gain_data)), np.finfo(float).tiny)
    gradient_relative = float(np.linalg.norm(gradient) / gradient_reference)
    history = []
    relative_change = 0.
    converged = gradient_relative <= tolerance
    descent_guard = linear_guard = False
    rejected_step = None
    for iteration in range(1, max_iter + 1):
        if converged:
            break
        quadratic = (sparse.diags(weight_a) + incidence.T @ sparse.diags(weight_e) @ incidence).tocsc()
        q_gain = splu(quadratic).solve(gain.T)
        sensor = np.eye(gain.shape[0]) + gain @ q_gain
        sensor = (sensor + sensor.T) * .5
        proposed = q_gain @ linalg.cho_solve(
            linalg.cho_factor(sensor, check_finite=False), data, check_finite=False)
        linear_residual = gain.T @ (gain @ proposed) + quadratic @ proposed - gain_data
        linear_relative = float(np.linalg.norm(linear_residual) / gradient_reference)
        if not np.isfinite(proposed).all() or not np.isfinite(linear_relative) or linear_relative > linear_tolerance:
            linear_guard = True
            rejected_step = dict(iteration=iteration, reason="linear_solve_residual", linear_relative=linear_relative)
            break
        next_objective, next_gradient, next_weight_a, next_weight_e = _objective_gradient_weights(
            proposed, data, gain, incidence, penalties, edge_penalty, eps_a, eps_e, delta_a, delta_e)
        allowed_increase = descent_tolerance * max(1., abs(objective))
        if not np.isfinite(next_objective) or next_objective > objective + allowed_increase:
            descent_guard = True
            rejected_step = dict(iteration=iteration, reason="objective_increase",
                                 previous_objective=objective, proposed_objective=next_objective,
                                 allowed_increase=allowed_increase)
            break
        relative_change = float(np.linalg.norm(proposed - x) / max(np.linalg.norm(proposed), 1e-12))
        gradient_relative = float(np.linalg.norm(next_gradient) / gradient_reference)
        converged = gradient_relative <= tolerance and relative_change <= tolerance
        history.append(dict(outer=iteration - 1, iterations=1, converged=True,
            linear_relative_residual=linear_relative, gradient_relative=gradient_relative,
            relative_change=relative_change, log_objective=next_objective,
            objective_decrease=objective - next_objective,
            objective_descent_slack=allowed_increase,
            active_rows=int(np.count_nonzero(np.linalg.norm(proposed, axis=1) >
                                            .01 * np.linalg.norm(proposed, axis=1).max(initial=0)))))
        x, objective, gradient = proposed, next_objective, next_gradient
        weight_a, weight_e = next_weight_a, next_weight_e
    return x, dict(
        mode="smooth_logsum_quadratic_majorization_irls", history=history,
        converged=bool(converged), inner_converged=not linear_guard,
        outer_converged=bool(converged), descent_guard_triggered=descent_guard,
        linear_guard_triggered=linear_guard, descent_guard_step=rejected_step,
        initial_log_objective=initial_objective, final_log_objective=objective,
        gradient_relative=gradient_relative, relative_change=relative_change,
        gradient_reference=gradient_reference, gradient_normalization="norm(G.T @ Y)",
        iterations=len(history), ridge_scale=float(ridge_scale),
        source_initial_scale=source_scale, edge_initial_scale=edge_scale,
        amplitude_epsilon=eps_a, edge_epsilon=eps_e,
        amplitude_delta=delta_a, edge_delta=delta_e,
        epsilon_fraction=epsilon_fraction, smoothing_fraction=smoothing_fraction,
        ridge_fraction=ridge_fraction, tolerance=tolerance,
        objective_descent_tolerance=descent_tolerance, linear_tolerance=linear_tolerance,
        source_quadratic_weight_range=[float(weight_a.min()), float(weight_a.max())],
        edge_quadratic_weight_range=([float(weight_e.min()), float(weight_e.max())] if weight_e.size else []),
        smoothing_note="Positive deltas change the objective and remove exact sparsity; no hard threshold applied.",
        convergence_note="Stationarity and iterate-change tolerances for this smooth nonconvex objective; no global-optimality claim.",
    )


if __name__ == "__main__":
    # Small runnable checks; no full-grid inverse or external data required.
    rng = np.random.default_rng(611)
    gain = rng.standard_normal((7, 12)) / np.sqrt(7)
    truth = np.zeros((12, 3))
    truth[3:6] = rng.standard_normal((1, 3))
    data = gain @ truth + .01 * rng.standard_normal((7, 3))
    incidence = sparse.diags((-np.ones(11), np.ones(11)), (0, 1), shape=(11, 12)).tocsr()
    penalties = np.linspace(.06, .12, 12)
    edge_penalty = .04
    trial = rng.standard_normal((12, 3)) * .2
    value, gradient, _, _ = _objective_gradient_weights(trial, data, gain, incidence, penalties, edge_penalty, .1, .12, .02, .03)
    numerical = np.empty_like(trial)
    for index in np.ndindex(trial.shape):
        plus, minus = trial.copy(), trial.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (
            _objective_gradient_weights(plus, data, gain, incidence, penalties, edge_penalty, .1, .12, .02, .03)[0]
            - _objective_gradient_weights(minus, data, gain, incidence, penalties, edge_penalty, .1, .12, .02, .03)[0]) / 2e-6
    assert np.allclose(gradient, numerical, atol=2e-8, rtol=2e-6), "analytic gradient disagrees with finite differences"
    estimate, diagnostics = solve_reweighted_graph_irls(
        data, gain, incidence, source_penalty=penalties, edge_penalty=edge_penalty, max_iter=400, tolerance=1e-7)
    objectives = [diagnostics["initial_log_objective"], *[item["log_objective"] for item in diagnostics["history"]]]
    assert np.max(np.diff(objectives)) <= 1e-10 * max(1., objectives[0]), "objective must descend"
    assert diagnostics["converged"] and diagnostics["gradient_relative"] <= 1e-7, diagnostics
    rotation = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    rotated, rotated_diagnostics = solve_reweighted_graph_irls(
        data @ rotation, gain, incidence, source_penalty=penalties, edge_penalty=edge_penalty, max_iter=400, tolerance=1e-7)
    assert np.allclose(rotated, estimate @ rotation, atol=2e-7, rtol=2e-6), "time-basis rotations must preserve the spatial solution"
    assert np.isclose(rotated_diagnostics["final_log_objective"], diagnostics["final_log_objective"], atol=1e-10)
    zero, zero_diagnostics = solve_reweighted_graph_irls(
        np.zeros_like(data), gain, incidence, source_penalty=penalties, edge_penalty=edge_penalty)
    assert not np.any(zero) and zero_diagnostics["converged"]
    print(f"PASS: gradient finite differences, objective descent, stationarity, temporal rotation, zero data; "
          f"iterations={diagnostics['iterations']}, relative_gradient={diagnostics['gradient_relative']:.3g}")
