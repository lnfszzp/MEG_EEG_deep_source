"""ERP-v5 experimental inverse: layerwise noise-calibrated graph reweighting.

The v4 implementation remains frozen. This variant changes penalty calibration
and temporal representation, not the grid, observations, window, or metrics.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.fft import dct
from scipy.sparse.linalg import splu

from algorithms.spatial_fused_fusion import _incidence_matrix
from candidates.oaster_rebuilt import _evoked_temporal_basis
from candidates.graph_reweight_solver import solve_reweighted_graph_v5


def _smooth_temporal_basis(data, baseline, active):
    """Keep the whole smooth ERP subspace, not just high-amplitude data modes.

    Uses the existing ERP smoothing scale (sigma=0.1*window width). Orthogonal
    DCT modes whose Gaussian attenuation is at least 1% are retained (about 10).
    This is an explicit temporal smoothness prior, not a learned significance
    cutoff or a spatial Gaussian template. Baseline uses exactly the same basis.
    """
    active = np.asarray(active, bool)
    if active.shape != baseline.shape or not active.any() or np.any(active & baseline):
        raise ValueError("active/baseline masks must be nonempty and disjoint")
    if any(np.any(np.diff(np.flatnonzero(mask)) != 1) for mask in (baseline, active)):
        raise ValueError("each mask must describe one continuous interval")
    width = int(active.sum())
    if width > baseline.sum():
        raise ValueError("baseline must be at least as long as the active window")
    attenuation = np.exp(-.5 * (np.pi * np.arange(width) * .1) ** 2)
    modes = dct(np.eye(width), type=2, norm="ortho", axis=0)[attenuation >= .01]
    basis = np.zeros((len(modes), data.shape[1]))
    basis[:, active] = modes
    return basis, dict(active_samples=width, temporal_rank=len(modes),
                       temporal_selection="amplitude_independent_smooth_subspace",
                       smoothing_fraction=.1, attenuation_floor=.01)


def reconstruct_evoked_oaster_v5_from_whitened(
        data, gain, n_surf, kernels=(), *, adjacency, baseline, active_windows,
        window_channel_weights=None, require_one=False, edge_fraction=0.5,
        noise_multiplier=1.0, mrf_strength=0.5, calibration="layer",
        temporal_mode="smooth", solver_kind="admm", surface_reweight_floor=0.,
        deep_reweight_floor=0., ridge_fraction=0., edge_penalty_mode="group",
        surface_penalty_multiplier=None, source_penalty_mode="group",
        **solver_settings):
    """Joint surface/deep solve; no truth, template selection or forced deep source.

    Calibration='global' is a numerical-solver-only ablation against v4.
    Calibration='layer' uses each layer's maximum baseline matched-filter score;
    7,498 cortical tests and cortical MRF amplification no longer set the penalty
    for the 16 deep candidates. These empirical cutoffs are not p-values.
    """
    data, gain = np.asarray(data, float), np.asarray(gain, float)
    baseline = np.asarray(baseline, bool)
    if (data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]
            or baseline.shape != (data.shape[1],) or not baseline.any()
            or not np.isfinite(data).all() or not np.isfinite(gain).all()
            or not 0 <= n_surf <= gain.shape[1] or len(active_windows) != 1
            or require_one or calibration not in {"global", "layer"}
            or temporal_mode not in {"v4", "smooth"}
            or solver_kind not in {"admm", "irls"}
            or edge_penalty_mode not in {"group", "elementwise"}
            or source_penalty_mode not in {"group", "surface_elementwise"}
            or not np.isfinite([
                edge_fraction, noise_multiplier, mrf_strength,
                surface_reweight_floor, deep_reweight_floor, ridge_fraction]).all()
            or edge_fraction < 0 or noise_multiplier <= 0 or not 0 <= mrf_strength < 1
            or not 0 <= surface_reweight_floor <= 1
            or not 0 <= deep_reweight_floor <= 1 or ridge_fraction < 0):
        raise ValueError("invalid v5 inputs; use one window and no forced source detection")
    if solver_kind != "admm" and (surface_reweight_floor or deep_reweight_floor):
        raise ValueError("reweight floors are supported only by the ADMM solver")
    if solver_kind != "admm" and ridge_fraction:
        raise ValueError("ridge_fraction is supported only by the ADMM solver")
    if solver_kind != "admm" and edge_penalty_mode != "group":
        raise ValueError("elementwise edge penalties are supported only by the ADMM solver")
    if solver_kind != "admm" and source_penalty_mode != "group":
        raise ValueError("surface-elementwise source penalties are supported only by the ADMM solver")
    if source_penalty_mode == "surface_elementwise" and not n_surf:
        raise ValueError("surface-elementwise source penalties require cortical sources")
    if "ridge_penalty" in solver_settings:
        raise ValueError("use the design-scaled ridge_fraction setting")
    graph = sparse.csr_matrix(adjacency)
    if graph.shape != (gain.shape[1], gain.shape[1]) or not np.isfinite(graph.data).all():
        raise ValueError("finite anatomical adjacency must match the complete source grid")
    upper = sparse.triu(graph[:n_surf, :n_surf].maximum(graph[:n_surf, :n_surf].T), k=1).tocoo()
    incidence = _incidence_matrix(gain.shape[1], upper.row, upper.col)
    mrf_factor = None
    if mrf_strength:
        neighbors = sparse.csr_matrix((np.ones(2 * upper.nnz),
            (np.r_[upper.row, upper.col], np.r_[upper.col, upper.row])), shape=graph.shape)
        degree = np.asarray(neighbors.sum(axis=1)).ravel()
        transition = sparse.diags(1 / np.maximum(degree, 1)) @ neighbors
        mrf_factor = splu((sparse.eye(gain.shape[1]) - mrf_strength * transition).tocsc())
    weights = (np.ones(data.shape[0]) if window_channel_weights is None
               else np.asarray(window_channel_weights[0], float))
    surface_penalty_multiplier = (np.ones(n_surf) if surface_penalty_multiplier is None
                                  else np.asarray(surface_penalty_multiplier, float))
    if weights.shape != (data.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("channel weights must be finite nonnegative sensor weights")
    if (surface_penalty_multiplier.shape != (n_surf,)
            or not np.isfinite(surface_penalty_multiplier).all()
            or np.any(surface_penalty_multiplier <= 0)):
        raise ValueError("surface penalty multipliers must be finite positive values")
    surface_penalty_range = ([float(surface_penalty_multiplier.min()),
                              float(surface_penalty_multiplier.max())]
                             if n_surf else [1., 1.])
    centered = data - data[:, baseline].mean(axis=1, keepdims=True)
    weighted_data = weights[:, None] * centered
    weighted_gain = weights[:, None] * gain
    active = np.asarray(active_windows[0], bool)
    basis, info = (_evoked_temporal_basis if temporal_mode == "v4" else _smooth_temporal_basis)(
        weighted_data, baseline, active)
    if solver_kind == "admm":
        solver_settings = {"outer_iterations": 20, "tolerance": .001, **solver_settings}
    metadata = dict(mode="joint_layer_calibrated_adaptive_graph", selected_templates=[],
                    spatial_templates_used=False, graph_edges=int(incidence.shape[0]),
                    deep_graph_edges=0, mrf_strength=mrf_strength, calibration=calibration,
                    surface_reweight_floor=float(surface_reweight_floor),
                    deep_reweight_floor=float(deep_reweight_floor),
                    ridge_fraction=float(ridge_fraction),
                    edge_penalty_mode=edge_penalty_mode,
                    source_penalty_mode=source_penalty_mode,
                    surface_penalty_multiplier_count=int(np.count_nonzero(
                        surface_penalty_multiplier != 1)),
                    surface_penalty_multiplier_range=surface_penalty_range,
                    temporal_mode=temporal_mode, solver_kind=solver_kind,
                    solver_settings=solver_settings,
                    weighting_coordinates="physical_current" if not mrf_strength else "MRF_current_innovation")
    if not basis.size:
        return np.zeros((gain.shape[1], data.shape[1])), dict(metadata, windows=[dict(info, solver=None)])
    calibration_basis = basis
    elementwise_coordinates = (source_penalty_mode == "surface_elementwise"
                               or edge_penalty_mode == "elementwise")
    if elementwise_coordinates:
        unrotated_response = weighted_data @ calibration_basis.T
        _, singular, rotation = np.linalg.svd(unrotated_response, full_matrices=True)
        basis = rotation @ calibration_basis
        info.update(temporal_rotation="training_sensor_svd_with_blockwise_null",
                    temporal_rotation_singular_values=singular.tolist())
    else:
        info.update(temporal_rotation="none")
    sensitivity = np.linalg.norm(weighted_gain, axis=0)
    if not np.any(sensitivity > 0):
        raise ValueError("lead field has no nonzero source columns")
    gain_scale = float(np.median(sensitivity[sensitivity > 0]))
    depth_weights = np.maximum(sensitivity / gain_scale, .1) ** .8
    design = weighted_gain / gain_scale
    if mrf_factor is not None:
        design = mrf_factor.solve(design.T, trans="T").T
    column_energy = np.sum(design ** 2, axis=0)
    ridge_scale = float(np.median(column_energy[column_energy > 0]))
    ridge_penalty = float(ridge_fraction * ridge_scale)
    response = weighted_data @ basis.T
    local_basis = basis[:, active].T
    calibration_local_basis = calibration_basis[:, active].T
    noise_projection = design.T @ weighted_data[:, baseline]
    baseline_data = weighted_data[:, baseline]
    width = local_basis.shape[0]
    starts = np.unique(np.linspace(0, noise_projection.shape[1] - width, 16).astype(int))
    layers = (slice(0, n_surf), slice(n_surf, gain.shape[1]))
    null, edge_null = [], []
    for start in starts:
        if elementwise_coordinates:
            block_response = baseline_data[:, start:start + width] @ calibration_local_basis
            block_rotation = np.linalg.svd(block_response, full_matrices=True)[2]
            projected = (noise_projection[:, start:start + width]
                         @ calibration_local_basis @ block_rotation.T)
        else:
            projected = noise_projection[:, start:start + width] @ local_basis
        surface_group_size = np.linalg.norm(projected[:n_surf], axis=1)
        surface_element_size = np.max(np.abs(projected[:n_surf]), axis=1, initial=0.)
        surface_size = (surface_element_size if source_penalty_mode == "surface_elementwise"
                        else surface_group_size)
        deep_size = np.linalg.norm(projected[n_surf:], axis=1)
        null.append([
            float((surface_size / depth_weights[:n_surf]).max(initial=0.)),
            float((deep_size / depth_weights[n_surf:]).max(initial=0.)),
        ])
        edge_size = surface_element_size if edge_penalty_mode == "elementwise" else surface_group_size
        edge_null.append(float((edge_size / depth_weights[:n_surf]).max(initial=0.)))
    null = np.asarray(null)
    layer_lambdas = noise_multiplier * np.quantile(null, .99, axis=0)
    global_lambda = float(noise_multiplier * np.quantile(null.max(axis=1), .99))
    edge_surface_lambda = float(noise_multiplier * np.quantile(edge_null, .99))
    if calibration == "global":
        layer_lambdas[:] = global_lambda
        edge_surface_lambda = global_lambda
    # An empty layer has no influence on the other layer's threshold.
    source_penalties = depth_weights.copy()
    for layer, threshold in zip(layers, layer_lambdas):
        source_penalties[layer] *= threshold
    source_penalties[:n_surf] *= surface_penalty_multiplier
    mean_degree = 2 * incidence.shape[0] / max(n_surf, 1)
    edge_penalty = float(edge_surface_lambda * edge_fraction / max(mean_degree, 1))
    spatial_solver = solve_reweighted_graph_v5
    if solver_kind == "irls":
        from candidates.graph_irls import solve_reweighted_graph_irls
        spatial_solver = solve_reweighted_graph_irls
    spatial_settings = dict(solver_settings)
    if solver_kind == "admm":
        amplitude_weight_floor = np.full(gain.shape[1], surface_reweight_floor)
        amplitude_weight_floor[n_surf:] = deep_reweight_floor
        spatial_settings["amplitude_weight_floor"] = amplitude_weight_floor
        spatial_settings["edge_penalty_mode"] = edge_penalty_mode
        if source_penalty_mode == "surface_elementwise":
            spatial_settings["source_penalty_mode"] = source_penalty_mode
            spatial_settings["elementwise_source_mask"] = (
                np.arange(gain.shape[1]) < n_surf)
        if ridge_penalty:
            spatial_settings["ridge_penalty"] = ridge_penalty
    coefficients, diagnostics = spatial_solver(
        response, design, incidence, source_penalty=source_penalties,
        edge_penalty=edge_penalty, **spatial_settings)
    if mrf_factor is not None:
        coefficients = mrf_factor.solve(coefficients)
    time_courses = np.linalg.pinv(response) @ weighted_data
    estimate = coefficients @ time_courses / gain_scale
    singular = np.linalg.svd(response, compute_uv=False)
    info.update(source_lambda_surface=float(layer_lambdas[0]),
                source_lambda_deep=float(layer_lambdas[1]),
                source_lambda_global_reference=global_lambda, edge_lambda=edge_penalty,
                edge_lambda_surface_reference=edge_surface_lambda,
                noise_projection_blocks=len(starts),
                ridge_penalty=ridge_penalty,
                ridge_scale=ridge_scale,
                response_condition=float(singular[0] / max(singular[-1], np.finfo(float).tiny)),
                null_rule=("separate layer maxima of baseline matched-filter norms; "
                           "elementwise modes repeat the sensor-SVD rotation in every null block; "
                           "empirical regularization, not p-values"),
                solver=diagnostics)
    return estimate, dict(metadata, windows=[info])
