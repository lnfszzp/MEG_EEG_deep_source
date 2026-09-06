from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.csgraph import connected_components


EPS = np.finfo(float).eps


@dataclass
class GraphFusedResult:
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    objective: float
    relative_residual: float
    iterations: int
    primal_residual: float
    dual_residual: float
    converged: bool
    edge_left: np.ndarray
    edge_right: np.ndarray
    edge_weights: np.ndarray


@dataclass
class SpatialPathPoint:
    index: int
    lambda_fraction: float
    beta_fraction: float
    deep_fusion_scale: float
    lam: float
    beta: float
    result: GraphFusedResult
    active_count: int
    jump_count: int
    fused_group_count: int
    bic: float


@dataclass
class SpatialPathSelection:
    index: int
    point: SpatialPathPoint
    points: list[SpatialPathPoint]


def extract_graph_edges(
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    deep_mask: np.ndarray | None = None,
    deep_fusion_scale: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract unique anatomical graph edges and their fusion weights."""
    graph = sparse.coo_matrix(adjacency)
    keep = (graph.row < graph.col) & (graph.data != 0)
    left = graph.row[keep].astype(int)
    right = graph.col[keep].astype(int)
    weights = np.asarray(graph.data[keep], dtype=float)
    weights = np.abs(weights)
    weights[weights == 0] = 1.0

    if deep_mask is not None:
        deep_mask = np.asarray(deep_mask, dtype=bool).ravel()
        if deep_mask.size != graph.shape[0]:
            raise ValueError("deep_mask must match adjacency size")
        cross_type = deep_mask[left] != deep_mask[right]
        retain = ~cross_type
        left = left[retain]
        right = right[retain]
        weights = weights[retain]
        deep_edges = deep_mask[left] & deep_mask[right]
        weights[deep_edges] *= float(deep_fusion_scale)
    return left, right, weights


def expand_candidates(
    seed_indices: Iterable[int],
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    hops: int,
) -> np.ndarray:
    """Expand seed locations by a fixed number of graph hops."""
    graph = sparse.csr_matrix(adjacency)
    selected = np.zeros(graph.shape[0], dtype=bool)
    seeds = np.asarray(list(seed_indices), dtype=int)
    seeds = seeds[(seeds >= 0) & (seeds < graph.shape[0])]
    selected[seeds] = True
    frontier = selected.copy()
    for _ in range(max(0, int(hops))):
        reached = np.asarray(graph @ frontier.astype(np.int8)).ravel() > 0
        reached &= ~selected
        selected |= reached
        frontier = reached
        if not np.any(frontier):
            break
    return np.flatnonzero(selected)


def vector_soft_threshold(
    values: np.ndarray,
    thresholds: float | np.ndarray,
) -> np.ndarray:
    """Apply row-wise L2 soft thresholding."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape rows x features")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full(values.shape[0], float(thresholds))
    thresholds = thresholds.ravel()
    if thresholds.size != values.shape[0]:
        raise ValueError("thresholds must be scalar or one value per row")
    norms = np.linalg.norm(values, axis=1)
    scale = np.maximum(0.0, 1.0 - thresholds / np.maximum(norms, EPS))
    return values * scale[:, None]


def _incidence_matrix(
    n_sources: int,
    left: np.ndarray,
    right: np.ndarray,
) -> sparse.csr_matrix:
    if left.size == 0:
        return sparse.csr_matrix((0, n_sources), dtype=float)
    rows = np.repeat(np.arange(left.size), 2)
    cols = np.column_stack([left, right]).ravel()
    values = np.tile(np.array([1.0, -1.0]), left.size)
    return sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(left.size, n_sources),
    ).tocsr()


def solve_graph_fused_lasso(
    data: np.ndarray,
    gain: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    lam: float,
    beta: float,
    deep_mask: np.ndarray | None = None,
    deep_fusion_scale: float = 0.35,
    rho: float = 1.0,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
    initial_coefficients: np.ndarray | None = None,
) -> GraphFusedResult:
    """Solve multitask Graph Fused Lasso with two-split ADMM."""
    data = np.asarray(data, dtype=float)
    gain = np.asarray(gain, dtype=float)
    if data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]:
        raise ValueError("data and gain must be 2D and share the sensor axis")
    if sparse.csr_matrix(adjacency).shape != (gain.shape[1], gain.shape[1]):
        raise ValueError("adjacency must match the source axis")
    n_sources = gain.shape[1]
    lam_array = np.asarray(lam, dtype=float)
    if lam_array.ndim == 0:
        lam_threshold = float(lam_array)
        lam_objective = np.full(n_sources, lam_threshold, dtype=float)
    else:
        lam_threshold = lam_array.ravel()
        if lam_threshold.size != n_sources:
            raise ValueError("lam vector must have one value per source")
        lam_objective = lam_threshold
    if np.any(np.asarray(lam_threshold) < 0) or beta < 0 or rho <= 0:
        raise ValueError("lam, beta and rho must be non-negative, with rho positive")

    n_times = data.shape[1]
    left, right, edge_weights = extract_graph_edges(
        adjacency,
        deep_mask=deep_mask,
        deep_fusion_scale=deep_fusion_scale,
    )
    incidence = _incidence_matrix(n_sources, left, right)
    laplacian = (incidence.T @ incidence).toarray()
    system = gain.T @ gain + rho * np.eye(n_sources) + rho * laplacian
    factor = linalg.cho_factor(system, lower=True, check_finite=False)
    gain_data = gain.T @ data

    if initial_coefficients is None:
        coefficients = np.zeros((n_sources, n_times), dtype=float)
    else:
        coefficients = np.asarray(initial_coefficients, dtype=float).copy()
        if coefficients.shape != (n_sources, n_times):
            raise ValueError("initial_coefficients has incompatible shape")
    source_split = coefficients.copy()
    edge_split = incidence @ coefficients
    source_dual = np.zeros_like(coefficients)
    edge_dual = np.zeros_like(edge_split)
    primal = np.inf
    dual = np.inf
    converged = False

    for iteration in range(1, int(max_iter) + 1):
        previous_source = source_split.copy()
        previous_edge = edge_split.copy()
        rhs = gain_data + rho * (source_split - source_dual)
        if incidence.shape[0]:
            rhs += rho * (incidence.T @ (edge_split - edge_dual))
        coefficients = linalg.cho_solve(
            factor,
            rhs,
            check_finite=False,
        )
        source_split = vector_soft_threshold(
            coefficients + source_dual,
            np.asarray(lam_threshold, dtype=float) / rho,
        )
        edge_values = incidence @ coefficients
        if incidence.shape[0]:
            edge_split = vector_soft_threshold(
                edge_values + edge_dual,
                float(beta) * edge_weights / rho,
            )
        source_dual += coefficients - source_split
        if incidence.shape[0]:
            edge_dual += edge_values - edge_split

        source_primal = np.linalg.norm(coefficients - source_split, "fro") ** 2
        edge_primal = (
            np.linalg.norm(edge_values - edge_split, "fro") ** 2
            if incidence.shape[0]
            else 0.0
        )
        primal = float(np.sqrt(source_primal + edge_primal))
        source_dual_norm = np.linalg.norm(source_split - previous_source, "fro") ** 2
        edge_dual_norm = (
            np.linalg.norm(incidence.T @ (edge_split - previous_edge), "fro") ** 2
            if incidence.shape[0]
            else 0.0
        )
        dual = float(rho * np.sqrt(source_dual_norm + edge_dual_norm))
        scale = max(
            1.0,
            float(np.linalg.norm(coefficients, "fro")),
            float(np.linalg.norm(source_split, "fro")),
            float(np.linalg.norm(edge_values, "fro")),
        )
        if primal <= tolerance * scale and dual <= tolerance * scale:
            converged = True
            break

    fitted = gain @ coefficients
    residual = data - fitted
    group_norms = np.linalg.norm(coefficients, axis=1)
    group_penalty = float(np.sum(lam_objective * group_norms))
    fused_penalty = (
        float(np.sum(edge_weights * np.linalg.norm(incidence @ coefficients, axis=1)))
        if incidence.shape[0]
        else 0.0
    )
    objective = (
        0.5 * float(np.linalg.norm(residual, "fro") ** 2)
        + group_penalty
        + float(beta) * fused_penalty
    )
    relative_residual = float(
        np.linalg.norm(residual, "fro") / max(np.linalg.norm(data, "fro"), EPS)
    )
    return GraphFusedResult(
        coefficients=coefficients,
        fitted=fitted,
        residual=residual,
        objective=objective,
        relative_residual=relative_residual,
        iterations=iteration,
        primal_residual=primal,
        dual_residual=dual,
        converged=converged,
        edge_left=left,
        edge_right=right,
        edge_weights=edge_weights,
    )


def select_spatial_bic(
    residual_sums: np.ndarray,
    active_counts: np.ndarray,
    jump_counts: np.ndarray,
    *,
    n_observations: int,
    temporal_rank: int,
) -> tuple[int, np.ndarray]:
    """Select spatial regularization without using source truth."""
    residual_sums = np.asarray(residual_sums, dtype=float).ravel()
    active_counts = np.asarray(active_counts, dtype=int).ravel()
    jump_counts = np.asarray(jump_counts, dtype=int).ravel()
    if not (
        residual_sums.size
        and residual_sums.shape == active_counts.shape == jump_counts.shape
    ):
        raise ValueError("path arrays must be non-empty and aligned")
    variance = np.maximum(residual_sums / float(n_observations), EPS)
    # For graph-fused estimates the effective degrees of freedom are governed
    # by fused plateaus, not by every grid point inside a plateau.  The caller
    # therefore passes fused-group counts through `active_counts`; jump counts
    # remain diagnostic and are validated for path alignment.
    parameter_counts = active_counts.astype(float) * float(temporal_rank)
    scores = (
        float(n_observations) * np.log(variance)
        + parameter_counts * np.log(float(n_observations))
    )
    return int(np.argmin(scores)), scores


def effective_fused_group_count(
    coefficients: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    active_threshold: float = 0.05,
    jump_threshold: float = 0.05,
) -> int:
    """Estimate Graph Fused degrees of freedom as nonzero fused plateaus."""
    coefficients = np.asarray(coefficients, dtype=float)
    amplitude = np.linalg.norm(coefficients, axis=1)
    maximum = float(amplitude.max(initial=0.0))
    if maximum <= 0:
        return 0
    active = np.flatnonzero(amplitude >= float(active_threshold) * maximum)
    if active.size == 0:
        return 0
    local_graph = sparse.coo_matrix(
        sparse.csr_matrix(adjacency)[active][:, active]
    )
    keep = local_graph.row < local_graph.col
    left = local_graph.row[keep]
    right = local_graph.col[keep]
    differences = np.linalg.norm(
        coefficients[active[left]] - coefficients[active[right]],
        axis=1,
    )
    fused = differences <= float(jump_threshold) * maximum
    rows = np.concatenate([left[fused], right[fused]])
    cols = np.concatenate([right[fused], left[fused]])
    plateau_graph = sparse.csr_matrix(
        (np.ones(rows.size), (rows, cols)),
        shape=(active.size, active.size),
    )
    return int(connected_components(plateau_graph, directed=False)[0])


def solve_spatial_fused_path(
    data: np.ndarray,
    gain: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    lambda_fractions: Iterable[float],
    beta_fractions: Iterable[float],
    lambda_weights: np.ndarray | None = None,
    deep_mask: np.ndarray | None = None,
    deep_fusion_scale: float = 0.35,
    deep_fusion_scales: Iterable[float] | None = None,
    active_threshold: float = 0.05,
    jump_threshold: float = 0.05,
    **solver_kwargs,
) -> SpatialPathSelection:
    """Solve a 2D truth-free lambda/beta path and select it by spatial BIC."""
    data = np.asarray(data, dtype=float)
    gain = np.asarray(gain, dtype=float)
    lambda_fractions = tuple(float(value) for value in lambda_fractions)
    beta_fractions = tuple(float(value) for value in beta_fractions)
    deep_fusion_scales = (
        (float(deep_fusion_scale),)
        if deep_fusion_scales is None
        else tuple(float(value) for value in deep_fusion_scales)
    )
    if not lambda_fractions or not beta_fractions or not deep_fusion_scales:
        raise ValueError("lambda and beta paths must be non-empty")
    correlations = np.linalg.norm(gain.T @ data, axis=1)
    if lambda_weights is None:
        lambda_weight_vector = np.ones(gain.shape[1], dtype=float)
    else:
        lambda_weight_vector = np.asarray(lambda_weights, dtype=float).ravel()
        if lambda_weight_vector.size != gain.shape[1]:
            raise ValueError("lambda_weights must have one value per source")
        if np.any(lambda_weight_vector <= 0):
            raise ValueError("lambda_weights must be positive")
    lambda_max = float(
        np.max(correlations / np.maximum(lambda_weight_vector, EPS), initial=0.0)
    )
    if lambda_max <= 0:
        raise ValueError("data has zero correlation with every candidate source")
    solver_kwargs = dict(solver_kwargs)
    solver_kwargs.setdefault("rho", max(1.0, 0.001 * lambda_max))

    points: list[SpatialPathPoint] = []
    initial = None
    for lambda_fraction in lambda_fractions:
        for beta_fraction in beta_fractions:
            for current_deep_scale in deep_fusion_scales:
                result = solve_graph_fused_lasso(
                    data,
                    gain,
                    adjacency,
                    lam=lambda_fraction * lambda_max * lambda_weight_vector,
                    beta=beta_fraction * lambda_max,
                    deep_mask=deep_mask,
                    deep_fusion_scale=current_deep_scale,
                    initial_coefficients=initial,
                    **solver_kwargs,
                )
                initial = result.coefficients
                amplitude = np.linalg.norm(result.coefficients, axis=1)
                amplitude_limit = active_threshold * amplitude.max(initial=0.0)
                active_count = int(np.sum(amplitude >= amplitude_limit)) if amplitude_limit > 0 else 0
                if result.edge_left.size:
                    jumps = result.coefficients[result.edge_left] - result.coefficients[result.edge_right]
                    jump_norm = np.linalg.norm(jumps, axis=1)
                    jump_limit = jump_threshold * amplitude.max(initial=0.0)
                    jump_count = int(np.sum(jump_norm >= jump_limit)) if jump_limit > 0 else 0
                else:
                    jump_count = 0
                fused_group_count = effective_fused_group_count(
                    result.coefficients,
                    adjacency,
                    active_threshold=active_threshold,
                    jump_threshold=jump_threshold,
                )
                points.append(
                    SpatialPathPoint(
                        index=len(points),
                        lambda_fraction=lambda_fraction,
                        beta_fraction=beta_fraction,
                        deep_fusion_scale=current_deep_scale,
                        lam=lambda_fraction * lambda_max,
                        beta=beta_fraction * lambda_max,
                        result=result,
                        active_count=active_count,
                        jump_count=jump_count,
                        fused_group_count=fused_group_count,
                        bic=np.nan,
                    )
                )
    selected_index, scores = select_spatial_bic(
        np.asarray([np.linalg.norm(point.result.residual, "fro") ** 2 for point in points]),
        np.asarray([point.fused_group_count for point in points]),
        np.asarray([point.jump_count for point in points]),
        n_observations=data.size,
        temporal_rank=data.shape[1],
    )
    for point, score in zip(points, scores):
        point.bic = float(score)
    return SpatialPathSelection(
        index=selected_index,
        point=points[selected_index],
        points=points,
    )


def region_mask_from_source(
    source: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    relative_threshold: float = 0.20,
    minimum_component_energy_fraction: float = 0.01,
) -> np.ndarray:
    """Convert source amplitudes into an energy-filtered connected range mask."""
    source = np.asarray(source, dtype=float)
    amplitude = (
        np.abs(source)
        if source.ndim == 1
        else np.linalg.norm(source, axis=1)
    )
    maximum = float(amplitude.max(initial=0.0))
    if maximum <= 0:
        return np.zeros(amplitude.size, dtype=bool)
    mask = amplitude >= float(relative_threshold) * maximum
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return mask
    graph = sparse.csr_matrix(adjacency)[selected][:, selected]
    count, labels = connected_components(graph, directed=False)
    total_energy = float(np.sum(amplitude[selected] ** 2))
    for label in range(count):
        members = selected[labels == label]
        energy = float(np.sum(amplitude[members] ** 2))
        if energy < float(minimum_component_energy_fraction) * max(total_energy, EPS):
            mask[members] = False
    return mask


def anatomical_region_mask(
    source: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    deep_mask: np.ndarray,
    *,
    surface_relative_threshold: float = 0.10,
    deep_relative_threshold: float = 0.20,
    minimum_domain_energy_fraction: float = 0.01,
    minimum_component_energy_fraction: float = 0.001,
) -> np.ndarray:
    """Extract surface and deep ranges relative to their own domain peaks."""
    source = np.asarray(source, dtype=float)
    deep_mask = np.asarray(deep_mask, dtype=bool).ravel()
    if source.shape[0] != deep_mask.size:
        raise ValueError("deep_mask must match the source axis")
    amplitude = (
        np.abs(source)
        if source.ndim == 1
        else np.linalg.norm(source, axis=1)
    )
    total_energy = float(np.sum(amplitude**2))
    output = np.zeros(amplitude.size, dtype=bool)
    graph = sparse.csr_matrix(adjacency)
    for domain_mask, threshold in (
        (~deep_mask, surface_relative_threshold),
        (deep_mask, deep_relative_threshold),
    ):
        indices = np.flatnonzero(domain_mask)
        if indices.size == 0:
            continue
        domain_energy = float(np.sum(amplitude[indices] ** 2))
        if domain_energy < float(minimum_domain_energy_fraction) * max(total_energy, EPS):
            continue
        domain_source = source[indices]
        domain_graph = graph[indices][:, indices]
        local = region_mask_from_source(
            domain_source,
            domain_graph,
            relative_threshold=threshold,
            minimum_component_energy_fraction=minimum_component_energy_fraction,
        )
        output[indices[local]] = True
    return output


def range_metrics(
    estimated_mask: np.ndarray,
    truth_mask: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
) -> dict:
    """Compute binary source-range overlap and connected-component metrics."""
    estimated = np.asarray(estimated_mask, dtype=bool).ravel()
    truth = np.asarray(truth_mask, dtype=bool).ravel()
    if estimated.shape != truth.shape:
        raise ValueError("estimated and truth masks must align")
    intersection = int(np.sum(estimated & truth))
    estimated_count = int(np.sum(estimated))
    truth_count = int(np.sum(truth))
    dice = 2.0 * intersection / max(estimated_count + truth_count, 1)
    precision = intersection / max(estimated_count, 1)
    recall = intersection / max(truth_count, 1)
    size_ratio = estimated_count / max(truth_count, 1)
    selected = np.flatnonzero(estimated)
    if selected.size:
        graph = sparse.csr_matrix(adjacency)[selected][:, selected]
        component_count = int(connected_components(graph, directed=False)[0])
    else:
        component_count = 0
    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "size_ratio": float(size_ratio),
        "estimated_count": estimated_count,
        "truth_count": truth_count,
        "intersection_count": intersection,
        "estimated_component_count": component_count,
    }


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC from ranks, including half credit for tied scores."""
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must align")
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size == 0 or negative.size == 0:
        return np.nan
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def _source_amplitude(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        return np.abs(source)
    if source.ndim == 2:
        return np.linalg.norm(source, axis=1)
    raise ValueError("source must be a vector or source-by-time matrix")


def _source_energy(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        return source**2
    if source.ndim == 2:
        return np.sum(source**2, axis=1)
    raise ValueError("source must be a vector or source-by-time matrix")


def _normalize_frobenius(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    norm = float(np.linalg.norm(source))
    return source / norm if norm > EPS else np.zeros_like(source)


def _graph_neighborhood_mask(
    adjacency: np.ndarray | sparse.spmatrix,
    centers: np.ndarray,
    *,
    order: int,
    domain_mask: np.ndarray,
) -> np.ndarray:
    graph = sparse.csr_matrix(adjacency)
    if graph.shape[0] != domain_mask.size or graph.shape[1] != domain_mask.size:
        raise ValueError("adjacency must be square and match source count")
    centers = np.asarray(centers, dtype=int).ravel()
    centers = centers[
        (centers >= 0) & (centers < domain_mask.size) & domain_mask[centers]
    ]
    reached = np.zeros(domain_mask.size, dtype=bool)
    if centers.size == 0:
        return reached
    reached[centers] = True
    frontier = reached.copy()
    for _ in range(int(order)):
        neighbors = np.asarray(graph @ frontier.astype(float)).ravel() > 0
        neighbors &= domain_mask
        new_frontier = neighbors & ~reached
        if not np.any(new_frontier):
            break
        reached |= new_frontier
        frontier = new_frontier
    return reached


def _sampled_auc(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    negative_pool: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> float:
    positives = np.flatnonzero(positive_mask)
    negatives = np.flatnonzero(negative_pool)
    if positives.size == 0 or negatives.size == 0:
        return np.nan
    values: list[float] = []
    for _ in range(int(repetitions)):
        sampled = rng.choice(
            negatives,
            size=positives.size,
            replace=negatives.size < positives.size,
        )
        labels = np.r_[np.ones(positives.size, dtype=bool), np.zeros(sampled.size, dtype=bool)]
        sampled_scores = np.r_[scores[positives], scores[sampled]]
        values.append(binary_roc_auc(labels, sampled_scores))
    return float(np.nanmean(values)) if values else np.nan


def pdf_source_metrics(
    estimated_source: np.ndarray,
    truth_source: np.ndarray,
    positions_m: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    domain_mask: np.ndarray | None = None,
    centers: np.ndarray | None = None,
    auc_repetitions: int = 50,
    random_state: int = 0,
    neighborhood_order: int = 10,
) -> dict:
    """Compute AUC, SD, DLE and RMSE using the provided PDF definitions."""
    estimated = np.asarray(estimated_source, dtype=float)
    truth = np.asarray(truth_source, dtype=float)
    if estimated.shape != truth.shape:
        raise ValueError("estimated_source and truth_source must align")
    n_sources = estimated.shape[0]
    positions = np.asarray(positions_m, dtype=float)
    if positions.shape != (n_sources, 3):
        raise ValueError("positions_m must have shape n_sources x 3")
    if domain_mask is None:
        domain = np.ones(n_sources, dtype=bool)
    else:
        domain = np.asarray(domain_mask, dtype=bool).ravel()
        if domain.size != n_sources:
            raise ValueError("domain_mask must match source count")

    truth_energy = _source_energy(truth)
    estimated_energy = _source_energy(estimated)
    active = domain & (truth_energy > EPS)
    if centers is None:
        centers = np.flatnonzero(active)
    centers = np.asarray(centers, dtype=int).ravel()

    close = _graph_neighborhood_mask(
        adjacency,
        centers,
        order=neighborhood_order,
        domain_mask=domain,
    )
    close_pool = close & ~active
    far_pool = domain & ~close & ~active
    rng = np.random.default_rng(int(random_state))
    auc_close = _sampled_auc(
        estimated_energy,
        active,
        close_pool,
        rng,
        auc_repetitions,
    )
    auc_far = _sampled_auc(
        estimated_energy,
        active,
        far_pool,
        rng,
        auc_repetitions,
    )
    auc_parts = [value for value in (auc_close, auc_far) if not np.isnan(value)]
    auc = float(np.mean(auc_parts)) if auc_parts else np.nan

    estimated_sum = float(np.sum(estimated_energy[domain]))
    domain_estimated = np.zeros_like(estimated)
    domain_truth = np.zeros_like(truth)
    domain_estimated[domain] = estimated[domain]
    domain_truth[domain] = truth[domain]
    normalized_estimated = _normalize_frobenius(domain_estimated)
    normalized_truth = _normalize_frobenius(domain_truth)
    truth_norm = float(np.linalg.norm(normalized_truth))
    rmse = (
        float(np.linalg.norm(normalized_estimated - normalized_truth) / truth_norm)
        if truth_norm > EPS
        else np.nan
    )

    active_indices = np.flatnonzero(active)
    estimated_positive = domain & (estimated_energy > EPS)
    if active_indices.size == 0 or estimated_sum <= EPS:
        return {
            "auc_close": auc_close,
            "auc_far": auc_far,
            "auc": auc,
            "sd_mm": np.nan,
            "dle_mm": np.nan,
            "rmse": rmse,
        }

    domain_indices = np.flatnonzero(domain)
    distances = np.linalg.norm(
        positions[domain_indices, None, :] - positions[active_indices][None, :, :],
        axis=2,
    )
    nearest_active_local = np.argmin(distances, axis=1)
    nearest_distance_mm = distances[
        np.arange(domain_indices.size),
        nearest_active_local,
    ] * 1000.0
    weights = estimated_energy[domain_indices] / estimated_sum
    sd_mm = float(np.sqrt(np.sum(weights * nearest_distance_mm**2)))

    dle_values: list[float] = []
    estimated_indices = np.flatnonzero(estimated_positive)
    if estimated_indices.size:
        estimated_distances = np.linalg.norm(
            positions[estimated_indices, None, :]
            - positions[active_indices][None, :, :],
            axis=2,
        )
        nearest_for_estimated = np.argmin(estimated_distances, axis=1)
        for active_local, active_index in enumerate(active_indices):
            assigned = estimated_indices[nearest_for_estimated == active_local]
            if assigned.size == 0:
                continue
            peak = int(assigned[np.argmax(estimated_energy[assigned])])
            dle_values.append(
                float(np.linalg.norm(positions[peak] - positions[active_index]) * 1000.0)
            )
    dle_mm = float(np.mean(dle_values)) if dle_values else np.nan

    return {
        "auc_close": auc_close,
        "auc_far": auc_far,
        "auc": auc,
        "sd_mm": sd_mm,
        "dle_mm": dle_mm,
        "rmse": rmse,
    }


def continuous_source_metrics(
    estimated_source: np.ndarray,
    truth_source: np.ndarray,
    positions_m: np.ndarray,
    domain_mask: np.ndarray,
) -> dict:
    """Evaluate continuous source amplitudes inside one anatomical domain."""
    estimated = _source_amplitude(estimated_source)
    truth = _source_amplitude(truth_source)
    positions = np.asarray(positions_m, dtype=float)
    domain = np.asarray(domain_mask, dtype=bool).ravel()
    if estimated.shape != truth.shape or estimated.shape != domain.shape:
        raise ValueError("source amplitudes and domain mask must align")
    if positions.shape != (estimated.size, 3):
        raise ValueError("positions_m must have shape n_sources x 3")

    indices = np.flatnonzero(domain)
    if indices.size == 0:
        return {
            "auc": np.nan,
            "rmse": np.nan,
            "sd_mm": np.nan,
            "peak_dle_mm": np.nan,
            "centroid_dle_mm": np.nan,
        }
    estimated_domain = estimated[indices]
    truth_domain = truth[indices]
    truth_positive = truth_domain > 0
    auc = binary_roc_auc(truth_positive, estimated_domain)

    estimated_sum = float(estimated_domain.sum())
    truth_sum = float(truth_domain.sum())
    estimated_normalized = (
        estimated_domain / estimated_sum
        if estimated_sum > EPS
        else np.zeros_like(estimated_domain)
    )
    truth_normalized = (
        truth_domain / truth_sum
        if truth_sum > EPS
        else np.zeros_like(truth_domain)
    )
    rmse = float(
        np.sqrt(np.mean((estimated_normalized - truth_normalized) ** 2))
    )

    if not np.any(truth_positive) or estimated_sum <= EPS:
        return {
            "auc": auc,
            "rmse": rmse,
            "sd_mm": np.nan,
            "peak_dle_mm": np.nan,
            "centroid_dle_mm": np.nan,
        }

    domain_positions = positions[indices]
    truth_positions = domain_positions[truth_positive]
    distances = np.linalg.norm(
        domain_positions[:, None, :] - truth_positions[None, :, :],
        axis=2,
    )
    nearest_mm = distances.min(axis=1) * 1000.0
    sd_mm = float(
        np.sqrt(np.sum(estimated_normalized * nearest_mm**2))
    )
    peak_local = int(np.argmax(estimated_domain))
    peak_dle_mm = float(nearest_mm[peak_local])
    estimated_centroid = np.average(
        domain_positions,
        axis=0,
        weights=estimated_domain,
    )
    truth_centroid = np.average(
        domain_positions,
        axis=0,
        weights=truth_domain,
    )
    centroid_dle_mm = float(
        np.linalg.norm(estimated_centroid - truth_centroid) * 1000.0
    )
    return {
        "auc": auc,
        "rmse": rmse,
        "sd_mm": sd_mm,
        "peak_dle_mm": peak_dle_mm,
        "centroid_dle_mm": centroid_dle_mm,
    }


def graph_total_variation(
    coefficients: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    *,
    deep_mask: np.ndarray | None = None,
    deep_fusion_scale: float = 0.35,
) -> float:
    """Return the weighted sum of coefficient jumps over anatomical edges."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, None]
    left, right, weights = extract_graph_edges(
        adjacency,
        deep_mask=deep_mask,
        deep_fusion_scale=deep_fusion_scale,
    )
    if left.size == 0:
        return 0.0
    jumps = np.linalg.norm(coefficients[left] - coefficients[right], axis=1)
    return float(np.sum(weights * jumps))
