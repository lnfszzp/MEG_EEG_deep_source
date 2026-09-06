from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = np.finfo(float).eps


@dataclass
class GroupLassoResult:
    coefficients: np.ndarray
    active_indices: np.ndarray
    group_norms: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    objective: float
    relative_residual: float
    iterations: int
    active_set_rounds: int
    max_kkt_violation: float
    converged: bool
    added_by_kkt: np.ndarray


@dataclass
class RegularizationPoint:
    index: int
    lambda_fraction: float
    lam: float
    result: GroupLassoResult


@dataclass
class PathSelection:
    index: int
    point: RegularizationPoint
    reason: str


@dataclass
class RankOneRefitResult:
    directions: np.ndarray
    time_courses: np.ndarray
    coefficients: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    relative_residual: float
    iterations: int


def _as_gain_3d(gain: np.ndarray) -> np.ndarray:
    gain = np.asarray(gain, dtype=float)
    if gain.ndim != 3 or gain.shape[2] != 3:
        raise ValueError("gain must have shape channels x sources x 3")
    return gain


def embed_scalar_gain(scalar_gain: np.ndarray) -> np.ndarray:
    """Represent one fixed orientation per source in the common 3D interface."""
    scalar_gain = np.asarray(scalar_gain, dtype=float)
    if scalar_gain.ndim != 2:
        raise ValueError("scalar_gain must have shape channels x sources")
    embedded = np.zeros((*scalar_gain.shape, 3), dtype=float)
    embedded[:, :, 0] = scalar_gain
    return embedded


def prepare_joint_gain(
    eeg_gain_3d: np.ndarray,
    meg_gain_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack EEG and MEG gains and normalize every 3D source group equally."""
    eeg_gain_3d = _as_gain_3d(eeg_gain_3d)
    meg_gain_3d = _as_gain_3d(meg_gain_3d)
    if eeg_gain_3d.shape[1:] != meg_gain_3d.shape[1:]:
        raise ValueError("EEG and MEG gains must share source and orientation axes")

    raw = np.concatenate([eeg_gain_3d, meg_gain_3d], axis=0)
    weights = np.linalg.norm(raw, axis=(0, 2))
    weights = np.maximum(weights, EPS)
    return raw / weights[None, :, None], weights


def group_correlations(residual: np.ndarray, normalized_gain: np.ndarray) -> np.ndarray:
    """Return ||G_i.T residual||_F for every source group."""
    residual = np.asarray(residual, dtype=float)
    normalized_gain = _as_gain_3d(normalized_gain)
    if residual.ndim != 2 or residual.shape[0] != normalized_gain.shape[0]:
        raise ValueError("residual and gain must share the channel axis")

    gradient = np.einsum("csk,ct->skt", normalized_gain, residual, optimize=True)
    return np.linalg.norm(gradient, axis=(1, 2))


def spectral_correlations(residual: np.ndarray, normalized_gain: np.ndarray) -> np.ndarray:
    """Return the spectral norm of G_i.T residual for every source."""
    residual = np.asarray(residual, dtype=float)
    normalized_gain = _as_gain_3d(normalized_gain)
    if residual.ndim != 2 or residual.shape[0] != normalized_gain.shape[0]:
        raise ValueError("residual and gain must share the channel axis")
    gradient = np.einsum("csk,ct->skt", normalized_gain, residual, optimize=True)
    singular_values = np.linalg.svd(gradient, compute_uv=False)
    return singular_values[:, 0]


def _validate_penalty(penalty: str) -> str:
    if penalty not in {"frobenius", "nuclear"}:
        raise ValueError("penalty must be 'frobenius' or 'nuclear'")
    return penalty


def source_correlations(
    residual: np.ndarray,
    normalized_gain: np.ndarray,
    penalty: str,
) -> np.ndarray:
    penalty = _validate_penalty(penalty)
    if penalty == "nuclear":
        return spectral_correlations(residual, normalized_gain)
    return group_correlations(residual, normalized_gain)


def compute_lambda_max(
    data: np.ndarray,
    normalized_gain: np.ndarray,
    penalty: str = "frobenius",
) -> float:
    """Smallest group-lasso lambda for which the all-zero solution is optimal."""
    scores = source_correlations(data, normalized_gain, penalty)
    return float(scores.max(initial=0.0))


def sisses_seed_indices(
    source_amplitude: np.ndarray,
    relative_threshold: float = 0.25,
    max_seeds: int = 64,
) -> np.ndarray:
    """Rank SISSES amplitudes globally without using anatomical source labels."""
    amplitude = np.asarray(source_amplitude, dtype=float).ravel()
    if amplitude.size == 0 or max_seeds <= 0:
        return np.zeros(0, dtype=int)
    maximum = float(amplitude.max(initial=0.0))
    if maximum <= 0:
        return np.zeros(0, dtype=int)
    keep = np.flatnonzero(amplitude >= float(relative_threshold) * maximum)
    order = keep[np.argsort(amplitude[keep])[::-1]]
    return np.asarray(order[:max_seeds], dtype=int)


def group_soft_threshold(blocks: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the proximal operator of the sum of source-group Frobenius norms."""
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3:
        raise ValueError("blocks must have shape groups x 3 x times")
    norms = np.linalg.norm(blocks, axis=(1, 2))
    scale = np.maximum(0.0, 1.0 - float(threshold) / np.maximum(norms, EPS))
    return blocks * scale[:, None, None]


def nuclear_soft_threshold(blocks: np.ndarray, threshold: float) -> np.ndarray:
    """Shrink singular values of each 3D current block."""
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1] != 3:
        raise ValueError("blocks must have shape groups x 3 x times")
    output = np.zeros_like(blocks)
    for idx, block in enumerate(blocks):
        left, singular_values, right = np.linalg.svd(block, full_matrices=False)
        singular_values = np.maximum(0.0, singular_values - float(threshold))
        output[idx] = (left * singular_values[None, :]) @ right
    return output


def _proximal_blocks(
    blocks: np.ndarray,
    threshold: float,
    penalty: str,
) -> np.ndarray:
    if _validate_penalty(penalty) == "nuclear":
        return nuclear_soft_threshold(blocks, threshold)
    return group_soft_threshold(blocks, threshold)


def _penalty_value(blocks: np.ndarray, penalty: str) -> float:
    if _validate_penalty(penalty) == "nuclear":
        return float(
            sum(np.linalg.svd(block, compute_uv=False).sum() for block in blocks)
        )
    return float(np.linalg.norm(blocks, axis=(1, 2)).sum())


def _restricted_kkt_violation(
    data: np.ndarray,
    design: np.ndarray,
    coefficients: np.ndarray,
    lam: float,
    lipschitz: float,
    penalty: str,
) -> float:
    fitted = design @ coefficients.reshape(design.shape[1], data.shape[1])
    gradient = (design.T @ (fitted - data)).reshape(coefficients.shape)
    proximal = _proximal_blocks(
        coefficients - gradient / lipschitz,
        lam / lipschitz,
        penalty,
    )
    mapping = lipschitz * (coefficients - proximal)
    return float(np.linalg.norm(mapping)) / max(lam, EPS)


def _solve_restricted_fista(
    data: np.ndarray,
    gain: np.ndarray,
    active_indices: np.ndarray,
    lam: float,
    initial_coefficients: np.ndarray | None,
    max_iter: int,
    tolerance: float,
    penalty: str,
) -> tuple[np.ndarray, int, float]:
    n_active = len(active_indices)
    n_times = data.shape[1]
    if n_active == 0:
        return np.zeros((0, 3, n_times), dtype=float), 0, 0.0

    design = gain[:, active_indices, :].reshape(gain.shape[0], n_active * 3)
    spectral_norm = float(np.linalg.norm(design, ord=2))
    lipschitz = max(spectral_norm * spectral_norm, EPS)

    if initial_coefficients is None:
        current = np.zeros((n_active, 3, n_times), dtype=float)
    else:
        current = np.asarray(initial_coefficients, dtype=float).copy()
        if current.shape != (n_active, 3, n_times):
            raise ValueError("initial_coefficients has an incompatible shape")

    extrapolated = current.copy()
    momentum = 1.0
    for iteration in range(1, max_iter + 1):
        extrapolated_flat = extrapolated.reshape(n_active * 3, n_times)
        gradient = (
            design.T @ (design @ extrapolated_flat - data)
        ).reshape(n_active, 3, n_times)
        updated = _proximal_blocks(
            extrapolated - gradient / lipschitz,
            lam / lipschitz,
            penalty,
        )

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated = updated + ((momentum - 1.0) / next_momentum) * (updated - current)
        current = updated
        momentum = next_momentum

        if iteration == 1 or iteration % 10 == 0:
            violation = _restricted_kkt_violation(
                data,
                design,
                current,
                lam,
                lipschitz,
                penalty,
            )
            if violation <= tolerance:
                return current, iteration, violation

    violation = _restricted_kkt_violation(
        data,
        design,
        current,
        lam,
        lipschitz,
        penalty,
    )
    return current, max_iter, violation


def _full_kkt_violation(
    active_indices: np.ndarray,
    scores: np.ndarray,
    lam: float,
    restricted_violation: float,
) -> float:
    inactive_violation = scores - lam
    if len(active_indices):
        inactive_violation[active_indices] = 0.0
    maximum = max(0.0, float(inactive_violation.max(initial=0.0)))
    return max(maximum / max(lam, EPS), restricted_violation)


def _unique_valid_indices(indices: Iterable[int], n_sources: int) -> list[int]:
    return list(dict.fromkeys(int(idx) for idx in indices if 0 <= int(idx) < n_sources))


def solve_active_set_group_lasso(
    data: np.ndarray,
    normalized_gain: np.ndarray,
    lam: float,
    initial_active: Iterable[int] = (),
    *,
    add_batch: int = 8,
    max_active: int = 256,
    max_active_rounds: int = 64,
    max_inner_iter: int = 3000,
    inner_tolerance: float = 2e-7,
    kkt_tolerance: float = 1e-6,
    penalty: str = "frobenius",
) -> GroupLassoResult:
    """Solve whole-brain group lasso with a globally checked active set."""
    data = np.asarray(data, dtype=float)
    gain = _as_gain_3d(normalized_gain)
    if data.ndim != 2 or data.shape[0] != gain.shape[0]:
        raise ValueError("data and gain must share the channel axis")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite")
    penalty = _validate_penalty(penalty)

    n_sources = gain.shape[1]
    active = _unique_valid_indices(initial_active, n_sources)[:max_active]
    coefficients = np.zeros((len(active), 3, data.shape[1]), dtype=float)
    added_by_kkt: list[int] = []
    total_iterations = 0
    converged = False
    max_violation = np.inf

    for active_round in range(1, max_active_rounds + 1):
        coefficients, used_iterations, restricted_violation = _solve_restricted_fista(
            data,
            gain,
            np.asarray(active, dtype=int),
            lam,
            coefficients,
            max_inner_iter,
            inner_tolerance,
            penalty,
        )
        total_iterations += used_iterations

        norms = np.linalg.norm(coefficients, axis=(1, 2))
        keep = norms > 1e-11
        if not np.all(keep):
            active = [idx for idx, retain in zip(active, keep) if retain]
            coefficients = coefficients[keep]

        active_array = np.asarray(active, dtype=int)
        if len(active):
            fitted = np.einsum(
                "csk,skt->ct",
                gain[:, active_array, :],
                coefficients,
                optimize=True,
            )
        else:
            fitted = np.zeros_like(data)
        residual = data - fitted
        scores = source_correlations(residual, gain, penalty)
        max_violation = _full_kkt_violation(
            active_array,
            scores,
            lam,
            restricted_violation,
        )
        if max_violation <= kkt_tolerance:
            converged = True
            break

        inactive = np.ones(n_sources, dtype=bool)
        inactive[active_array] = False
        violating = np.flatnonzero(inactive & (scores > lam * (1.0 + kkt_tolerance)))
        if violating.size == 0 or len(active) >= max_active:
            break

        order = violating[np.argsort(scores[violating])[::-1]]
        room = max_active - len(active)
        new_indices = [int(idx) for idx in order[: min(add_batch, room)]]
        active.extend(new_indices)
        added_by_kkt.extend(idx for idx in new_indices if idx not in added_by_kkt)
        coefficients = np.concatenate(
            [
                coefficients,
                np.zeros((len(new_indices), 3, data.shape[1]), dtype=float),
            ],
            axis=0,
        )
    else:
        active_round = max_active_rounds

    active_array = np.asarray(active, dtype=int)
    group_norms = np.linalg.norm(coefficients, axis=(1, 2))
    if len(active):
        fitted = np.einsum(
            "csk,skt->ct",
            gain[:, active_array, :],
            coefficients,
            optimize=True,
        )
    else:
        fitted = np.zeros_like(data)
    residual = data - fitted
    objective = 0.5 * float(np.linalg.norm(residual, "fro") ** 2)
    objective += float(lam * _penalty_value(coefficients, penalty))
    relative_residual = float(
        np.linalg.norm(residual, "fro") / max(np.linalg.norm(data, "fro"), EPS)
    )

    return GroupLassoResult(
        coefficients=coefficients,
        active_indices=active_array,
        group_norms=group_norms,
        fitted=fitted,
        residual=residual,
        objective=objective,
        relative_residual=relative_residual,
        iterations=total_iterations,
        active_set_rounds=active_round,
        max_kkt_violation=float(max_violation),
        converged=converged,
        added_by_kkt=np.asarray(added_by_kkt, dtype=int),
    )


def solve_regularization_path(
    data: np.ndarray,
    normalized_gain: np.ndarray,
    lambda_fractions: Iterable[float],
    initial_active: Iterable[int] = (),
    **solver_kwargs,
) -> list[RegularizationPoint]:
    """Solve a descending lambda path using each active set to warm the next."""
    fractions = np.asarray(list(lambda_fractions), dtype=float)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("lambda_fractions must be a non-empty sequence")
    if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0):
        raise ValueError("lambda_fractions must be positive and finite")
    if np.any(np.diff(fractions) > 0):
        raise ValueError("lambda_fractions must be in descending order")

    penalty = solver_kwargs.get("penalty", "frobenius")
    lam_max = compute_lambda_max(data, normalized_gain, penalty=penalty)
    if lam_max <= 0:
        raise ValueError("data has zero correlation with every source group")

    active = list(initial_active)
    path = []
    for index, fraction in enumerate(fractions):
        lam = float(fraction * lam_max)
        result = solve_active_set_group_lasso(
            data,
            normalized_gain,
            lam,
            initial_active=active,
            **solver_kwargs,
        )
        path.append(
            RegularizationPoint(
                index=index,
                lambda_fraction=float(fraction),
                lam=lam,
                result=result,
            )
        )
        active = result.active_indices.tolist()
    return path


def choose_path_solution(
    path: list[RegularizationPoint],
    noise_target: float,
) -> PathSelection:
    """Choose the sparsest path point that reaches a truth-free noise target."""
    if not path:
        raise ValueError("path must not be empty")
    if not np.isfinite(noise_target) or noise_target < 0:
        raise ValueError("noise_target must be non-negative and finite")

    feasible = [
        point
        for point in path
        if point.result.converged and point.result.relative_residual <= noise_target
    ]
    if feasible:
        minimum_groups = min(len(point.result.active_indices) for point in feasible)
        selected = next(
            point for point in feasible if len(point.result.active_indices) == minimum_groups
        )
        reason = "sparsest_noise_consistent"
    else:
        selected = min(
            path,
            key=lambda point: (
                point.result.relative_residual,
                len(point.result.active_indices),
            ),
        )
        reason = "minimum_residual_no_noise_consistent_solution"
    return PathSelection(index=selected.index, point=selected, reason=reason)


def select_bic_index(
    residual_sums: np.ndarray,
    active_counts: np.ndarray,
    *,
    n_observations: int,
    parameters_per_source: int,
) -> tuple[int, np.ndarray]:
    """Select a path point by BIC without using source-location truth."""
    residual_sums = np.asarray(residual_sums, dtype=float).ravel()
    active_counts = np.asarray(active_counts, dtype=int).ravel()
    if residual_sums.size == 0 or residual_sums.shape != active_counts.shape:
        raise ValueError("residual_sums and active_counts must be non-empty and aligned")
    if n_observations <= 0 or parameters_per_source <= 0:
        raise ValueError("observation and parameter counts must be positive")
    variance = np.maximum(residual_sums / float(n_observations), EPS)
    parameter_counts = active_counts.astype(float) * float(parameters_per_source)
    scores = (
        float(n_observations) * np.log(variance)
        + parameter_counts * np.log(float(n_observations))
    )
    return int(np.argmin(scores)), scores


def normalized_to_physical_currents(
    normalized_coefficients: np.ndarray,
    active_indices: np.ndarray,
    group_weights: np.ndarray,
) -> np.ndarray:
    """Undo leadfield group normalization for the active current estimates."""
    coefficients = np.asarray(normalized_coefficients, dtype=float)
    active_indices = np.asarray(active_indices, dtype=int).ravel()
    group_weights = np.asarray(group_weights, dtype=float).ravel()
    if coefficients.ndim != 3 or coefficients.shape[1] != 3:
        raise ValueError("coefficients must have shape active_sources x 3 x times")
    if coefficients.shape[0] != active_indices.size:
        raise ValueError("active_indices and coefficients must have the same length")
    if active_indices.size and (
        active_indices.min() < 0 or active_indices.max() >= group_weights.size
    ):
        raise ValueError("active source index is outside group_weights")
    return coefficients / group_weights[active_indices, None, None]


def coefficients_to_scalar_source(
    physical_coefficients: np.ndarray,
    active_indices: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    """Convert active 3D currents to a full source-by-time magnitude matrix."""
    physical_coefficients = np.asarray(physical_coefficients, dtype=float)
    active_indices = np.asarray(active_indices, dtype=int).ravel()
    if physical_coefficients.shape[0] != active_indices.size:
        raise ValueError("active_indices and coefficients must have the same length")
    source = np.zeros((int(n_sources), physical_coefficients.shape[2]), dtype=float)
    source[active_indices] = np.linalg.norm(physical_coefficients, axis=1)
    return source


def _ridge_solve(design: np.ndarray, data: np.ndarray, ridge_fraction: float) -> np.ndarray:
    gram = design.T @ design
    ridge = float(ridge_fraction) * float(np.trace(gram)) / max(gram.shape[0], 1)
    return np.linalg.solve(
        gram + max(ridge, EPS) * np.eye(gram.shape[0]),
        design.T @ data,
    )


def refit_rank_one_currents(
    data: np.ndarray,
    gain_3d: np.ndarray,
    active_indices: np.ndarray,
    initial_time_courses: np.ndarray,
    *,
    ridge_fraction: float = 1e-10,
    max_iter: int = 50,
    tolerance: float = 1e-10,
) -> RankOneRefitResult:
    """Jointly refit fixed 3D directions and time courses for active sources."""
    data = np.asarray(data, dtype=float)
    gain_3d = _as_gain_3d(gain_3d)
    active_indices = np.asarray(active_indices, dtype=int).ravel()
    time_courses = np.asarray(initial_time_courses, dtype=float).copy()
    if data.ndim != 2 or data.shape[0] != gain_3d.shape[0]:
        raise ValueError("data and gain must share the channel axis")
    if time_courses.shape != (active_indices.size, data.shape[1]):
        raise ValueError("initial_time_courses must have shape active_sources x times")
    if active_indices.size == 0:
        zeros = np.zeros_like(data)
        return RankOneRefitResult(
            directions=np.zeros((0, 3), dtype=float),
            time_courses=time_courses,
            coefficients=np.zeros((0, 3, data.shape[1]), dtype=float),
            fitted=zeros,
            residual=data.copy(),
            relative_residual=1.0,
            iterations=0,
        )

    active_gain = gain_3d[:, active_indices, :]
    previous_residual = None
    directions = np.zeros((active_indices.size, 3), dtype=float)
    fitted = np.zeros_like(data)

    for iteration in range(1, max_iter + 1):
        orientation_design = np.einsum(
            "csk,st->ctsk",
            active_gain,
            time_courses,
            optimize=True,
        ).reshape(data.size, active_indices.size * 3)
        directions = _ridge_solve(
            orientation_design,
            data.reshape(-1, 1),
            ridge_fraction,
        ).reshape(active_indices.size, 3)
        direction_norms = np.linalg.norm(directions, axis=1)
        nonzero = direction_norms > EPS
        directions[nonzero] /= direction_norms[nonzero, None]
        directions[~nonzero, 0] = 1.0

        oriented_gain = np.einsum(
            "csk,sk->cs",
            active_gain,
            directions,
            optimize=True,
        )
        time_courses = _ridge_solve(oriented_gain, data, ridge_fraction)
        fitted = oriented_gain @ time_courses
        residual_norm = float(np.linalg.norm(data - fitted, "fro"))
        if previous_residual is not None and abs(
            previous_residual - residual_norm
        ) <= tolerance * max(1.0, previous_residual):
            break
        previous_residual = residual_norm

    residual = data - fitted
    coefficients = directions[:, :, None] * time_courses[:, None, :]
    relative_residual = float(
        np.linalg.norm(residual, "fro") / max(np.linalg.norm(data, "fro"), EPS)
    )
    return RankOneRefitResult(
        directions=directions,
        time_courses=time_courses,
        coefficients=coefficients,
        fitted=fitted,
        residual=residual,
        relative_residual=relative_residual,
        iterations=iteration,
    )


def _normalize_directions(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=float)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.maximum(norms, EPS)


def _modality_directions(
    data: np.ndarray,
    gain_3d: np.ndarray,
    active_indices: np.ndarray,
    time_courses: np.ndarray,
    joint_directions: np.ndarray,
    ridge_fraction: float,
) -> np.ndarray:
    active_gain = gain_3d[:, active_indices, :]
    full_fitted = np.einsum(
        "csk,sk,st->ct",
        active_gain,
        joint_directions,
        time_courses,
        optimize=True,
    )
    estimated = np.zeros_like(joint_directions)
    for pos in range(len(active_indices)):
        contribution = np.einsum(
            "ck,k,t->ct",
            active_gain[:, pos, :],
            joint_directions[pos],
            time_courses[pos],
            optimize=True,
        )
        partial_data = data - full_fitted + contribution
        design = np.einsum(
            "ck,t->ctk",
            active_gain[:, pos, :],
            time_courses[pos],
            optimize=True,
        ).reshape(data.size, 3)
        estimated[pos] = _ridge_solve(
            design,
            partial_data.reshape(-1, 1),
            ridge_fraction,
        ).ravel()
    return _normalize_directions(estimated)


def modality_direction_confidence(
    eeg_data: np.ndarray,
    eeg_gain_3d: np.ndarray,
    meg_data: np.ndarray,
    meg_gain_3d: np.ndarray,
    active_indices: np.ndarray,
    time_courses: np.ndarray,
    joint_directions: np.ndarray,
    *,
    ridge_fraction: float = 1e-10,
) -> dict:
    """Compare EEG-only, MEG-only, and joint directions for each active source."""
    active_indices = np.asarray(active_indices, dtype=int).ravel()
    joint_directions = _normalize_directions(joint_directions)
    eeg_directions = _modality_directions(
        np.asarray(eeg_data, dtype=float),
        _as_gain_3d(eeg_gain_3d),
        active_indices,
        np.asarray(time_courses, dtype=float),
        joint_directions,
        ridge_fraction,
    )
    meg_directions = _modality_directions(
        np.asarray(meg_data, dtype=float),
        _as_gain_3d(meg_gain_3d),
        active_indices,
        np.asarray(time_courses, dtype=float),
        joint_directions,
        ridge_fraction,
    )

    def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.abs(np.sum(left * right, axis=1))

    return {
        "eeg_directions": eeg_directions,
        "meg_directions": meg_directions,
        "joint_directions": joint_directions,
        "eeg_meg_cosine": cosine(eeg_directions, meg_directions),
        "eeg_joint_cosine": cosine(eeg_directions, joint_directions),
        "meg_joint_cosine": cosine(meg_directions, joint_directions),
    }
