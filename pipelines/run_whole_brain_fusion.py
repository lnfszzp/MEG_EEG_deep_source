from __future__ import annotations

from pathlib import Path
import csv
import time

import h5py
import numpy as np
import scipy.io as sio

from algorithms.whole_brain_adaptive_fusion import (
    PathSelection,
    coefficients_to_scalar_source,
    embed_scalar_gain,
    modality_direction_confidence,
    normalized_to_physical_currents,
    prepare_joint_gain,
    refit_rank_one_currents,
    select_bic_index,
    sisses_seed_indices,
    solve_active_set_group_lasso,
    solve_regularization_path,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "generated"
RESULTS_ROOT = ROOT / "results" / "latest"
OUT_ROOT = RESULTS_ROOT / "whole_brain_adaptive_fusion"
SCENARIOS = (
    "deep_only",
    "surface_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
)

NOISE_SAMPLES = 200
ANALYSIS_START = 200
LAMBDA_FRACTIONS = (0.35, 0.22, 0.14, 0.09, 0.06, 0.04)
MAX_TIME_COMPONENTS = 32
TEMPORAL_VARIANCE = 0.995


def load_mat(path: Path) -> dict:
    path = Path(path)
    try:
        return {key: value for key, value in sio.loadmat(path).items() if not key.startswith("__")}
    except NotImplementedError:
        result = {}
        with h5py.File(path, "r") as handle:
            for key in handle.keys():
                if isinstance(handle[key], h5py.Dataset):
                    result[key] = np.array(handle[key])
        return result


def load_sisses(path: Path, n_sources: int) -> np.ndarray:
    source = np.asarray(load_mat(path)["s_wen"], dtype=float)
    if source.shape[0] != n_sources and source.shape[1] == n_sources:
        source = source.T
    if source.shape[0] != n_sources:
        raise ValueError("SISSES source matrix does not match the forward model")
    return source


def whitening_matrix(data: np.ndarray, noise_samples: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    count = min(int(noise_samples), data.shape[1])
    if count < 2:
        raise ValueError("at least two noise samples are required")
    noise = data[:, :count]
    noise = noise - noise.mean(axis=1, keepdims=True)
    covariance = noise @ noise.T / (count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = eigenvalues > eigenvalues.max() * 1e-8
    return np.diag(1.0 / np.sqrt(eigenvalues[keep])) @ eigenvectors[:, keep].T


def whiten_modality(
    data: np.ndarray,
    gain_3d: np.ndarray,
    noise_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    gain_3d = np.asarray(gain_3d, dtype=float)
    whitening = whitening_matrix(data, noise_samples)
    n_sources = gain_3d.shape[1]
    flat_gain = gain_3d.reshape(gain_3d.shape[0], n_sources * 3)
    white_gain = (whitening @ flat_gain).reshape(whitening.shape[0], n_sources, 3)
    return whitening @ data, white_gain


def temporal_svd_compress(
    data: np.ndarray,
    variance_fraction: float = TEMPORAL_VARIANCE,
    max_components: int = MAX_TIME_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.asarray(data, dtype=float)
    _, singular_values, right_vectors = np.linalg.svd(data, full_matrices=False)
    energy = singular_values**2
    cumulative = np.cumsum(energy) / max(float(energy.sum()), np.finfo(float).eps)
    rank = int(np.searchsorted(cumulative, variance_fraction) + 1)
    rank = max(1, min(rank, int(max_components), len(singular_values)))
    temporal_basis = right_vectors[:rank]
    compressed = data @ temporal_basis.T
    retained = float(energy[:rank].sum() / max(float(energy.sum()), np.finfo(float).eps))
    return compressed, temporal_basis, retained


def estimate_noise_target(compressed_data: np.ndarray, safety_factor: float = 1.10) -> float:
    expected_noise_norm = np.sqrt(float(compressed_data.size))
    return float(
        min(
            0.999,
            safety_factor
            * expected_noise_norm
            / max(float(np.linalg.norm(compressed_data, "fro")), np.finfo(float).eps),
        )
    )


def _sisses_amplitude(sisses_source: np.ndarray | None, n_sources: int) -> np.ndarray:
    if sisses_source is None:
        return np.zeros(n_sources, dtype=float)
    source = np.asarray(sisses_source, dtype=float)
    if source.ndim == 1:
        amplitude = np.abs(source)
    elif source.ndim == 2:
        amplitude = np.linalg.norm(source, axis=1)
    else:
        raise ValueError("sisses_source must be a vector or source-by-time matrix")
    if amplitude.size != n_sources:
        raise ValueError("sisses_source does not match the number of sources")
    return amplitude


def _solve_prewhitened_modalities(
    white_data: np.ndarray,
    white_gain_3d: np.ndarray,
    white_localization_gain_3d: np.ndarray,
    *,
    sisses_source: np.ndarray | None,
    analysis_start: int,
    lambda_fractions: tuple[float, ...],
    max_time_components: int,
    temporal_variance: float,
) -> dict:
    analysis_start = int(analysis_start)
    analysis_data = white_data[:, analysis_start:]
    compressed_data, temporal_basis, retained_variance = temporal_svd_compress(
        analysis_data,
        variance_fraction=temporal_variance,
        max_components=max_time_components,
    )
    empty_gain = np.zeros(
        (0, white_localization_gain_3d.shape[1], 3),
        dtype=float,
    )
    normalized_gain, group_weights = prepare_joint_gain(
        white_localization_gain_3d,
        empty_gain,
    )
    amplitude = _sisses_amplitude(sisses_source, normalized_gain.shape[1])
    seeds = sisses_seed_indices(amplitude, relative_threshold=0.25, max_seeds=64)
    noise_target = estimate_noise_target(compressed_data)

    started = time.perf_counter()
    path = solve_regularization_path(
        compressed_data,
        normalized_gain,
        lambda_fractions,
        initial_active=seeds,
        penalty="frobenius",
    )
    runtime = time.perf_counter() - started

    path_refits = []
    for point in path:
        physical_location = normalized_to_physical_currents(
            point.result.coefficients,
            point.result.active_indices,
            group_weights,
        )
        path_refits.append(
            refit_rank_one_currents(
                compressed_data,
                white_gain_3d,
                point.result.active_indices,
                physical_location[:, 0, :],
            )
        )

    selected_index, bic_scores = select_bic_index(
        np.asarray(
            [np.linalg.norm(refit.residual, "fro") ** 2 for refit in path_refits]
        ),
        np.asarray([len(point.result.active_indices) for point in path]),
        n_observations=compressed_data.size,
        parameters_per_source=compressed_data.shape[1] + 3,
    )
    selection = PathSelection(
        index=selected_index,
        point=path[selected_index],
        reason="rank_one_refit_bic",
    )
    selected = selection.point.result
    refit = path_refits[selected_index]
    physical_time = np.einsum(
        "skr,rt->skt",
        refit.coefficients,
        temporal_basis,
        optimize=True,
    )
    analysis_source = coefficients_to_scalar_source(
        physical_time,
        selected.active_indices,
        normalized_gain.shape[1],
    )
    source = np.zeros((normalized_gain.shape[1], white_data.shape[1]), dtype=float)
    source[:, analysis_start:] = analysis_source

    return {
        "source": source,
        "active_indices": selected.active_indices,
        "group_norms": selected.group_norms,
        "physical_currents": physical_time,
        "directions": refit.directions,
        "path": path,
        "path_refits": path_refits,
        "selection": selection,
        "bic_scores": bic_scores,
        "seed_indices": seeds,
        "added_by_kkt": selected.added_by_kkt,
        "noise_target": noise_target,
        "retained_temporal_variance": retained_variance,
        "temporal_rank": temporal_basis.shape[0],
        "runtime_seconds": runtime,
        "rank_one_refit_residual": refit.relative_residual,
        "rank_one_refit_iterations": refit.iterations,
    }


def run_single_modality_solver(
    data: np.ndarray,
    gain_3d: np.ndarray,
    localization_gain: np.ndarray,
    sisses_source: np.ndarray | None = None,
    *,
    noise_samples: int = NOISE_SAMPLES,
    analysis_start: int = ANALYSIS_START,
    lambda_fractions: tuple[float, ...] = LAMBDA_FRACTIONS,
    max_time_components: int = MAX_TIME_COMPONENTS,
    temporal_variance: float = TEMPORAL_VARIANCE,
) -> dict:
    """Run the adaptive whole-brain method using one sensor modality."""
    white_data, white_gain = whiten_modality(data, gain_3d, noise_samples)
    _, white_location = whiten_modality(
        data,
        embed_scalar_gain(localization_gain),
        noise_samples,
    )
    return _solve_prewhitened_modalities(
        white_data,
        white_gain,
        white_location,
        sisses_source=sisses_source,
        analysis_start=analysis_start,
        lambda_fractions=lambda_fractions,
        max_time_components=max_time_components,
        temporal_variance=temporal_variance,
    )


def run_whole_brain_solver(
    eeg_data: np.ndarray,
    eeg_gain_3d: np.ndarray,
    meg_data: np.ndarray,
    meg_gain_3d: np.ndarray,
    sisses_source: np.ndarray | None = None,
    *,
    eeg_localization_gain: np.ndarray | None = None,
    meg_localization_gain: np.ndarray | None = None,
    noise_samples: int = NOISE_SAMPLES,
    analysis_start: int = ANALYSIS_START,
    lambda_fractions: tuple[float, ...] = LAMBDA_FRACTIONS,
    max_time_components: int = MAX_TIME_COMPONENTS,
    temporal_variance: float = TEMPORAL_VARIANCE,
    run_cold_comparison: bool = True,
) -> dict:
    """Run truth-free whole-brain fusion; SISSES only seeds the active set."""
    white_eeg, white_eeg_gain = whiten_modality(eeg_data, eeg_gain_3d, noise_samples)
    white_meg, white_meg_gain = whiten_modality(meg_data, meg_gain_3d, noise_samples)
    if white_eeg.shape[1] != white_meg.shape[1]:
        raise ValueError("EEG and MEG data must share the time axis")

    analysis_start = int(analysis_start)
    joint_data = np.vstack(
        [
            white_eeg[:, analysis_start:],
            white_meg[:, analysis_start:],
        ]
    )
    compressed_data, temporal_basis, retained_variance = temporal_svd_compress(
        joint_data,
        variance_fraction=temporal_variance,
        max_components=max_time_components,
    )
    if (eeg_localization_gain is None) != (meg_localization_gain is None):
        raise ValueError("EEG and MEG localization gains must be provided together")
    if eeg_localization_gain is not None:
        _, white_eeg_location = whiten_modality(
            eeg_data,
            embed_scalar_gain(eeg_localization_gain),
            noise_samples,
        )
        _, white_meg_location = whiten_modality(
            meg_data,
            embed_scalar_gain(meg_localization_gain),
            noise_samples,
        )
        localization_penalty = "frobenius"
    else:
        white_eeg_location = white_eeg_gain
        white_meg_location = white_meg_gain
        localization_penalty = "nuclear"

    normalized_gain, group_weights = prepare_joint_gain(
        white_eeg_location,
        white_meg_location,
    )
    joint_free_gain = np.concatenate([white_eeg_gain, white_meg_gain], axis=0)

    amplitude = _sisses_amplitude(sisses_source, normalized_gain.shape[1])
    seeds = sisses_seed_indices(amplitude, relative_threshold=0.25, max_seeds=64)
    noise_target = estimate_noise_target(compressed_data)

    started = time.perf_counter()
    warm_path = solve_regularization_path(
        compressed_data,
        normalized_gain,
        lambda_fractions,
        initial_active=seeds,
        penalty=localization_penalty,
    )
    warm_runtime = time.perf_counter() - started
    path_refits = []
    for point in warm_path:
        physical_location = normalized_to_physical_currents(
            point.result.coefficients,
            point.result.active_indices,
            group_weights,
        )
        if localization_penalty == "frobenius":
            initial_time_courses = physical_location[:, 0, :]
        else:
            initial_time_courses = np.zeros(
                (len(point.result.active_indices), compressed_data.shape[1]),
                dtype=float,
            )
            for idx, block in enumerate(physical_location):
                _, singular_values, right = np.linalg.svd(block, full_matrices=False)
                initial_time_courses[idx] = singular_values[0] * right[0]
        path_refits.append(
            refit_rank_one_currents(
                compressed_data,
                joint_free_gain,
                point.result.active_indices,
                initial_time_courses,
            )
        )

    selected_index, bic_scores = select_bic_index(
        np.asarray(
            [np.linalg.norm(refit.residual, "fro") ** 2 for refit in path_refits]
        ),
        np.asarray([len(point.result.active_indices) for point in warm_path]),
        n_observations=compressed_data.size,
        parameters_per_source=compressed_data.shape[1] + 3,
    )
    selection = PathSelection(
        index=selected_index,
        point=warm_path[selected_index],
        reason="rank_one_refit_bic",
    )
    selected = selection.point.result
    rank_one_refit = path_refits[selected_index]
    compressed_eeg = white_eeg[:, analysis_start:] @ temporal_basis.T
    compressed_meg = white_meg[:, analysis_start:] @ temporal_basis.T
    direction_confidence = modality_direction_confidence(
        compressed_eeg,
        white_eeg_gain,
        compressed_meg,
        white_meg_gain,
        selected.active_indices,
        rank_one_refit.time_courses,
        rank_one_refit.directions,
    )
    physical_time = np.einsum(
        "skr,rt->skt",
        rank_one_refit.coefficients,
        temporal_basis,
        optimize=True,
    )
    analysis_source = coefficients_to_scalar_source(
        physical_time,
        selected.active_indices,
        normalized_gain.shape[1],
    )
    source = np.zeros((normalized_gain.shape[1], white_eeg.shape[1]), dtype=float)
    source[:, analysis_start:] = analysis_source

    cold = None
    cold_runtime = np.nan
    if run_cold_comparison:
        started = time.perf_counter()
        cold = solve_active_set_group_lasso(
            compressed_data,
            normalized_gain,
            selection.point.lam,
            initial_active=(),
            penalty=localization_penalty,
        )
        cold_runtime = time.perf_counter() - started

    return {
        "source": source,
        "active_indices": selected.active_indices,
        "group_norms": selected.group_norms,
        "physical_currents": physical_time,
        "directions": rank_one_refit.directions,
        "direction_confidence": direction_confidence,
        "rank_one_refit_residual": rank_one_refit.relative_residual,
        "rank_one_refit_iterations": rank_one_refit.iterations,
        "path_refits": path_refits,
        "bic_scores": bic_scores,
        "warm_path": warm_path,
        "selection": selection,
        "cold_result": cold,
        "seed_indices": seeds,
        "added_by_kkt": selected.added_by_kkt,
        "noise_target": noise_target,
        "retained_temporal_variance": retained_variance,
        "temporal_rank": temporal_basis.shape[0],
        "warm_runtime_seconds": warm_runtime,
        "cold_runtime_seconds": cold_runtime,
        "compressed_data": compressed_data,
        "normalized_gain": normalized_gain,
        "group_weights": group_weights,
        "localization_penalty": localization_penalty,
    }


def source_amplitude(source: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(source, dtype=float), axis=1)


def surface_patch_metrics(
    source: np.ndarray,
    truth: dict,
    *,
    detection_radius_mm: float = 15.0,
) -> dict:
    centers = np.asarray(
        truth.get("true_surface_centers0", np.zeros((1, 0), dtype=int)),
        dtype=int,
    ).ravel()
    if centers.size == 0:
        return {
            "true_surface_patch_count": 0,
            "detected_surface_patch_count": 0,
            "surface_patch_center_dle_mean_mm": np.nan,
            "surface_patch_center_dle_max_mm": np.nan,
        }

    vertices = np.asarray(truth["src_vertices"], dtype=float)
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    amplitude = source_amplitude(source)
    active_surface = np.flatnonzero(amplitude[:n_surf] > 0)
    if active_surface.size == 0:
        distances = np.full(centers.size, np.inf)
    else:
        distances = np.asarray(
            [
                np.linalg.norm(
                    vertices[active_surface] - vertices[center],
                    axis=1,
                ).min()
                * 1000
                for center in centers
            ]
        )
    return {
        "true_surface_patch_count": int(centers.size),
        "detected_surface_patch_count": int(
            np.sum(distances <= float(detection_radius_mm))
        ),
        "surface_patch_center_dle_mean_mm": float(np.mean(distances)),
        "surface_patch_center_dle_max_mm": float(np.max(distances)),
    }


def metric_row(
    scenario: str,
    source: np.ndarray,
    truth: dict,
) -> dict:
    amplitude = source_amplitude(source)
    vertices = np.asarray(truth["src_vertices"], dtype=float)
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    true_deep = int(np.asarray(truth["true_deep_idx0"]).ravel()[0])
    has_deep = bool(int(np.asarray(truth.get("has_deep_source", [[1]])).ravel()[0]))
    true_surface = np.asarray(truth["true_surface_indices0"], dtype=int).ravel()
    order = np.argsort(amplitude)[::-1]
    global_peak = int(order[0])
    deep_indices = np.arange(n_surf, len(amplitude))
    has_estimated_deep = bool(np.any(amplitude[deep_indices] > 0))
    deep_peak = (
        int(deep_indices[np.argmax(amplitude[deep_indices])])
        if has_estimated_deep
        else -1
    )

    surface_peak = -1
    surface_dle = np.nan
    surface_rank = -1
    if true_surface.size and np.any(amplitude[:n_surf] > 0):
        surface_peak = int(np.argmax(amplitude[:n_surf]))
        surface_dle = float(
            np.linalg.norm(
                vertices[surface_peak][None, :] - vertices[true_surface],
                axis=1,
            ).min()
            * 1000
        )
        surface_rank = int(min(np.where(order == idx)[0][0] + 1 for idx in true_surface))

    row = {
        "scenario": scenario,
        "method": "Whole_brain_adaptive_fusion",
        "global_peak_idx1": global_peak + 1,
        "global_peak_is_deep": int(global_peak >= n_surf),
        "deep_peak_idx1": deep_peak + 1 if deep_peak >= 0 else -1,
        "surface_peak_idx1": surface_peak + 1 if surface_peak >= 0 else -1,
        "surface_dle_to_patch_mm": surface_dle,
        "surface_best_true_rank": surface_rank,
        "deep_energy_ratio": float(
            amplitude[n_surf:].sum() / max(amplitude.sum(), np.finfo(float).eps)
        ),
        "surface_energy_ratio": float(
            amplitude[:n_surf].sum() / max(amplitude.sum(), np.finfo(float).eps)
        ),
    }
    if has_deep and deep_peak >= 0:
        row["global_dle_to_true_deep_mm"] = float(
            np.linalg.norm(vertices[global_peak] - vertices[true_deep]) * 1000
        )
        row["deep_dle_mm"] = float(
            np.linalg.norm(vertices[deep_peak] - vertices[true_deep]) * 1000
        )
        row["true_deep_rank"] = int(np.where(order == true_deep)[0][0] + 1)
    else:
        row["global_dle_to_true_deep_mm"] = np.nan
        row["deep_dle_mm"] = np.nan
        row["true_deep_rank"] = -1
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    path_rows = []
    warm_cold_rows = []

    for scenario in SCENARIOS:
        print(f"\n==== {scenario} ====")
        data_dir = DATA_ROOT / scenario
        out_dir = OUT_ROOT / scenario
        out_dir.mkdir(parents=True, exist_ok=True)

        eeg = load_mat(data_dir / "sub_EEG.mat")
        meg = load_mat(data_dir / "sub_MEG.mat")
        truth = load_mat(data_dir / "s_true.mat")
        n_sources = np.asarray(eeg["Gain3D"]).shape[1]
        sisses = load_sisses(
            RESULTS_ROOT
            / "sisses_warm_start"
            / scenario
            / "s_wen_sisses_fusion_quick_best.mat",
            n_sources,
        )

        result = run_whole_brain_solver(
            eeg["F"],
            eeg["Gain3D"],
            meg["F"],
            meg["Gain3D"],
            sisses,
            eeg_localization_gain=eeg["Gain"],
            meg_localization_gain=meg["Gain"],
        )
        source = result["source"]
        row = metric_row(scenario, source, truth)
        row.update(surface_patch_metrics(source, truth))
        selected = result["selection"].point.result
        true_deep = int(np.asarray(truth["true_deep_idx0"]).ravel()[0])
        has_deep = bool(int(np.asarray(truth.get("has_deep_source", [[1]])).ravel()[0]))
        row.update(
            {
                "selected_lambda_fraction": result["selection"].point.lambda_fraction,
                "selection_reason": result["selection"].reason,
                "active_source_count": len(result["active_indices"]),
                "relative_residual": selected.relative_residual,
                "rank_one_refit_residual": result["rank_one_refit_residual"],
                "rank_one_refit_iterations": result["rank_one_refit_iterations"],
                "localization_penalty": result["localization_penalty"],
                "mean_eeg_meg_direction_cos": float(
                    np.mean(result["direction_confidence"]["eeg_meg_cosine"])
                ),
                "min_eeg_meg_direction_cos": float(
                    np.min(result["direction_confidence"]["eeg_meg_cosine"])
                ),
                "max_kkt_violation": selected.max_kkt_violation,
                "converged": int(selected.converged),
                "sisses_seed_count": len(result["seed_indices"]),
                "true_deep_in_sisses_seed": int(true_deep in result["seed_indices"])
                if has_deep
                else -1,
                "true_deep_added_by_kkt": int(true_deep in result["added_by_kkt"])
                if has_deep
                else -1,
                "temporal_rank": result["temporal_rank"],
                "retained_temporal_variance": result["retained_temporal_variance"],
                "noise_target": result["noise_target"],
                "warm_runtime_seconds": result["warm_runtime_seconds"],
                "cold_runtime_seconds": result["cold_runtime_seconds"],
            }
        )
        summary_rows.append(row)

        for point, refit, bic_score in zip(
            result["warm_path"],
            result["path_refits"],
            result["bic_scores"],
        ):
            path_rows.append(
                {
                    "scenario": scenario,
                    "index": point.index,
                    "lambda_fraction": point.lambda_fraction,
                    "lambda": point.lam,
                    "active_source_count": len(point.result.active_indices),
                    "relative_residual": point.result.relative_residual,
                    "rank_one_refit_residual": refit.relative_residual,
                    "rank_one_bic": float(bic_score),
                    "objective": point.result.objective,
                    "max_kkt_violation": point.result.max_kkt_violation,
                    "converged": int(point.result.converged),
                    "iterations": point.result.iterations,
                    "active_set_rounds": point.result.active_set_rounds,
                    "selected": int(point.index == result["selection"].index),
                }
            )

        cold = result["cold_result"]
        warm_cold_rows.append(
            {
                "scenario": scenario,
                "lambda_fraction": result["selection"].point.lambda_fraction,
                "warm_objective": selected.objective,
                "cold_objective": cold.objective,
                "relative_objective_difference": abs(selected.objective - cold.objective)
                / max(abs(cold.objective), np.finfo(float).eps),
                "warm_active_count": len(selected.active_indices),
                "cold_active_count": len(cold.active_indices),
                "warm_runtime_seconds": result["warm_runtime_seconds"],
                "cold_runtime_seconds": result["cold_runtime_seconds"],
                "warm_converged": int(selected.converged),
                "cold_converged": int(cold.converged),
            }
        )

        if has_deep:
            omitted_seed = [
                int(idx) for idx in result["seed_indices"] if int(idx) != true_deep
            ]
            omitted_result = solve_active_set_group_lasso(
                result["compressed_data"],
                result["normalized_gain"],
                result["selection"].point.lam,
                initial_active=omitted_seed,
                penalty=result["localization_penalty"],
            )
            row["omitted_seed_true_deep_recovered"] = int(
                true_deep in omitted_result.active_indices
            )
            row["omitted_seed_true_deep_added_by_kkt"] = int(
                true_deep in omitted_result.added_by_kkt
            )
        else:
            row["omitted_seed_true_deep_recovered"] = -1
            row["omitted_seed_true_deep_added_by_kkt"] = -1

        if has_deep and true_deep in result["active_indices"]:
            deep_pos = int(np.where(result["active_indices"] == true_deep)[0][0])
            row["true_deep_eeg_meg_direction_cos"] = float(
                result["direction_confidence"]["eeg_meg_cosine"][deep_pos]
            )
            row["true_deep_eeg_joint_direction_cos"] = float(
                result["direction_confidence"]["eeg_joint_cosine"][deep_pos]
            )
            row["true_deep_meg_joint_direction_cos"] = float(
                result["direction_confidence"]["meg_joint_cosine"][deep_pos]
            )
        else:
            row["true_deep_eeg_meg_direction_cos"] = np.nan
            row["true_deep_eeg_joint_direction_cos"] = np.nan
            row["true_deep_meg_joint_direction_cos"] = np.nan

        active_rows = []
        for pos, idx in enumerate(result["active_indices"]):
            active_rows.append(
                {
                    "source_idx0": int(idx),
                    "source_idx1": int(idx) + 1,
                    "group_norm": float(result["group_norms"][pos]),
                    "direction_x": float(result["directions"][pos, 0]),
                    "direction_y": float(result["directions"][pos, 1]),
                    "direction_z": float(result["directions"][pos, 2]),
                    "eeg_meg_direction_cos": float(
                        result["direction_confidence"]["eeg_meg_cosine"][pos]
                    ),
                    "eeg_joint_direction_cos": float(
                        result["direction_confidence"]["eeg_joint_cosine"][pos]
                    ),
                    "meg_joint_direction_cos": float(
                        result["direction_confidence"]["meg_joint_cosine"][pos]
                    ),
                    "seeded_by_sisses": int(idx in result["seed_indices"]),
                    "added_by_kkt": int(idx in result["added_by_kkt"]),
                }
            )
        write_csv(out_dir / "active_sources.csv", active_rows)

        times = np.asarray(truth["times"], dtype=float).ravel()
        np.savez(
            out_dir / "whole_brain_result.npz",
            S=source,
            times=times,
            active_indices0=result["active_indices"],
            group_norms=result["group_norms"],
            directions=result["directions"],
            eeg_directions=result["direction_confidence"]["eeg_directions"],
            meg_directions=result["direction_confidence"]["meg_directions"],
            eeg_meg_direction_cos=result["direction_confidence"]["eeg_meg_cosine"],
            seed_indices0=result["seed_indices"],
            added_by_kkt0=result["added_by_kkt"],
        )
        sio.savemat(
            out_dir / "whole_brain_result.mat",
            {
                "S": source,
                "times": times[np.newaxis, :],
                "active_indices0": result["active_indices"][np.newaxis, :],
                "group_norms": result["group_norms"][np.newaxis, :],
                "directions": result["directions"],
                "eeg_directions": result["direction_confidence"]["eeg_directions"],
                "meg_directions": result["direction_confidence"]["meg_directions"],
                "eeg_meg_direction_cos": result["direction_confidence"]["eeg_meg_cosine"][
                    np.newaxis, :
                ],
            },
            do_compression=True,
        )

        print(
            f"{scenario}: peak={row['global_peak_idx1']} "
            f"deepDLE={row['deep_dle_mm']:.3f} mm "
            f"surfaceDLE={row['surface_dle_to_patch_mm']:.3f} mm "
            f"active={row['active_source_count']} "
            f"KKT={row['max_kkt_violation']:.2e}"
        )

    write_csv(OUT_ROOT / "whole_brain_summary.csv", summary_rows)
    write_csv(OUT_ROOT / "regularization_path.csv", path_rows)
    write_csv(OUT_ROOT / "warm_cold_comparison.csv", warm_cold_rows)
    print(f"\nSaved: {OUT_ROOT}")


if __name__ == "__main__":
    main()
