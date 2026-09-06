from __future__ import annotations

from pathlib import Path
import csv
import sys
import time

import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.spatial_fused_fusion import (
    anatomical_region_mask,
    continuous_source_metrics,
    expand_candidates,
    pdf_source_metrics,
    range_metrics,
    region_mask_from_source,
    solve_graph_fused_lasso,
    solve_spatial_fused_path,
)
from algorithms.external_metrics import external_full_head_metrics
from algorithms.whole_brain_adaptive_fusion import (
    embed_scalar_gain,
    modality_direction_confidence,
    prepare_joint_gain,
    refit_rank_one_currents,
    sisses_seed_indices,
)
from pipelines.run_whole_brain_fusion import (
    ANALYSIS_START,
    DATA_ROOT,
    MAX_TIME_COMPONENTS,
    NOISE_SAMPLES,
    RESULTS_ROOT,
    SCENARIOS,
    TEMPORAL_VARIANCE,
    load_mat,
    load_sisses,
    source_amplitude,
    temporal_svd_compress,
    whiten_modality,
)


OUT_ROOT = RESULTS_ROOT / "spatial_fused_fusion"
POINT_ROOT = RESULTS_ROOT / "whole_brain_adaptive_fusion"
SISSES_ROOT = RESULTS_ROOT / "sisses_warm_start"

LAMBDA_FRACTIONS = (0.18, 0.10, 0.06, 0.035)
BETA_FRACTIONS = (0.01, 0.03, 0.08, 0.15)
DEEP_FUSION_SCALES = (0.15, 0.35, 0.70)


def whole_brain_screening_seeds(
    data: np.ndarray,
    normalized_gain: np.ndarray,
    n_surf: int,
    *,
    surface_count: int = 96,
    deep_count: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Select high-correlation surface and deep seeds without source truth."""
    data = np.asarray(data, dtype=float)
    gain = np.asarray(normalized_gain, dtype=float)
    if data.ndim != 2 or gain.ndim != 2 or data.shape[0] != gain.shape[0]:
        raise ValueError("data and normalized_gain must share the sensor axis")
    scores = np.linalg.norm(gain.T @ data, axis=1)
    n_surf = int(n_surf)
    surface_order = np.argsort(scores[:n_surf])[::-1][
        : min(int(surface_count), n_surf)
    ]
    deep_order = np.argsort(scores[n_surf:])[::-1][
        : min(int(deep_count), scores.size - n_surf)
    ] + n_surf
    return np.union1d(surface_order, deep_order), scores


def spatially_diverse_seeds(
    scores: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    domain_indices: np.ndarray,
    *,
    count: int,
    suppression_hops: int,
) -> np.ndarray:
    """Select high-score centers while suppressing redundant graph neighbors."""
    scores = np.asarray(scores, dtype=float).ravel()
    graph = sparse.csr_matrix(adjacency)
    domain_indices = np.asarray(domain_indices, dtype=int).ravel()
    available = np.zeros(scores.size, dtype=bool)
    available[domain_indices] = True
    selected: list[int] = []
    for _ in range(min(int(count), domain_indices.size)):
        candidates = np.flatnonzero(available)
        if candidates.size == 0:
            break
        center = int(candidates[np.argmax(scores[candidates])])
        selected.append(center)
        suppressed = expand_candidates(
            [center],
            graph,
            hops=int(suppression_hops),
        )
        available[suppressed] = False
    return np.asarray(sorted(selected), dtype=int)


def selection_probability(masks: np.ndarray) -> np.ndarray:
    """Return the source-wise selection frequency across perturbation runs."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or masks.shape[0] == 0:
        raise ValueError("masks must have shape repetitions x sources")
    return np.mean(masks, axis=0)


def stable_region_mask(
    probability: np.ndarray,
    raw_mask: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    deep_mask: np.ndarray,
    *,
    surface_probability_threshold: float = 0.55,
    deep_probability_threshold: float = 0.50,
) -> np.ndarray:
    """Keep reproducible connected support separately by anatomical domain."""
    probability = np.asarray(probability, dtype=float).ravel()
    raw_mask = np.asarray(raw_mask, dtype=bool).ravel()
    deep_mask = np.asarray(deep_mask, dtype=bool).ravel()
    if not (probability.shape == raw_mask.shape == deep_mask.shape):
        raise ValueError("probability, raw_mask and deep_mask must align")
    graph = sparse.csr_matrix(adjacency)
    output = np.zeros_like(raw_mask)
    for domain, threshold, minimum_size in (
        (~deep_mask, surface_probability_threshold, 2),
        (deep_mask, deep_probability_threshold, 1),
    ):
        selected = np.flatnonzero(
            domain & raw_mask & (probability >= float(threshold))
        )
        if selected.size == 0:
            continue
        count, labels = connected_components(
            graph[selected][:, selected],
            directed=False,
        )
        for label in range(count):
            members = selected[labels == label]
            if members.size >= minimum_size:
                output[members] = True
    return output


def _amplitude(source: np.ndarray | None, n_sources: int) -> np.ndarray:
    if source is None:
        return np.zeros(n_sources, dtype=float)
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        amplitude = np.abs(source)
    elif source.ndim == 2:
        amplitude = np.linalg.norm(source, axis=1)
    else:
        raise ValueError("source must be a vector or source-by-time matrix")
    if amplitude.size != n_sources:
        raise ValueError("source does not match source space")
    return amplitude


def build_spatial_candidates(
    point_source: np.ndarray | None,
    sisses_source: np.ndarray | None,
    adjacency: np.ndarray | sparse.spmatrix,
    n_surf: int,
    *,
    screening_seed_sets: tuple[np.ndarray, ...] = (),
    screening_score_sets: tuple[np.ndarray, ...] = (),
    surface_hops: int = 3,
    deep_hops: int = 1,
    max_candidates: int = 768,
) -> np.ndarray:
    """Build a truth-free graph neighborhood from point and SISSES centers."""
    graph = sparse.csr_matrix(adjacency)
    n_sources = graph.shape[0]
    point_amplitude = _amplitude(point_source, n_sources)
    sisses_amplitude = _amplitude(sisses_source, n_sources)
    point_centers = np.flatnonzero(point_amplitude > 0)
    sisses_centers = sisses_seed_indices(
        sisses_amplitude,
        relative_threshold=0.25,
        max_seeds=64,
    )
    centers = np.union1d(point_centers, sisses_centers)
    for screening_seeds in screening_seed_sets:
        centers = np.union1d(
            centers,
            np.asarray(screening_seeds, dtype=int).ravel(),
        )
    if centers.size == 0:
        combined = point_amplitude + sisses_amplitude
        centers = np.argsort(combined)[-8:]

    surface = expand_candidates(
        centers[centers < int(n_surf)],
        graph,
        hops=surface_hops,
    )
    deep = expand_candidates(
        centers[centers >= int(n_surf)],
        graph,
        hops=deep_hops,
    )
    candidates = np.union1d(surface, deep)
    if candidates.size <= max_candidates:
        return candidates

    center_score = point_amplitude + sisses_amplitude
    for scores in screening_score_sets:
        scores = np.asarray(scores, dtype=float).ravel()
        if scores.size != n_sources:
            raise ValueError("screening scores must match source space")
        maximum = float(scores.max(initial=0.0))
        if maximum > 0:
            center_score += scores / maximum
    order = centers[np.argsort(center_score[centers])[::-1]]
    selected: set[int] = set()
    for center in order:
        hops = surface_hops if center < int(n_surf) else deep_hops
        neighborhood = expand_candidates([center], graph, hops=hops)
        if len(selected | set(neighborhood.tolist())) > max_candidates and selected:
            continue
        selected.update(int(idx) for idx in neighborhood)
        if len(selected) >= max_candidates:
            break
    return np.asarray(sorted(selected), dtype=int)


def _prepare_modalities(
    modality_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    noise_samples: int,
    analysis_start: int,
    max_time_components: int,
    temporal_variance: float,
) -> dict:
    white_data_parts = []
    white_free_parts = []
    white_location_parts = []
    for data, gain_3d, scalar_gain in modality_specs:
        white_data, white_free = whiten_modality(data, gain_3d, noise_samples)
        _, white_location = whiten_modality(
            data,
            embed_scalar_gain(scalar_gain),
            noise_samples,
        )
        white_data_parts.append(white_data)
        white_free_parts.append(white_free)
        white_location_parts.append(white_location)

    joint_analysis = np.vstack(
        [part[:, int(analysis_start) :] for part in white_data_parts]
    )
    compressed, temporal_basis, retained = temporal_svd_compress(
        joint_analysis,
        variance_fraction=temporal_variance,
        max_components=max_time_components,
    )
    if len(white_location_parts) == 1:
        empty = np.zeros((0, white_location_parts[0].shape[1], 3))
        normalized, weights = prepare_joint_gain(white_location_parts[0], empty)
    else:
        normalized, weights = prepare_joint_gain(
            white_location_parts[0],
            white_location_parts[1],
        )
    compressed_parts = [
        part[:, int(analysis_start) :] @ temporal_basis.T
        for part in white_data_parts
    ]
    screening_seed_sets = []
    screening_score_sets = []
    for compressed_part, white_location in zip(
        compressed_parts,
        white_location_parts,
    ):
        seeds, scores = whole_brain_screening_seeds(
            compressed_part,
            white_location[:, :, 0],
            white_location.shape[1],
            surface_count=0,
            deep_count=0,
        )
        screening_seed_sets.append(seeds)
        screening_score_sets.append(scores)
    return {
        "compressed_data": compressed,
        "temporal_basis": temporal_basis,
        "retained_temporal_variance": retained,
        "normalized_scalar_gain": normalized[:, :, 0],
        "group_weights": weights,
        "joint_free_gain": np.concatenate(white_free_parts, axis=0),
        "white_data_parts": white_data_parts,
        "white_free_parts": white_free_parts,
        "white_location_parts": white_location_parts,
        "compressed_parts": compressed_parts,
        "screening_score_sets": tuple(screening_score_sets),
        "n_times": white_data_parts[0].shape[1],
    }


def run_spatial_fused_solver(
    modality_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    adjacency: np.ndarray | sparse.spmatrix,
    n_surf: int,
    *,
    point_source: np.ndarray | None = None,
    sisses_source: np.ndarray | None = None,
    screening_seed_sets: tuple[np.ndarray, ...] | None = None,
    screening_score_sets: tuple[np.ndarray, ...] | None = None,
    noise_samples: int = NOISE_SAMPLES,
    analysis_start: int = ANALYSIS_START,
    lambda_fractions: tuple[float, ...] = LAMBDA_FRACTIONS,
    beta_fractions: tuple[float, ...] = BETA_FRACTIONS,
    deep_fusion_scales: tuple[float, ...] = DEEP_FUSION_SCALES,
    max_time_components: int = MAX_TIME_COMPONENTS,
    temporal_variance: float = TEMPORAL_VARIANCE,
    deep_fusion_scale: float = 0.35,
    stability_repetitions: int = 6,
    stability_noise_fraction: float = 0.08,
    stability_random_state: int = 20260625,
    forced_candidate_mask: np.ndarray | None = None,
    lambda_weight_vector: np.ndarray | None = None,
    surface_region_threshold: float = 0.10,
    deep_region_threshold: float = 0.20,
) -> dict:
    """Run truth-free spatial fused source-range reconstruction."""
    prepared = _prepare_modalities(
        modality_specs,
        noise_samples=noise_samples,
        analysis_start=analysis_start,
        max_time_components=max_time_components,
        temporal_variance=temporal_variance,
    )
    n_sources = prepared["normalized_scalar_gain"].shape[1]
    if screening_score_sets is None:
        screening_score_sets = prepared["screening_score_sets"]
    if screening_seed_sets is None:
        generated_seeds = []
        for scores in screening_score_sets:
            surface = spatially_diverse_seeds(
                scores,
                adjacency,
                np.arange(int(n_surf)),
                count=64,
                suppression_hops=2,
            )
            deep = spatially_diverse_seeds(
                scores,
                adjacency,
                np.arange(int(n_surf), n_sources),
                count=16,
                suppression_hops=1,
            )
            generated_seeds.append(np.union1d(surface, deep))
        _, joint_scores = whole_brain_screening_seeds(
            prepared["compressed_data"],
            prepared["normalized_scalar_gain"],
            n_surf,
            surface_count=0,
            deep_count=0,
        )
        joint_surface = spatially_diverse_seeds(
            joint_scores,
            adjacency,
            np.arange(int(n_surf)),
            count=96,
            suppression_hops=2,
        )
        joint_deep = spatially_diverse_seeds(
            joint_scores,
            adjacency,
            np.arange(int(n_surf), n_sources),
            count=24,
            suppression_hops=1,
        )
        joint_seeds = np.union1d(joint_surface, joint_deep)
        screening_seed_sets = tuple(generated_seeds) + (joint_seeds,)
        screening_score_sets = tuple(screening_score_sets) + (joint_scores,)
    if forced_candidate_mask is None:
        candidates = build_spatial_candidates(
            point_source,
            sisses_source,
            adjacency,
            n_surf,
            screening_seed_sets=screening_seed_sets,
            screening_score_sets=screening_score_sets,
        )
    else:
        forced = np.asarray(forced_candidate_mask, dtype=bool).ravel()
        if forced.size != n_sources:
            raise ValueError("forced_candidate_mask must match source count")
        candidates = np.flatnonzero(forced)
        if candidates.size == 0:
            raise ValueError("forced_candidate_mask selects no candidate sources")
    if lambda_weight_vector is None:
        candidate_lambda_weights = None
        full_lambda_weights = np.ones(n_sources, dtype=float)
    else:
        full_lambda_weights = np.asarray(lambda_weight_vector, dtype=float).ravel()
        if full_lambda_weights.size != n_sources:
            raise ValueError("lambda_weight_vector must match source count")
        candidate_lambda_weights = full_lambda_weights[candidates]
    candidate_graph = sparse.csr_matrix(adjacency)[candidates][:, candidates]
    candidate_gain = prepared["normalized_scalar_gain"][:, candidates]
    candidate_deep = candidates >= int(n_surf)

    started = time.perf_counter()
    selection = solve_spatial_fused_path(
        prepared["compressed_data"],
        candidate_gain,
        candidate_graph,
        lambda_fractions=lambda_fractions,
        beta_fractions=beta_fractions,
        lambda_weights=candidate_lambda_weights,
        deep_mask=candidate_deep,
        deep_fusion_scale=deep_fusion_scale,
        deep_fusion_scales=deep_fusion_scales,
        max_iter=3000,
        tolerance=1e-3,
    )
    runtime = time.perf_counter() - started
    selected_point = selection.point
    selected = selected_point.result
    physical_compressed = (
        selected.coefficients
        / prepared["group_weights"][candidates, None]
    )
    # Graph fusion is imposed on the leadfield-normalized optimization
    # variable.  Extract its spatial support before undoing depth
    # normalization; physical currents are used below for amplitudes/refits.
    raw_local_region = anatomical_region_mask(
        selected.coefficients,
        candidate_graph,
        candidate_deep,
        surface_relative_threshold=surface_region_threshold,
        deep_relative_threshold=deep_region_threshold,
        minimum_domain_energy_fraction=0.01,
        minimum_component_energy_fraction=0.001,
    )
    if not np.any(raw_local_region):
        raw_local_region[
            int(np.argmax(np.linalg.norm(physical_compressed, axis=1)))
        ] = True

    perturbation_masks = [raw_local_region]
    noise_scale = float(np.std(selected.residual))
    if stability_repetitions > 0 and noise_scale > 0:
        rng = np.random.default_rng(int(stability_random_state))
        lambda_max = selected_point.lam / max(
            selected_point.lambda_fraction,
            np.finfo(float).eps,
        )
        for _ in range(int(stability_repetitions)):
            perturbed_data = prepared["compressed_data"] + rng.normal(
                scale=float(stability_noise_fraction) * noise_scale,
                size=prepared["compressed_data"].shape,
            )
            perturbed = solve_graph_fused_lasso(
                perturbed_data,
                candidate_gain,
                candidate_graph,
                lam=(
                    selected_point.lam
                    if candidate_lambda_weights is None
                    else selected_point.lam * candidate_lambda_weights
                ),
                beta=selected_point.beta,
                deep_mask=candidate_deep,
                deep_fusion_scale=selected_point.deep_fusion_scale,
                rho=max(1.0, 0.001 * lambda_max),
                max_iter=1500,
                tolerance=1e-3,
                initial_coefficients=selected.coefficients,
            )
            perturbation_masks.append(
                anatomical_region_mask(
                    perturbed.coefficients,
                    candidate_graph,
                    candidate_deep,
                    surface_relative_threshold=surface_region_threshold,
                    deep_relative_threshold=deep_region_threshold,
                    minimum_domain_energy_fraction=0.01,
                    minimum_component_energy_fraction=0.001,
                )
            )
    local_stability = selection_probability(
        np.asarray(perturbation_masks, dtype=bool)
    )
    local_region = stable_region_mask(
        local_stability,
        raw_local_region,
        candidate_graph,
        candidate_deep,
    )
    for domain in (~candidate_deep, candidate_deep):
        domain_raw = np.flatnonzero(raw_local_region & domain)
        if domain_raw.size and not np.any(local_region & domain):
            local_region[
                domain_raw[np.argmax(local_stability[domain_raw])]
            ] = True
    region_indices = candidates[local_region]

    physical_full_time = physical_compressed @ prepared["temporal_basis"]
    source = np.zeros((n_sources, prepared["n_times"]), dtype=float)
    source[candidates, int(analysis_start) :] = np.abs(physical_full_time)
    region_mask = np.zeros(n_sources, dtype=bool)
    region_mask[region_indices] = True
    raw_region_mask = np.zeros(n_sources, dtype=bool)
    raw_region_mask[candidates[raw_local_region]] = True
    stability_probability = np.zeros(n_sources, dtype=float)
    stability_probability[candidates] = local_stability

    local_positions = np.flatnonzero(local_region)
    refit = refit_rank_one_currents(
        prepared["compressed_data"],
        prepared["joint_free_gain"],
        region_indices,
        physical_compressed[local_positions],
        max_iter=25,
        tolerance=1e-8,
    )
    direction_confidence = None
    if len(modality_specs) == 2:
        compressed_parts = [
            part[:, int(analysis_start) :] @ prepared["temporal_basis"].T
            for part in prepared["white_data_parts"]
        ]
        direction_confidence = modality_direction_confidence(
            compressed_parts[0],
            prepared["white_free_parts"][0],
            compressed_parts[1],
            prepared["white_free_parts"][1],
            region_indices,
            refit.time_courses,
            refit.directions,
        )

    return {
        "source": source,
        "region_mask": region_mask,
        "raw_region_mask": raw_region_mask,
        "stability_probability": stability_probability,
        "region_indices": region_indices,
        "candidate_indices": candidates,
        "lambda_weight_vector": full_lambda_weights,
        "candidate_coefficients": physical_compressed,
        "selection": selection,
        "directions": refit.directions,
        "direction_confidence": direction_confidence,
        "rank_one_refit_residual": refit.relative_residual,
        "temporal_rank": prepared["temporal_basis"].shape[0],
        "retained_temporal_variance": prepared["retained_temporal_variance"],
        "runtime_seconds": runtime,
    }


def evaluate_spatial_result(
    source: np.ndarray,
    region_mask: np.ndarray,
    truth: dict,
    adjacency: np.ndarray | sparse.spmatrix,
) -> dict:
    """Apply truth only after the reconstruction is fixed."""
    vertices = np.asarray(truth["src_vertices"], dtype=float)
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    n_sources = len(region_mask)
    true_surface_indices = np.asarray(
        truth["true_surface_indices0"],
        dtype=int,
    ).ravel()
    true_surface_mask = np.zeros(n_sources, dtype=bool)
    true_surface_mask[true_surface_indices] = True
    cortical_estimated = np.asarray(region_mask, dtype=bool).copy()
    cortical_estimated[n_surf:] = False
    cortical_truth = true_surface_mask.copy()
    cortical_metrics = range_metrics(
        cortical_estimated,
        cortical_truth,
        adjacency,
    )
    row = {
        f"cortical_{key}": value
        for key, value in cortical_metrics.items()
    }

    labels = np.asarray(
        truth.get("true_surface_patch_labels", np.zeros((1, 0))),
        dtype=int,
    ).ravel()
    true_groups = []
    for label in sorted(set(labels.tolist()) - {0}):
        patch_truth = np.zeros(n_sources, dtype=bool)
        patch_indices = true_surface_indices[labels == label]
        patch_truth[patch_indices] = True
        true_groups.append(patch_indices)
        patch_metrics = range_metrics(
            cortical_estimated,
            patch_truth,
            adjacency,
        )
        row[f"patch_{label}_dice"] = patch_metrics["dice"]
        row[f"patch_{label}_recall"] = patch_metrics["recall"]
    if not true_groups and true_surface_indices.size:
        true_groups.append(true_surface_indices)

    has_deep = bool(
        int(np.asarray(truth.get("has_deep_source", [[1]])).ravel()[0])
    )
    true_deep = int(np.asarray(truth["true_deep_idx0"]).ravel()[0])
    deep_estimated = np.flatnonzero(region_mask & (np.arange(n_sources) >= n_surf))
    amplitude = source_amplitude(source)
    if deep_estimated.size:
        deep_peak = int(deep_estimated[np.argmax(amplitude[deep_estimated])])
        weights = amplitude[deep_estimated]
        centroid = np.average(vertices[deep_estimated], axis=0, weights=weights)
    else:
        deep_peak = -1
        centroid = np.full(3, np.nan)
    row.update(
        {
            "deep_range_count": int(deep_estimated.size),
            "true_deep_covered": int(true_deep in deep_estimated) if has_deep else -1,
            "deep_peak_idx1": deep_peak + 1 if deep_peak >= 0 else -1,
            "deep_peak_dle_mm": (
                float(np.linalg.norm(vertices[deep_peak] - vertices[true_deep]) * 1000)
                if has_deep and deep_peak >= 0
                else np.nan
            ),
            "deep_centroid_dle_mm": (
                float(np.linalg.norm(centroid - vertices[true_deep]) * 1000)
                if has_deep and deep_estimated.size
                else np.nan
            ),
        }
    )
    total = max(float(amplitude.sum()), np.finfo(float).eps)
    row["surface_energy_ratio"] = float(amplitude[:n_surf].sum() / total)
    row["deep_energy_ratio"] = float(amplitude[n_surf:].sum() / total)

    truth_source = np.asarray(truth["s_true"], dtype=float)
    range_source = np.asarray(source, dtype=float) * np.asarray(
        region_mask,
        dtype=bool,
    )[:, None]
    surface_domain = np.arange(n_sources) < n_surf
    deep_domain = ~surface_domain
    true_surface_centers = np.asarray(
        truth.get("true_surface_centers0", true_surface_indices[:1]),
        dtype=int,
    ).ravel()
    all_centers = true_surface_centers.copy()
    if has_deep:
        all_centers = np.r_[all_centers, true_deep]
        true_groups.append(np.array([true_deep], dtype=int))
    for prefix, domain in (
        ("all", np.ones(n_sources, dtype=bool)),
        ("surface", surface_domain),
        ("deep", deep_domain),
    ):
        continuous = continuous_source_metrics(
            source,
            truth_source,
            vertices,
            domain,
        )
        row.update(
            {
                f"{prefix}_{key}": value
                for key, value in continuous.items()
            }
        )
        range_continuous = continuous_source_metrics(
            range_source,
            truth_source,
            vertices,
            domain,
        )
        for key in ("sd_mm", "peak_dle_mm", "centroid_dle_mm"):
            row[f"{prefix}_range_{key}"] = range_continuous[key]
        if prefix == "all":
            centers = all_centers
        elif prefix == "surface":
            centers = true_surface_centers
        else:
            centers = np.array([true_deep], dtype=int) if has_deep else np.array([], dtype=int)
        pdf_metrics = pdf_source_metrics(
            source,
            truth_source,
            vertices,
            adjacency,
            domain_mask=domain,
            centers=centers,
            auc_repetitions=50,
            random_state=20260625,
            neighborhood_order=10,
        )
        row.update(
            {
                f"pdf_{prefix}_{key}": value
                for key, value in pdf_metrics.items()
            }
        )
        pdf_range_metrics = pdf_source_metrics(
            range_source,
            truth_source,
            vertices,
            adjacency,
            domain_mask=domain,
            centers=centers,
            auc_repetitions=50,
            random_state=20260625,
            neighborhood_order=10,
        )
        row.update(
            {
                f"pdf_range_{prefix}_{key}": value
                for key, value in pdf_range_metrics.items()
            }
        )
        if prefix == "all":
            external = external_full_head_metrics(
                range_source,
                truth_source,
                vertices,
                true_groups=true_groups,
            )
            row.update(
                {
                    f"pdf_range_all_{key}": value
                    for key, value in external.items()
                }
            )
            row.update(
                {
                    "primary_auc": external["auc"],
                    "primary_rmse": external["rmse"],
                    "primary_sd_mm": external["sd_mm"],
                    "primary_dle_mm": external["dle_mm"],
                }
            )

    surface_detected = bool(np.any(region_mask & surface_domain))
    deep_detected = bool(np.any(region_mask & deep_domain))
    has_surface = bool(true_surface_indices.size)
    row.update(
        {
            "surface_detected": int(surface_detected),
            "deep_detected": int(deep_detected),
            "simultaneous_surface_deep_detected": (
                int(surface_detected and deep_detected)
                if has_surface and has_deep
                else -1
            ),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_spatial_result(
    out_dir: Path,
    result: dict,
    times: np.ndarray,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    confidence = result["direction_confidence"]
    eeg_meg = (
        confidence["eeg_meg_cosine"]
        if confidence is not None
        else np.zeros(0, dtype=float)
    )
    npz_arrays = {
        "S": result["source"],
        "times": times,
        "region_mask": result["region_mask"],
        "raw_region_mask": result["raw_region_mask"],
        "stability_probability": result["stability_probability"],
        "region_indices0": result["region_indices"],
        "candidate_indices0": result["candidate_indices"],
        "directions": result["directions"],
        "eeg_meg_direction_cos": eeg_meg,
        "selected_lambda_fraction": np.array(
            result["selection"].point.lambda_fraction
        ),
        "selected_beta_fraction": np.array(
            result["selection"].point.beta_fraction
        ),
        "selected_deep_fusion_scale": np.array(
            result["selection"].point.deep_fusion_scale
        ),
    }
    if extra_arrays:
        npz_arrays.update(extra_arrays)
    np.savez(out_dir / "spatial_fused_result.npz", **npz_arrays)
    sio.savemat(
        out_dir / "spatial_fused_result.mat",
        {
            "S": result["source"],
            "times": times[np.newaxis, :],
            "region_mask": result["region_mask"][np.newaxis, :],
            "raw_region_mask": result["raw_region_mask"][np.newaxis, :],
            "stability_probability": result["stability_probability"][
                np.newaxis, :
            ],
            "region_indices0": result["region_indices"][np.newaxis, :],
            "candidate_indices0": result["candidate_indices"][np.newaxis, :],
            "directions": result["directions"],
            "selected_lambda_fraction": result["selection"].point.lambda_fraction,
            "selected_beta_fraction": result["selection"].point.beta_fraction,
            "selected_deep_fusion_scale": (
                result["selection"].point.deep_fusion_scale
            ),
        },
        do_compression=True,
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    metric_rows = []
    path_rows = []
    comparison_rows = []
    for scenario in SCENARIOS:
        print(f"\n==== spatial fused: {scenario} ====")
        data_dir = DATA_ROOT / scenario
        eeg = load_mat(data_dir / "sub_EEG.mat")
        meg = load_mat(data_dir / "sub_MEG.mat")
        truth = load_mat(data_dir / "s_true.mat")
        adjacency = sparse.csr_matrix(eeg["VertConn"])
        n_surf = int(np.asarray(eeg["n_surf"]).ravel()[0])
        n_sources = np.asarray(eeg["Gain"]).shape[1]
        point = np.load(
            POINT_ROOT / scenario / "whole_brain_result.npz",
            allow_pickle=True,
        )
        point_source = np.asarray(point["S"], dtype=float)
        sisses_source = load_sisses(
            SISSES_ROOT / scenario / "s_wen_sisses_fusion_quick_best.mat",
            n_sources,
        )
        result = run_spatial_fused_solver(
            [
                (eeg["F"], eeg["Gain3D"], eeg["Gain"]),
                (meg["F"], meg["Gain3D"], meg["Gain"]),
            ],
            adjacency,
            n_surf,
            point_source=point_source,
            sisses_source=sisses_source,
        )
        times = np.asarray(truth["times"], dtype=float).ravel()
        save_spatial_result(OUT_ROOT / scenario, result, times)
        metrics = evaluate_spatial_result(
            result["source"],
            result["region_mask"],
            truth,
            adjacency,
        )
        selected = result["selection"].point
        row = {
            "scenario": scenario,
            "method": "Spatial_fused_fusion",
            "selected_lambda_fraction": selected.lambda_fraction,
            "selected_beta_fraction": selected.beta_fraction,
            "selected_deep_fusion_scale": selected.deep_fusion_scale,
            "candidate_count": len(result["candidate_indices"]),
            "region_count": len(result["region_indices"]),
            "relative_residual": selected.result.relative_residual,
            "rank_one_refit_residual": result["rank_one_refit_residual"],
            "admm_converged": int(selected.result.converged),
            "admm_iterations": selected.result.iterations,
            "runtime_seconds": result["runtime_seconds"],
            **metrics,
        }
        summary_rows.append(row)
        metric_rows.append({"scenario": scenario, **metrics})
        comparison_rows.append(
            {
                "scenario": scenario,
                "point_active_count": len(np.asarray(point["active_indices0"]).ravel()),
                "spatial_region_count": len(result["region_indices"]),
                **metrics,
            }
        )
        for path_point in result["selection"].points:
            path_rows.append(
                {
                    "scenario": scenario,
                    "index": path_point.index,
                    "lambda_fraction": path_point.lambda_fraction,
                    "beta_fraction": path_point.beta_fraction,
                    "deep_fusion_scale": path_point.deep_fusion_scale,
                    "active_count": path_point.active_count,
                    "jump_count": path_point.jump_count,
                    "fused_group_count": path_point.fused_group_count,
                    "relative_residual": path_point.result.relative_residual,
                    "bic": path_point.bic,
                    "converged": int(path_point.result.converged),
                    "iterations": path_point.result.iterations,
                    "selected": int(path_point.index == result["selection"].index),
                }
            )
        print(
            f"{scenario}: candidate={len(result['candidate_indices'])} "
            f"range={len(result['region_indices'])} "
            f"lambda={selected.lambda_fraction:.3f} "
            f"beta={selected.beta_fraction:.3f} "
            f"dice={metrics['cortical_dice']:.3f} "
            f"deepDLE={metrics['deep_peak_dle_mm']}"
        )
    write_csv(OUT_ROOT / "spatial_fused_summary.csv", summary_rows)
    write_csv(OUT_ROOT / "source_range_metrics.csv", metric_rows)
    write_csv(OUT_ROOT / "spatial_fused_parameter_path.csv", path_rows)
    write_csv(
        OUT_ROOT / "spatial_fused_vs_point_summary.csv",
        comparison_rows,
    )
    print("\nSaved:", OUT_ROOT)


if __name__ == "__main__":
    main()
