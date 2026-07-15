from __future__ import annotations

import csv
from collections import deque
from itertools import groupby
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.external_metrics import DLE_an, external_full_head_metrics
from auc_metric import an_auc_from_cortex, auc_cortex
from pipelines.sisses_direct_utils import best_estimated_waveform, group_waveform
from protected_multilayer import (
    DATA_ROOT,
    OUT_ROOT,
    V8_FACTORS,
    V9_FACTORS,
    V9B_VARIANTS,
    component_refit_select,
    component_refit_select_v3,
    component_refit_select_v4_deep_rescue,
    component_refit_select_v5_compact_deep_prior,
    component_refit_select_v6_sisses_refit,
    component_refit_select_v7_tbf_refit,
    component_refit_select_v8_protected_sisses,
    component_refit_select_v9_layerwise_sisses,
    component_refit_select_v11_compactness_sisses,
    component_refit_select_v14_local_evidence,
    connected_components,
    evidence_aware_compact_mask,
    load_mat,
    load_source,
    source_amplitude,
    threshold_mask,
    whitened_joint_system,
)
from refined_grid import load_refined_forward, refined_deep_evidence, refined_sd_dle

VAL_ROOT = OUT_ROOT / "position_validation"
VAL_DATA = VAL_ROOT / "generated"
VAL_RUNS = VAL_ROOT / "sisses_runs"
VAL_FIGS = VAL_ROOT / "waveforms"
V11_ROOT = OUT_ROOT / "v11"
V14_ROOT = OUT_ROOT / "v14"
NOISE_SAMPLES = 200


def _neighbors(adjacency: np.ndarray, start: int, count: int) -> np.ndarray:
    seen = {int(start)}
    queue: deque[int] = deque([int(start)])
    while queue and len(seen) < count:
        item = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[item]):
            if int(neighbor) not in seen:
                seen.add(int(neighbor))
                queue.append(int(neighbor))
                if len(seen) >= count:
                    break
    return np.array(sorted(seen), dtype=int)


def _wave(times: np.ndarray, freq: float, phase: float = 0.0) -> np.ndarray:
    wave = np.zeros_like(times, dtype=float)
    active = np.arange(times.size) >= NOISE_SAMPLES
    t = times[active] - times[NOISE_SAMPLES]
    envelope = np.sin(np.linspace(0, np.pi, active.sum()))
    wave[active] = envelope * np.sin(2 * np.pi * freq * t + phase)
    return wave


def _noise_like(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = np.asarray(data[:, :NOISE_SAMPLES], dtype=float)
    noise = noise - noise.mean(axis=1, keepdims=True)
    cov = noise @ noise.T / max(1, noise.shape[1] - 1)
    cov = 0.5 * (cov + cov.T) + np.eye(noise.shape[0]) * np.trace(cov) * 1e-6 / cov.shape[0]
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=data.shape[1]).T


def _true_groups(truth: dict) -> list[np.ndarray]:
    groups = []
    labels = np.asarray(truth["true_surface_patch_labels"], dtype=int).ravel()
    surface = np.asarray(truth["true_surface_indices0"], dtype=int).ravel()
    for label in sorted(set(labels.tolist()) - {0}):
        groups.append(surface[labels == label])
    if int(np.asarray(truth["has_deep_source"]).ravel()[0]):
        groups.append(np.array([int(np.asarray(truth["true_deep_idx0"]).ravel()[0])], dtype=int))
    return groups


def _component_centroid_dle(
    source: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
    true_groups: list[np.ndarray],
    n_surf: int,
    vert_conn: np.ndarray,
    *,
    deep_position: np.ndarray | None = None,
) -> dict[str, float]:
    """Match predicted components to true groups and compare their centroids."""
    source = np.asarray(source, dtype=float)
    mask = np.asarray(mask, dtype=bool).ravel()
    positions = np.asarray(positions, dtype=float)
    energy = np.sum(source * source, axis=1)
    components = [("surface", component) for component in connected_components(mask[:n_surf], vert_conn[:n_surf, :n_surf])]
    components += [("deep", np.array([index], dtype=int)) for index in np.flatnonzero(mask[n_surf:]) + n_surf]
    deep_position = np.asarray(deep_position, dtype=float).reshape(-1, 3) if deep_position is not None else np.empty((0, 3))
    deep_component_count = sum(layer == "deep" for layer, _component in components)

    predicted: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"surface": [], "deep": []}
    for layer, component in components:
        xyz = positions[component]
        if layer == "deep" and deep_position.shape[0] == 1 and deep_component_count == 1:
            xyz = deep_position
        support_center = xyz.mean(axis=0)
        weights = np.maximum(energy[component], np.finfo(float).eps)
        energy_center = np.average(xyz, axis=0, weights=weights[: xyz.shape[0]])
        predicted[layer].append((support_center, energy_center))

    truth: dict[str, list[np.ndarray]] = {"surface": [], "deep": []}
    for group in true_groups:
        group = np.asarray(group, dtype=int).ravel()
        layer = "deep" if np.all(group >= n_surf) else "surface"
        truth[layer].append(positions[group].mean(axis=0))

    def matched_distances(center_index: int, layer: str) -> list[float]:
        if not truth[layer] or not predicted[layer]:
            return []
        true_xyz = np.vstack(truth[layer])
        predicted_xyz = np.vstack([item[center_index] for item in predicted[layer]])
        cost = np.linalg.norm(true_xyz[:, None, :] - predicted_xyz[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(cost)
        return cost[rows, columns].tolist()

    support_by_layer = {layer: matched_distances(0, layer) for layer in ("surface", "deep")}
    energy_by_layer = {layer: matched_distances(1, layer) for layer in ("surface", "deep")}
    support_distances = support_by_layer["surface"] + support_by_layer["deep"]
    energy_distances = energy_by_layer["surface"] + energy_by_layer["deep"]
    matched = len(support_distances)
    count = len(true_groups)

    def mean_mm(values: list[float]) -> float:
        return float(np.mean(values) * 1000.0) if values else np.nan

    return {
        "support_centroid_dle_mm": mean_mm(support_distances),
        "energy_centroid_dle_mm": mean_mm(energy_distances),
        "centroid_match_rate": matched / count if count else np.nan,
        "surface_support_centroid_dle_mm": mean_mm(support_by_layer["surface"]),
        "deep_support_centroid_dle_mm": mean_mm(support_by_layer["deep"]),
        "surface_energy_centroid_dle_mm": mean_mm(energy_by_layer["surface"]),
        "deep_energy_centroid_dle_mm": mean_mm(energy_by_layer["deep"]),
        "surface_centroid_match_rate": len(support_by_layer["surface"]) / len(truth["surface"]) if truth["surface"] else np.nan,
        "deep_centroid_match_rate": len(support_by_layer["deep"]) / len(truth["deep"]) if truth["deep"] else np.nan,
    }


def _layerwise_peak_dle(
    source: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
    true_groups: list[np.ndarray],
    n_surf: int,
    vert_conn: np.ndarray,
    *,
    deep_position: np.ndarray | None = None,
) -> dict[str, float]:
    """Run the requested peak DLE independently in surface and deep spaces."""
    source = np.asarray(source, dtype=float)
    mask = np.asarray(mask, dtype=bool).ravel()
    positions = np.asarray(positions, dtype=float)
    deep_position = np.asarray(deep_position, dtype=float).reshape(-1, 3) if deep_position is not None else np.empty((0, 3))
    groups = {
        "surface": [np.asarray(group, dtype=int).ravel() for group in true_groups if np.all(np.asarray(group) < n_surf)],
        "deep": [np.asarray(group, dtype=int).ravel() for group in true_groups if np.all(np.asarray(group) >= n_surf)],
    }

    def evaluate(layer: str) -> tuple[float, float]:
        layer_groups = groups[layer]
        if not layer_groups:
            return np.nan, np.nan
        if layer == "surface":
            layer_source = source[:n_surf] * mask[:n_surf, None]
            layer_positions = positions[:n_surf]
            local_groups = [group + 1 for group in layer_groups]
            active_positions = layer_positions[mask[:n_surf]]
            component_count = len(connected_components(mask[:n_surf], vert_conn[:n_surf, :n_surf]))
        else:
            active = np.flatnonzero(mask[n_surf:]) + n_surf
            active_positions = positions[active]
            if deep_position.shape[0] == 1 and active.size == 1:
                active_positions = deep_position
                layer_source = source[active]
                true_positions = np.vstack([positions[group] for group in layer_groups])
                layer_positions = np.vstack([active_positions, true_positions])
                layer_source = np.vstack([layer_source, np.zeros((true_positions.shape[0], source.shape[1]))])
                local_groups = []
                start = 1
                for group in layer_groups:
                    local_groups.append(np.arange(start, start + group.size) + 1)
                    start += group.size
            else:
                layer_source = source[n_surf:] * mask[n_surf:, None]
                layer_positions = positions[n_surf:]
                local_groups = [group - n_surf + 1 for group in layer_groups]
            component_count = active.size

        hit = np.mean(
            [
                active_positions.size > 0
                and float(np.linalg.norm(active_positions[:, None, :] - positions[group][None, :, :], axis=2).min()) <= 0.010
                for group in layer_groups
            ]
        )
        if component_count < len(layer_groups) or not np.any(np.sum(layer_source * layer_source, axis=1) > 0):
            return np.nan, float(hit)
        return float(DLE_an(layer_source, local_groups, layer_positions) * 1000.0), float(hit)

    surface_dle, surface_hit = evaluate("surface")
    deep_dle, deep_hit = evaluate("deep")
    return {
        "surface_dle_mm": surface_dle,
        "deep_dle_mm": deep_dle,
        "surface_hit_rate_10mm": surface_hit,
        "deep_hit_rate_10mm": deep_hit,
    }


def _group_report_rows(case_id: str, method: str, source: np.ndarray, mask: np.ndarray, candidate: np.ndarray, truth: dict, vert_conn: np.ndarray) -> list[dict]:
    positions = np.asarray(truth["src_vertices"], dtype=float) * 1000.0
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    support = np.flatnonzero(mask)
    candidates = np.flatnonzero(candidate)
    amp = source_amplitude(source)
    clusters = connected_components(mask[:n_surf], vert_conn[:n_surf, :n_surf])
    clusters += [np.array([idx], dtype=int) for idx in np.flatnonzero(mask[n_surf:]) + n_surf]
    rows = []

    def group_sd(indices: np.ndarray, true_xyz: np.ndarray) -> float:
        if indices.size == 0:
            return float("nan")
        weights = amp[indices]
        if float(weights.sum()) <= 0:
            return float("nan")
        dist = np.linalg.norm(positions[indices, None, :] - true_xyz[None, :, :], axis=2).min(axis=1)
        return float(np.sqrt(np.average(dist**2, weights=weights)))

    for group_id, group in enumerate(_true_groups(truth), start=1):
        group = np.asarray(group, dtype=int).ravel()
        group_type = "deep" if np.all(group >= n_surf) else "surface"
        true_xyz = positions[group]

        def nearest(indices: np.ndarray) -> float:
            if indices.size == 0:
                return float("nan")
            dist = np.linalg.norm(positions[indices, None, :] - true_xyz[None, :, :], axis=2)
            return float(dist.min())

        nearest_support = nearest(support)
        peak_dist = float("nan")
        cluster_peak_dist = float("nan")
        centroid_dist = float("nan")
        if support.size:
            peak = int(support[np.argmax(amp[support])])
            peak_dist = nearest(np.array([peak], dtype=int))
            weights = amp[support]
            if float(weights.sum()) > 0:
                centroid = np.average(positions[support], axis=0, weights=weights)
                centroid_dist = float(np.linalg.norm(centroid[None, :] - true_xyz, axis=1).min())
        if clusters:
            nearest_cluster = min(clusters, key=nearest)
            cluster_peak = int(nearest_cluster[np.argmax(amp[nearest_cluster])])
            cluster_peak_dist = nearest(np.array([cluster_peak], dtype=int))
            cluster_sd = group_sd(nearest_cluster, true_xyz)
            group_active_count = int(nearest_cluster.size)
        else:
            cluster_sd = float("nan")
            group_active_count = 0
        rows.append(
            {
                "scenario": case_id.rsplit("_", 1)[0],
                "case_id": case_id,
                "group_id": group_id,
                "group_type": group_type,
                "method": method,
                "nearest_support_dist_mm": nearest_support,
                "global_peak_dist_mm": peak_dist,
                "cluster_peak_dist_mm": cluster_peak_dist,
                "group_sd_mm": cluster_sd,
                "group_active_count": group_active_count,
                "peak_dist_mm": peak_dist,
                "centroid_dist_mm": centroid_dist,
                "support_hit_true": int(np.isfinite(nearest_support) and nearest_support <= 10.0),
                "candidate_hit_true": int(np.isfinite(nearest(candidates)) and nearest(candidates) <= 10.0),
                "active_count": int(mask.sum()),
                "surface_active_count": int(mask[:n_surf].sum()),
                "deep_active_count": int(mask[n_surf:].sum()),
            }
        )
    return rows


def _prediction_components(mask: np.ndarray, n_surf: int, vert_conn: np.ndarray) -> list[np.ndarray]:
    components = connected_components(mask[:n_surf], vert_conn[:n_surf, :n_surf])
    components += [np.array([idx], dtype=int) for idx in np.flatnonzero(mask[n_surf:]) + n_surf]
    return components


def _component_evidence(b: np.ndarray, l: np.ndarray, source: np.ndarray, components: list[np.ndarray]) -> list[dict]:
    full_rss = float(np.linalg.norm(b - l @ source, "fro") ** 2)
    amp = source_amplitude(source)
    rows = []
    for component_id, component in enumerate(components, start=1):
        without = source.copy()
        without[component] = 0.0
        residual_drop = max(0.0, float(np.linalg.norm(b - l @ without, "fro") ** 2) - full_rss)
        energy = float(np.linalg.norm(source[component]))
        complexity = float(np.sqrt(max(1, component.size)))
        score = residual_drop * energy / complexity
        peak = int(component[np.argmax(amp[component])])
        rows.append(
            {
                "component_id": component_id,
                "peak_index": peak,
                "n_points": int(component.size),
                "source_energy": energy,
                "residual_drop": residual_drop,
                "complexity": complexity,
                "evidence_score": score,
            }
        )
    return sorted(rows, key=lambda row: row["evidence_score"], reverse=True)


def _evidence_peak_rows(case_id: str, method: str, source: np.ndarray, mask: np.ndarray, truth: dict, vert_conn: np.ndarray, eeg: dict, meg: dict) -> list[dict]:
    positions = np.asarray(truth["src_vertices"], dtype=float) * 1000.0
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    components = _prediction_components(mask, n_surf, vert_conn)
    b, l = whitened_joint_system(eeg, meg)
    ranked = _component_evidence(b, l, source * mask[:, None], components)
    if not ranked:
        return []
    amp = source_amplitude(source)
    amp_peak = int(np.flatnonzero(mask)[np.argmax(amp[mask])]) if mask.any() else -1
    evidence_peak = int(ranked[0]["peak_index"])
    groups = [(i, np.asarray(group, dtype=int).ravel()) for i, group in enumerate(_true_groups(truth), start=1)]
    assigned: dict[int, int] = {}
    unused_groups = set(group_id for group_id, _group in groups)
    for component in ranked:
        if not unused_groups:
            break
        peak = int(component["peak_index"])
        best_group = min(
            unused_groups,
            key=lambda gid: float(np.linalg.norm(positions[peak][None, :] - positions[groups[gid - 1][1]], axis=1).min()),
        )
        assigned[best_group] = peak
        unused_groups.remove(best_group)
    rows = []
    for group_id, group in groups:
        group = np.asarray(group, dtype=int).ravel()
        group_type = "deep" if np.all(group >= n_surf) else "surface"
        true_xyz = positions[group]

        def dist_to(index: int) -> float:
            if index < 0:
                return float("nan")
            return float(np.linalg.norm(positions[index][None, :] - true_xyz, axis=1).min())

        rows.append(
            {
                "scenario": case_id.rsplit("_", 1)[0],
                "case_id": case_id,
                "method": method,
                "group_id": group_id,
                "group_type": group_type,
                "amplitude_peak_index": amp_peak,
                "evidence_peak_index": evidence_peak,
                "amplitude_peak_dist_mm": dist_to(amp_peak),
                "evidence_peak_dist_mm": dist_to(evidence_peak),
                "evidence_assigned_peak_dist_mm": dist_to(assigned.get(group_id, evidence_peak)),
                "top_component_id": int(ranked[0]["component_id"]),
                "top_component_points": int(ranked[0]["n_points"]),
                "top_component_score": float(ranked[0]["evidence_score"]),
                "top_component_residual_drop": float(ranked[0]["residual_drop"]),
                "top_component_energy": float(ranked[0]["source_energy"]),
            }
        )
    return rows


def _layer_sd_row(case_id: str, method: str, metrics: dict, mask: np.ndarray, report: list[dict], n_surf: int, vert_conn: np.ndarray) -> dict:
    def mean_for(group_type: str) -> float:
        values = [float(row["group_sd_mm"]) for row in report if row["group_type"] == group_type and np.isfinite(float(row["group_sd_mm"]))]
        return float(np.mean(values)) if values else float("nan")

    surface_components = connected_components(mask[:n_surf], vert_conn[:n_surf, :n_surf])
    return {
        "scenario": case_id.rsplit("_", 1)[0],
        "case_id": case_id,
        "method": method,
        "surface_sd_mm": mean_for("surface"),
        "deep_sd_mm": mean_for("deep"),
        "surface_active_count": int(mask[:n_surf].sum()),
        "deep_active_count": int(mask[n_surf:].sum()),
        "surface_component_count": len(surface_components),
        "deep_component_count": int(mask[n_surf:].sum()),
        "global_sd_mm": float(metrics["sd_mm"]),
        "dle_mm": float(metrics["dle_mm"]),
    }


def _mesh_resolution_row(case_id: str, truth: dict, vert_conn: np.ndarray) -> dict:
    positions = np.asarray(truth["src_vertices"], dtype=float) * 1000.0
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    surface_edges = np.array(np.nonzero(np.triu(np.asarray(vert_conn != 0)[:n_surf, :n_surf], 1))).T
    surface_lengths = np.linalg.norm(positions[surface_edges[:, 0]] - positions[surface_edges[:, 1]], axis=1) if surface_edges.size else np.array([])
    deep = positions[n_surf:]
    if deep.shape[0] > 1:
        dist = np.linalg.norm(deep[:, None, :] - deep[None, :, :], axis=2)
        dist[dist == 0] = np.nan
        deep_nn = np.nanmin(dist, axis=1)
    else:
        deep_nn = np.array([])
    oracle = []
    for group in _true_groups(truth):
        group = np.asarray(group, dtype=int).ravel()
        centroid = positions[group].mean(axis=0)
        pool = positions[n_surf:] if np.all(group >= n_surf) else positions[:n_surf]
        oracle.append(float(np.linalg.norm(pool - centroid, axis=1).min()))
    return {
        "scenario": case_id.rsplit("_", 1)[0],
        "case_id": case_id,
        "median_surface_edge_mm": float(np.nanmedian(surface_lengths)) if surface_lengths.size else float("nan"),
        "median_deep_nn_dist_mm": float(np.nanmedian(deep_nn)) if deep_nn.size else float("nan"),
        "oracle_nearest_grid_dist_mm": float(np.mean(oracle)) if oracle else float("nan"),
    }


def make_jobs() -> list[dict]:
    base = load_mat(DATA_ROOT / "deep_plus_two_surface" / "sub_EEG.mat")
    n_surf = int(np.asarray(base["n_surf"]).ravel()[0])
    vertices = np.asarray(base["src_vertices"], dtype=float)
    surface = vertices[:n_surf]
    deep = np.arange(n_surf, vertices.shape[0])
    surface_pick = np.linspace(0, n_surf - 1, 6, dtype=int)
    deep_pick = deep[[0, 3, 6, 9, 12]]
    return (
        [{"job": f"surface_{i:02d}", "surface": [int(s)], "deep": []} for i, s in enumerate(surface_pick)]
        + [{"job": f"deep_{i:02d}", "surface": [], "deep": [int(d)]} for i, d in enumerate(deep_pick)]
        + [
            {"job": f"mixed_{i:02d}", "surface": [int(surface_pick[i % surface_pick.size])], "deep": [int(deep_pick[i % deep_pick.size])]}
            for i in range(6)
        ]
        + [
            {
                "job": f"mixed2_{i:02d}",
                "surface": [int(surface_pick[i % surface_pick.size]), int(surface_pick[(i + 2) % surface_pick.size])],
                "deep": [int(deep_pick[(i + 1) % deep_pick.size])],
            }
            for i in range(4)
        ]
    )


def generate() -> None:
    eeg = load_mat(DATA_ROOT / "deep_plus_two_surface" / "sub_EEG.mat")
    meg = load_mat(DATA_ROOT / "deep_plus_two_surface" / "sub_MEG.mat")
    adjacency = np.asarray(eeg["VertConn"] != 0)
    times = np.asarray(eeg["times"], dtype=float).ravel()
    n_sources = np.asarray(eeg["Gain"]).shape[1]
    n_surf = int(np.asarray(eeg["n_surf"]).ravel()[0])
    rng = np.random.default_rng(20260707)
    jobs = make_jobs()
    VAL_DATA.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        out = VAL_DATA / job["job"]
        out.mkdir(parents=True, exist_ok=True)
        s_true = np.zeros((n_sources, times.size), dtype=float)
        surface_indices: list[int] = []
        labels: list[int] = []
        for label, center in enumerate(job["surface"], start=1):
            patch = _neighbors(adjacency[:n_surf, :n_surf], center, 24)
            wave = _wave(times, 11.0 + label)
            s_true[patch] = wave
            surface_indices.extend(patch.tolist())
            labels.extend([label] * len(patch))
        has_deep = bool(job["deep"])
        deep_idx = int(job["deep"][0]) if has_deep else n_surf
        if has_deep:
            s_true[deep_idx] = 0.9 * _wave(times, 17.0, phase=np.pi / 4)

        sub_eeg = dict(eeg)
        sub_meg = dict(meg)
        sub_eeg["F"] = np.asarray(eeg["Gain"], dtype=float) @ s_true + _noise_like(eeg["F"], rng)
        sub_meg["F"] = np.asarray(meg["Gain"], dtype=float) @ s_true + _noise_like(meg["F"], rng)
        for sub in (sub_eeg, sub_meg):
            sub["true_surface_indices0"] = np.array(surface_indices, dtype=np.int64)[None, :]
            sub["true_surface_indices1"] = sub["true_surface_indices0"] + 1
            sub["true_surface_patch_labels"] = np.array(labels, dtype=np.int64)[None, :]
            sub["true_surface_centers0"] = np.array(job["surface"], dtype=np.int64)[None, :]
            sub["true_surface_centers1"] = sub["true_surface_centers0"] + 1
            sub["true_deep_idx0"] = np.array([[deep_idx]], dtype=np.int64)
            sub["true_deep_idx1"] = np.array([[deep_idx + 1]], dtype=np.int64)
            sub["has_deep_source"] = np.array([[int(has_deep)]], dtype=np.int64)

        truth = {
            "s_true": s_true,
            "src_vertices": np.asarray(eeg["src_vertices"], dtype=float),
            "sfreq": eeg["sfreq"],
            "times": eeg["times"],
            "t_idx0": np.arange(NOISE_SAMPLES, times.size, dtype=np.int64)[None, :],
            "n_surf": eeg["n_surf"],
            "n_deep": eeg["n_deep"],
            "true_surface_indices0": sub_eeg["true_surface_indices0"],
            "true_surface_indices1": sub_eeg["true_surface_indices1"],
            "true_surface_patch_labels": sub_eeg["true_surface_patch_labels"],
            "true_surface_centers0": sub_eeg["true_surface_centers0"],
            "true_surface_centers1": sub_eeg["true_surface_centers1"],
            "true_deep_idx0": sub_eeg["true_deep_idx0"],
            "true_deep_idx1": sub_eeg["true_deep_idx1"],
            "has_deep_source": sub_eeg["has_deep_source"],
        }
        sio.savemat(out / "sub_EEG.mat", sub_eeg, do_compression=True)
        sio.savemat(out / "sub_MEG.mat", sub_meg, do_compression=True)
        sio.savemat(out / "s_true.mat", truth, do_compression=True)
    with (VAL_ROOT / "jobs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["job", "surface", "deep"])
        writer.writeheader()
        writer.writerows(jobs)


def summarize(*, include_v6: bool = True) -> None:
    rows = []
    group_rows = []
    layer_rows = []
    mesh_rows = []
    evidence_rows = []
    threshold_rows = []
    rng = np.random.default_rng(7)
    for job_dir in sorted(VAL_DATA.iterdir()):
        if not job_dir.is_dir():
            continue
        run_file = VAL_RUNS / job_dir.name / "s_wen.mat"
        if not run_file.exists():
            continue
        truth = load_mat(job_dir / "s_true.mat")
        eeg = load_mat(job_dir / "sub_EEG.mat")
        mesh_rows.append(_mesh_resolution_row(job_dir.name, truth, np.asarray(eeg["VertConn"], dtype=float)))
        n_sources = np.asarray(truth["s_true"]).shape[0]
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
        cortex = auc_cortex(
            np.asarray(truth["src_vertices"], dtype=float),
            np.asarray(eeg["VertConn"], dtype=float),
            n_surf,
        )
        sisses = load_source(run_file, n_sources)
        random_source = rng.normal(size=sisses.shape)
        methods = {
            "random_baseline": (random_source, threshold_mask(random_source, 0.10)),
            "SISSES_0.10": (sisses, threshold_mask(sisses, 0.10)),
            "Compact": (sisses, evidence_aware_compact_mask(sisses, np.asarray(eeg["VertConn"], dtype=float), n_surf)),
        }
        component_refit, component_mask = component_refit_select(sio.loadmat(job_dir / "sub_EEG.mat"), sio.loadmat(job_dir / "sub_MEG.mat"), sisses, np.asarray(eeg["VertConn"], dtype=float), n_surf)
        methods["ComponentRefit_v2"] = (component_refit, component_mask)
        component_v3, component_v3_mask, component_v3_candidate = component_refit_select_v3(
            sio.loadmat(job_dir / "sub_EEG.mat"),
            sio.loadmat(job_dir / "sub_MEG.mat"),
            sisses,
            np.asarray(eeg["VertConn"], dtype=float),
            n_surf,
            np.asarray(truth["src_vertices"], dtype=float),
            return_candidate=True,
        )
        methods["ComponentRefit_v3"] = (component_v3, component_v3_mask)
        component_v4, component_v4_mask, component_v4_candidate = component_refit_select_v4_deep_rescue(
            sio.loadmat(job_dir / "sub_EEG.mat"),
            sio.loadmat(job_dir / "sub_MEG.mat"),
            sisses,
            np.asarray(eeg["VertConn"], dtype=float),
            n_surf,
            np.asarray(truth["src_vertices"], dtype=float),
            return_candidate=True,
        )
        methods["ComponentRefit_v4"] = (component_v4, component_v4_mask)
        component_v5, component_v5_mask, component_v5_candidate = component_refit_select_v5_compact_deep_prior(
            sio.loadmat(job_dir / "sub_EEG.mat"),
            sio.loadmat(job_dir / "sub_MEG.mat"),
            sisses,
            np.asarray(eeg["VertConn"], dtype=float),
            n_surf,
            np.asarray(truth["src_vertices"], dtype=float),
            return_candidate=True,
        )
        methods["ComponentRefit_v5"] = (component_v5, component_v5_mask)
        component_v6 = {}
        if include_v6:
            component_v6 = component_refit_select_v6_sisses_refit(
                sio.loadmat(job_dir / "sub_EEG.mat"),
                sio.loadmat(job_dir / "sub_MEG.mat"),
                sisses,
                np.asarray(eeg["VertConn"], dtype=float),
                n_surf,
                np.asarray(truth["src_vertices"], dtype=float),
                return_candidate=True,
            )
            for variant, (source, mask, _candidate) in component_v6.items():
                methods[f"ComponentRefit_v6_{variant}"] = (source, mask)
        component_v7, component_v7_mask, component_v7_candidate = component_refit_select_v7_tbf_refit(
            sio.loadmat(job_dir / "sub_EEG.mat"),
            sio.loadmat(job_dir / "sub_MEG.mat"),
            sisses,
            np.asarray(eeg["VertConn"], dtype=float),
            n_surf,
            np.asarray(truth["src_vertices"], dtype=float),
            return_candidate=True,
        )
        methods["ComponentRefit_v7"] = (component_v7, component_v7_mask)
        component_v8 = {}
        component_v8_candidates = {}
        for name, factor in V8_FACTORS.items():
            source, mask, candidate = component_refit_select_v8_protected_sisses(
                sio.loadmat(job_dir / "sub_EEG.mat"),
                sio.loadmat(job_dir / "sub_MEG.mat"),
                sisses,
                np.asarray(eeg["VertConn"], dtype=float),
                n_surf,
                np.asarray(truth["src_vertices"], dtype=float),
                deep_protect_factor=factor,
                return_candidate=True,
            )
            method = f"ComponentRefit_v8_{name}"
            component_v8[method] = (source, mask)
            component_v8_candidates[method] = candidate
            methods[method] = (source, mask)
        component_v9 = {}
        component_v9_candidates = {}
        for name, factor in V9_FACTORS.items():
            source, mask, candidate = component_refit_select_v9_layerwise_sisses(
                sio.loadmat(job_dir / "sub_EEG.mat"),
                sio.loadmat(job_dir / "sub_MEG.mat"),
                sisses,
                np.asarray(eeg["VertConn"], dtype=float),
                n_surf,
                np.asarray(truth["src_vertices"], dtype=float),
                deep_protect_factor=factor,
                return_candidate=True,
            )
            method = f"ComponentRefit_v9_{name}"
            component_v9[method] = (source, mask)
            component_v9_candidates[method] = candidate
            methods[method] = (source, mask)
        component_v9b = {}
        component_v9b_candidates = {}
        for name, params in V9B_VARIANTS.items():
            source, mask, candidate = component_refit_select_v9_layerwise_sisses(
                sio.loadmat(job_dir / "sub_EEG.mat"),
                sio.loadmat(job_dir / "sub_MEG.mat"),
                sisses,
                np.asarray(eeg["VertConn"], dtype=float),
                n_surf,
                np.asarray(truth["src_vertices"], dtype=float),
                **params,
                return_candidate=True,
            )
            method = f"ComponentRefit_v9b_{name}"
            component_v9b[method] = (source, mask)
            component_v9b_candidates[method] = candidate
            methods[method] = (source, mask)
        for method, (source, mask) in methods.items():
            metrics = external_full_head_metrics(
                source * mask[:, None],
                np.asarray(truth["s_true"], dtype=float),
                np.asarray(truth["src_vertices"], dtype=float),
                true_groups=_true_groups(truth),
            )
            metrics["auc"] = an_auc_from_cortex(
                np.asarray(truth["s_true"], dtype=float), source * mask[:, None], cortex, _true_groups(truth)
            )
            candidate = component_v9b_candidates[method] if method in component_v9b_candidates else component_v9_candidates[method] if method in component_v9_candidates else component_v8_candidates[method] if method in component_v8_candidates else component_v7_candidate if method == "ComponentRefit_v7" else component_v5_candidate if method == "ComponentRefit_v5" else component_v4_candidate if method == "ComponentRefit_v4" else component_v3_candidate if method == "ComponentRefit_v3" else component_v6[method.removeprefix("ComponentRefit_v6_")][2] if method.startswith("ComponentRefit_v6_") and include_v6 else mask
            report = _group_report_rows(job_dir.name, method, source * mask[:, None], mask, candidate, truth, np.asarray(eeg["VertConn"], dtype=float))
            group_rows.extend(report)
            layer_row = _layer_sd_row(job_dir.name, method, metrics, mask, report, n_surf, np.asarray(eeg["VertConn"], dtype=float))
            layer_rows.append(layer_row)
            evidence_rows.extend(
                _evidence_peak_rows(
                    job_dir.name,
                    method,
                    source,
                    mask,
                    truth,
                    np.asarray(eeg["VertConn"], dtype=float),
                    sio.loadmat(job_dir / "sub_EEG.mat"),
                    sio.loadmat(job_dir / "sub_MEG.mat"),
                )
            )
            rows.append(
                {
                    "job": job_dir.name,
                    "method": method,
                    "active_count": int(mask.sum()),
                    "auc": metrics["auc"],
                    "rmse": metrics["rmse"],
                    "surface_sd_mm": layer_row["surface_sd_mm"],
                    "deep_sd_mm": layer_row["deep_sd_mm"],
                    "global_sd_reference_mm": metrics["sd_mm"],
                    "dle_mm": metrics["dle_mm"],
                }
            )
        sweep_sources = [("ComponentRefit_v4", component_v4), ("ComponentRefit_v5", component_v5), ("ComponentRefit_v7", component_v7)]
        if "ComponentRefit_v8_p025" in component_v8:
            sweep_sources.append(("ComponentRefit_v8_p025", component_v8["ComponentRefit_v8_p025"][0]))
        if "ComponentRefit_v9_p025" in component_v9:
            sweep_sources.append(("ComponentRefit_v9_p025", component_v9["ComponentRefit_v9_p025"][0]))
        if "ComponentRefit_v9b_balanced" in component_v9b:
            sweep_sources.append(("ComponentRefit_v9b_balanced", component_v9b["ComponentRefit_v9b_balanced"][0]))
        if include_v6:
            sweep_sources.append(("ComponentRefit_v6_mid", component_v6["mid"][0]))
        for method, source in sweep_sources:
            for rel in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90):
                mask = threshold_mask(source, rel)
                metrics = external_full_head_metrics(
                    source * mask[:, None],
                    np.asarray(truth["s_true"], dtype=float),
                    np.asarray(truth["src_vertices"], dtype=float),
                    true_groups=_true_groups(truth),
                )
                metrics["auc"] = an_auc_from_cortex(
                    np.asarray(truth["s_true"], dtype=float), source * mask[:, None], cortex, _true_groups(truth)
                )
                report = _group_report_rows(job_dir.name, method, source * mask[:, None], mask, mask, truth, np.asarray(eeg["VertConn"], dtype=float))
                threshold_rows.append(
                    {
                        "job": job_dir.name,
                        "method": method,
                        "rel": rel,
                        "active_count": int(mask.sum()),
                        "auc": metrics["auc"],
                        "rmse": metrics["rmse"],
                        "sd_mm": metrics["sd_mm"],
                        "dle_mm": metrics["dle_mm"],
                        "nearest_support_dist_mm": float(np.nanmean([row["nearest_support_dist_mm"] for row in report])),
                        "support_hit_true": float(np.mean([row["support_hit_true"] for row in report])),
                    }
                )
    with (VAL_ROOT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (VAL_ROOT / "per_group_localization_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    with (VAL_ROOT / "layerwise_sd_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    with (VAL_ROOT / "mesh_resolution_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mesh_rows[0]))
        writer.writeheader()
        writer.writerows(mesh_rows)
    with (VAL_ROOT / "evidence_peak_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence_rows[0]))
        writer.writeheader()
        writer.writerows(evidence_rows)
    scenario_rows = []
    for key, group in groupby(
        sorted(evidence_rows, key=lambda row: (row["scenario"], row["method"])),
        key=lambda row: (row["scenario"], row["method"]),
    ):
        items = list(group)
        scenario, method = key
        scenario_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "metric_dle_mm": float(np.nanmean([row["dle_mm"] for row in rows if row["method"] == method and row["job"].rsplit("_", 1)[0] == scenario])),
                "amplitude_global_peak_dle_mm": float(np.nanmean([row["amplitude_peak_dist_mm"] for row in items])),
                "evidence_global_peak_dle_mm": float(np.nanmean([row["evidence_peak_dist_mm"] for row in items])),
                "evidence_assigned_peak_dle_mm": float(np.nanmean([row["evidence_assigned_peak_dist_mm"] for row in items])),
            }
        )
    with (VAL_ROOT / "dle_by_scenario_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenario_rows[0]))
        writer.writeheader()
        writer.writerows(scenario_rows)
    with (VAL_ROOT / "threshold_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)


def summarize_v11(*, include_v14: bool = False) -> None:
    from SD import SD

    refined = load_refined_forward()
    rows = []
    for job_dir in sorted(VAL_DATA.iterdir()):
        run_file = VAL_RUNS / job_dir.name / "s_wen.mat"
        if not job_dir.is_dir() or not run_file.exists():
            continue
        truth = load_mat(job_dir / "s_true.mat")
        eeg = load_mat(job_dir / "sub_EEG.mat")
        meg = load_mat(job_dir / "sub_MEG.mat")
        true_source = np.asarray(truth["s_true"], dtype=float)
        vertices = np.asarray(truth["src_vertices"], dtype=float)
        groups = _true_groups(truth)
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
        vert_conn = np.asarray(eeg["VertConn"], dtype=float)
        sisses = load_source(run_file, true_source.shape[0])
        cortex = auc_cortex(vertices, vert_conn, n_surf)

        v9, v9_mask = component_refit_select_v9_layerwise_sisses(
            eeg,
            meg,
            sisses,
            vert_conn,
            n_surf,
            vertices,
            **V9B_VARIANTS["balanced"],
        )
        v11, range_mask, candidate = component_refit_select_v11_compactness_sisses(
            eeg,
            meg,
            sisses,
            vert_conn,
            n_surf,
            vertices,
            surface_compactness=0.10,
            sigma_surface_component=0.10,
            deep_compactness=0.50,
            return_candidate=True,
        )
        compact_mask = threshold_mask(v11, 0.50) & candidate
        compact_mask[n_surf:] = False
        compact = v11 * compact_mask[:, None]
        deep = refined_deep_evidence(eeg, meg, v11, vertices, n_surf, refined)
        if deep is not None:
            compact[int(deep["coarse_index"])] = deep["timecourse"]
            compact_mask[int(deep["coarse_index"])] = True

        methods = {
            "SISSES_0.10": (sisses, threshold_mask(sisses, 0.10)),
            "V9b_balanced": (v9, v9_mask),
            "V11_range": (v11, range_mask),
        }
        for method, (source, mask) in methods.items():
            used = source * mask[:, None]
            metrics = external_full_head_metrics(used, true_source, vertices, true_groups=groups)
            metrics["auc"] = an_auc_from_cortex(true_source, used, cortex, groups)
            centroid = _component_centroid_dle(used, mask, vertices, groups, n_surf, vert_conn)
            layerwise_dle = _layerwise_peak_dle(used, mask, vertices, groups, n_surf, vert_conn)
            report = _group_report_rows(job_dir.name, method, used, mask, mask, truth, vert_conn)
            surface_values = [r["group_sd_mm"] for r in report if r["group_type"] == "surface"]
            deep_values = [r["group_sd_mm"] for r in report if r["group_type"] == "deep"]
            rows.append(
                {
                    "case_id": job_dir.name,
                    "scenario": job_dir.name.rsplit("_", 1)[0],
                    "method": method,
                    "auc": metrics["auc"],
                    "rmse": metrics["rmse"],
                    "surface_sd_mm": float(np.mean(surface_values)) if surface_values else np.nan,
                    "deep_sd_mm": float(np.mean(deep_values)) if deep_values else np.nan,
                    "dle_mm": metrics["dle_mm"],
                    **layerwise_dle,
                    **centroid,
                    "active_count": int(mask.sum()),
                    "deep_grid": "coarse",
                    "eeg_drop": np.nan,
                    "meg_drop": np.nan,
                }
            )

        surface_groups = [group for group in groups if np.all(group < n_surf)]
        deep_groups = [group for group in groups if np.all(group >= n_surf)]
        compact_methods = [("V11_compact_refined", compact, compact_mask, V11_ROOT)]
        if include_v14:
            v14, v14_mask = component_refit_select_v14_local_evidence(
                eeg,
                meg,
                compact,
                compact_mask,
                vert_conn,
                n_surf,
            )
            compact_methods.append(("V14_local_evidence", v14, v14_mask, V14_ROOT))

        for method, compact_source, method_mask, result_root in compact_methods:
            predicted_vertices = vertices[:n_surf]
            predicted_source = compact_source[:n_surf]
            predicted_mask = method_mask[:n_surf]
            if deep is not None:
                predicted_vertices = np.vstack([predicted_vertices, deep["position"]])
                predicted_source = np.vstack([predicted_source, compact_source[int(deep["coarse_index"])]])
                predicted_mask = np.r_[predicted_mask, True]
            _sd_mm, dle_mm = refined_sd_dle(predicted_source, predicted_mask, predicted_vertices, vertices, groups)
            surface_sd = (
                float(SD(compact_source[:n_surf], np.vstack([vertices[group] for group in surface_groups]), vertices[:n_surf], 0.0) * 1000.0)
                if surface_groups
                else np.nan
            )
            deep_sd = (
                float(np.linalg.norm(deep["position"] - vertices[deep_groups[0]], axis=1).min() * 1000.0)
                if deep is not None and deep_groups
                else np.nan
            )
            mapped_metrics = external_full_head_metrics(compact_source, true_source, vertices, true_groups=groups)
            centroid = _component_centroid_dle(
                compact_source,
                method_mask,
                vertices,
                groups,
                n_surf,
                vert_conn,
                deep_position=deep["position"] if deep is not None else None,
            )
            layerwise_dle = _layerwise_peak_dle(
                compact_source,
                method_mask,
                vertices,
                groups,
                n_surf,
                vert_conn,
                deep_position=deep["position"] if deep is not None else None,
            )
            rows.append(
                {
                    "case_id": job_dir.name,
                    "scenario": job_dir.name.rsplit("_", 1)[0],
                    "method": method,
                    "auc": an_auc_from_cortex(true_source, compact_source, cortex, groups),
                    "rmse": mapped_metrics["rmse"],
                    "surface_sd_mm": surface_sd,
                    "deep_sd_mm": deep_sd,
                    "dle_mm": dle_mm,
                    **layerwise_dle,
                    **centroid,
                    "active_count": int(predicted_mask.sum()),
                    "deep_grid": deep["grid"] if deep is not None else "none",
                    "eeg_drop": deep["eeg_drop"] if deep is not None else np.nan,
                    "meg_drop": deep["meg_drop"] if deep is not None else np.nan,
                }
            )
            result_root.joinpath("sources").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                result_root / "sources" / f"{job_dir.name}.npz",
                S=compact_source,
                region_mask=method_mask,
                deep_position=np.asarray(deep["position"]) if deep is not None else np.empty((0, 3)),
            )
        print("V14" if include_v14 else "V11", job_dir.name, flush=True)

    summary_metrics = (
        "dle_mm",
        "surface_dle_mm",
        "deep_dle_mm",
        "surface_support_centroid_dle_mm",
        "deep_support_centroid_dle_mm",
        "surface_energy_centroid_dle_mm",
        "deep_energy_centroid_dle_mm",
        "surface_hit_rate_10mm",
        "deep_hit_rate_10mm",
    )
    summary_rows = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        for scenario in ("all", "deep", "surface", "mixed", "mixed2"):
            items = method_rows if scenario == "all" else [row for row in method_rows if row["scenario"] == scenario]
            if not items:
                continue
            summary = {"method": method, "scenario": scenario, "cases": len(items)}
            for metric in summary_metrics:
                values = [float(row[metric]) for row in items if np.isfinite(float(row[metric]))]
                summary[metric] = float(np.mean(values)) if values else np.nan
            summary_rows.append(summary)
    spacing_rows = [
        {"layer": "surface", "coarse_mm": 5.065213849225115, "refined_mm": float(refined["surface_spacing_mm"][0])},
        {"layer": "deep", "coarse_mm": 9.999999399353555, "refined_mm": float(refined["deep_spacing_mm"][0])},
    ]
    result_roots = (V14_ROOT,) if include_v14 else (V11_ROOT,)
    for result_root in result_roots:
        result_root.mkdir(parents=True, exist_ok=True)
        for filename, output_rows in (
            ("metrics.csv", rows),
            ("layerwise_dle_summary.csv", summary_rows),
            ("grid_resolution.csv", spacing_rows),
        ):
            with (result_root / filename).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
                writer.writeheader()
                writer.writerows(output_rows)


def plot_waveforms(limit: int | None = None) -> None:
    VAL_FIGS.mkdir(parents=True, exist_ok=True)
    for job_dir in sorted(VAL_DATA.iterdir())[:limit]:
        if not (VAL_RUNS / job_dir.name / "s_wen.mat").exists():
            continue
        truth = load_mat(job_dir / "s_true.mat")
        sisses = load_source(VAL_RUNS / job_dir.name / "s_wen.mat", np.asarray(truth["s_true"]).shape[0])
        eeg = load_mat(job_dir / "sub_EEG.mat")
        compact_mask = evidence_aware_compact_mask(sisses, np.asarray(eeg["VertConn"], dtype=float), int(np.asarray(truth["n_surf"]).ravel()[0]))
        compact = sisses * compact_mask[:, None]
        component, _ = component_refit_select_v9_layerwise_sisses(
            load_mat(job_dir / "sub_EEG.mat"),
            load_mat(job_dir / "sub_MEG.mat"),
            sisses,
            np.asarray(eeg["VertConn"], dtype=float),
            int(np.asarray(truth["n_surf"]).ravel()[0]),
            np.asarray(truth["src_vertices"], dtype=float),
            deep_protect_factor=0.25,
        )
        times = np.asarray(truth["times"], dtype=float).ravel()
        groups = _true_groups(truth)
        fig, axes = plt.subplots(len(groups), 1, figsize=(8.5, 2.3 * len(groups)), squeeze=False)
        for row, group in enumerate(groups):
            ax = axes[row, 0]
            true_wave = group_waveform(np.asarray(truth["s_true"], dtype=float), group)
            sis_idx, sis_wave = best_estimated_waveform(sisses, group)
            comp_idx, comp_wave = best_estimated_waveform(compact, group)
            refit_idx, refit_wave = best_estimated_waveform(component, group)
            for label, wave, color in (("truth", true_wave, "black"), (f"SISSES {sis_idx + 1}", sis_wave, "#0072b2"), (f"Compact {comp_idx + 1}", comp_wave, "#d55e00"), (f"V9 p025 {refit_idx + 1}", refit_wave, "#009e73")):
                scale = max(float(np.max(np.abs(wave), initial=0.0)), np.finfo(float).eps)
                ax.plot(times, wave / scale, label=label, color=color, linewidth=1.5)
            ax.axvline(times[NOISE_SAMPLES], color="0.75", linewidth=1)
            ax.set_ylim(-1.15, 1.15)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.legend(frameon=False, loc="upper right")
        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle(job_dir.name)
        fig.tight_layout()
        fig.savefig(VAL_FIGS / f"{job_dir.name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if command == "generate":
        generate()
    elif command == "summarize":
        summarize()
    elif command == "summarize_fast":
        summarize(include_v6=False)
    elif command == "waveforms":
        plot_waveforms()
    elif command == "v11":
        summarize_v11()
    elif command == "v14":
        summarize_v11(include_v14=True)
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
