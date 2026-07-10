from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.external_metrics import external_full_head_metrics
from pipelines.sisses_direct_utils import best_estimated_waveform, group_waveform
from protected_multilayer import (
    DATA_ROOT,
    OUT_ROOT,
    V8_FACTORS,
    V9_FACTORS,
    component_refit_select,
    component_refit_select_v3,
    component_refit_select_v4_deep_rescue,
    component_refit_select_v5_compact_deep_prior,
    component_refit_select_v6_sisses_refit,
    component_refit_select_v7_tbf_refit,
    component_refit_select_v8_protected_sisses,
    component_refit_select_v9_layerwise_sisses,
    connected_components,
    evidence_aware_compact_mask,
    load_mat,
    load_source,
    source_amplitude,
    threshold_mask,
)

VAL_ROOT = OUT_ROOT / "position_validation"
VAL_DATA = VAL_ROOT / "generated"
VAL_RUNS = VAL_ROOT / "sisses_runs"
VAL_FIGS = VAL_ROOT / "waveforms"
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
        n_sources = np.asarray(truth["s_true"]).shape[0]
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
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
        for method, (source, mask) in methods.items():
            metrics = external_full_head_metrics(
                source * mask[:, None],
                np.asarray(truth["s_true"], dtype=float),
                np.asarray(truth["src_vertices"], dtype=float),
                true_groups=_true_groups(truth),
            )
            rows.append(
                {
                    "job": job_dir.name,
                    "method": method,
                    "active_count": int(mask.sum()),
                    "auc": metrics["auc"],
                    "rmse": metrics["rmse"],
                    "sd_mm": metrics["sd_mm"],
                    "dle_mm": metrics["dle_mm"],
                }
            )
            candidate = component_v9_candidates[method] if method in component_v9_candidates else component_v8_candidates[method] if method in component_v8_candidates else component_v7_candidate if method == "ComponentRefit_v7" else component_v5_candidate if method == "ComponentRefit_v5" else component_v4_candidate if method == "ComponentRefit_v4" else component_v3_candidate if method == "ComponentRefit_v3" else component_v6[method.removeprefix("ComponentRefit_v6_")][2] if method.startswith("ComponentRefit_v6_") and include_v6 else mask
            report = _group_report_rows(job_dir.name, method, source * mask[:, None], mask, candidate, truth, np.asarray(eeg["VertConn"], dtype=float))
            group_rows.extend(report)
            layer_rows.append(_layer_sd_row(job_dir.name, method, metrics, mask, report, n_surf, np.asarray(eeg["VertConn"], dtype=float)))
        sweep_sources = [("ComponentRefit_v4", component_v4), ("ComponentRefit_v5", component_v5), ("ComponentRefit_v7", component_v7)]
        if "ComponentRefit_v8_p025" in component_v8:
            sweep_sources.append(("ComponentRefit_v8_p025", component_v8["ComponentRefit_v8_p025"][0]))
        if "ComponentRefit_v9_p025" in component_v9:
            sweep_sources.append(("ComponentRefit_v9_p025", component_v9["ComponentRefit_v9_p025"][0]))
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
    with (VAL_ROOT / "threshold_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)


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
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
