from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.sisses_direct_utils import best_estimated_waveform, group_waveform, true_source_groups
from visualization.visualize_modality_comparison import dominant_truth_hemi, load_surface_context
from visualization.visualize_spatial_fused_cortical import (
    ACTIVATION_CMAP,
    load_anatomical_context,
    render_surface_view,
    source_to_continuous_surface,
)
from visualization.visualize_spatial_fused_mri import draw_slice, plane_image, pos_to_vox
from protected_multilayer import DATA_ROOT, OUT_ROOT, SCENARIOS, load_mat, load_source, source_amplitude, threshold_mask
from validation_batch import VAL_DATA, VAL_RUNS, V11_ROOT, V14_ROOT

FIG_ROOT = OUT_ROOT / "figures"
V11_CASES = ("deep_00", "surface_04", "mixed_04", "mixed2_02")
V11_DEEP_CASES = ("deep_00", "mixed_04", "mixed2_02")
V14_CASES = ("surface_00", "mixed_02", "mixed2_00")


def load_npz_source(path: Path) -> np.ndarray:
    result = np.load(path, allow_pickle=True)
    source = np.asarray(result["S"], dtype=float)
    mask = np.asarray(result["region_mask"], dtype=bool).ravel()
    return source * mask[:, None]


def rows_for(scenario: str, truth: dict) -> list[tuple[str, np.ndarray]]:
    base = OUT_ROOT / "protected" / scenario
    return [
        ("Ground truth", np.asarray(truth["s_true"], dtype=float)),
        ("SISSES EEG+MEG", load_npz_source(base / "sisses_eeg_meg_direct.npz")),
        ("Adaptive SISSES", load_npz_source(base / "sisses_adaptive_threshold.npz")),
        ("Compact SISSES", load_npz_source(base / "sisses_evidence_compact.npz")),
        ("Component Refit v2", load_npz_source(base / "sisses_component_refit_v2.npz")),
        ("Component Refit v3", load_npz_source(base / "sisses_component_refit_v3_localize.npz")),
        ("Component Refit v4", load_npz_source(base / "sisses_component_refit_v4_deep_rescue.npz")),
        ("Component Refit v5", load_npz_source(base / "sisses_component_refit_v5_compact_deep_prior.npz")),
        ("Component Refit v6 mid", load_npz_source(base / "sisses_component_refit_v6_sisses_refit_mid.npz")),
        ("Component Refit v7 TBF", load_npz_source(base / "sisses_component_refit_v7_tbf_refit.npz")),
        ("Protected SISSES v8", load_npz_source(base / "sisses_component_refit_v8_protected_sisses_p025.npz")),
        ("Layer-wise SISSES v9", load_npz_source(base / "sisses_component_refit_v9_layerwise_sisses_p025.npz")),
        ("Layer-wise SISSES v9b", load_npz_source(base / "sisses_component_refit_v9b_layerwise_sisses_balanced.npz")),
        ("Weighted SISSES", load_npz_source(base / "sisses_weighted_multilayer.npz")),
        ("Protected SISSES", load_npz_source(base / "sisses_protected_multilayer.npz")),
        ("SISSES MEG-only", load_npz_source(base / "sisses_meg_only.npz")),
        ("SISSES EEG-only", load_npz_source(base / "sisses_eeg_only.npz")),
    ]


def render_deep_activation(truth: dict, source: np.ndarray, anatomical: dict) -> np.ndarray:
    positions = mne.transforms.apply_trans(
        anatomical["trans"],
        np.asarray(truth["src_vertices"], dtype=float),
    ) * 1000.0
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    amp = source_amplitude(source)
    deep_amp = amp[n_surf:]
    keep = deep_amp > 0
    points = positions[n_surf:][keep]
    weights = deep_amp[keep]
    if weights.size:
        weights = weights / max(float(weights.max()), np.finfo(float).eps)

    plotter = pv.Plotter(off_screen=True, window_size=(760, 590))
    plotter.set_background("black")
    plotter.enable_anti_aliasing("ssaa")
    cortex_vertices = []
    for hemi in ("lh", "rh"):
        info = anatomical[hemi]
        cortex_vertices.append(info["vertices"])
        plotter.add_mesh(
            pv.PolyData(info["vertices"], info["faces"]),
            color="#d8d8d8",
            opacity=0.12,
            smooth_shading=True,
            show_scalar_bar=False,
        )
    if points.size:
        w = weights / max(float(weights.sum()), np.finfo(float).eps)
        center = np.average(points, axis=0, weights=w)
        centered = points - center
        cov = (centered * w[:, None]).T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        radii = np.clip(np.sqrt(np.maximum(eigvals, 0.0)) * 2.2 + 3.0, 3.5, 11.0)
        sphere = pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=24)
        sphere.points = center + (sphere.points * radii) @ eigvecs.T
        plotter.add_mesh(
            sphere,
            color="#00bfff",
            opacity=0.82,
            smooth_shading=True,
            show_scalar_bar=False,
        )
    cortex = np.vstack(cortex_vertices)
    center = cortex.mean(axis=0)
    span = float(np.ptp(cortex, axis=0).max())
    plotter.camera_position = [
        center + np.array([1.7 * span, -2.0 * span, 1.0 * span]),
        center,
        [0.0, 0.0, 1.0],
    ]
    plotter.camera.zoom(1.18)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def render_surface_medial(
    lh_values: np.ndarray,
    rh_values: np.ndarray,
    context: dict,
    hemi: str,
) -> np.ndarray:
    info = context[hemi]
    activation = lh_values if hemi == "lh" else rh_values
    mesh = pv.PolyData(info["vertices"], info["face_array"])
    mesh.point_data["curvature"] = info["curvature"]

    plotter = pv.Plotter(off_screen=True, window_size=(760, 590))
    plotter.set_background("black")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        mesh,
        scalars="curvature",
        cmap=["#666666", "#dedede"],
        clim=(-1, 1),
        smooth_shading=True,
        show_scalar_bar=False,
    )
    overlay = mesh.copy()
    overlay.point_data["activation"] = activation
    plotter.add_mesh(
        overlay,
        scalars="activation",
        cmap=ACTIVATION_CMAP,
        clim=(0, 1),
        opacity=np.where(activation >= 0.035, 0.32 + 0.68 * activation, 0.0),
        smooth_shading=True,
        show_scalar_bar=False,
    )
    center = info["vertices"].mean(axis=0)
    span = float(np.ptp(info["vertices"], axis=0).max())
    direction = -1.0 if hemi == "rh" else 1.0
    plotter.camera_position = [center + np.array([direction * 2.7 * span, 0.0, 0.0]), center, [0.0, 0.0, 1.0]]
    plotter.camera.zoom(1.17)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def draw_localization() -> None:
    pv.global_theme.allow_empty_mesh = True
    surface_context = load_surface_context()
    anatomical_context = load_anatomical_context()
    out_dir = FIG_ROOT / "localization"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        print("Rendering localization", scenario)
        truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
        lateral_hemi = dominant_truth_hemi(truth, surface_context)
        rows = rows_for(scenario, truth)
        fig, axes = plt.subplots(len(rows), 3, figsize=(14.8, 3.6 * len(rows)), facecolor="white")
        for row_index, (label, source) in enumerate(rows):
            lh, rh = source_to_continuous_surface(source, surface_context, smoothing_steps=5)
            images = (
                render_surface_view(lh, rh, surface_context, "ventral", lateral_hemi),
                render_surface_view(lh, rh, surface_context, "lateral", lateral_hemi),
                render_deep_activation(truth, source, anatomical_context),
            )
            for col, image in enumerate(images):
                axes[row_index, col].imshow(image)
                axes[row_index, col].set_axis_off()
            axes[row_index, 0].text(
                -0.03, 0.5, label, rotation=90, va="center", ha="right",
                transform=axes[row_index, 0].transAxes, fontsize=12, fontweight="bold"
            )
        axes[0, 0].set_title("Inflated ventral", fontsize=13)
        axes[0, 1].set_title(f"Inflated {lateral_hemi.upper()} lateral", fontsize=13)
        axes[0, 2].set_title("Deep activation", fontsize=13)
        fig.suptitle(f"SISSES protected multilayer: {scenario}", fontsize=15, y=0.995)
        fig.subplots_adjust(left=0.065, right=0.995, top=0.965, bottom=0.005, wspace=0.02, hspace=0.035)
        fig.savefig(out_dir / f"{scenario}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def draw_v11_localization() -> None:
    pv.global_theme.allow_empty_mesh = True
    surface_context = load_surface_context()
    anatomical_context = load_anatomical_context()
    for case_id in V11_CASES:
        truth = load_mat(VAL_DATA / case_id / "s_true.mat")
        n_sources = np.asarray(truth["s_true"]).shape[0]
        sisses = load_source(VAL_RUNS / case_id / "s_wen.mat", n_sources)
        sisses *= threshold_mask(sisses, 0.10)[:, None]
        with np.load(V11_ROOT / "sources" / f"{case_id}.npz") as saved:
            v11 = np.asarray(saved["S"], dtype=float) * np.asarray(saved["region_mask"], dtype=bool)[:, None]
        rows = (("Ground truth", np.asarray(truth["s_true"], dtype=float)), ("SISSES", sisses), ("V11 compact + refined", v11))
        lateral_hemi = dominant_truth_hemi(truth, surface_context)
        fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.5), facecolor="white")
        for row_index, (label, source) in enumerate(rows):
            lh, rh = source_to_continuous_surface(source, surface_context, smoothing_steps=5)
            images = (
                render_surface_view(lh, rh, surface_context, "ventral", lateral_hemi),
                render_surface_view(lh, rh, surface_context, "lateral", lateral_hemi),
                render_deep_activation(truth, source, anatomical_context),
            )
            for col, image in enumerate(images):
                axes[row_index, col].imshow(image)
                axes[row_index, col].set_axis_off()
            axes[row_index, 0].text(
                -0.03,
                0.5,
                label,
                rotation=90,
                va="center",
                ha="right",
                transform=axes[row_index, 0].transAxes,
                fontsize=12,
                fontweight="bold",
            )
        axes[0, 0].set_title("Inflated ventral", fontsize=13)
        axes[0, 1].set_title(f"Inflated {lateral_hemi.upper()} lateral", fontsize=13)
        axes[0, 2].set_title("Deep activation", fontsize=13)
        fig.suptitle(f"V11 compactness + refined grid: {case_id}", fontsize=15, y=0.995)
        fig.subplots_adjust(left=0.085, right=0.995, top=0.91, bottom=0.01, wspace=0.02, hspace=0.035)
        fig.savefig(V11_ROOT / f"localization_{case_id}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def draw_v11_deep_multiview() -> None:
    data_path = mne.datasets.sample.data_path()
    t1 = nib.load(str(data_path / "subjects" / "sample" / "mri" / "T1.mgz"))
    volume = np.asarray(t1.get_fdata(), dtype=float)
    vox2ras_tkr = t1.header.get_vox2ras_tkr()
    head_to_mri = mne.read_trans(data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif")["trans"]
    anatomical = load_anatomical_context()
    cortex_tree = cKDTree(np.vstack([anatomical["lh"]["vertices"], anatomical["rh"]["vertices"]]))
    planes = ("coronal", "sagittal", "axial")

    for case_id in V11_DEEP_CASES:
        truth = load_mat(VAL_DATA / case_id / "s_true.mat")
        vertices = np.asarray(truth["src_vertices"], dtype=float)
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
        deep_index = int(np.asarray(truth["true_deep_idx0"]).ravel()[0])
        true_mri = mne.transforms.apply_trans(head_to_mri, vertices[deep_index])
        center_voxel = pos_to_vox(true_mri, vox2ras_tkr)
        sisses = load_source(VAL_RUNS / case_id / "s_wen.mat", vertices.shape[0])
        sisses *= threshold_mask(sisses, 0.10)[:, None]
        with np.load(V11_ROOT / "sources" / f"{case_id}.npz") as saved:
            v11 = np.asarray(saved["S"], dtype=float) * np.asarray(saved["region_mask"], dtype=bool)[:, None]
            refined_deep = np.asarray(saved["deep_position"], dtype=float).reshape(-1, 3)
        rows = (
            ("Ground truth", np.asarray(truth["s_true"], dtype=float), None),
            ("SISSES", sisses, None),
            ("V11 compact + refined", v11, refined_deep),
        )
        fig, axes = plt.subplots(3, 3, figsize=(12.3, 9.6), dpi=170, facecolor="black", squeeze=False)
        for row, (label, source, position_override) in enumerate(rows):
            amplitude = source_amplitude(source)[n_surf:]
            active = np.flatnonzero(amplitude > 0)
            points = vertices[n_surf:][active]
            weights = amplitude[active]
            if position_override is not None and position_override.shape[0] == 1 and active.size:
                points = position_override
                weights = np.array([float(weights.max())])
            points_mri = mne.transforms.apply_trans(head_to_mri, points) if points.size else np.empty((0, 3))
            source_voxels = np.vstack([pos_to_vox(point, vox2ras_tkr) for point in points_mri]) if points_mri.size else np.empty((0, 3))
            if weights.size:
                weights = weights / max(float(weights.max()), np.finfo(float).eps)
                center_mm = np.average(points_mri * 1000.0, axis=0, weights=weights)
                depth_mm = float(cortex_tree.query(center_mm)[0])
                depth_label = f"depth to cortex {depth_mm:.1f} mm"
            else:
                depth_label = "no deep estimate"
            for column, plane in enumerate(planes):
                ax = axes[row, column]
                draw_slice(ax, volume, plane, center_voxel, source_voxels, weights)
                _image, _axis, _slice, point_2d = plane_image(volume, plane, center_voxel)
                x, y = point_2d(center_voxel)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="#ffb000",
                    markersize=12,
                    markeredgewidth=2.0,
                )
                if row == 0:
                    ax.set_title(plane.capitalize(), color="white", fontsize=12)
            axes[row, 0].text(
                -0.04,
                0.5,
                f"{label}\n{depth_label}",
                rotation=90,
                va="center",
                ha="right",
                transform=axes[row, 0].transAxes,
                color="white",
                fontsize=10.5,
                fontweight="bold",
            )
        fig.suptitle(
            f"Deep-source orthogonal MRI views: {case_id}\norange ring = true deep center, cyan = source estimate",
            color="white",
            fontsize=13,
            y=0.985,
        )
        fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.01, wspace=0.02, hspace=0.04)
        fig.savefig(V11_ROOT / f"deep_multiview_{case_id}.png", facecolor="black", bbox_inches="tight")
        plt.close(fig)


def normalize(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float).ravel()
    scale = float(np.max(np.abs(wave), initial=0.0))
    return wave / scale if scale > 0 else wave


def draw_v11_waveforms() -> None:
    for case_id in V11_CASES:
        truth = load_mat(VAL_DATA / case_id / "s_true.mat")
        true_source = np.asarray(truth["s_true"], dtype=float)
        sisses = load_source(VAL_RUNS / case_id / "s_wen.mat", true_source.shape[0])
        with np.load(V11_ROOT / "sources" / f"{case_id}.npz") as saved:
            v11 = np.asarray(saved["S"], dtype=float) * np.asarray(saved["region_mask"], dtype=bool)[:, None]
        groups = true_source_groups(truth)
        times = np.asarray(truth["times"], dtype=float).ravel()
        fig, axes = plt.subplots(len(groups), 1, figsize=(8.4, 2.35 * len(groups)), squeeze=False)
        for row, group in enumerate(groups):
            truth_wave = group_waveform(true_source, group)
            sisses_index, sisses_wave = best_estimated_waveform(sisses, group)
            v11_index, v11_wave = best_estimated_waveform(v11, group)
            ax = axes[row, 0]
            ax.plot(times, normalize(truth_wave), color="black", linewidth=2.0, label="truth")
            ax.plot(times, normalize(sisses_wave), color="#0072b2", linewidth=1.3, label=f"SISSES {sisses_index + 1}")
            ax.plot(times, normalize(v11_wave), color="#d55e00", linewidth=1.5, label=f"V11 {v11_index + 1}")
            ax.axvline(times[min(200, times.size - 1)], color="0.75", linewidth=1.0)
            ax.set_ylim(-1.15, 1.15)
            ax.set_ylabel(f"group {row + 1}")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.legend(frameon=False, loc="upper right")
        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle(f"Source waveform comparison: {case_id}", fontsize=13)
        fig.tight_layout()
        fig.savefig(V11_ROOT / f"waveforms_{case_id}.png", dpi=180)
        plt.close(fig)


def draw_v14_localization() -> None:
    pv.global_theme.allow_empty_mesh = True
    surface_context = load_surface_context()
    anatomical_context = load_anatomical_context()
    for case_id in V14_CASES:
        truth = load_mat(VAL_DATA / case_id / "s_true.mat")
        with np.load(V11_ROOT / "sources" / f"{case_id}.npz") as saved:
            v11 = np.asarray(saved["S"], dtype=float) * np.asarray(saved["region_mask"], dtype=bool)[:, None]
        with np.load(V14_ROOT / "sources" / f"{case_id}.npz") as saved:
            v14 = np.asarray(saved["S"], dtype=float) * np.asarray(saved["region_mask"], dtype=bool)[:, None]
        rows = (
            ("Ground truth", np.asarray(truth["s_true"], dtype=float)),
            ("V11 compact + refined", v11),
            ("V14 local evidence", v14),
        )
        lateral_hemi = dominant_truth_hemi(truth, surface_context)
        fig, axes = plt.subplots(3, 4, figsize=(18.5, 10.5), facecolor="white")
        for row, (label, source) in enumerate(rows):
            lh, rh = source_to_continuous_surface(source, surface_context, smoothing_steps=5)
            images = (
                render_surface_view(lh, rh, surface_context, "ventral", lateral_hemi),
                render_surface_view(lh, rh, surface_context, "lateral", lateral_hemi),
                render_surface_medial(lh, rh, surface_context, lateral_hemi),
                render_deep_activation(truth, source, anatomical_context),
            )
            for column, image in enumerate(images):
                axes[row, column].imshow(image)
                axes[row, column].set_axis_off()
            axes[row, 0].text(
                -0.03,
                0.5,
                label,
                rotation=90,
                va="center",
                ha="right",
                transform=axes[row, 0].transAxes,
                fontsize=12,
                fontweight="bold",
            )
        axes[0, 0].set_title("Inflated ventral", fontsize=13)
        axes[0, 1].set_title(f"Inflated {lateral_hemi.upper()} lateral", fontsize=13)
        axes[0, 2].set_title(f"Inflated {lateral_hemi.upper()} medial", fontsize=13)
        axes[0, 3].set_title("Deep activation", fontsize=13)
        fig.suptitle(f"V14 residual-guided surface localization: {case_id}", fontsize=15, y=0.995)
        fig.subplots_adjust(left=0.085, right=0.995, top=0.91, bottom=0.01, wspace=0.02, hspace=0.035)
        fig.savefig(V14_ROOT / f"localization_{case_id}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def draw_v14_waveforms() -> None:
    for case_id in V14_CASES:
        truth = load_mat(VAL_DATA / case_id / "s_true.mat")
        true_source = np.asarray(truth["s_true"], dtype=float)
        with np.load(V11_ROOT / "sources" / f"{case_id}.npz") as saved:
            v11 = np.asarray(saved["S"], dtype=float) * np.asarray(saved["region_mask"], dtype=bool)[:, None]
        with np.load(V14_ROOT / "sources" / f"{case_id}.npz") as saved:
            v14 = np.asarray(saved["S"], dtype=float) * np.asarray(saved["region_mask"], dtype=bool)[:, None]
        groups = true_source_groups(truth)
        times = np.asarray(truth["times"], dtype=float).ravel()
        fig, axes = plt.subplots(len(groups), 1, figsize=(8.4, 2.35 * len(groups)), squeeze=False)
        for row, group in enumerate(groups):
            truth_wave = group_waveform(true_source, group)
            v11_index, v11_wave = best_estimated_waveform(v11, group)
            v14_index, v14_wave = best_estimated_waveform(v14, group)
            ax = axes[row, 0]
            ax.plot(times, normalize(truth_wave), color="black", linewidth=2.0, label="truth")
            ax.plot(times, normalize(v11_wave), color="#0072b2", linewidth=1.3, label=f"V11 {v11_index + 1}")
            ax.plot(times, normalize(v14_wave), color="#d55e00", linewidth=1.5, linestyle="--", label=f"V14 {v14_index + 1}")
            ax.axvline(times[min(200, times.size - 1)], color="0.75", linewidth=1.0)
            ax.set_ylim(-1.15, 1.15)
            ax.set_ylabel(f"group {row + 1}")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.legend(frameon=False, loc="upper right")
        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle(f"V14 source waveform comparison: {case_id}", fontsize=13)
        fig.tight_layout()
        fig.savefig(V14_ROOT / f"waveforms_{case_id}.png", dpi=180)
        plt.close(fig)


def draw_waveforms() -> None:
    out_dir = FIG_ROOT / "waveforms"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        print("Rendering waveforms", scenario)
        truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
        times = np.asarray(truth["times"], dtype=float).ravel()
        true_source = np.asarray(truth["s_true"], dtype=float)
        direct = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_eeg_meg_direct.npz")
        protected = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_protected_multilayer.npz")
        component = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v2.npz")
        component_v3 = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v3_localize.npz")
        component_v4 = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v4_deep_rescue.npz")
        component_v5 = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v5_compact_deep_prior.npz")
        component_v6 = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v6_sisses_refit_mid.npz")
        component_v7 = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v7_tbf_refit.npz")
        component_v8 = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v8_protected_sisses_p025.npz")
        component_v9 = load_npz_source(OUT_ROOT / "protected" / scenario / "sisses_component_refit_v9_layerwise_sisses_p025.npz")
        groups = true_source_groups(truth)
        fig, axes = plt.subplots(len(groups), 1, figsize=(8.5, 2.5 * len(groups)), squeeze=False)
        for row, group in enumerate(groups):
            ax = axes[row, 0]
            truth_wave = group_waveform(true_source, group)
            direct_peak, direct_wave = best_estimated_waveform(direct, group)
            protected_peak, protected_wave = best_estimated_waveform(protected, group)
            component_peak, component_wave = best_estimated_waveform(component, group)
            component_v3_peak, component_v3_wave = best_estimated_waveform(component_v3, group)
            component_v4_peak, component_v4_wave = best_estimated_waveform(component_v4, group)
            component_v5_peak, component_v5_wave = best_estimated_waveform(component_v5, group)
            component_v6_peak, component_v6_wave = best_estimated_waveform(component_v6, group)
            component_v7_peak, component_v7_wave = best_estimated_waveform(component_v7, group)
            component_v8_peak, component_v8_wave = best_estimated_waveform(component_v8, group)
            component_v9_peak, component_v9_wave = best_estimated_waveform(component_v9, group)
            ax.plot(times, normalize(truth_wave), color="black", linewidth=2.0, label="truth")
            ax.plot(times, normalize(direct_wave), color="#0072b2", linewidth=1.4, label=f"SISSES {direct_peak + 1}")
            ax.plot(times, normalize(protected_wave), color="#d55e00", linewidth=1.4, label=f"protected {protected_peak + 1}")
            ax.plot(times, normalize(component_wave), color="#009e73", linewidth=1.2, label=f"refit v2 {component_peak + 1}")
            ax.plot(times, normalize(component_v3_wave), color="#cc79a7", linewidth=1.4, label=f"refit v3 {component_v3_peak + 1}")
            ax.plot(times, normalize(component_v4_wave), color="#56b4e9", linewidth=1.4, label=f"refit v4 {component_v4_peak + 1}")
            ax.plot(times, normalize(component_v5_wave), color="#000000", linewidth=1.0, linestyle="--", label=f"refit v5 {component_v5_peak + 1}")
            ax.plot(times, normalize(component_v6_wave), color="#999999", linewidth=1.0, linestyle=":", label=f"refit v6 {component_v6_peak + 1}")
            ax.plot(times, normalize(component_v7_wave), color="#882255", linewidth=1.2, linestyle="-.", label=f"refit v7 {component_v7_peak + 1}")
            ax.plot(times, normalize(component_v8_wave), color="#117733", linewidth=1.3, linestyle="-", label=f"v8 {component_v8_peak + 1}")
            ax.plot(times, normalize(component_v9_wave), color="#332288", linewidth=1.3, linestyle="--", label=f"v9 {component_v9_peak + 1}")
            ax.axvline(times[min(200, len(times) - 1)], color="0.75", linewidth=1.0)
            ax.set_ylim(-1.15, 1.15)
            ax.set_ylabel(f"group {row + 1}")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.legend(loc="upper right", frameon=False)
        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle(f"SISSES waveform comparison: {scenario}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scenario}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "v14":
        draw_v14_localization()
        draw_v14_waveforms()
        print("Saved:", V14_ROOT)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "v11":
        draw_v11_localization()
        draw_v11_deep_multiview()
        draw_v11_waveforms()
        print("Saved:", V11_ROOT)
        return
    draw_localization()
    draw_waveforms()
    print("Saved:", FIG_ROOT)


if __name__ == "__main__":
    main()
