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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.sisses_direct_utils import best_estimated_waveform, group_waveform, true_source_groups
from visualization.visualize_modality_comparison import dominant_truth_hemi, load_surface_context
from visualization.visualize_spatial_fused_cortical import (
    load_anatomical_context,
    render_surface_view,
    source_to_continuous_surface,
)
from protected_multilayer import DATA_ROOT, OUT_ROOT, SCENARIOS, load_mat, source_amplitude

FIG_ROOT = OUT_ROOT / "figures"


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


def normalize(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float).ravel()
    scale = float(np.max(np.abs(wave), initial=0.0))
    return wave / scale if scale > 0 else wave


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
    draw_localization()
    draw_waveforms()
    print("Saved:", FIG_ROOT)


if __name__ == "__main__":
    main()
