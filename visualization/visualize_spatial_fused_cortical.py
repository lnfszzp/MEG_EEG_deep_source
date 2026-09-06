from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne
import nibabel as nib
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.run_whole_brain_fusion import DATA_ROOT, load_mat, source_amplitude
from visualization.visualize_modality_comparison import (
    diffuse,
    dominant_truth_hemi,
    load_surface_context,
)


RESULT_ROOT = ROOT / "results" / "latest" / "spatial_fused_fusion"
OUT_ROOT = ROOT / "results" / "latest" / "figures" / "spatial_fused_cortical"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")
ACTIVATION_CMAP = LinearSegmentedColormap.from_list(
    "source_activation",
    ["#003cff", "#006dff", "#00c8ff", "#20fff0"],
)
DEEP_KERNEL_SIGMA_MM = 3.0
DEEP_CONTOUR_LEVEL = 0.35
DEEP_KEEP_FRACTION = 0.25


def _face_array(faces: np.ndarray) -> np.ndarray:
    return np.c_[np.full(len(faces), 3), faces].ravel()


def source_to_continuous_surface(
    source: np.ndarray,
    context: dict,
    *,
    smoothing_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate discrete source vertices into an MNE-style surface field."""
    amplitude = source_amplitude(source)
    n_lh = context["n_lh_sources"]
    fields = []
    offset = 0
    for hemi in ("lh", "rh"):
        info = context[hemi]
        source_vertices = info["source_vertices"]
        count = len(source_vertices)
        values = np.zeros(len(info["vertices"]), dtype=float)
        values[source_vertices] = amplitude[offset : offset + count]
        values = diffuse(
            values,
            info["adjacency"],
            iterations=smoothing_steps,
        )
        fields.append(np.power(values, 0.35))
        offset += count
    return fields[0], fields[1]


def render_surface_view(
    lh_values: np.ndarray,
    rh_values: np.ndarray,
    context: dict,
    view: str,
    lateral_hemi: str,
) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=(760, 590))
    plotter.set_background("black")
    plotter.enable_anti_aliasing("ssaa")
    hemis = ("lh", "rh") if view == "ventral" else (lateral_hemi,)
    for hemi in hemis:
        info = context[hemi]
        mesh = pv.PolyData(info["vertices"], info["face_array"])
        mesh.point_data["curvature"] = info["curvature"]
        plotter.add_mesh(
            mesh,
            scalars="curvature",
            cmap=["#666666", "#dedede"],
            clim=(-1, 1),
            smooth_shading=True,
            show_scalar_bar=False,
        )
        activation = lh_values if hemi == "lh" else rh_values
        overlay = mesh.copy()
        overlay.point_data["activation"] = activation
        alpha = np.where(activation >= 0.035, 0.32 + 0.68 * activation, 0.0)
        plotter.add_mesh(
            overlay,
            scalars="activation",
            cmap=ACTIVATION_CMAP,
            clim=(0, 1),
            opacity=alpha,
            smooth_shading=True,
            show_scalar_bar=False,
        )

    all_vertices = np.vstack([context[hemi]["vertices"] for hemi in hemis])
    center = all_vertices.mean(axis=0)
    span = float(np.ptp(all_vertices, axis=0).max())
    if view == "ventral":
        position = center + np.array([0.0, 0.0, -2.7 * span])
        up = [0.0, 1.0, 0.0]
    else:
        direction = 1.0 if lateral_hemi == "rh" else -1.0
        position = center + np.array([direction * 2.7 * span, 0.0, 0.0])
        up = [0.0, 0.0, 1.0]
    plotter.camera_position = [position, center, up]
    plotter.camera.zoom(1.17)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def load_anatomical_context() -> dict:
    data_path = mne.datasets.sample.data_path()
    surf_dir = data_path / "subjects" / "sample" / "surf"
    context = {
        "trans": mne.read_trans(
            data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
        )["trans"],
    }
    for hemi in ("lh", "rh"):
        vertices, faces = nib.freesurfer.read_geometry(
            str(surf_dir / f"{hemi}.pial")
        )
        context[hemi] = {
            "vertices": vertices,
            "faces": _face_array(faces),
        }
    return context


def _deep_isosurface(
    points_mm: np.ndarray,
    weights: np.ndarray,
    bounds: np.ndarray,
) -> pv.PolyData | None:
    if points_mm.size == 0 or float(weights.max(initial=0.0)) <= 0:
        return None
    dimensions = np.array([52, 52, 52])
    lower = bounds[0]
    upper = bounds[1]
    spacing = (upper - lower) / (dimensions - 1)
    grid = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=lower,
    )
    coordinates = grid.points
    field = np.zeros(len(coordinates), dtype=float)
    normalized = weights / max(float(weights.max()), np.finfo(float).eps)
    for point, weight in zip(points_mm, normalized):
        distance2 = np.sum((coordinates - point) ** 2, axis=1)
        field += float(weight) * np.exp(
            -distance2 / (2.0 * DEEP_KERNEL_SIGMA_MM**2)
        )
    maximum = float(field.max(initial=0.0))
    if maximum <= 0:
        return None
    grid.point_data["activation"] = field / maximum
    return grid.contour([DEEP_CONTOUR_LEVEL], scalars="activation")


def render_deep_volume(
    truth: dict,
    source: np.ndarray,
    anatomical: dict,
) -> np.ndarray:
    positions = (
        mne.transforms.apply_trans(
            anatomical["trans"],
            np.asarray(truth["src_vertices"], dtype=float),
        )
        * 1000.0
    )
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    amplitude = source_amplitude(source)
    deep_amplitude = amplitude[n_surf:]
    keep = deep_amplitude >= DEEP_KEEP_FRACTION * deep_amplitude.max(initial=0.0)
    deep_points = positions[n_surf:][keep]
    deep_weights = deep_amplitude[keep]

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
            opacity=0.16,
            smooth_shading=True,
            show_scalar_bar=False,
        )
    cortex = np.vstack(cortex_vertices)
    margin = 4.0
    bounds = np.vstack([cortex.min(axis=0) - margin, cortex.max(axis=0) + margin])
    surface = _deep_isosurface(deep_points, deep_weights, bounds)
    if surface is not None and surface.n_points:
        plotter.add_mesh(
            surface,
            color="#00bfff",
            opacity=0.88,
            smooth_shading=True,
            show_scalar_bar=False,
        )
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


def draw_scenario(
    scenario: str,
    surface_context: dict,
    anatomical_context: dict,
) -> None:
    truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
    result = np.load(
        RESULT_ROOT / scenario / "spatial_fused_result.npz",
        allow_pickle=True,
    )
    reconstructed = np.asarray(result["S"], dtype=float)
    region = np.asarray(result["region_mask"], dtype=bool).ravel()
    reconstructed = reconstructed * region[:, None]
    rows = (
        ("Ground truth", np.asarray(truth["s_true"], dtype=float)),
        ("Spatial Fused reconstruction", reconstructed),
    )
    lateral_hemi = dominant_truth_hemi(truth, surface_context)
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.0), facecolor="white")

    for row_index, (label, source) in enumerate(rows):
        lh_values, rh_values = source_to_continuous_surface(
            source,
            surface_context,
        )
        images = (
            render_surface_view(
                lh_values,
                rh_values,
                surface_context,
                "ventral",
                lateral_hemi,
            ),
            render_surface_view(
                lh_values,
                rh_values,
                surface_context,
                "lateral",
                lateral_hemi,
            ),
            render_deep_volume(truth, source, anatomical_context),
        )
        for column, image in enumerate(images):
            axes[row_index, column].imshow(image)
            axes[row_index, column].set_axis_off()
        axes[row_index, 0].text(
            -0.03,
            0.5,
            label,
            rotation=90,
            va="center",
            ha="right",
            transform=axes[row_index, 0].transAxes,
            fontsize=13,
            fontweight="bold",
        )

    axes[0, 0].set_title("Inflated ventral", fontsize=14)
    axes[0, 1].set_title(f"Inflated {lateral_hemi.upper()} lateral", fontsize=14)
    axes[0, 2].set_title("Deep-source volume", fontsize=14)
    fig.suptitle(
        f"Continuous source-range reconstruction: {scenario}",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.92,
        bottom=0.01,
        wspace=0.02,
        hspace=0.04,
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_ROOT / f"spatial_fused_cortical_{scenario}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    pv.global_theme.allow_empty_mesh = True
    surface_context = load_surface_context()
    anatomical_context = load_anatomical_context()
    for scenario in SCENARIOS:
        print("Rendering continuous cortical/deep", scenario)
        draw_scenario(scenario, surface_context, anatomical_context)
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
