from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import sparse

from pipelines.run_whole_brain_fusion import DATA_ROOT, load_mat, source_amplitude


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "latest"
MODALITY_ROOT = RESULTS_ROOT / "modality_comparison"
OUT_ROOT = RESULTS_ROOT / "figures" / "modality_comparison"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")
METHODS = ("Ground truth", "Both", "MEG-only", "EEG-only")
ACTIVATION_CMAP = LinearSegmentedColormap.from_list(
    "activation",
    ["#06186b", "#004dff", "#00b9ff", "#19f6e8"],
)


def triangle_adjacency(n_vertices: int, faces: np.ndarray) -> sparse.csr_matrix:
    row = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]] * 2)
    col = np.concatenate(
        [
            faces[:, 1],
            faces[:, 2],
            faces[:, 0],
            faces[:, 0],
            faces[:, 1],
            faces[:, 2],
        ]
    )
    adjacency = sparse.coo_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    adjacency.data[:] = 1.0
    return adjacency


def diffuse(values: np.ndarray, adjacency: sparse.csr_matrix, iterations: int = 8) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    seeds = values.copy()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree = np.maximum(degree, 1.0)
    for _ in range(iterations):
        neighbors = adjacency @ values / degree
        values = np.maximum(seeds, 0.45 * values + 0.55 * neighbors)
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0 else values


def load_surface_context() -> dict:
    data_path = mne.datasets.sample.data_path()
    subjects_dir = data_path / "subjects"
    surf_dir = subjects_dir / "sample" / "surf"
    forward = mne.read_forward_solution(
        data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif",
        verbose=False,
    )

    context = {"subjects_dir": subjects_dir}
    for hemi, src in zip(("lh", "rh"), forward["src"]):
        vertices, faces = nib.freesurfer.read_geometry(str(surf_dir / f"{hemi}.inflated"))
        vertices = vertices.copy()
        vertices[:, 0] += -58.0 if hemi == "lh" else 58.0
        curvature = nib.freesurfer.read_morph_data(str(surf_dir / f"{hemi}.curv"))
        face_array = np.c_[np.full(len(faces), 3), faces].ravel()
        context[hemi] = {
            "vertices": vertices,
            "faces": faces,
            "face_array": face_array,
            "curvature": np.sign(curvature),
            "source_vertices": np.asarray(src["vertno"], dtype=int),
            "adjacency": triangle_adjacency(len(vertices), faces),
        }
    context["n_lh_sources"] = len(context["lh"]["source_vertices"])
    return context


def source_to_surface(
    source: np.ndarray,
    context: dict,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = source_amplitude(source)
    n_lh = context["n_lh_sources"]
    lh = np.zeros(len(context["lh"]["vertices"]), dtype=float)
    rh = np.zeros(len(context["rh"]["vertices"]), dtype=float)
    lh[context["lh"]["source_vertices"]] = amplitude[:n_lh]
    rh[context["rh"]["source_vertices"]] = amplitude[
        n_lh : n_lh + len(context["rh"]["source_vertices"])
    ]
    return (
        np.power(diffuse(lh, context["lh"]["adjacency"], iterations=24), 0.30),
        np.power(diffuse(rh, context["rh"]["adjacency"], iterations=24), 0.30),
    )


def active_surface_points(
    active_indices: np.ndarray,
    context: dict,
) -> dict[str, np.ndarray]:
    active_indices = np.asarray(active_indices, dtype=int).ravel()
    n_lh = context["n_lh_sources"]
    lh_local = active_indices[active_indices < n_lh]
    rh_local = active_indices[
        (active_indices >= n_lh)
        & (active_indices < n_lh + len(context["rh"]["source_vertices"]))
    ] - n_lh
    return {
        "lh": context["lh"]["source_vertices"][lh_local],
        "rh": context["rh"]["source_vertices"][rh_local],
    }


def truth_surface_points(truth: dict, context: dict) -> dict[str, np.ndarray]:
    indices = np.asarray(truth["true_surface_indices0"], dtype=int).ravel()
    return active_surface_points(indices, context)


def add_hemisphere(
    plotter: pv.Plotter,
    hemi: str,
    values: np.ndarray,
    truth_points: np.ndarray,
    active_points: np.ndarray,
    context: dict,
) -> None:
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

    activation = np.asarray(values, dtype=float)
    alpha = np.where(activation >= 0.04, 0.15 + 0.85 * activation, 0.0)
    overlay = mesh.copy()
    overlay.point_data["activation"] = activation
    plotter.add_mesh(
        overlay,
        scalars="activation",
        cmap=ACTIVATION_CMAP,
        clim=(0, 1),
        opacity=alpha,
        smooth_shading=True,
        show_scalar_bar=False,
    )

    if truth_points.size:
        plotter.add_points(
            info["vertices"][truth_points],
            color="#31ffd1",
            point_size=9,
            render_points_as_spheres=True,
        )
    if active_points.size:
        plotter.add_points(
            info["vertices"][active_points],
            color="#ffe14d",
            point_size=10,
            render_points_as_spheres=True,
        )
    if activation.max(initial=0.0) > 0:
        peak = int(np.argmax(activation))
        plotter.add_points(
            info["vertices"][peak][None, :],
            color="#18306f",
            point_size=14,
            render_points_as_spheres=True,
        )


def render_panel(
    lh_values: np.ndarray,
    rh_values: np.ndarray,
    truth_points: dict[str, np.ndarray],
    active_points: dict[str, np.ndarray],
    context: dict,
    view: str,
    lateral_hemi: str,
) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=(700, 560))
    plotter.set_background("black")
    plotter.enable_anti_aliasing("ssaa")
    hemis = ("lh", "rh") if view == "ventral" else (lateral_hemi,)
    for hemi in hemis:
        add_hemisphere(
            plotter,
            hemi,
            lh_values if hemi == "lh" else rh_values,
            truth_points[hemi],
            active_points[hemi],
            context,
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
    plotter.camera.zoom(1.15)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def load_result(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return (
        np.asarray(data["S"], dtype=float),
        np.asarray(data["active_indices0"], dtype=int).ravel(),
    )


def dominant_truth_hemi(truth: dict, context: dict) -> str:
    indices = np.asarray(truth["true_surface_indices0"], dtype=int).ravel()
    return "lh" if np.sum(indices < context["n_lh_sources"]) > np.sum(
        indices >= context["n_lh_sources"]
    ) else "rh"


def draw_scenario(scenario: str, context: dict) -> None:
    truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
    truth_source = np.asarray(truth["s_true_surface"], dtype=float)
    out_dir = MODALITY_ROOT / scenario
    maps = {
        "Ground truth": (truth_source, np.zeros(0, dtype=int)),
        "Both": load_result(out_dir / "both_result.npz"),
        "MEG-only": load_result(out_dir / "meg_only_result.npz"),
        "EEG-only": load_result(out_dir / "eeg_only_result.npz"),
    }

    truth_points = truth_surface_points(truth, context)
    lateral_hemi = dominant_truth_hemi(truth, context)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="white")

    for column, method in enumerate(METHODS):
        source, active = maps[method]
        lh_values, rh_values = source_to_surface(source, context)
        active_points = active_surface_points(active, context)
        ventral = render_panel(
            lh_values,
            rh_values,
            truth_points,
            active_points,
            context,
            "ventral",
            lateral_hemi,
        )
        lateral = render_panel(
            lh_values,
            rh_values,
            truth_points,
            active_points,
            context,
            "lateral",
            lateral_hemi,
        )
        axes[0, column].imshow(ventral)
        axes[1, column].imshow(lateral)
        axes[0, column].set_title(
            method if method == "Ground truth" else f"{method}\nactive={len(active)}",
            fontsize=16,
        )
        for row in range(2):
            axes[row, column].set_axis_off()

    fig.text(0.015, 0.72, "Ventral", rotation=90, va="center", fontsize=13)
    fig.text(
        0.015,
        0.28,
        f"{lateral_hemi.upper()} lateral",
        rotation=90,
        va="center",
        fontsize=13,
    )
    fig.suptitle(
        f"Cortical activity comparison: {scenario}\n"
        "cyan points = true patch, yellow points = selected grid points, maps normalized to peak",
        fontsize=17,
        y=0.99,
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.02, wspace=0.03, hspace=0.04)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_ROOT / f"cortical_modality_{scenario}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    pv.global_theme.allow_empty_mesh = True
    context = load_surface_context()
    for scenario in SCENARIOS:
        print("Rendering", scenario)
        draw_scenario(scenario, context)
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
