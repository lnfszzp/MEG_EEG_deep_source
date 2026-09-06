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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.run_whole_brain_fusion import load_mat, source_amplitude


DATA_ROOT = ROOT / "generated"
RESULT_ROOT = ROOT / "results" / "latest" / "spatial_fused_fusion"
OUT_ROOT = ROOT / "results" / "latest" / "figures" / "spatial_fused_mri"
SCENARIOS = (
    "deep_only",
    "surface_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
)
ACTIVATION_CMAP = LinearSegmentedColormap.from_list(
    "mri_activation",
    ["#06186b", "#004dff", "#00b9ff", "#19f6e8"],
)
OVERLAY_CUTOFF = 0.12
OVERLAY_ALPHA = 0.58
OVERLAY_SIGMA = 2.8


def pos_to_vox(position_m: np.ndarray, vox2ras_tkr: np.ndarray) -> np.ndarray:
    position_mm = np.asarray(position_m, dtype=float) * 1000.0
    return (np.linalg.inv(vox2ras_tkr) @ np.r_[position_mm, 1.0])[:3]


def normalize_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / (high - low + 1e-20), 0, 1)


def plane_image(volume: np.ndarray, plane: str, voxel: np.ndarray):
    x, y, z = [int(round(value)) for value in voxel]
    x = int(np.clip(x, 0, volume.shape[0] - 1))
    y = int(np.clip(y, 0, volume.shape[1] - 1))
    z = int(np.clip(z, 0, volume.shape[2] - 1))
    if plane == "coronal":
        return (
            normalize_image(volume[:, y, :].T),
            1,
            y,
            lambda v: (v[0], volume.shape[2] - 1 - v[2]),
        )
    if plane == "sagittal":
        return (
            normalize_image(volume[x, :, :].T),
            0,
            x,
            lambda v: (v[1], volume.shape[2] - 1 - v[2]),
        )
    return (
        normalize_image(volume[:, :, z].T),
        2,
        z,
        lambda v: (v[0], volume.shape[1] - 1 - v[1]),
    )


def gaussian_overlay(
    shape: tuple[int, int],
    points: list[tuple[float, float]],
    weights: list[float],
    sigma: float = 4.0,
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    overlay = np.zeros(shape, dtype=float)
    for (px, py), weight in zip(points, weights):
        overlay += weight * np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma**2)
        )
    maximum = float(overlay.max(initial=0.0))
    return overlay / maximum if maximum > 0 else overlay


def source_overlay_for_slice(
    image_shape: tuple[int, int],
    source_voxels: np.ndarray,
    source_weights: np.ndarray,
    axis_index: int,
    slice_index: int,
    point_2d,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    weights: list[float] = []
    for voxel, weight in zip(source_voxels, source_weights):
        distance = abs(float(voxel[axis_index]) - float(slice_index))
        if distance <= 9:
            points.append(point_2d(voxel))
            weights.append(float(weight) * np.exp(-(distance**2) / 32.0))
    return gaussian_overlay(image_shape, points, weights, sigma=OVERLAY_SIGMA)


def visible_overlay_mask(overlay: np.ndarray) -> np.ndarray:
    return np.asarray(overlay, dtype=float) >= OVERLAY_CUTOFF


def draw_slice(
    ax,
    volume: np.ndarray,
    plane: str,
    center_voxel: np.ndarray,
    source_voxels: np.ndarray,
    source_weights: np.ndarray,
) -> None:
    image, axis_index, slice_index, point_2d = plane_image(
        volume,
        plane,
        center_voxel,
    )
    ax.imshow(image, cmap="gray", origin="upper")
    overlay = source_overlay_for_slice(
        image.shape,
        source_voxels,
        source_weights,
        axis_index,
        slice_index,
        point_2d,
    )
    if float(overlay.max(initial=0.0)) > 0:
        ax.imshow(
            np.ma.masked_where(~visible_overlay_mask(overlay), overlay),
            cmap=ACTIVATION_CMAP,
            origin="upper",
            alpha=OVERLAY_ALPHA,
            vmin=0,
            vmax=1,
        )
    ax.set_axis_off()


def _focus_indices(truth: dict) -> list[tuple[str, int]]:
    true_surface = np.asarray(
        truth["true_surface_indices0"],
        dtype=int,
    ).ravel()
    has_deep = bool(int(np.asarray(truth["has_deep_source"]).ravel()[0]))
    focuses: list[tuple[str, int]] = []
    if true_surface.size:
        centers = np.asarray(
            truth.get("true_surface_centers0", true_surface[:1]),
            dtype=int,
        ).ravel()
        focuses.append(("Surface focus", int(centers[0])))
    if has_deep:
        true_deep = int(np.asarray(truth["true_deep_idx0"]).ravel()[0])
        focuses.append(("Deep focus", true_deep))
    return focuses


def draw_scenario(
    scenario: str,
    volume: np.ndarray,
    vox2ras_tkr: np.ndarray,
    head_to_mri: np.ndarray,
) -> None:
    truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
    result = np.load(
        RESULT_ROOT / scenario / "spatial_fused_result.npz",
        allow_pickle=True,
    )
    vertices = mne.transforms.apply_trans(
        head_to_mri,
        np.asarray(truth["src_vertices"], dtype=float),
    )
    region_mask = np.asarray(result["region_mask"], dtype=bool).ravel()
    reconstructed = np.asarray(result["S"], dtype=float) * region_mask[:, None]
    rows = (
        ("Ground truth", np.asarray(truth["s_true"], dtype=float)),
        ("Spatial Fused reconstruction", reconstructed),
    )
    focuses = _focus_indices(truth)
    planes = ("coronal", "sagittal", "axial")
    n_columns = len(focuses) * len(planes)
    fig, axes = plt.subplots(
        2,
        n_columns,
        figsize=(4.0 * n_columns, 7.4),
        dpi=170,
        facecolor="black",
        squeeze=False,
    )

    for row_index, (row_label, source) in enumerate(rows):
        amplitude = source_amplitude(source)
        active = amplitude > 0
        source_voxels = np.asarray(
            [pos_to_vox(position, vox2ras_tkr) for position in vertices[active]]
        )
        source_weights = amplitude[active]
        if source_weights.size:
            source_weights = source_weights / max(
                float(source_weights.max()),
                np.finfo(float).eps,
            )
        for focus_index, (focus_label, center_index) in enumerate(focuses):
            center_voxel = pos_to_vox(vertices[center_index], vox2ras_tkr)
            for plane_index, plane in enumerate(planes):
                column = focus_index * len(planes) + plane_index
                draw_slice(
                    axes[row_index, column],
                    volume,
                    plane,
                    center_voxel,
                    source_voxels,
                    source_weights,
                )
                if row_index == 0:
                    axes[row_index, column].set_title(
                        f"{focus_label}\n{plane.capitalize()}",
                        color="white",
                        fontsize=11,
                    )
        axes[row_index, 0].text(
            -0.05,
            0.5,
            row_label,
            rotation=90,
            va="center",
            ha="right",
            transform=axes[row_index, 0].transAxes,
            color="white",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle(
        f"Continuous MRI source-range reconstruction: {scenario}",
        color="white",
        fontsize=15,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.90,
        bottom=0.015,
        wspace=0.02,
        hspace=0.04,
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_ROOT / f"spatial_fused_{scenario}.png",
        facecolor="black",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    data_path = mne.datasets.sample.data_path()
    t1 = nib.load(str(data_path / "subjects" / "sample" / "mri" / "T1.mgz"))
    volume = np.asarray(t1.get_fdata(), dtype=float)
    vox2ras_tkr = t1.header.get_vox2ras_tkr()
    head_to_mri = mne.read_trans(
        data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    )["trans"]
    for scenario in SCENARIOS:
        print("Rendering continuous MRI", scenario)
        draw_scenario(scenario, volume, vox2ras_tkr, head_to_mri)
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
