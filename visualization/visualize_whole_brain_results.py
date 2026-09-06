"""MRI-style visualization for the latest whole-brain adaptive model only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "generated"
RESULT_ROOT = ROOT / "results" / "latest" / "whole_brain_adaptive_fusion"
OUT_ROOT = ROOT / "results" / "latest" / "figures" / "whole_brain_mri"
SCENARIOS = (
    "deep_only",
    "surface_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
)


def position_to_voxel(position_m: np.ndarray, vox2ras_tkr: np.ndarray) -> np.ndarray:
    position_mm = np.asarray(position_m, dtype=float) * 1000.0
    return (np.linalg.inv(vox2ras_tkr) @ np.r_[position_mm, 1.0])[:3]


def normalized_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, volume.shape[axis] - 1))
    image = np.take(volume, index, axis=axis).T
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / (high - low + 1e-20), 0, 1)


def draw_scenario(scenario: str, volume: np.ndarray, vox2ras_tkr: np.ndarray) -> None:
    truth = sio.loadmat(DATA_ROOT / scenario / "s_true.mat")
    result = np.load(RESULT_ROOT / scenario / "whole_brain_result.npz")
    vertices = np.asarray(truth["src_vertices"], dtype=float)
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    source = np.asarray(result["S"], dtype=float)
    active = np.asarray(result["active_indices0"], dtype=int).ravel()
    amplitude = np.linalg.norm(source, axis=1)
    peak = int(np.argmax(amplitude))
    peak_voxel = position_to_voxel(vertices[peak], vox2ras_tkr)

    fig = plt.figure(figsize=(12, 9), dpi=160, facecolor="black")
    planes = (
        ("Sagittal", 0, int(round(peak_voxel[0]))),
        ("Coronal", 1, int(round(peak_voxel[1]))),
        ("Axial", 2, int(round(peak_voxel[2]))),
    )
    for panel, (title, axis, index) in enumerate(planes, start=1):
        ax = fig.add_subplot(2, 2, panel)
        ax.imshow(normalized_slice(volume, axis, index), cmap="gray", origin="upper")
        ax.set_title(f"{title}: peak grid {peak + 1}", color="white")
        ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    surface = vertices[:n_surf]
    deep = vertices[n_surf:]
    ax.scatter(surface[::8, 0], surface[::8, 1], surface[::8, 2], s=0.6, color="lightgray", alpha=0.12)
    ax.scatter(deep[:, 0], deep[:, 1], deep[:, 2], s=15, color="purple", alpha=0.5)
    if active.size:
        points = vertices[active]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=45,
            facecolors="none",
            edgecolors="gold",
            label="selected active grids",
        )
    ax.scatter(
        *vertices[peak],
        marker="x",
        s=120,
        linewidths=2.5,
        color="deepskyblue",
        label="global peak",
    )
    ax.set_title("Latest whole-brain result", color="white")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(loc="upper right")

    fig.suptitle(f"{scenario} — whole-brain adaptive EEG-MEG fusion", color="white", fontsize=14)
    fig.tight_layout()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_ROOT / f"whole_brain_{scenario}.png", facecolor="black")
    plt.close(fig)


def main() -> None:
    sample_path = mne.datasets.sample.data_path()
    t1 = nib.load(str(sample_path / "subjects" / "sample" / "mri" / "T1.mgz"))
    volume = np.asarray(t1.get_fdata(), dtype=float)
    vox2ras_tkr = t1.header.get_vox2ras_tkr()
    for scenario in SCENARIOS:
        draw_scenario(scenario, volume, vox2ras_tkr)
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
