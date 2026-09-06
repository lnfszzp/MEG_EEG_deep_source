from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pyvista as pv

from pipelines.run_whole_brain_fusion import DATA_ROOT, load_mat, source_amplitude
from visualization.source_utils import load_masked_source
from visualization.visualize_modality_comparison import (
    dominant_truth_hemi,
    load_surface_context,
)
from visualization.visualize_spatial_fused_cortical import (
    load_anatomical_context,
    render_deep_volume,
    render_surface_view,
    source_to_continuous_surface,
)
from visualization.visualize_spatial_fused_mri import (
    _focus_indices,
    draw_slice,
    pos_to_vox,
)

RowSpecs = tuple[tuple[str, Path | None], ...]
RowFactory = Callable[[str], RowSpecs]


def _rows(truth: dict, specs: RowSpecs) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple(
        (
            label,
            np.asarray(truth["s_true"], dtype=float)
            if path is None
            else load_masked_source(path),
        )
        for label, path in specs
    )


def render_cortical_set(
    scenarios: tuple[str, ...],
    out_root: Path,
    row_specs: RowFactory,
    *,
    title_template: str,
    filename_template: str,
    print_label: str,
) -> None:
    pv.global_theme.allow_empty_mesh = True
    surface_context = load_surface_context()
    anatomical_context = load_anatomical_context()
    for scenario in scenarios:
        print(f"Rendering {print_label}", scenario)
        truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
        lateral_hemi = dominant_truth_hemi(truth, surface_context)
        rows = _rows(truth, row_specs(scenario))
        fig, axes = plt.subplots(4, 3, figsize=(14.8, 15.0), facecolor="white")
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
        fig.suptitle(title_template.format(scenario=scenario), fontsize=16, y=0.995)
        fig.subplots_adjust(
            left=0.065,
            right=0.995,
            top=0.965,
            bottom=0.005,
            wspace=0.02,
            hspace=0.035,
        )
        out_root.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_root / filename_template.format(scenario=scenario),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)
    print("Saved:", out_root)


def render_mri_set(
    scenarios: tuple[str, ...],
    out_root: Path,
    row_specs: RowFactory,
    *,
    title_template: str,
    filename_template: str,
    print_label: str,
) -> None:
    data_path = mne.datasets.sample.data_path()
    t1 = nib.load(str(data_path / "subjects" / "sample" / "mri" / "T1.mgz"))
    volume = np.asarray(t1.get_fdata(), dtype=float)
    vox2ras_tkr = t1.header.get_vox2ras_tkr()
    head_to_mri = mne.read_trans(
        data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    )["trans"]
    for scenario in scenarios:
        print(f"Rendering {print_label}", scenario)
        truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
        vertices = mne.transforms.apply_trans(
            head_to_mri,
            np.asarray(truth["src_vertices"], dtype=float),
        )
        focuses = _focus_indices(truth)
        planes = ("coronal", "sagittal", "axial")
        fig, axes = plt.subplots(
            4,
            len(focuses) * len(planes),
            figsize=(4.0 * len(focuses) * len(planes), 13.8),
            dpi=170,
            facecolor="black",
            squeeze=False,
        )
        for row_index, (label, source) in enumerate(_rows(truth, row_specs(scenario))):
            amplitude = source_amplitude(source)
            active = amplitude > 0
            source_voxels = np.asarray(
                [pos_to_vox(position, vox2ras_tkr) for position in vertices[active]]
            )
            weights = amplitude[active]
            if weights.size:
                weights = weights / max(float(weights.max()), np.finfo(float).eps)
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
                        weights,
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
                label,
                rotation=90,
                va="center",
                ha="right",
                transform=axes[row_index, 0].transAxes,
                color="white",
                fontsize=13,
                fontweight="bold",
            )
        fig.suptitle(
            title_template.format(scenario=scenario),
            color="white",
            fontsize=15,
            y=0.995,
        )
        fig.subplots_adjust(
            left=0.055,
            right=0.995,
            top=0.955,
            bottom=0.005,
            wspace=0.02,
            hspace=0.035,
        )
        out_root.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_root / filename_template.format(scenario=scenario),
            facecolor="black",
            bbox_inches="tight",
        )
        plt.close(fig)
    print("Saved:", out_root)
