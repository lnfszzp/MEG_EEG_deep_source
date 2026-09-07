"""Render completed strict-benchmark estimates on the same anatomical MRI slices.

The default query is a representative 0/0 dB deep-plus-two-surface location.
SISSES is loaded read-only when available; Python methods are recomputed from
that case's immutable archived EEG/MEG observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mne
from mne.transforms import apply_trans
import nibabel as nib
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

import plot_strict_case as strict_plot
import plot_strict_metrics as metric_plots
import run_strict_comparators as comparators
from benchmark import methods as comparator_methods
from benchmark import metrics as benchmark_metrics
from visualization.visualize_spatial_fused_mri import (
    plane_image,
    pos_to_vox,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_QUERY = (0, 0, "deep_plus_two_surface", 13)
DEFAULT_SISSES_ROOT = Path(r"D:\oaster_strict_blind_sisses")
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "strict_blind" / "brain_maps_v2"
DEFAULT_SAMPLE_PATH = Path(r"D:\mne_data\MNE-sample-data")
OASTER_LABELS = ("OASTER V20", "OASTER V19")
METHOD_ORDER = (
    OASTER_LABELS[0],
    "SISSES",
    *comparators.METHODS,
)
METHOD_SLUGS = {
    "OASTER V20": "oaster_v20",
    "OASTER V19": "oaster_v19",
    "SISSES": "sisses",
    **comparators.METHOD_SLUGS,
}
PLANES = ("coronal", "sagittal", "axial")
ESTIMATE_CMAP = "magma"
TRUTH_COLOR = "#00E5A8"
SCENARIO_LABELS = {
    "surface_only": "Cortical surface only",
    "deep_only": "Thalamic deep source only",
    "deep_plus_surface": "Thalamic + one cortical source",
    "deep_plus_two_surface": "Thalamic + two cortical sources",
}


def _case_title(case: dict) -> str:
    scenario = str(case["scenario"])
    return (
        f"Case {int(case['case_number']):05d} | "
        f"{SCENARIO_LABELS.get(scenario, scenario.replace('_', ' ').title())} | "
        f"EEG SNR {int(case['eeg_snr_db']):+d} dB | "
        f"MEG SNR {int(case['meg_snr_db']):+d} dB"
    )


def _legend_handles(
    include_truth: bool = True, energy_label: str = "Estimated source energy"
) -> list:
    handles = []
    if include_truth:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    linestyle="none",
                    markerfacecolor=TRUTH_COLOR,
                    markeredgecolor="#07130F",
                    markersize=13,
                    label="Simulated source center",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor=TRUTH_COLOR,
                    markeredgewidth=1.8,
                    markersize=9,
                    label="Simulated source parcel",
                ),
            ]
        )
    handles.append(
        Patch(
            facecolor=plt.get_cmap(ESTIMATE_CMAP)(0.72),
            label=energy_label,
        )
    )
    return handles


def _default_data_root() -> Path:
    """Reuse the geometry path recorded by the completed strict run if present."""
    for directory in ("oaster_v20_final", "oaster_v19_final"):
        metadata = ROOT / "results" / "strict_blind" / directory / "metadata.json"
        if metadata.exists():
            path = Path(json.loads(metadata.read_text(encoding="utf-8"))["geometry_reference"])
            if path.exists():
                return path.parents[1]
    return strict_plot.strict.protocol.DEFAULT_DATA_ROOT


def _sisses_path(loaded: dict, sisses_root: Path) -> Path:
    spec = loaded["chunk"]
    local_number = int(loaded["case"]["case_number"]) - int(spec.start) + 1
    return (
        Path(sisses_root)
        / "matlab_output"
        / spec.path.stem
        / f"case_{local_number:04d}.mat"
    )


def load_sisses_estimate(loaded: dict, sisses_root: Path) -> tuple[np.ndarray, Path, dict]:
    """Read and validate one preserved SISSES estimate without changing the archive."""
    path = _sisses_path(loaded, sisses_root)
    if not path.is_file():
        raise FileNotFoundError(f"preserved SISSES estimate not found: {path}")
    before = path.stat()
    values = sio.loadmat(
        path,
        variable_names=(
            "source_estimates",
            "method_names",
            "case_id",
            "success",
            "source_shape",
            "format_version",
            "metadata",
        ),
        simplify_cells=True,
    )
    expected_shape = loaded["truth"].shape
    estimate = np.asarray(values.get("source_estimates"), dtype=float).squeeze()
    if estimate.shape == expected_shape[::-1]:
        estimate = estimate.T
    if estimate.shape != expected_shape or not np.all(np.isfinite(estimate)):
        raise ValueError(f"invalid SISSES source_estimates in {path}: {estimate.shape}")
    if strict_plot.strict._text(values.get("case_id")) != loaded["case"]["case_id"]:
        raise ValueError(f"SISSES case_id does not match strict manifest: {path}")
    methods = strict_plot.strict._string_list(values.get("method_names", []))
    if methods != ["SISSES"] or not bool(np.asarray(values.get("success", 0)).item()):
        raise ValueError(f"SISSES output is not a successful single-method result: {path}")
    recorded_shape = tuple(int(value) for value in np.asarray(values["source_shape"]).ravel())
    if recorded_shape[:2] != expected_shape:
        raise ValueError(f"SISSES source_shape metadata disagrees with data: {path}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"SISSES archive changed while being read: {path}")
    return estimate, path, {
        "format_version": strict_plot.strict._text(values.get("format_version", "")),
        "variant": dict(values.get("metadata", {})).get("variant", ""),
        "bytes": before.st_size,
    }


def _completed_method_order(results_root: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    results, availability = metric_plots.load_results_with_availability(results_root)
    oaster_source = next(
        (Path(row["source"]).parent.name for row in availability if row["method"] == "OASTER"),
        "",
    )
    aliases = {
        "OASTER": "OASTER V20" if oaster_source == "oaster_v20_final" else "OASTER V19"
    }
    return tuple(aliases.get(method, method) for method in results), availability


def reconstruct_all(
    loaded: dict,
    sisses_root: Path,
    methods: tuple[str, ...] = METHOD_ORDER,
    *,
    sisses_mode: str = "auto",
) -> tuple[dict[str, np.ndarray], dict]:
    """Reconstruct only requested methods; unavailable optional SISSES becomes N/A."""
    if sisses_mode not in {"auto", "require", "skip"}:
        raise ValueError("sisses_mode must be auto, require, or skip")
    unknown = set(methods) - (set(METHOD_ORDER) | set(OASTER_LABELS))
    selected_oaster = tuple(method for method in methods if method in OASTER_LABELS)
    if not methods or unknown or len(selected_oaster) > 1:
        raise ValueError(f"invalid method selection: {sorted(unknown)}")

    runtime = strict_plot.strict._runtime(loaded["geometry"], loaded["reference"])
    computed: dict[str, np.ndarray] = {}
    provenance: dict[str, object] = {}
    if selected_oaster:
        oaster_label = selected_oaster[0]
        estimate, diagnostics = strict_plot.strict.oaster.reconstruct(
            loaded["eeg"],
            loaded["meg"],
            loaded["gain_eeg"],
            loaded["gain_meg"],
            loaded["geometry"]["n_surf"],
            runtime["kernels"],
        )
        computed[oaster_label] = estimate
        provenance["oaster_diagnostics"] = {
            "version_label": oaster_label,
            "temporal_rank": int(diagnostics["temporal_rank"]),
            "selected_templates": int(diagnostics["selected_templates"]),
        }

    if "SISSES" in methods and sisses_mode != "skip":
        try:
            estimate, path, metadata = load_sisses_estimate(loaded, sisses_root)
            computed["SISSES"] = estimate
            provenance.update(
                {"sisses_status": "available", "sisses_path": str(path.resolve()), "sisses": metadata}
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            if sisses_mode == "require":
                raise
            provenance.update({"sisses_status": "N/A", "sisses_reason": str(error)})
    elif "SISSES" not in methods or sisses_mode == "skip":
        provenance["sisses_status"] = "N/A"
        provenance["sisses_reason"] = (
            "disabled by --sisses-mode skip"
            if sisses_mode == "skip"
            else "not completed for this results root"
        )

    requested_comparators = set(methods) & set(comparators.METHODS)
    if requested_comparators:
        data, gain = comparator_methods.joint_whiten(
            loaded["eeg"], loaded["meg"], loaded["gain_eeg"], loaded["gain_meg"]
        )
        family = comparator_methods.minimum_norm_family(data, gain)
        computed.update(
            {method: estimate for method, estimate in family.items() if method in requested_comparators}
        )
        active = runtime["active"]
        if "LCMV" in requested_comparators:
            computed["LCMV"] = comparator_methods.lcmv(data, gain, active)
        if "Dipole fitting (grid)" in requested_comparators:
            computed["Dipole fitting (grid)"] = comparator_methods.dipole_fit(data, gain, active)
        if "RAP-MUSIC" in requested_comparators:
            computed["RAP-MUSIC"] = comparator_methods.rap_music(data, gain, active)

    estimates = {method: computed[method] for method in methods if method in computed}
    if not estimates:
        raise ValueError("no requested method estimate is available")
    return estimates, provenance


def evaluate_all(loaded: dict, estimates: dict[str, np.ndarray]) -> list[dict]:
    runtime = comparators._runtime(loaded["geometry"], loaded["reference"])
    rows = []
    for method in estimates:
        rows.append(
            {
                "case_id": loaded["case"]["case_id"],
                "method": method,
                **benchmark_metrics.evaluate_estimate(
                    estimates[method],
                    loaded["truth"],
                    loaded["geometry"]["vertices"],
                    loaded["groups"],
                    loaded["geometry"]["n_surf"],
                    runtime["active"],
                    runtime["cortex"],
                ),
            }
        )
    return rows


def load_anatomy(sample_path: Path) -> dict:
    sample_path = Path(sample_path)
    t1_path = sample_path / "subjects" / "sample" / "mri" / "T1.mgz"
    trans_path = sample_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    if not t1_path.is_file() or not trans_path.is_file():
        raise FileNotFoundError(f"MNE sample anatomy or head-to-MRI transform missing: {sample_path}")
    t1 = nib.load(str(t1_path))
    return {
        "volume": np.asarray(t1.get_fdata(), dtype=float),
        "vox2ras_tkr": t1.header.get_vox2ras_tkr(),
        "head_to_mri": mne.read_trans(trans_path)["trans"],
        "sample_path": str(sample_path.resolve()),
    }


def _validate_surface_source_space(geometry: dict, src, subjects_dir: Path) -> dict:
    """Match corrected source indices to the two FreeSurfer hemispheres."""
    if geometry.get("deep_aseg_labels") is None:
        raise ValueError("surface maps require corrected geometry with anatomical deep labels")
    if len(src) != 2:
        raise ValueError("surface forward must contain exactly left and right hemispheres")
    vertices = tuple(np.asarray(part["vertno"], dtype=int).ravel() for part in src)
    n_surf = int(geometry["n_surf"])
    if sum(map(len, vertices)) != n_surf:
        raise ValueError("surface forward source count disagrees with corrected geometry")
    forward_positions = np.vstack(
        [np.asarray(part["rr"], dtype=float)[vertno] for part, vertno in zip(src, vertices)]
    )
    geometry_positions = np.asarray(geometry["vertices"], dtype=float)[:n_surf]
    if geometry_positions.shape != forward_positions.shape or not np.allclose(
        geometry_positions, forward_positions, rtol=0.0, atol=1e-12
    ):
        raise ValueError("corrected geometry source order disagrees with the surface forward")
    subjects = {str(part.get("subject_his_id", "")) for part in src}
    if len(subjects) != 1 or not next(iter(subjects)):
        raise ValueError("surface forward does not identify one FreeSurfer subject")
    subject = subjects.pop()
    for hemi in ("lh", "rh"):
        if not (Path(subjects_dir) / subject / "surf" / f"{hemi}.pial").is_file():
            raise FileNotFoundError(f"missing {hemi}.pial for subject {subject}")
    return {
        "subject": subject,
        "subjects_dir": Path(subjects_dir),
        "vertices": vertices,
    }


def load_surface_source_space(sample_path: Path, geometry: dict) -> dict:
    sample_path = Path(sample_path)
    forward_path = sample_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    if not forward_path.is_file():
        raise FileNotFoundError(f"MNE sample surface forward missing: {forward_path}")
    forward = mne.read_forward_solution(forward_path, verbose="ERROR")
    return _validate_surface_source_space(geometry, forward["src"], sample_path / "subjects")


def _surface_truth_overlays(loaded: dict, surface: dict) -> list[dict]:
    """Return cortical truth only; deep indices are deliberately discarded."""
    n_surf = int(loaded["geometry"]["n_surf"])
    left_count = len(surface["vertices"][0])

    def mapped(index: int) -> tuple[str, int]:
        if not 0 <= index < n_surf:
            raise ValueError(f"surface truth index outside [0,{n_surf}): {index}")
        hemi = 0 if index < left_count else 1
        local = index if hemi == 0 else index - left_count
        return ("lh", "rh")[hemi], int(surface["vertices"][hemi][local])

    overlays = []
    for number, center in enumerate(loaded["case"].get("surface_centers", []), start=1):
        center = int(center)
        indices = np.array([center], dtype=int)
        for group in loaded["groups"]:
            candidate = np.asarray(group, dtype=int).ravel()
            if np.any(candidate == center):
                indices = np.unique(candidate[(candidate >= 0) & (candidate < n_surf)])
                break
        center_hemi, center_vertex = mapped(center)
        mapped_patch = [mapped(int(index)) for index in indices]
        if any(hemi != center_hemi for hemi, _vertex in mapped_patch):
            raise ValueError("one simulated surface patch crosses hemispheres")
        overlays.append(
            {
                "name": f"simulated_surface_{number}",
                "hemi": center_hemi,
                "patch_vertices": np.asarray([vertex for _hemi, vertex in mapped_patch], dtype=int),
                "center_vertex": center_vertex,
            }
        )
    return overlays


def _focuses(loaded: dict) -> list[tuple[str, int]]:
    case = loaded["case"]
    focuses = [
        (f"Surface source {number}", int(index))
        for number, index in enumerate(case.get("surface_centers", []), start=1)
    ]
    if case.get("deep_index") is not None:
        corrected = loaded["geometry"].get("deep_aseg_labels") is not None
        focuses.append(("Thalamic source" if corrected else "Non-cortical source", int(case["deep_index"])))
    if not focuses:
        raise ValueError("strict case has no source focus")
    return focuses


def _source_projection(
    source: np.ndarray,
    loaded: dict,
    anatomy: dict,
    relative_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.arange(strict_plot.strict.protocol.ACTIVE_START, source.shape[1])
    amplitude = benchmark_metrics.source_amplitude(source, active)
    peak = float(amplitude.max(initial=0.0))
    relative = amplitude / peak if peak > 0 else np.zeros_like(amplitude)
    keep = relative >= float(relative_threshold)
    positions_mri = apply_trans(
        anatomy["head_to_mri"], loaded["geometry"]["vertices"]
    )
    voxels = np.asarray(
        [pos_to_vox(position, anatomy["vox2ras_tkr"]) for position in positions_mri[keep]],
        dtype=float,
    )
    return voxels.reshape(-1, 3), relative[keep]


def _truth_voxels(loaded: dict, anatomy: dict, focus_index: int) -> np.ndarray:
    indices = np.asarray([focus_index], dtype=int)
    for group in loaded["groups"]:
        candidate = np.asarray(group, dtype=int)
        if np.any(candidate == focus_index):
            indices = np.unique(candidate)
            break
    positions = apply_trans(
        anatomy["head_to_mri"], loaded["geometry"]["vertices"][indices]
    )
    return np.asarray(
        [pos_to_vox(position, anatomy["vox2ras_tkr"]) for position in positions],
        dtype=float,
    )


def _focus_voxel(index: int, loaded: dict, anatomy: dict) -> np.ndarray:
    position = apply_trans(
        anatomy["head_to_mri"], loaded["geometry"]["vertices"][int(index)]
    )
    return pos_to_vox(position, anatomy["vox2ras_tkr"])


def _draw_slice(
    ax,
    plane: str,
    center_voxel: np.ndarray,
    estimate_voxels: np.ndarray,
    estimate_weights: np.ndarray,
    truth_voxels: np.ndarray | None,
    volume: np.ndarray,
) -> None:
    image, axis, index, point_2d = plane_image(volume, plane, center_voxel)
    ax.imshow(image, cmap="gray", origin="upper", vmin=0, vmax=1)
    distance = np.abs(estimate_voxels[:, axis] - index)
    near_estimate = distance <= 9.0
    overlay = np.zeros(image.shape, dtype=float)
    if np.any(near_estimate):
        points = np.asarray([point_2d(voxel) for voxel in estimate_voxels[near_estimate]])
        x = np.rint(points[:, 0]).astype(int)
        y = np.rint(points[:, 1]).astype(int)
        visible = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        weights = estimate_weights[near_estimate] * np.exp(-(distance[near_estimate] ** 2) / 32.0)
        np.add.at(overlay, (y[visible], x[visible]), weights[visible])
        overlay = gaussian_filter(overlay, sigma=2.8)
        maximum = float(overlay.max(initial=0.0))
        if maximum > 0:
            overlay /= maximum
    if float(overlay.max(initial=0.0)) > 0:
        ax.imshow(
            np.ma.masked_less(overlay, 0.10),
            cmap=ESTIMATE_CMAP,
            origin="upper",
            alpha=0.78,
            vmin=0.10,
            vmax=1.0,
        )
    if truth_voxels is not None:
        near = truth_voxels[np.abs(truth_voxels[:, axis] - index) <= 3.0]
        if near.size:
            points = np.asarray([point_2d(voxel) for voxel in near])
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=58,
                facecolors="none",
                edgecolors="#07130F",
                linewidths=3.8,
                zorder=6,
            )
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=58,
                facecolors="none",
                edgecolors=TRUTH_COLOR,
                linewidths=1.8,
                zorder=7,
            )
        focus_xy = point_2d(center_voxel)
        ax.scatter(
            [focus_xy[0]],
            [focus_xy[1]],
            s=190,
            marker="*",
            facecolor=TRUTH_COLOR,
            edgecolor="#07130F",
            linewidths=1.4,
            zorder=8,
        )
    ax.set_axis_off()


def _number(value: object, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "--"


def render_method(
    method: str,
    estimate: np.ndarray,
    metrics: dict | None,
    loaded: dict,
    anatomy: dict,
    output: Path,
    relative_threshold: float,
    dpi: int,
    *,
    truth_overlay: bool = False,
) -> Path:
    focuses = _focuses(loaded)
    estimate_voxels, estimate_weights = _source_projection(
        estimate, loaded, anatomy, relative_threshold
    )
    fig, axes = plt.subplots(
        len(focuses),
        len(PLANES),
        figsize=(12.0, 3.35 * len(focuses) + 1.25),
        facecolor="#090A0F",
        squeeze=False,
    )
    for row, (focus_label, source_index) in enumerate(focuses):
        center = _focus_voxel(source_index, loaded, anatomy)
        truth_voxels = (
            _truth_voxels(loaded, anatomy, source_index) if truth_overlay else None
        )
        for column, plane in enumerate(PLANES):
            _draw_slice(
                axes[row, column],
                plane,
                center,
                estimate_voxels,
                estimate_weights,
                truth_voxels,
                anatomy["volume"],
            )
            if row == 0:
                axes[row, column].set_title(
                    plane.capitalize(), color="#ECEEF4", fontsize=12, pad=5
                )
        axes[row, 0].text(
            -0.03,
            0.5,
            focus_label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            color="#ECEEF4",
            fontsize=10,
        )
    case = loaded["case"]
    metric_line = ""
    if metrics is not None:
        metric_line = (
            f"\nAn_auc {_number(metrics['auc_tie_corrected'])}   "
            f"RMSE {_number(metrics['rmse'])}   "
            f"Surface DLE {_number(metrics['surface_dle_mm'], 1)} mm   "
            f"Deep DLE {_number(metrics['deep_dle_mm'], 1)} mm"
        )
    fig.suptitle(
        f"{method}\n{_case_title(case)}{metric_line}",
        color="#F6F7FB",
        fontsize=13,
        y=0.995,
    )
    colorbar = fig.colorbar(
        ScalarMappable(Normalize(0.10, 1.0), cmap=ESTIMATE_CMAP),
        ax=axes,
        fraction=0.015,
        pad=0.012,
        aspect=35,
    )
    energy_label = "Simulated source energy" if truth_overlay else "Estimated source energy"
    colorbar.set_label(f"Relative {energy_label.lower()}", color="#ECEEF4", fontsize=9)
    colorbar.ax.tick_params(colors="#ECEEF4", labelsize=8)
    legend = fig.legend(
        handles=_legend_handles(truth_overlay, energy_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3 if truth_overlay else 1,
        frameon=True,
        fontsize=9,
        handletextpad=0.6,
        columnspacing=1.7,
    )
    legend.get_frame().set_facecolor("#151821")
    legend.get_frame().set_edgecolor("#5D6372")
    for text in legend.get_texts():
        text.set_color("#F6F7FB")
    fig.text(
        0.995,
        0.008,
        f"Heat shown >= {relative_threshold:.0%} of source-map peak; "
        + ("green marks identify truth" if truth_overlay else "simulation truth is shown separately"),
        ha="right",
        color="#B8BBC6",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.055, right=0.94, top=0.855, bottom=0.03, wspace=0.025, hspace=0.035)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def render_surface_method(
    method: str,
    estimate: np.ndarray,
    loaded: dict,
    surface: dict,
    output: Path,
    dpi: int = 160,
    *,
    truth_overlay: bool = False,
) -> Path:
    """Render either a cortical estimate or the separate simulation truth."""
    active = np.arange(strict_plot.strict.protocol.ACTIVE_START, estimate.shape[1])
    amplitude = benchmark_metrics.source_amplitude(estimate, active)
    n_surf = int(loaded["geometry"]["n_surf"])
    cortical = amplitude[:n_surf]
    peak = float(cortical.max(initial=0.0))
    relative = cortical / peak if peak > 0 else np.zeros_like(cortical)
    stc = mne.SourceEstimate(
        relative[:, np.newaxis],
        vertices=list(surface["vertices"]),
        tmin=0.0,
        tstep=1.0,
        subject=surface["subject"],
    )
    brain = None
    try:
        brain = stc.plot(
            surface="inflated",
            hemi="split",
            colormap="inferno",
            time_label=None,
            smoothing_steps=10,
            transparent=True,
            subjects_dir=surface["subjects_dir"],
            size=(1200, 800),
            clim={
                "kind": "value",
                "lims": [0.10, 0.55, 1.0],
            },
            background="white",
            foreground="black",
            cortex="classic",
            initial_time=0.0,
            time_viewer=False,
            show_traces=False,
            views=("lateral", "medial"),
            view_layout="horizontal",
            backend="pyvistaqt",
            brain_kwargs={"show": False, "theme": "light"},
        )
        overlays = _surface_truth_overlays(loaded, surface) if truth_overlay else []
        for overlay in overlays:
            label = mne.Label(
                overlay["patch_vertices"],
                hemi=overlay["hemi"],
                name=overlay["name"],
                subject=surface["subject"],
            )
            brain.add_label(label, color=TRUTH_COLOR, alpha=0.9, borders=True)
            brain.add_foci(
                [overlay["center_vertex"]],
                coords_as_verts=True,
                hemi=overlay["hemi"],
                scale_factor=0.7,
                color=TRUTH_COLOR,
                name=f"{overlay['name']}_center",
            )
        image = brain.screenshot(mode="rgb", time_viewer=False)
    finally:
        if brain is not None:
            brain.close()
    case = loaded["case"]
    notes = []
    if truth_overlay and overlays:
        notes.append("Green outline/sphere: simulated cortical patch/center.")
    if not truth_overlay:
        notes.append("Simulation truth is shown separately.")
    if case.get("deep_index") is not None:
        notes.append(
            "No cortical truth; see MRI."
            if truth_overlay and not overlays
            else "Deep sources are not projected to cortex; see MRI."
        )
    fig, ax = plt.subplots(figsize=(12.0, 8.6), facecolor="white")
    ax.imshow(image)
    ax.set_axis_off()
    fig.suptitle(f"{method}\n{_case_title(case)}", fontsize=15, fontweight="bold")
    if notes:
        fig.text(0.5, 0.015, " ".join(notes), ha="center", fontsize=10, color="#202020")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.045)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def render_montage(
    paths: list[tuple[str, Path]], output: Path, dpi: int, title: str
) -> Path:
    columns = min(3, len(paths))
    rows = math.ceil(len(paths) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.4 * columns, 6.0 * rows),
        facecolor="white",
        squeeze=False,
    )
    for ax, (method, path) in zip(axes.ravel(), paths):
        ax.imshow(plt.imread(path))
        ax.set_title(method, fontsize=13, fontweight="bold", pad=5)
        ax.set_axis_off()
    for ax in axes.ravel()[len(paths) :]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.015, hspace=0.04)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_brain_maps(
    manifest_path: Path = strict_plot.strict.DEFAULT_MANIFEST,
    input_root: Path = strict_plot.strict.DEFAULT_INPUT_ROOT,
    data_root: Path | None = None,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
    sisses_root: Path = DEFAULT_SISSES_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    results_root: Path | None = None,
    *,
    case_id: str | None = None,
    case_number: int | None = None,
    eeg_snr_db: int | None = None,
    meg_snr_db: int | None = None,
    scenario: str | None = None,
    location: int | None = None,
    sisses_mode: str = "auto",
    relative_threshold: float = 0.10,
    dpi: int = 160,
    surface_maps: bool = False,
) -> Path:
    if (
        case_id is None
        and case_number is None
        and all(value is None for value in (eeg_snr_db, meg_snr_db, scenario, location))
    ):
        eeg_snr_db, meg_snr_db, scenario, location = DEFAULT_CASE_QUERY
    if not 0 < relative_threshold < 1 or dpi < 72:
        raise ValueError("relative_threshold must be in (0,1) and dpi must be >= 72")
    sisses_root = Path(sisses_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == sisses_root or sisses_root in output_root.parents:
        raise ValueError("output_root must stay outside the preserved SISSES archive")
    loaded = strict_plot.load_strict_case(
        Path(manifest_path),
        Path(input_root),
        Path(data_root or _default_data_root()),
        case_id=case_id,
        case_number=case_number,
        eeg_snr_db=eeg_snr_db,
        meg_snr_db=meg_snr_db,
        scenario=scenario,
        location=location,
    )
    if results_root is None:
        method_order, availability = METHOD_ORDER, []
    else:
        method_order, availability = _completed_method_order(Path(results_root))
    estimates, provenance = reconstruct_all(
        loaded, sisses_root, method_order, sisses_mode=sisses_mode
    )
    rows = evaluate_all(loaded, estimates)
    anatomy = load_anatomy(Path(sample_path))
    surface = (
        load_surface_source_space(Path(sample_path), loaded["geometry"])
        if surface_maps
        else None
    )
    case = loaded["case"]
    case_dir = output_root / f"case_{int(case['case_number']):05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    rendered_surface = []
    by_method = {row["method"]: row for row in rows}
    render_method(
        "Simulated truth",
        loaded["truth"],
        None,
        loaded,
        anatomy,
        case_dir / "simulation_truth_mri.png",
        relative_threshold,
        dpi,
        truth_overlay=True,
    )
    if surface is not None:
        render_surface_method(
            "Simulated truth",
            loaded["truth"],
            loaded,
            surface,
            case_dir / "simulation_truth_surface.png",
            dpi,
            truth_overlay=True,
        )
    for method in estimates:
        path = render_method(
            method,
            estimates[method],
            by_method[method],
            loaded,
            anatomy,
            case_dir / f"{METHOD_SLUGS[method]}.png",
            relative_threshold,
            dpi,
        )
        rendered.append((method, path))
        if surface is not None:
            surface_path = render_surface_method(
                method,
                estimates[method],
                loaded,
                surface,
                case_dir / f"{METHOD_SLUGS[method]}_surface.png",
                dpi,
            )
            rendered_surface.append((method, surface_path))
    render_montage(
        rendered,
        case_dir / "all_methods_brain_mri.png",
        dpi,
        f"Algorithm estimates on MRI slices\n{_case_title(case)}",
    )
    if rendered_surface:
        render_montage(
            rendered_surface,
            case_dir / "all_methods_brain_surface.png",
            dpi,
            f"Algorithm estimates on cortical surfaces\n{_case_title(case)}",
        )
    _write_metrics(case_dir / "metrics.csv", rows)
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "case": case,
                "methods": list(estimates),
                "method_availability": availability,
                "normalization": "per-method active-minus-baseline RMS, normalized to global peak",
                "display_threshold": relative_threshold,
                "algorithm_truth_overlay": "none; simulation truth is stored in separate files",
                "simulation_truth_mri": "simulation_truth_mri.png",
                "simulation_truth_surface": (
                    "simulation_truth_surface.png" if surface_maps else "not requested"
                ),
                "surface_maps": surface_maps,
                "surface_truth_overlay": (
                    "only simulation_truth_surface.png has green cortical truth marks; deep sources are never projected"
                    if surface_maps
                    else "not requested"
                ),
                "surface_normalization": (
                    "per-method cortical RMS amplitude divided by its cortical peak"
                    if surface_maps
                    else "not requested"
                ),
                "surface_rendering": (
                    "inflated cortex; split hemispheres; lateral and medial views; "
                    "classic cortex; inferno colormap; 10 smoothing steps"
                    if surface_maps
                    else "not requested"
                ),
                "anatomy": anatomy["sample_path"],
                "observations": "immutable archived F_EEG/F_MEG; never regenerated",
                "sisses_archive_access": "read-only; output root is explicitly rejected inside archive",
                "case_metrics": "recomputed uniformly with the current benchmark.metrics scorer; primary AUC is auc_tie_corrected (An_auc)",
                "deep_source_anatomy": (
                    "bilateral thalamus (aseg labels 10/49)"
                    if loaded["geometry"].get("deep_aseg_labels") is not None
                    else "legacy non-cortical grid; see SOURCE_SPACE_AUDIT.md"
                ),
                "legacy_auc_note": "the preserved SISSES table used an older parcel-AUC aggregation for multi-source cases; do not substitute that historical auc column for the common-scoring value here",
                **provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--case-id")
    selector.add_argument("--case-number", type=int)
    parser.add_argument("--eeg-snr-db", type=int)
    parser.add_argument("--meg-snr-db", type=int)
    parser.add_argument("--scenario")
    parser.add_argument("--location", type=int)
    parser.add_argument("--manifest", type=Path, default=strict_plot.strict.DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=strict_plot.strict.DEFAULT_INPUT_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--sisses-root", type=Path, default=DEFAULT_SISSES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--sisses-mode", choices=("auto", "require", "skip"), default="auto")
    parser.add_argument("--relative-threshold", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--surface-maps",
        action="store_true",
        help="also render split-hemisphere inflated maps (requires the PyVistaQt backend)",
    )
    args = parser.parse_args()
    output = plot_brain_maps(
        args.manifest,
        args.input_root,
        args.data_root,
        args.sample_path,
        args.sisses_root,
        args.output_root,
        args.results_root,
        case_id=args.case_id,
        case_number=args.case_number,
        eeg_snr_db=args.eeg_snr_db,
        meg_snr_db=args.meg_snr_db,
        scenario=args.scenario,
        location=args.location,
        sisses_mode=args.sisses_mode,
        relative_threshold=args.relative_threshold,
        dpi=args.dpi,
        surface_maps=args.surface_maps,
    )
    print(f"Saved brain maps: {output}")


if __name__ == "__main__":
    main()
