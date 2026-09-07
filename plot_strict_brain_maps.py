"""Render nine strict-benchmark estimates on the same anatomical MRI slices.

The default case is the representative 0/0 dB deep-plus-two-surface case 3124.
SISSES is loaded read-only from its preserved MATLAB output; all other methods
are recomputed from that case's immutable archived EEG/MEG observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import mne
from mne.transforms import apply_trans
import nibabel as nib
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

import plot_strict_case as strict_plot
import run_strict_comparators as comparators
from benchmark import methods as comparator_methods
from benchmark import metrics as benchmark_metrics
from visualization.visualize_spatial_fused_mri import (
    plane_image,
    pos_to_vox,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_NUMBER = 3124
DEFAULT_SISSES_ROOT = Path(r"D:\oaster_strict_blind_sisses")
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "strict_blind" / "brain_maps_v2"
DEFAULT_SAMPLE_PATH = Path(r"D:\mne_data\MNE-sample-data")
METHOD_ORDER = (
    "OASTER V19",
    "SISSES",
    *comparators.METHODS,
)
METHOD_SLUGS = {
    "OASTER V19": "oaster_v19",
    "SISSES": "sisses",
    **comparators.METHOD_SLUGS,
}
PLANES = ("coronal", "sagittal", "axial")
ESTIMATE_CMAP = "magma"
TRUTH_COLOR = "#00E5A8"


def _default_data_root() -> Path:
    """Reuse the geometry path recorded by the completed strict run if present."""
    metadata = ROOT / "results" / "strict_blind" / "oaster_v19_final" / "metadata.json"
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


def reconstruct_all(loaded: dict, sisses_root: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Reconstruct the same observation with OASTER, SISSES and seven comparators."""
    runtime = strict_plot.strict._runtime(loaded["geometry"], loaded["reference"])
    oaster, diagnostics = strict_plot.strict.oaster.reconstruct(
        loaded["eeg"],
        loaded["meg"],
        loaded["gain_eeg"],
        loaded["gain_meg"],
        loaded["geometry"]["n_surf"],
        runtime["kernels"],
    )
    sisses, sisses_path, sisses_metadata = load_sisses_estimate(loaded, sisses_root)
    data, gain = comparator_methods.joint_whiten(
        loaded["eeg"], loaded["meg"], loaded["gain_eeg"], loaded["gain_meg"]
    )
    family = comparator_methods.minimum_norm_family(data, gain)
    active = runtime["active"]
    estimates = {
        "OASTER V19": oaster,
        "SISSES": sisses,
        **family,
        "LCMV": comparator_methods.lcmv(data, gain, active),
        "Dipole fitting (grid)": comparator_methods.dipole_fit(data, gain, active),
        "RAP-MUSIC": comparator_methods.rap_music(data, gain, active),
    }
    if tuple(estimates) != METHOD_ORDER:
        raise RuntimeError(f"nine-method ordering changed: {tuple(estimates)}")
    return estimates, {
        "oaster_diagnostics": {
            "temporal_rank": int(diagnostics["temporal_rank"]),
            "selected_templates": int(diagnostics["selected_templates"]),
        },
        "sisses_path": str(sisses_path.resolve()),
        "sisses": sisses_metadata,
    }


def evaluate_all(loaded: dict, estimates: dict[str, np.ndarray]) -> list[dict]:
    runtime = comparators._runtime(loaded["geometry"], loaded["reference"])
    rows = []
    for method in METHOD_ORDER:
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


def _focuses(loaded: dict) -> list[tuple[str, int]]:
    case = loaded["case"]
    focuses = [
        (f"Surface source {number}", int(index))
        for number, index in enumerate(case.get("surface_centers", []), start=1)
    ]
    if case.get("deep_index") is not None:
        focuses.append(("Deep source", int(case["deep_index"])))
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


def _truth_voxels(loaded: dict, anatomy: dict) -> np.ndarray:
    indices = np.unique(np.concatenate([np.asarray(group, dtype=int) for group in loaded["groups"]]))
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
    truth_voxels: np.ndarray,
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
    near = truth_voxels[np.abs(truth_voxels[:, axis] - index) <= 3.0]
    if near.size:
        points = np.asarray([point_2d(voxel) for voxel in near])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=34,
            facecolors="none",
            edgecolors=TRUTH_COLOR,
            linewidths=1.25,
        )
    focus_xy = point_2d(center_voxel)
    ax.scatter(
        [focus_xy[0]],
        [focus_xy[1]],
        s=70,
        marker="+",
        color=TRUTH_COLOR,
        linewidths=1.8,
    )
    ax.set_axis_off()


def _number(value: object, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "--"


def render_method(
    method: str,
    estimate: np.ndarray,
    metrics: dict,
    loaded: dict,
    anatomy: dict,
    output: Path,
    relative_threshold: float,
    dpi: int,
) -> Path:
    focuses = _focuses(loaded)
    estimate_voxels, estimate_weights = _source_projection(
        estimate, loaded, anatomy, relative_threshold
    )
    truth_voxels = _truth_voxels(loaded, anatomy)
    fig, axes = plt.subplots(
        len(focuses),
        len(PLANES),
        figsize=(12.0, 3.35 * len(focuses) + 1.25),
        facecolor="#090A0F",
        squeeze=False,
    )
    for row, (focus_label, source_index) in enumerate(focuses):
        center = _focus_voxel(source_index, loaded, anatomy)
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
    fig.suptitle(
        f"{method}  |  {case['scenario']}  |  EEG {case['eeg_snr_db']:+d} dB, "
        f"MEG {case['meg_snr_db']:+d} dB\n"
        f"An_auc {_number(metrics['auc_tie_corrected'])}   RMSE {_number(metrics['rmse'])}   "
        f"Surface DLE {_number(metrics['surface_dle_mm'], 1)} mm   "
        f"Deep DLE {_number(metrics['deep_dle_mm'], 1)} mm",
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
    colorbar.set_label("Relative estimated source energy", color="#ECEEF4", fontsize=9)
    colorbar.ax.tick_params(colors="#ECEEF4", labelsize=8)
    fig.text(
        0.995,
        0.008,
        f"green rings/+ = truth; heat = estimate >= {relative_threshold:.0%} of method peak",
        ha="right",
        color="#B8BBC6",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.055, right=0.94, top=0.89, bottom=0.03, wspace=0.025, hspace=0.035)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def render_montage(paths: list[tuple[str, Path]], output: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(19.2, 18.0), facecolor="white")
    for ax, (method, path) in zip(axes.ravel(), paths):
        ax.imshow(plt.imread(path))
        ax.set_title(method, fontsize=13, fontweight="bold", pad=5)
        ax.set_axis_off()
    fig.suptitle(
        "Nine-method source localization on one frozen EEG-MEG observation",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005, wspace=0.015, hspace=0.04)
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
    *,
    case_id: str | None = None,
    case_number: int | None = None,
    relative_threshold: float = 0.10,
    dpi: int = 160,
) -> Path:
    if case_id is None and case_number is None:
        case_number = DEFAULT_CASE_NUMBER
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
    )
    estimates, provenance = reconstruct_all(loaded, sisses_root)
    rows = evaluate_all(loaded, estimates)
    anatomy = load_anatomy(Path(sample_path))
    case = loaded["case"]
    case_dir = output_root / f"case_{int(case['case_number']):05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    by_method = {row["method"]: row for row in rows}
    for method in METHOD_ORDER:
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
    render_montage(rendered, case_dir / "all_methods_brain_mri.png", dpi)
    _write_metrics(case_dir / "metrics.csv", rows)
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "case": case,
                "methods": list(METHOD_ORDER),
                "normalization": "per-method active-minus-baseline RMS, normalized to global peak",
                "display_threshold": relative_threshold,
                "truth_overlay": "green rings and focus plus signs",
                "anatomy": anatomy["sample_path"],
                "observations": "immutable archived F_EEG/F_MEG; never regenerated",
                "sisses_archive_access": "read-only; output root is explicitly rejected inside archive",
                "case_metrics": "recomputed uniformly with the current benchmark.metrics scorer; primary AUC is auc_tie_corrected (An_auc)",
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
    parser.add_argument("--manifest", type=Path, default=strict_plot.strict.DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=strict_plot.strict.DEFAULT_INPUT_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--sisses-root", type=Path, default=DEFAULT_SISSES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--relative-threshold", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    output = plot_brain_maps(
        args.manifest,
        args.input_root,
        args.data_root,
        args.sample_path,
        args.sisses_root,
        args.output_root,
        case_id=args.case_id,
        case_number=args.case_number,
        relative_threshold=args.relative_threshold,
        dpi=args.dpi,
    )
    print(f"Saved nine-method brain maps: {output}")


if __name__ == "__main__":
    main()
