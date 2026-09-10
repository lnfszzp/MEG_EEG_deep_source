"""Preliminary real-data test of OASTER on synchronized ds006035 EEG-MEG.

The dataset has no source ground truth or FreeSurfer derivatives.  This script
therefore uses a rigidly coregistered MNE ``sample`` cortex for a smoke test and
reports anatomical plausibility/repeatability, never AUC or DLE.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.forward import _merge_fwds
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from candidates import oaster_rebuilt as oaster
import protected_multilayer as protected


DEFAULT_DATASET = Path(r"D:\博士\工作＆汇报\源定位\开源数据\v1.0.0")
DEFAULT_SUBJECTS_DIR = Path(r"C:\Users\zzp\mne_data\MNE-sample-data\subjects")
METHODS = (
    "OASTER ERP Joint", "OASTER Joint", "OASTER EEG", "OASTER MAG",
    "dSPM Joint", "eLORETA Joint",
)
COLORS = {
    "OASTER ERP Joint": "#0072B2",
    "OASTER Joint": "#6A3D9A",
    "OASTER EEG": "#009E73",
    "OASTER MAG": "#E69F00",
    "dSPM Joint": "#CC79A7",
    "eLORETA Joint": "#D55E00",
}
TARGET_TIMES = np.r_[np.arange(-250, -50), np.arange(15, 46)].astype(float) / 1000.0
N20 = (TARGET_TIMES >= 0.018) & (TARGET_TIMES <= 0.024)
P30 = (TARGET_TIMES >= 0.028) & (TARGET_TIMES <= 0.040)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subject", default="sm09")
    parser.add_argument("--runs", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--spacing", default="ico4")
    parser.add_argument("--skip-brain-render", action="store_true")
    parser.add_argument("--rerender-saved", action="store_true")
    parser.add_argument("--recompute-saved-metrics", action="store_true")
    return parser.parse_args()


def read_somatosensory_events(path: Path, first_samp: int) -> np.ndarray:
    """Read BIDS-relative samples and return MNE absolute sample indices."""
    events = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("trial_type") == "somatosensory" or row.get("value") == "32":
                events.append((int(float(row["sample"])) + int(first_samp), 0, 1))
    if not events:
        raise ValueError(f"no somatosensory events in {path}")
    return np.asarray(events, dtype=int)


def _paths(dataset: Path, subject: str, run: int) -> tuple[Path, Path]:
    stem = f"sub-{subject}_ses-meeg_task-somatomotor_run-{run}"
    folder = dataset / f"sub-{subject}" / "ses-meeg" / "meg"
    return folder / f"{stem}_meg.fif", folder / f"{stem}_events.tsv"


def preprocess_run(raw_path: Path, events_path: Path) -> tuple:
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
    first_samp = int(raw.first_samp)
    bads = tuple(raw.info["bads"])
    picks = mne.pick_types(
        raw.info, meg="mag", eeg=True, eog=False, ecg=False, stim=False,
        ref_meg=False, exclude="bads",
    )
    raw.pick(picks).load_data(verbose=False)
    line_frequency = float(raw.info.get("line_freq") or 60.0)
    if line_frequency < raw.info["sfreq"] / 2.0:
        raw.notch_filter([line_frequency], n_jobs=1, verbose=False)
    raw.filter(1.0, 100.0, n_jobs=1, verbose=False)
    events = read_somatosensory_events(events_path, first_samp)
    epoch_kwargs = dict(
        raw=raw,
        events=events,
        event_id={"somatosensory": 1},
        tmin=-0.4,
        tmax=0.1,
        baseline=(-0.4, -0.05),
        reject_tmin=-0.2,
        reject_tmax=0.08,
        preload=True,
        proj=False,
        event_repeated="drop",
        verbose=False,
    )
    probe = mne.Epochs(**epoch_kwargs)
    eeg_probe = mne.pick_types(probe.info, meg=False, eeg=True, exclude=[])
    reject_window = (probe.times >= -0.35) & (probe.times <= -0.05)
    median_ptp = np.median(
        np.ptp(probe.get_data(copy=False)[:, eeg_probe][:, :, reject_window], axis=2),
        axis=0,
    )
    center = float(np.median(median_ptp))
    mad = float(np.median(np.abs(median_ptp - center)))
    cutoff = max(100e-6, center + 6.0 * 1.4826 * mad)
    auto_bads = [
        probe.ch_names[index] for index, value in zip(eeg_probe, median_ptp) if value > cutoff
    ]
    if auto_bads:
        raw.drop_channels(auto_bads)
    epochs = mne.Epochs(**{**epoch_kwargs, "raw": raw, "reject": {"eeg": 200e-6, "mag": 6e-12}})
    if len(epochs) < 15:
        raise RuntimeError(f"only {len(epochs)} usable epochs in {raw_path.name}")
    epochs.set_eeg_reference("average", projection=True, verbose=False)
    noise_cov = mne.compute_covariance(
        epochs, tmin=-0.4, tmax=-0.05, method="shrunk", rank="info", verbose=False
    )
    evoked = epochs.average()
    data = np.vstack([np.interp(TARGET_TIMES, evoked.times, row) for row in evoked.data])
    noise_epochs = epochs.get_data(tmin=-0.4, tmax=-0.05, copy=True, verbose=False)
    noise = noise_epochs.transpose(1, 0, 2).reshape(len(evoked.ch_names), -1)
    eeg = mne.pick_types(evoked.info, meg=False, eeg=True, exclude=[])
    mag = mne.pick_types(evoked.info, meg="mag", eeg=False, exclude=[])

    def window_snr(indices: np.ndarray) -> float:
        active = np.sqrt(np.mean(data[indices][:, N20] ** 2))
        baseline = np.sqrt(np.mean(data[indices, :200] ** 2))
        return float(active / max(baseline, np.finfo(float).eps))

    return data, noise, evoked, noise_cov, {
        "epochs": len(epochs),
        "events": len(events),
        "first_samp": first_samp,
        "bad_channels": list(bads),
        "auto_bad_channels": auto_bads,
        "auto_bad_eeg_ptp_cutoff_uv": cutoff * 1e6,
        "noise_samples": noise.shape[1],
        "eeg_n20_sensor_snr": window_snr(eeg),
        "mag_n20_sensor_snr": window_snr(mag),
    }


def coregister(info: mne.Info, subjects_dir: Path) -> tuple[mne.transforms.Transform, dict]:
    coreg = mne.coreg.Coregistration(
        info, "sample", subjects_dir=subjects_dir, fiducials="estimated"
    )
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=20, nasion_weight=2.0, verbose=False)
    coreg.omit_head_shape_points(distance=0.02)
    coreg.fit_icp(n_iterations=30, nasion_weight=2.0, verbose=False)
    distances = np.asarray(coreg.compute_dig_mri_distances(), dtype=float) * 1000.0
    return coreg.trans, {
        "mean_mm": float(np.mean(distances)),
        "median_mm": float(np.median(distances)),
        "p95_mm": float(np.percentile(distances, 95)),
        "max_mm": float(np.max(distances)),
    }


def make_fixed_forwards(
    info: mne.Info,
    trans: mne.transforms.Transform,
    src: mne.SourceSpaces,
    subjects_dir: Path,
) -> tuple[mne.Forward, mne.Forward, mne.Forward, mne.Forward]:
    bem_dir = subjects_dir / "sample" / "bem"
    specifications = (
        ("eeg", bem_dir / "sample-5120-5120-5120-bem-sol.fif"),
        ("mag", bem_dir / "sample-5120-bem-sol.fif"),
    )
    fixed_forwards, free_forwards = [], []
    for modality, bem in specifications:
        picks = mne.pick_types(
            info, eeg=modality == "eeg", meg="mag" if modality == "mag" else False,
            ref_meg=False, exclude="bads",
        )
        modality_info = mne.pick_info(info, picks, copy=True)
        forward = mne.make_forward_solution(
            modality_info,
            trans,
            src,
            bem,
            meg=modality == "mag",
            eeg=modality == "eeg",
            mindist=5.0,
            n_jobs=1,
            on_inside="raise",
            verbose=False,
        )
        free = mne.convert_forward_solution(
            forward, surf_ori=True, force_fixed=False, use_cps=True, verbose=False
        )
        free_forwards.append(free)
        fixed_forwards.append(mne.convert_forward_solution(
            free, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
        ))
    eeg_fwd, mag_fwd = fixed_forwards
    for hemi in range(2):
        if not np.array_equal(eeg_fwd["src"][hemi]["vertno"], mag_fwd["src"][hemi]["vertno"]):
            raise RuntimeError("EEG and MAG forwards do not share cortical vertices")
    return eeg_fwd, mag_fwd, free_forwards[0], free_forwards[1]


def _aligned_modality(
    data: np.ndarray, noise: np.ndarray, info: mne.Info, fwd: mne.Forward, *, eeg: bool
) -> tuple:
    order = np.asarray([info["ch_names"].index(name) for name in fwd["info"]["ch_names"]])
    selected = data[order].copy()
    selected_noise = noise[order].copy()
    gain = np.asarray(fwd["sol"]["data"], dtype=float).copy()
    model_info = mne.pick_info(info, order, copy=True)
    if model_info["projs"]:
        projector = mne.EvokedArray(np.eye(len(order)), model_info, tmin=0.0, verbose=False)
        projector.apply_proj(verbose=False)
        selected = projector.data @ selected
        selected_noise = projector.data @ selected_noise
        gain = projector.data @ gain
    if eeg:
        selected -= selected.mean(axis=0, keepdims=True)
        selected_noise -= selected_noise.mean(axis=0, keepdims=True)
        gain -= gain.mean(axis=0, keepdims=True)
    return selected, gain, selected_noise


def aligned_data_and_gain(
    data: np.ndarray,
    noise: np.ndarray,
    info: mne.Info,
    forwards: tuple[mne.Forward, mne.Forward],
) -> tuple:
    eeg, ge, eeg_noise = _aligned_modality(data, noise, info, forwards[0], eeg=True)
    mag, gm, mag_noise = _aligned_modality(data, noise, info, forwards[1], eeg=False)
    return eeg, mag, ge, gm, eeg_noise, mag_noise


def official_mne_minimum_norm(
    evoked: mne.Evoked,
    noise_cov: mne.Covariance,
    forwards: tuple[mne.Forward, mne.Forward, mne.Forward, mne.Forward],
) -> dict[str, np.ndarray]:
    joint = _merge_fwds(
        {"meg": forwards[3].copy(), "eeg": forwards[2].copy()}, verbose=False
    )
    row_names = joint["sol"]["row_names"]
    ordered = evoked.copy().reorder_channels(row_names)
    joint["info"] = ordered.info
    covariance = mne.pick_channels_cov(
        noise_cov, include=row_names, exclude=[], ordered=True, copy=True, verbose=False
    )
    inverse = mne.minimum_norm.make_inverse_operator(
        ordered.info,
        joint,
        covariance,
        loose=0.0,
        fixed=True,
        depth=0.8,
        rank="info",
        use_cps=True,
        verbose=False,
    )
    estimates = {}
    for method in ("dSPM", "eLORETA"):
        stc = mne.minimum_norm.apply_inverse(
            ordered,
            inverse,
            lambda2=1.0 / 9.0,
            method=method,
            use_cps=True,
            verbose=False,
        )
        estimates[f"{method} Joint"] = np.vstack([
            np.interp(TARGET_TIMES, stc.times, row) for row in stc.data
        ])
    return estimates


def whiten_from_trials(
    data: np.ndarray, gain: np.ndarray, noise: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    whitener = protected.whitening_matrix(noise, noise.shape[1])
    return whitener @ data, whitener @ gain


def solve_methods(
    eeg: np.ndarray,
    mag: np.ndarray,
    ge: np.ndarray,
    gm: np.ndarray,
    eeg_noise: np.ndarray,
    mag_noise: np.ndarray,
    kernels,
    mne_estimates: dict[str, np.ndarray],
):
    sources = ge.shape[1]
    eeg_white, eeg_gain = whiten_from_trials(eeg, ge, eeg_noise)
    mag_white, mag_gain = whiten_from_trials(mag, gm, mag_noise)
    joint = np.vstack((eeg_white, mag_white))
    joint_gain = np.vstack((eeg_gain, mag_gain))
    erp_baseline = np.arange(joint.shape[1]) < 200
    joint_erp, joint_erp_diag = oaster.reconstruct_evoked_oaster_from_whitened(
        joint,
        joint_gain,
        sources,
        kernels,
        baseline=erp_baseline,
        active_windows=(N20, P30),
    )
    joint_oaster, joint_diag = oaster.reconstruct_from_whitened(
        joint, joint_gain, sources, kernels
    )
    eeg_oaster, eeg_diag = oaster.reconstruct_from_whitened(
        eeg_white, eeg_gain, sources, kernels
    )
    mag_oaster, mag_diag = oaster.reconstruct_from_whitened(
        mag_white, mag_gain, sources, kernels
    )
    estimates = {
        "OASTER ERP Joint": joint_erp,
        "OASTER Joint": joint_oaster,
        "OASTER EEG": eeg_oaster,
        "OASTER MAG": mag_oaster,
        **mne_estimates,
    }
    return estimates, {
        "OASTER ERP Joint": joint_erp_diag,
        "OASTER Joint": joint_diag,
        "OASTER EEG": eeg_diag,
        "OASTER MAG": mag_diag,
    }


def label_geometry(src: mne.SourceSpaces, subjects_dir: Path) -> dict:
    vertices = [np.asarray(space["vertno"], dtype=int) for space in src]
    xyz = np.vstack([space["rr"][space["vertno"]] for space in src])
    names = np.full(xyz.shape[0], "unknown", dtype=object)
    left = np.zeros(xyz.shape[0], dtype=bool)
    right = np.zeros(xyz.shape[0], dtype=bool)
    offset = (0, len(vertices[0]))
    for label in mne.read_labels_from_annot(
        "sample", parc="aparc", subjects_dir=subjects_dir, verbose=False
    ):
        hemi = 0 if label.hemi == "lh" else 1
        local = np.flatnonzero(np.isin(vertices[hemi], label.vertices)) + offset[hemi]
        names[local] = label.name
        if label.name == "postcentral-lh":
            left[local] = True
        elif label.name == "postcentral-rh":
            right[local] = True
    if not left.any() or not right.any():
        raise RuntimeError("sample aparc postcentral labels were not found")
    return {"vertices": vertices, "xyz": xyz, "names": names, "left": left, "right": right}


def _amplitude_metrics(amplitude: np.ndarray, geometry: dict, prefix: str) -> dict:
    power = amplitude**2
    total = float(power.sum())
    left_power = float(power[geometry["left"]].sum())
    right_power = float(power[geometry["right"]].sum())
    left_density = float(power[geometry["left"]].mean())
    right_density = float(power[geometry["right"]].mean())
    roi_total = left_density + right_density
    peak = int(np.argmax(amplitude))
    xyz = geometry["xyz"]
    distance = np.linalg.norm(xyz[geometry["left"]] - xyz[peak], axis=1).min() * 1000.0
    l1, l2, n = float(amplitude.sum()), float(np.linalg.norm(amplitude)), amplitude.size
    sparsity = (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1.0) if l2 > 0 else 0.0
    peak_amplitude = float(amplitude.max(initial=0.0))
    return {
        f"{prefix}_left_s1_mass_pct": 100.0 * left_power / total if total > 0 else 0.0,
        f"{prefix}_left_s1_enrichment": ((left_power / total) / float(np.mean(geometry["left"]))) if total > 0 else 0.0,
        f"{prefix}_postcentral_laterality": (left_density - right_density) / roi_total if roi_total > 0 else 0.0,
        f"{prefix}_peak_euclidean_distance_to_left_s1_mm": float(distance),
        f"{prefix}_peak_label": str(geometry["names"][peak]),
        f"{prefix}_peak_hemi": "left" if peak < len(geometry["vertices"][0]) else "right",
        f"{prefix}_hoyer_sparsity": float(sparsity),
        f"{prefix}_vertices_above_50pct": int(np.sum(amplitude >= 0.50 * peak_amplitude)) if peak_amplitude > 0 else 0,
    }


def _window_metrics(source: np.ndarray, window: np.ndarray, geometry: dict, prefix: str) -> tuple[dict, np.ndarray]:
    baseline_power = np.mean(source[:, :200] ** 2, axis=1)
    active_power = np.mean(source[:, window] ** 2, axis=1)
    amplitude = np.sqrt(np.maximum(active_power - baseline_power, 0.0))
    return _amplitude_metrics(amplitude, geometry, prefix), amplitude


def source_metrics(source: np.ndarray, geometry: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    n20_metrics, n20_map = _window_metrics(source, N20, geometry, "n20")
    p30_metrics, p30_map = _window_metrics(source, P30, geometry, "p30")
    return {**n20_metrics, **p30_metrics}, n20_map, p30_map


def _mean_pairwise(maps: list[np.ndarray], xyz: np.ndarray) -> dict:
    correlations, dice, peak_distances = [], [], []
    for first, second in combinations(maps, 2):
        correlations.append(float(spearmanr(first, second).statistic))
        count = max(1, int(np.ceil(0.05 * first.size)))
        a = set(np.argpartition(first, -count)[-count:])
        b = set(np.argpartition(second, -count)[-count:])
        dice.append(2.0 * len(a & b) / (len(a) + len(b)))
        peak_distances.append(float(np.linalg.norm(xyz[np.argmax(first)] - xyz[np.argmax(second)]) * 1000.0))
    return {
        "run_map_spearman_mean": float(np.mean(correlations)) if correlations else np.nan,
        "run_top5pct_dice_mean": float(np.mean(dice)) if dice else np.nan,
        "run_peak_distance_mm_mean": float(np.mean(peak_distances)) if peak_distances else np.nan,
    }


def summarize(rows: list[dict], maps: dict, xyz: np.ndarray) -> list[dict]:
    numeric = tuple(
        f"{window}_{field}"
        for window in ("n20", "p30")
        for field in (
            "left_s1_mass_pct", "left_s1_enrichment", "postcentral_laterality",
            "peak_euclidean_distance_to_left_s1_mm", "hoyer_sparsity",
            "vertices_above_50pct",
        )
    )
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        item = {"method": method, "runs": len(selected)}
        for field in numeric:
            values = np.asarray([float(row[field]) for row in selected])
            item[f"{field}_mean"] = float(values.mean())
            item[f"{field}_median"] = float(np.median(values))
            item[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        for window in ("n20", "p30"):
            item.update({
                f"{window}_{key}": value
                for key, value in _mean_pairwise(maps[window][method], xyz).items()
            })
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows: list[dict], output: Path, subject: str) -> None:
    panels = (
        ("n20_left_s1_enrichment", "N20 left-S1 enrichment (× uniform-source null)"),
        ("p30_left_s1_enrichment", "P30/P35 left-S1 enrichment (× uniform-source null)"),
        ("n20_peak_euclidean_distance_to_left_s1_mm", "N20 peak-to-left-S1 Euclidean distance (mm)"),
        ("p30_peak_euclidean_distance_to_left_s1_mm", "P30/P35 peak-to-left-S1 Euclidean distance (mm)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="white")
    x = np.arange(len(METHODS))
    runs = sorted({int(row["run"]) for row in rows})
    for ax, (field, title) in zip(axes.ravel(), panels):
        for run in runs:
            values = [float(next(row[field] for row in rows if row["method"] == method and int(row["run"]) == run)) for method in METHODS]
            ax.plot(x, values, color="#B8C2CC", linewidth=1.2, alpha=0.7, zorder=1)
        for index, method in enumerate(METHODS):
            values = [float(row[field]) for row in rows if row["method"] == method]
            ax.scatter(np.full(len(values), index), values, s=55, color=COLORS[method], edgecolor="white", linewidth=0.8, zorder=3)
            ax.plot(index, np.mean(values), marker="_", markersize=22, markeredgewidth=3, color="#202A35", zorder=4)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, METHODS, rotation=22, ha="right")
        ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"sub-{subject} synchronized EEG–MEG | fixed N20 and P30/P35 windows\nEach grey line is one run; template-sample preliminary model", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_brains(
    maps: dict[str, list[np.ndarray]], geometry: dict, subjects_dir: Path,
    output: Path, subject: str, window_label: str,
    surface: str = "pial", view: str = "lateral",
) -> None:
    images = []
    for method in METHODS:
        normalized = [values / max(values.max(initial=0.0), np.finfo(float).eps) for values in maps[method]]
        values = np.mean(normalized, axis=0)
        floor = float(np.percentile(values, 95))
        ceiling = max(float(values.max(initial=0.0)), np.finfo(float).eps)
        if ceiling <= floor:
            floor = 0.0
        displayed = np.where(values >= floor, values, 0.0)
        stc = mne.SourceEstimate(
            displayed[:, None], geometry["vertices"], tmin=0.0, tstep=1.0, subject="sample"
        )
        brain = None
        try:
            brain = stc.plot(
                surface=surface, hemi="split", views=view, colormap="inferno",
                time_label=None, smoothing_steps=5, transparent=True,
                subjects_dir=subjects_dir, size=(1000, 520),
                clim={"kind": "value", "lims": [floor, 0.5 * (floor + ceiling), ceiling]},
                background="white", foreground="black", cortex="classic",
                initial_time=0.0, time_viewer=False, show_traces=False,
                colorbar=False,
                backend="pyvistaqt", brain_kwargs={"show": False, "theme": "light"},
            )
            images.append((method, brain.screenshot(mode="rgb", time_viewer=False)))
        finally:
            if brain is not None:
                brain.close()
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 13), facecolor="white", constrained_layout=True
    )
    for ax, (method, image) in zip(axes.ravel(), images):
        ax.imshow(image)
        ax.set_title(f"{method} | estimate only", fontweight="bold")
        ax.set_axis_off()
    for ax in axes.ravel()[len(images):]:
        ax.set_visible(False)
    fig.suptitle(
        f"sub-{subject} | {window_label} | mean of run-normalized excess-power maps\n"
        f"Rendered {surface} anatomy ({view}); display ≥ P95; each map independently normalized",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def render_all_brains(
    maps: dict[str, dict[str, list[np.ndarray]]], geometry: dict,
    subjects_dir: Path, output: Path, subject: str,
) -> None:
    for window, label in (("n20", "18–24 ms N20/N20m"), ("p30", "28–40 ms P30/P35")):
        render_brains(
            maps[window], geometry, subjects_dir,
            output / f"{window}_brain_maps.png", subject, label,
        )
        render_brains(
            maps[window], geometry, subjects_dir,
            output / f"{window}_brain_maps_inflated_dorsal.png", subject, label,
            surface="inflated", view="dorsal",
        )


def load_saved_maps(output: Path) -> tuple[list[np.ndarray], dict, list[int]]:
    with np.load(output / "source_maps.npz") as archive:
        vertices = [archive["vertices_lh"].copy(), archive["vertices_rh"].copy()]
        maps = {
            window: {
                method: list(archive[f"{window}_{method.lower().replace(' ', '_').replace('-', '_')}"])
                for method in METHODS
            }
            for window in ("n20", "p30")
        }
        runs = list(map(int, archive["runs"]))
    return vertices, maps, runs


def rerender_saved_brains(output: Path, subjects_dir: Path, subject: str) -> None:
    vertices, maps, _ = load_saved_maps(output)
    geometry = {"vertices": vertices}
    render_all_brains(maps, geometry, subjects_dir, output, subject)


def recompute_saved_metrics(
    output: Path, subjects_dir: Path, subject: str, spacing: str,
) -> None:
    vertices, maps, runs = load_saved_maps(output)
    full_src = mne.setup_source_space(
        "sample", spacing=spacing, subjects_dir=subjects_dir,
        add_dist=False, n_jobs=1, verbose=False,
    )
    geometry = label_geometry([
        {"vertno": vertices[hemi], "rr": full_src[hemi]["rr"]} for hemi in range(2)
    ], subjects_dir)
    with (output / "run_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    run_index = {run: index for index, run in enumerate(runs)}
    for row in rows:
        index = run_index[int(row["run"])]
        method = row["method"]
        row.update(_amplitude_metrics(maps["n20"][method][index], geometry, "n20"))
        row.update(_amplitude_metrics(maps["p30"][method][index], geometry, "p30"))
    summary = summarize(rows, maps, geometry["xyz"])
    write_csv(output / "run_metrics.csv", rows)
    write_csv(output / "method_comparison.csv", summary)
    plot_comparison(rows, output / "method_metrics.png", subject)
    coreg = {
        field: float(rows[0][field]) for field in ("mean_mm", "median_mm", "p95_mm", "max_mm")
    }
    write_report(output / "REPORT.md", subject, rows, summary, coreg, runs)


def write_report(path: Path, subject: str, rows: list[dict], summary: list[dict], coreg: dict, runs: list[int]) -> None:
    by_method = {row["method"]: row for row in summary}
    erp = by_method["OASTER ERP Joint"]
    spectral = by_method["OASTER Joint"]
    lines = [
        f"# ds006035 sub-{subject} 同步 EEG–MEG 初步结果",
        "",
        f"分析了 run {', '.join(map(str, runs))}。OASTER-ERP 对 N20、P30 分别建立多尺度时间基并用 EBIC 选择空间模板，最后对完整有符号 ERP 回归；原频谱 OASTER 仍使用 −250 至 −51 ms 噪声段和 15–45 ms 频谱证据。主指标固定为 18–24 ms（N20/N20m）。",
        "这是 MNE sample 模板脑 + 自动刚性配准的流程验证，不是个体 MRI 最终结果；数据没有源真值，因此不计算 AUC/DLE。",
        "",
        f"配准点到模板头表面：均值 {coreg['mean_mm']:.2f} mm，中位数 {coreg['median_mm']:.2f} mm，P95 {coreg['p95_mm']:.2f} mm。",
        "",
        "| 方法 | N20 左S1富集 | N20侧化 | N20欧氏峰距(mm) | P30左S1富集 | P30欧氏峰距(mm) | N20 run相关 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_method[method]
        lines.append(
            f"| {method} | {row['n20_left_s1_enrichment_mean']:.2f} | "
            f"{row['n20_postcentral_laterality_mean']:.3f} | "
            f"{row['n20_peak_euclidean_distance_to_left_s1_mm_mean']:.2f} | "
            f"{row['p30_left_s1_enrichment_mean']:.2f} | "
            f"{row['p30_peak_euclidean_distance_to_left_s1_mm_mean']:.2f} | "
            f"{row['n20_run_map_spearman_mean']:.3f} |"
        )
    erp_interpretation = (
        f"OASTER ERP Joint 的 N20 左 S1 富集为 {erp['n20_left_s1_enrichment_mean']:.2f}（1 代表均匀源点零假设），"
        + (
            "没有显示可信的 S1 富集。"
            if erp["n20_left_s1_enrichment_mean"] < 1.0
            else "显示出高于均匀源点零假设的 S1 富集。"
        )
    )
    lines += [
        "",
        "## 初步判断",
        "",
        erp_interpretation,
        f"原频谱 OASTER 的 N20/P30 富集分别为 {spectral['n20_left_s1_enrichment_mean']:.2f}/"
        f"{spectral['p30_left_s1_enrichment_mean']:.2f}；官方 dSPM 为 {by_method['dSPM Joint']['n20_left_s1_enrichment_mean']:.2f}/"
        f"{by_method['dSPM Joint']['p30_left_s1_enrichment_mean']:.2f}。",
        "是否真正具有同步融合优势，应以全部 5 名受试者、个体 FreeSurfer/BEM 和留一 run 验证为准。",
        "run 不是独立受试者，本结果不做显著性检验。",
        "",
        "脑图只显示 P95 以上；表中指标使用未阈值化的 `sqrt(max(响应功率−基线功率, 0))` 图。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _arguments()
    args.output = args.output or ROOT / "results" / "real_data" / "ds006035" / f"sub-{args.subject}_final"
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.subjects_dir / "sample").is_dir():
        raise FileNotFoundError(f"MNE sample subject not found: {args.subjects_dir}")
    mne.set_log_level("WARNING")
    if args.rerender_saved:
        rerender_saved_brains(args.output, args.subjects_dir, args.subject)
        print(f"re-rendered: {args.output}", flush=True)
        return
    if args.recompute_saved_metrics:
        recompute_saved_metrics(args.output, args.subjects_dir, args.subject, args.spacing)
        print(f"recomputed metrics: {args.output}", flush=True)
        return
    raw_paths = [_paths(args.dataset, args.subject, run) for run in args.runs]
    missing = [str(path) for pair in raw_paths for path in pair if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing ds006035 files:\n" + "\n".join(missing))
    first_data, first_noise, first_evoked, first_cov, first_qc = preprocess_run(*raw_paths[0])
    first_info = first_evoked.info
    trans, coreg = coregister(first_info, args.subjects_dir)
    mne.write_trans(args.output / f"sub-{args.subject}_to_sample-trans.fif", trans, overwrite=True)
    src = mne.setup_source_space(
        "sample", spacing=args.spacing, subjects_dir=args.subjects_dir,
        add_dist=False, n_jobs=1, verbose=False,
    )
    first_forwards = make_fixed_forwards(first_info, trans, src, args.subjects_dir)
    analysis_src = first_forwards[0]["src"]
    geometry = label_geometry(analysis_src, args.subjects_dir)
    adjacency = mne.spatial_src_adjacency(analysis_src, verbose=False).toarray()
    kernels = protected.connected_euclidean_surface_kernels(
        geometry["xyz"], adjacency, len(geometry["xyz"])
    )

    rows, diagnostics = [], {"coregistration": coreg, "runs": {}}
    maps = {
        window: {method: [] for method in METHODS} for window in ("n20", "p30")
    }
    prepared = {
        args.runs[0]: (
            first_data, first_noise, first_evoked, first_cov, first_qc, first_forwards
        )
    }
    for run, paths in zip(args.runs, raw_paths):
        if run in prepared:
            data, noise, evoked, noise_cov, qc, forwards = prepared[run]
        else:
            data, noise, evoked, noise_cov, qc = preprocess_run(*paths)
            info = evoked.info
            forwards = make_fixed_forwards(info, trans, src, args.subjects_dir)
        info = evoked.info
        print(f"sub-{args.subject} run-{run}: {qc['epochs']}/{qc['events']} epochs; solving", flush=True)
        if forwards[0]["nsource"] != len(geometry["xyz"]):
            raise RuntimeError(f"forward source count changed: {forwards[0]['nsource']}")
        for hemi in range(2):
            if not np.array_equal(forwards[0]["src"][hemi]["vertno"], analysis_src[hemi]["vertno"]):
                raise RuntimeError("run forward does not share the analysis source grid")
        eeg, mag, ge, gm, eeg_noise, mag_noise = aligned_data_and_gain(
            data, noise, info, forwards
        )
        estimates, run_diagnostics = solve_methods(
            eeg,
            mag,
            ge,
            gm,
            eeg_noise,
            mag_noise,
            kernels,
            official_mne_minimum_norm(evoked, noise_cov, forwards),
        )
        diagnostics["runs"][str(run)] = {"qc": qc, "oaster": run_diagnostics}
        for method in METHODS:
            metric, n20_map, p30_map = source_metrics(estimates[method], geometry)
            rows.append({"subject": args.subject, "run": run, "method": method, **qc, **coreg, **metric})
            maps["n20"][method].append(n20_map)
            maps["p30"][method].append(p30_map)
            check_total = float(np.sum(n20_map**2))
            check = (
                100.0 * np.sum(n20_map[geometry["left"]] ** 2) / check_total
                if check_total > 0 else 0.0
            )
            if not np.isclose(check, metric["n20_left_s1_mass_pct"], rtol=1e-10, atol=1e-12):
                raise RuntimeError("saved N20 map and metric disagree")
            print(f"  {method}: N20 enrichment={metric['n20_left_s1_enrichment']:.2f} "
                  f"distance={metric['n20_peak_euclidean_distance_to_left_s1_mm']:.1f} mm; "
                  f"P30 enrichment={metric['p30_left_s1_enrichment']:.2f}", flush=True)

    summary = summarize(rows, maps, geometry["xyz"])
    write_csv(args.output / "run_metrics.csv", rows)
    write_csv(args.output / "method_comparison.csv", summary)
    map_arrays = {
        f"{window}_{method.lower().replace(' ', '_').replace('-', '_')}": np.asarray(maps[window][method])
        for window in ("n20", "p30") for method in METHODS
    }
    np.savez_compressed(
        args.output / "source_maps.npz",
        methods=np.asarray(METHODS), runs=np.asarray(args.runs),
        vertices_lh=geometry["vertices"][0], vertices_rh=geometry["vertices"][1],
        **map_arrays,
    )
    (args.output / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=lambda value: value.item()),
        encoding="utf-8",
    )
    plot_comparison(rows, args.output / "method_metrics.png", args.subject)
    if not args.skip_brain_render:
        try:
            render_all_brains(maps, geometry, args.subjects_dir, args.output, args.subject)
        except Exception as error:  # plotting must not discard completed numerical results
            (args.output / "brain_render_error.txt").write_text(repr(error), encoding="utf-8")
            print(f"brain render failed: {error!r}", file=sys.stderr)
    write_report(args.output / "REPORT.md", args.subject, rows, summary, coreg, list(args.runs))
    print(f"complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
