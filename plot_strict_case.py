"""Plot one frozen strict-blind case from archived EEG/MEG observations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import run_strict_oaster as strict
from pipelines.sisses_direct_utils import group_waveform


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "strict_blind" / "figures"
ESTIMATE_KEYS = ("estimate", "source_estimates", "S", "source", "s_wen")


def _select_case(
    cases: list[dict], case_id: str | None, case_number: int | None
) -> dict:
    if (case_id is None) == (case_number is None):
        raise ValueError("specify exactly one of case_id or case_number")
    if case_number is not None:
        if case_number < 0 or case_number >= len(cases):
            raise ValueError(f"case_number must be in [0,{len(cases) - 1}]")
        return cases[case_number]
    matches = [case for case in cases if str(case["case_id"]) == case_id]
    if len(matches) != 1:
        raise ValueError(f"case_id not found: {case_id}")
    return matches[0]


def load_strict_case(
    manifest_path: Path,
    input_root: Path,
    data_root: Path,
    *,
    case_id: str | None = None,
    case_number: int | None = None,
) -> dict:
    """Load one archived observation pair and reconstruct only its source truth."""
    cases, manifest_sha256 = strict._load_manifest(Path(manifest_path))
    case = _select_case(cases, case_id, case_number)
    chunks = strict._discover_chunks(Path(input_root), len(cases))
    number = int(case["case_number"])
    spec = next(chunk for chunk in chunks if chunk.start <= number < chunk.stop)
    geometry = strict._load_geometry(Path(data_root))
    reference = strict._load_chunk(chunks[0].path, observations=False)
    fingerprints = strict._fingerprints(reference)
    strict._validate_chunk(
        chunks[0], reference, cases, manifest_sha256, geometry, fingerprints
    )
    chunk = strict._load_chunk(spec.path, observations=True)
    strict._validate_chunk(
        spec, chunk, cases, manifest_sha256, geometry, fingerprints
    )
    local_index = number - spec.start
    shared = {
        "times": geometry["times"],
        "vertices": geometry["vertices"],
        "adjacency": reference["adjacency"],
        "n_surf": geometry["n_surf"],
        "n_deep": geometry["n_deep"],
        "active_start": strict.protocol.ACTIVE_START,
    }
    # Frozen truth is deterministic scoring metadata.  F_EEG/F_MEG above are
    # the archived observations; simulate_case is deliberately never called.
    truth, groups, truth_meta = strict.protocol.truth_for_case(shared, case)
    return {
        "case": case,
        "chunk": spec,
        "geometry": geometry,
        "reference": reference,
        "shared": shared,
        "truth": truth,
        "groups": groups,
        "truth_meta": truth_meta,
        "eeg": chunk["f_eeg"][:, :, local_index],
        "meg": chunk["f_meg"][:, :, local_index],
        "gain_eeg": chunk["gain_eeg"],
        "gain_meg": chunk["gain_meg"],
    }


def _load_estimate(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        estimate = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as values:
            key = next((name for name in ESTIMATE_KEYS if name in values), None)
            if key is None and len(values.files) == 1:
                key = values.files[0]
            if key is None:
                raise ValueError(
                    f"{path} must contain one of {ESTIMATE_KEYS} or one array"
                )
            estimate = values[key]
    elif suffix == ".mat":
        values = sio.loadmat(path, variable_names=ESTIMATE_KEYS, simplify_cells=True)
        key = next((name for name in ESTIMATE_KEYS if name in values), None)
        if key is None:
            raise ValueError(f"{path} must contain one of {ESTIMATE_KEYS}")
        estimate = values[key]
    else:
        raise ValueError("estimate must be .npy, .npz, or .mat")
    estimate = np.asarray(estimate, dtype=float).squeeze()
    if estimate.shape == expected_shape[::-1]:
        estimate = estimate.T
    if estimate.shape != expected_shape:
        raise ValueError(f"estimate shape {estimate.shape} != {expected_shape}")
    if not np.all(np.isfinite(estimate)):
        raise ValueError("estimate contains NaN or infinity")
    return estimate


def _energy(source: np.ndarray) -> np.ndarray:
    active = np.arange(strict.protocol.ACTIVE_START, source.shape[1])
    return strict.benchmark_metrics.source_amplitude(source, active)


def _location_panel(
    ax,
    vertices: np.ndarray,
    truth_energy: np.ndarray,
    estimate_energy: np.ndarray,
    start: int,
    stop: int,
    title: str,
) -> None:
    indices = np.arange(start, stop)
    points = vertices[indices] * 1000.0
    step = max(1, len(indices) // 1600)
    ax.scatter(
        points[::step, 0],
        points[::step, 1],
        points[::step, 2],
        s=3,
        color="#c8c8c8",
        alpha=0.22,
        depthshade=False,
    )

    truth_peak = float(truth_energy.max(initial=0.0))
    true = indices[truth_energy[indices] > np.finfo(float).eps * truth_peak]
    if true.size:
        weight = truth_energy[true] / max(truth_peak, np.finfo(float).eps)
        ax.scatter(
            *(vertices[true].T * 1000.0),
            s=28 + 80 * weight,
            facecolors="none",
            edgecolors="#009e73",
            linewidths=1.4,
            label="truth energy",
            depthshade=False,
        )

    estimate_peak = float(estimate_energy.max(initial=0.0))
    relative = estimate_energy / max(estimate_peak, np.finfo(float).eps)
    selected = indices[relative[indices] >= 0.10] if estimate_peak > 0 else indices[:0]
    if selected.size:
        ax.scatter(
            *(vertices[selected].T * 1000.0),
            s=30 + 75 * relative[selected],
            c=relative[selected],
            cmap="magma",
            vmin=0.10,
            vmax=1.0,
            marker="x",
            linewidths=1.5,
            label="estimate ≥10% global peak",
            depthshade=False,
        )

    span = np.ptp(points, axis=0)
    ax.set_box_aspect(np.maximum(span, max(float(span.max()), 1.0) * 0.12))
    ax.view_init(elev=22, azim=-62)
    ax.set_title(f"{title}  |  truth={true.size}, estimate={selected.size}", loc="left")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.grid(False)
    if true.size or selected.size:
        ax.legend(loc="upper right", frameon=False, fontsize=8)


def _waveform_panel(
    ax,
    times: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    groups: list[np.ndarray],
    n_surf: int,
) -> None:
    labels = []
    for row, group in enumerate(groups):
        group = np.asarray(group, dtype=int)
        true_wave = group_waveform(truth, group)
        estimate_wave = group_waveform(estimate, group)
        scale = max(
            float(np.max(np.abs(true_wave), initial=0.0)),
            float(np.max(np.abs(estimate_wave), initial=0.0)),
            np.finfo(float).eps,
        )
        offset = 2.4 * row
        ax.plot(
            times,
            true_wave / scale + offset,
            color="#202020",
            linewidth=1.7,
            label="truth" if row == 0 else None,
        )
        ax.plot(
            times,
            estimate_wave / scale + offset,
            color="#d55e00",
            linewidth=1.35,
            linestyle="--",
            label="estimate at truth support" if row == 0 else None,
        )
        layer = "deep" if np.all(group >= n_surf) else "surface"
        labels.append(f"{layer} {sum(label.startswith(layer) for label in labels) + 1}")
    active_time = times[strict.protocol.ACTIVE_START]
    ax.axvspan(times[0], active_time, color="#e8e8e8", alpha=0.65, label="baseline")
    ax.axvline(active_time, color="#777777", linewidth=0.8)
    ax.set_yticks(2.4 * np.arange(len(groups)), labels)
    ax.set_ylim(-1.25, 2.4 * max(len(groups) - 1, 0) + 1.25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized waveform (offset)")
    ax.set_title("Temporal comparison at each true source support", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncols=3, fontsize=8)


def render_case(loaded: dict, estimate: np.ndarray, output: Path, source: str) -> Path:
    truth = loaded["truth"]
    geometry = loaded["geometry"]
    case = loaded["case"]
    n_surf = int(geometry["n_surf"])
    truth_energy = _energy(truth)
    estimate_energy = _energy(estimate)

    fig = plt.figure(figsize=(13.2, 8.6), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 0.75))
    _location_panel(
        fig.add_subplot(grid[0, 0], projection="3d"),
        geometry["vertices"],
        truth_energy,
        estimate_energy,
        0,
        n_surf,
        "A  Surface sources",
    )
    _location_panel(
        fig.add_subplot(grid[0, 1], projection="3d"),
        geometry["vertices"],
        truth_energy,
        estimate_energy,
        n_surf,
        n_surf + int(geometry["n_deep"]),
        "B  Deep sources",
    )
    _waveform_panel(
        fig.add_subplot(grid[1, :]),
        geometry["times"],
        truth,
        estimate,
        loaded["groups"],
        n_surf,
    )
    fig.suptitle(
        f"Strict-blind source localization case {case['case_number']}: {case['case_id']}\n"
        f"{case['scenario']}  |  EEG {case['eeg_snr_db']} dB  |  "
        f"MEG {case['meg_snr_db']} dB  |  estimate: {source}",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.995,
        0.005,
        f"Archived observations: EEG {loaded['eeg'].shape}, MEG {loaded['meg'].shape}; "
        "marker size encodes source energy",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    output = Path(output)
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_case(
    manifest_path: Path = strict.DEFAULT_MANIFEST,
    input_root: Path = strict.DEFAULT_INPUT_ROOT,
    data_root: Path = strict.protocol.DEFAULT_DATA_ROOT,
    *,
    case_id: str | None = None,
    case_number: int | None = None,
    estimate_path: Path | None = None,
    output: Path | None = None,
) -> Path:
    loaded = load_strict_case(
        manifest_path,
        input_root,
        data_root,
        case_id=case_id,
        case_number=case_number,
    )
    expected_shape = loaded["truth"].shape
    if estimate_path is None:
        runtime = strict._runtime(loaded["geometry"], loaded["reference"])
        estimate, diagnostics = strict.oaster.reconstruct(
            loaded["eeg"],
            loaded["meg"],
            loaded["gain_eeg"],
            loaded["gain_meg"],
            loaded["geometry"]["n_surf"],
            runtime["kernels"],
        )
        source = (
            "OASTER rebuilt now "
            f"(rank={diagnostics['temporal_rank']}, templates={diagnostics['selected_templates']})"
        )
    else:
        estimate = _load_estimate(Path(estimate_path), expected_shape)
        source = Path(estimate_path).name
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(loaded["case"]["case_id"]))
    if output is None:
        output = DEFAULT_OUTPUT_ROOT / f"strict_{loaded['case']['case_number']:05d}_{safe_id}.png"
    return render_case(loaded, estimate, output, source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--case-id")
    selector.add_argument("--case-number", type=int)
    parser.add_argument("--manifest", type=Path, default=strict.DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=strict.DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--data-root", type=Path, default=strict.protocol.DEFAULT_DATA_ROOT
    )
    parser.add_argument(
        "--estimate",
        type=Path,
        help="saved .npy/.npz/.mat source estimate; omit to run OASTER",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = plot_case(
        args.manifest,
        args.input_root,
        args.data_root,
        case_id=args.case_id,
        case_number=args.case_number,
        estimate_path=args.estimate,
        output=args.output,
    )
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
