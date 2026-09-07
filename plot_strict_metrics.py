"""Create publication-style comparisons for all nine strict-blind methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SNR_LEVELS = (-10, -5, 0, 5, 10, 15, 20)
METHODS = (
    "OASTER",
    "SISSES",
    "dSPM",
    "eLORETA",
    "sLORETA",
    "MNE",
    "LCMV",
    "RAP-MUSIC",
    "Dipole fitting (grid)",
)
DISPLAY = {"Dipole fitting (grid)": "Dipole grid"}
# Paul Tol's colour-blind-safe muted palette; symbols and labels also encode method.
COLORS = dict(
    zip(
        METHODS,
        (
            "#332288",
            "#88CCEE",
            "#44AA99",
            "#117733",
            "#999933",
            "#DDCC77",
            "#CC6677",
            "#882255",
            "#AA4499",
        ),
        strict=True,
    )
)
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "v", "P", "X", "<", ">"), strict=True))
METRICS = {
    "auc_tie_corrected": ("An_auc", True, "score"),
    "auc": ("Parcel AUC", True, "score"),
    "rmse": ("RMSE", False, "error"),
    "surface_sd_mm_penalized": ("Surface SD\npenalized (mm)", False, "distance"),
    "surface_dle_mm_penalized": ("Surface DLE\npenalized (mm)", False, "distance"),
    "deep_sd_mm_penalized": ("Deep SD\npenalized (mm)", False, "distance"),
    "deep_dle_mm_penalized": ("Deep DLE\npenalized (mm)", False, "distance"),
    "deep_sensitivity": ("Deep sensitivity", True, "detection"),
    "deep_specificity": ("Deep specificity", True, "detection"),
    "deep_balanced_accuracy": ("Deep balanced\naccuracy", True, "detection"),
}


def _name(method: str) -> str:
    return DISPLAY.get(method, method)


def _read(path: Path, fixed_method: str | None) -> list[tuple[str, dict[str, float]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"eeg_snr_db", "meg_snr_db", *METRICS}
        if fixed_method is None:
            required.add("method")
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        parsed = []
        for line, row in enumerate(reader, 2):
            method = fixed_method or row["method"].strip()
            if method not in METHODS:
                raise ValueError(f"{path}:{line}: unexpected method {method!r}")
            try:
                values = {key: float(row[key]) for key in METRICS}
                values["eeg_snr_db"] = float(row["eeg_snr_db"])
                values["meg_snr_db"] = float(row["meg_snr_db"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid numeric value") from exc
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"{path}:{line}: non-finite value")
            parsed.append((method, values))
    return parsed


def load_results(root: Path) -> dict[str, list[dict[str, float]]]:
    """Load and validate the frozen scenario-macro 7x7 summaries."""
    root = Path(root)
    sources = (
        (root / "oaster_v19_final/summary_by_snr_pair_scenario_macro.csv", "OASTER"),
        (root / "sisses_preserved/summary_by_snr_pair_scenario_macro.csv", "SISSES"),
        (root / "comparators_final/summary_by_snr_pair_scenario_macro.csv", None),
    )
    rows = {method: [] for method in METHODS}
    for path, fixed_method in sources:
        for method, values in _read(path, fixed_method):
            rows[method].append(values)

    expected = {(eeg, meg) for eeg in SNR_LEVELS for meg in SNR_LEVELS}
    for method, method_rows in rows.items():
        pairs = {(int(row["eeg_snr_db"]), int(row["meg_snr_db"])) for row in method_rows}
        if len(method_rows) != 49 or pairs != expected:
            raise ValueError(f"{method}: expected every 7x7 SNR pair exactly once")
        method_rows.sort(key=lambda row: (row["eeg_snr_db"], row["meg_snr_db"]))
    return rows


def _values(results: dict[str, list[dict[str, float]]], method: str, metric: str) -> np.ndarray:
    return np.asarray([row[metric] for row in results[method]], dtype=float)


def _matrix(results: dict[str, list[dict[str, float]]], method: str, metric: str) -> np.ndarray:
    index = {snr: position for position, snr in enumerate(SNR_LEVELS)}
    matrix = np.empty((7, 7), dtype=float)
    for row in results[method]:
        matrix[index[int(row["eeg_snr_db"])], index[int(row["meg_snr_db"])]] = row[metric]
    return matrix


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "white",
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.7,
            "savefig.dpi": 260,
        }
    )


def _save(figure: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def write_statistics(results: dict[str, list[dict[str, float]]], output: Path) -> Path:
    """Write descriptive statistics across the 49 frozen SNR pairs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("method", "metric", "n", "mean", "sd", "median", "q1", "q3", "min", "max", "worst")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            for metric, (_, higher, _) in METRICS.items():
                values = _values(results, method, metric)
                q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "n": values.size,
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "median": float(median),
                        "q1": float(q1),
                        "q3": float(q3),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "worst": float(values.min() if higher else values.max()),
                    }
                )
    return output


def plot_metric_table(results: dict[str, list[dict[str, float]]], output: Path) -> Path:
    keys = tuple(METRICS)
    means = np.asarray([[_values(results, method, key).mean() for key in keys] for method in METHODS])
    scores = np.empty_like(means)
    for column, key in enumerate(keys):
        values = means[:, column]
        span = np.ptp(values)
        normalized = (values - values.min()) / span if span else np.ones_like(values)
        scores[:, column] = normalized if METRICS[key][1] else 1.0 - normalized

    figure, axis = plt.subplots(figsize=(17, 6.2), layout="constrained")
    image = axis.imshow(scores, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(keys)), [METRICS[key][0] for key in keys], rotation=28, ha="right")
    axis.set_yticks(range(len(METHODS)), [_name(method) for method in METHODS])
    axis.tick_params(length=0)
    for row, method in enumerate(METHODS):
        for column, key in enumerate(keys):
            value = means[row, column]
            text = f"{value:.1f}" if METRICS[key][2] == "distance" else f"{value:.3f}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if scores[row, column] == 1 else "normal",
                color="white" if scores[row, column] < 0.52 else "#111111",
            )
    axis.set_title("Nine-method metric comparison — scenario-macro mean over 49 SNR pairs", pad=14)
    axis.set_xlabel(
        "Cells show original values; colour is rank-normalized within each metric. Spatial metrics are miss-penalized.",
        labelpad=12,
        color="#444444",
    )
    figure.colorbar(image, ax=axis, shrink=0.78, label="Within-metric performance (worst → best)")
    return _save(figure, output)


def plot_score_distributions(results: dict[str, list[dict[str, float]]], output: Path) -> Path:
    keys = ("auc_tie_corrected", "auc", "rmse")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 6.2), sharey=True, layout="constrained")
    positions = np.arange(len(METHODS))
    for axis, key in zip(axes, keys, strict=True):
        distributions = [_values(results, method, key) for method in METHODS]
        parts = axis.violinplot(
            distributions,
            positions=positions,
            orientation="horizontal",
            showextrema=False,
            widths=0.72,
            points=80,
        )
        for body, method in zip(parts["bodies"], METHODS, strict=True):
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("white")
            body.set_alpha(0.72)
        for position, values in zip(positions, distributions, strict=True):
            q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
            axis.hlines(position, q1, q3, color="#202020", linewidth=2.8)
            axis.scatter(median, position, marker="|", s=55, color="white", linewidth=1.8, zorder=3)
            axis.scatter(values.mean(), position, marker="D", s=18, color="#111111", zorder=3)
        axis.set_title(METRICS[key][0].replace("\n", " "))
        axis.grid(axis="x")
        axis.set_axisbelow(True)
        axis.set_yticks(positions, [_name(method) for method in METHODS])
        axis.invert_yaxis()
        if key in {"auc_tie_corrected", "auc"}:
            axis.axvline(0.9, color="#D55E00", linestyle="--", linewidth=1.2, label="0.90 target")
            axis.set_xlim(0.5, 1.01)
        else:
            axis.set_xlim(left=0)
    axes[0].legend(frameon=False, loc="lower left")
    figure.suptitle("Distribution across 49 EEG × MEG SNR pairs", fontsize=14, fontweight="semibold")
    figure.legend(
        handles=(
            Line2D([], [], color="#202020", linewidth=3, label="IQR"),
            Line2D([], [], marker="|", color="#202020", linestyle="None", markersize=9, label="Median"),
            Line2D([], [], marker="D", color="#111111", linestyle="None", markersize=4, label="Mean"),
        ),
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    return _save(figure, output)


def plot_spatial_forest(results: dict[str, list[dict[str, float]]], output: Path) -> Path:
    keys = (
        "surface_sd_mm_penalized",
        "surface_dle_mm_penalized",
        "deep_sd_mm_penalized",
        "deep_dle_mm_penalized",
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharey=True, layout="constrained")
    positions = np.arange(len(METHODS))
    for axis, key in zip(axes.flat, keys, strict=True):
        for position, method in zip(positions, METHODS, strict=True):
            values = _values(results, method, key)
            low, q1, median, q3, high = np.quantile(values, (0, 0.25, 0.5, 0.75, 1))
            axis.hlines(position, low, high, color=COLORS[method], linewidth=1.2, alpha=0.42)
            axis.hlines(position, q1, q3, color=COLORS[method], linewidth=5.2)
            axis.scatter(median, position, color=COLORS[method], marker=MARKERS[method], s=42, zorder=3)
        axis.set_xscale("log")
        axis.set_title(METRICS[key][0])
        axis.set_xlabel("Distance (mm, log scale)")
        axis.set_yticks(positions, [_name(method) for method in METHODS])
        axis.grid(axis="x", which="both")
        axis.set_axisbelow(True)
    axes[0, 0].invert_yaxis()
    figure.suptitle("Layer-specific localization error across 49 SNR pairs", fontsize=14, fontweight="semibold")
    figure.legend(
        handles=(
            Line2D([], [], color="#777777", linewidth=1.2, label="Range"),
            Line2D([], [], color="#444444", linewidth=5.2, label="IQR"),
            Line2D([], [], marker="o", color="#444444", linestyle="None", label="Median"),
        ),
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    return _save(figure, output)


def plot_deep_detection(results: dict[str, list[dict[str, float]]], output: Path) -> Path:
    keys = ("deep_sensitivity", "deep_specificity", "deep_balanced_accuracy")
    colors = ("#0072B2", "#D55E00", "#009E73")
    markers = ("o", "s", "D")
    offsets = (-0.19, 0.0, 0.19)
    positions = np.arange(len(METHODS))
    figure, axis = plt.subplots(figsize=(10.5, 6.5), layout="constrained")
    for key, color, marker, offset in zip(keys, colors, markers, offsets, strict=True):
        for position, method in zip(positions, METHODS, strict=True):
            values = _values(results, method, key)
            q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
            axis.errorbar(
                median,
                position + offset,
                xerr=np.asarray([[median - q1], [q3 - median]]),
                fmt=marker,
                color=color,
                markersize=5,
                capsize=2,
                linewidth=1.4,
            )
    axis.axvline(0.9, color="#555555", linestyle="--", linewidth=1, label="0.90 reference")
    axis.set_xlim(0, 1.01)
    axis.set_yticks(positions, [_name(method) for method in METHODS])
    axis.invert_yaxis()
    axis.set_xlabel("Score (median and IQR across 49 SNR pairs)")
    axis.set_title("Deep-source detection trade-off")
    axis.grid(axis="x")
    axis.set_axisbelow(True)
    figure.legend(
        handles=[
            Line2D([], [], marker=marker, color=color, linestyle="None", label=METRICS[key][0].replace("\n", " "))
            for key, color, marker in zip(keys, colors, markers, strict=True)
        ]
        + [Line2D([], [], color="#555555", linestyle="--", label="0.90 reference")],
        loc="outside lower center",
        frameon=False,
        ncols=2,
    )
    return _save(figure, output)


def plot_snr_robustness(results: dict[str, list[dict[str, float]]], output: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True, layout="constrained")
    for axis, varying, averaged, title in (
        (axes[0], "eeg_snr_db", "MEG", "Vary EEG SNR (average over MEG)"),
        (axes[1], "meg_snr_db", "EEG", "Vary MEG SNR (average over EEG)"),
    ):
        for method in METHODS:
            means = [
                np.mean([row["auc_tie_corrected"] for row in results[method] if int(row[varying]) == snr])
                for snr in SNR_LEVELS
            ]
            axis.plot(
                SNR_LEVELS,
                means,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.7 if method in {"OASTER", "SISSES"} else 1.15,
                markersize=5,
                label=_name(method),
            )
        axis.axhline(0.9, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel(f"SNR (dB); {averaged} marginalized")
        axis.set_xticks(SNR_LEVELS)
        axis.grid()
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Mean An_auc")
    axes[0].set_ylim(0.55, 1.005)
    figure.suptitle("Modality-specific SNR robustness", fontsize=14, fontweight="semibold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False)
    return _save(figure, output)


def plot_auc_heatmaps(results: dict[str, list[dict[str, float]]], output: Path) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(12.4, 11.2), sharex=True, sharey=True, layout="constrained")
    image = None
    for axis, method in zip(axes.flat, METHODS, strict=True):
        matrix = _matrix(results, method, "auc_tie_corrected")
        image = axis.imshow(matrix, origin="lower", cmap="cividis", vmin=0.5, vmax=1.0, aspect="equal")
        if matrix.min() < 0.9 < matrix.max():
            axis.contour(matrix, levels=[0.9], colors="white", linewidths=1.0, origin="lower")
        for row in range(7):
            for column in range(7):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.4,
                    color="white" if value < 0.78 else "#111111",
                )
        axis.set_title(f"{_name(method)}  ·  mean {matrix.mean():.3f}")
        axis.set_xticks(range(7), SNR_LEVELS)
        axis.set_yticks(range(7), SNR_LEVELS)
        axis.tick_params(labelsize=7)
    for axis in axes[-1, :]:
        axis.set_xlabel("MEG SNR (dB)")
    for axis in axes[:, 0]:
        axis.set_ylabel("EEG SNR (dB)")
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.78, label="An_auc (white contour = 0.90)")
    figure.suptitle("An_auc across the full 7 × 7 SNR matrix", fontsize=14, fontweight="semibold")
    return _save(figure, output)


def generate(root: Path, output: Path) -> list[Path]:
    _style()
    results = load_results(root)
    output = Path(output)
    return [
        write_statistics(results, output / "method_metric_statistics.csv"),
        plot_metric_table(results, output / "metric_mean_rank_heatmap.png"),
        plot_score_distributions(results, output / "auc_rmse_distributions.png"),
        plot_spatial_forest(results, output / "spatial_error_forest.png"),
        plot_deep_detection(results, output / "deep_detection_profile.png"),
        plot_snr_robustness(results, output / "snr_robustness_lines.png"),
        plot_auc_heatmaps(results, output / "an_auc_snr_heatmaps.png"),
    ]


def main(argv: Sequence[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/strict_blind"))
    parser.add_argument("--output", type=Path, default=Path("results/strict_blind/figures_v2"))
    args = parser.parse_args(argv)
    outputs = generate(args.root, args.output)
    print("\n".join(map(str, outputs)))
    return outputs


if __name__ == "__main__":
    main()
