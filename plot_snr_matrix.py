"""Plot 7x7 EEG-by-MEG SNR matrices from scenario-macro summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNR_LEVELS = (-10, -5, 0, 5, 10, 15, 20)
METRICS = {
    "auc_tie_corrected": ("An_auc (preferred)", True, (0.5, 1.0)),
    "auc": ("Requested parcel AUC", True, (0.5, 1.0)),
    "rmse": ("RMSE", False, None),
    "surface_sd_mm": ("Surface SD (mm)", False, None),
    "surface_dle_mm": ("Surface DLE (mm)", False, None),
    "deep_sd_mm": ("Deep SD (mm)", False, None),
    "deep_dle_mm": ("Deep DLE (mm)", False, None),
    "surface_sd_mm_penalized": ("Surface SD, miss-penalized (mm)", False, None),
    "surface_dle_mm_penalized": ("Surface DLE, miss-penalized (mm)", False, None),
    "deep_sd_mm_penalized": ("Deep SD, miss-penalized (mm)", False, None),
    "deep_dle_mm_penalized": ("Deep DLE, miss-penalized (mm)", False, None),
    "deep_balanced_accuracy": ("Deep balanced accuracy", True, (0.0, 1.0)),
}


def parse_input(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("input must be LABEL=path/to/summary.csv")
    return label.strip(), Path(path)


def load_series(label: str, path: Path, metric: str) -> dict[str, np.ndarray]:
    """Load one CSV; split comparator summaries by their optional method column."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"eeg_snr_db", "meg_snr_db", metric}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty summary")

    methods = list(dict.fromkeys(row.get("method", "").strip() for row in rows))
    methods = [method for method in methods if method]
    groups = methods or [""]
    result: dict[str, np.ndarray] = {}
    index = {snr: position for position, snr in enumerate(SNR_LEVELS)}
    for method in groups:
        name = label if len(groups) == 1 else f"{label}: {method}"
        matrix = np.full((len(SNR_LEVELS), len(SNR_LEVELS)), np.nan)
        seen: set[tuple[int, int]] = set()
        for row in rows:
            if method and row.get("method", "").strip() != method:
                continue
            try:
                eeg_value = float(row["eeg_snr_db"])
                meg_value = float(row["meg_snr_db"])
                eeg_snr, meg_snr = int(eeg_value), int(meg_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid SNR value") from exc
            if eeg_value != eeg_snr or meg_value != meg_snr:
                raise ValueError(f"{path}: SNR values must be integers")
            if eeg_snr not in index or meg_snr not in index:
                raise ValueError(
                    f"{path}: unsupported SNR pair ({eeg_snr}, {meg_snr})"
                )
            cell = (index[eeg_snr], index[meg_snr])
            if cell in seen:
                raise ValueError(
                    f"{path}: duplicate {name} SNR pair ({eeg_snr}, {meg_snr})"
                )
            seen.add(cell)
            text = row[metric].strip()
            try:
                matrix[cell] = float(text) if text else np.nan
            except ValueError as exc:
                raise ValueError(f"{path}: invalid {metric} value {text!r}") from exc
        result[name] = matrix
    return result


def _limits(metric: str, matrices: Sequence[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    if not values.size:
        raise ValueError(f"no finite {metric} values to plot")
    fixed = METRICS[metric][2]
    if fixed:
        return min(fixed[0], float(values.min())), max(fixed[1], float(values.max()))
    lower, upper = min(0.0, float(values.min())), float(values.max())
    return (lower, upper) if upper > lower else (lower, lower + 1.0)


def plot(
    inputs: Sequence[tuple[str, Path]],
    output: Path,
    metric: str = "auc_tie_corrected",
    title: str | None = None,
) -> Path:
    series: dict[str, np.ndarray] = {}
    for label, path in inputs:
        for name, matrix in load_series(label, path, metric).items():
            if name in series:
                raise ValueError(f"duplicate plotted method label: {name}")
            series[name] = matrix
    if not series:
        raise ValueError("at least one --input is required")

    metric_title, higher_is_better, _ = METRICS[metric]
    matrices = list(series.values())
    vmin, vmax = _limits(metric, matrices)
    count = len(series)
    columns = min(3, count)
    heat_rows = math.ceil(count / columns)
    comparison = count > 1
    figure = plt.figure(
        figsize=(4.1 * columns, 3.45 * heat_rows + (3.0 if comparison else 0.4)),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        heat_rows + int(comparison),
        columns,
        height_ratios=[1.0] * heat_rows + ([0.8] if comparison else []),
    )
    cmap = plt.get_cmap("viridis" if higher_is_better else "viridis_r").copy()
    cmap.set_bad("#e6e6e6")
    heat_axes, image = [], None
    for position, (name, matrix) in enumerate(series.items()):
        axis = figure.add_subplot(grid[position // columns, position % columns])
        heat_axes.append(axis)
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        valid = int(np.isfinite(matrix).sum())
        axis.set_title(name + ("" if valid == 49 else f" ({valid}/49 pairs)"), fontsize=10)
        axis.set_xticks(range(7), SNR_LEVELS)
        axis.set_yticks(range(7), SNR_LEVELS)
        axis.set_xlabel("MEG SNR (dB)")
        axis.set_ylabel("EEG SNR (dB)")
        axis.tick_params(labelsize=8)
        for row in range(7):
            for column in range(7):
                value = matrix[row, column]
                if np.isfinite(value):
                    normalized = (value - vmin) / (vmax - vmin)
                    color = "white" if 0.15 < normalized < 0.72 else "black"
                    text = f"{value:.2f}"
                else:
                    color, text = "#666666", "-"
                axis.text(column, row, text, ha="center", va="center", fontsize=6, color=color)

    for position in range(count, heat_rows * columns):
        figure.add_subplot(grid[position // columns, position % columns]).axis("off")
    assert image is not None
    figure.colorbar(image, ax=heat_axes, shrink=0.82, label=metric_title)

    if comparison:
        axis = figure.add_subplot(grid[heat_rows, :])
        names = list(series)
        finite = [series[name][np.isfinite(series[name])] for name in names]
        means = [float(values.mean()) if values.size else np.nan for values in finite]
        worst = [
            float((values.min() if higher_is_better else values.max()))
            if values.size
            else np.nan
            for values in finite
        ]
        x = np.arange(count)
        width = 0.36
        bars = (
            axis.bar(x - width / 2, means, width, label="Mean", color="#4472C4"),
            axis.bar(x + width / 2, worst, width, label="Worst", color="#ED7D31"),
        )
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.set_ylabel(metric_title)
        axis.set_title("Across available SNR pairs")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncols=2)
        axis.tick_params(labelsize=8)
        for group in bars:
            axis.bar_label(group, fmt="%.3g", padding=2, fontsize=7)

    figure.suptitle(title or f"{metric_title}: scenario-macro EEG x MEG SNR", fontsize=14)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input,
        metavar="LABEL=CSV",
        help="scenario-macro summary; repeat to compare methods",
    )
    parser.add_argument("--metric", choices=METRICS, default="auc_tie_corrected")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    result = plot(args.input, args.output, args.metric, args.title)
    print(result)
    return result


if __name__ == "__main__":
    main()
