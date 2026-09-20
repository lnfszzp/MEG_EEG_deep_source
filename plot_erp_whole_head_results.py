"""Plot the whole-head ERP benchmark for OASTER and seven comparators."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


SNR_LEVELS = (-10, -5, 0, 5, 10, 15, 20)
SCENARIOS = (
    "surface_only",
    "deep_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
)
METHODS = (
    "OASTER-ERP",
    "MNE",
    "dSPM",
    "sLORETA",
    "eLORETA",
    "LCMV",
    "Dipole fitting (grid)",
    "RAP-MUSIC",
)
DISPLAY = {"Dipole fitting (grid)": "Dipole grid"}
SCENARIO_TITLES = {
    "surface_only": "纯表层源 / Surface only",
    "deep_only": "纯深层源 / Deep only",
    "deep_plus_surface": "深层 + 单表层源 / Deep + surface",
    "deep_plus_two_surface": "深层 + 双表层源 / Deep + two surfaces",
}
COLORS = dict(
    zip(
        METHODS,
        (
            "#332288",
            "#4477AA",
            "#66CCEE",
            "#228833",
            "#CCBB44",
            "#EE6677",
            "#AA3377",
            "#BBBBBB",
        ),
        strict=True,
    )
)
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "v", "P", "X", "<"), strict=True))
MACRO_NAME = "summary_by_snr_pair_scenario_macro.csv"
SCENARIO_NAME = "summary_by_snr_scenario.csv"
ROWS_NAME = "rows.csv"
# Frozen corrected-v2 source-space bounding-box diagonal, used only when a
# rows-only export needs the same miss penalty as the strict runners.
DEFAULT_MISS_PENALTY_MM = 234.02911185627738
DISTANCE_METRICS = (
    ("surface_sd_mm_penalized", "表层 SD / Surface SD"),
    ("surface_dle_mm_penalized", "表层 DLE / Surface DLE"),
    ("deep_sd_mm_penalized", "深层 SD / Deep SD"),
    ("deep_dle_mm_penalized", "深层 DLE / Deep DLE"),
)


def _canonical_method(value: str) -> str:
    value = value.strip()
    return "OASTER-ERP" if value.upper().startswith("OASTER") else value


def _read(path: Path, scenario_table: bool) -> list[dict[str, float | str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"eeg_snr_db", "meg_snr_db", "auc_tie_corrected"}
        if scenario_table:
            required.add("scenario")
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        for line, row in enumerate(reader, 2):
            method = _canonical_method(row.get("method", "OASTER-ERP"))
            if method not in METHODS:
                continue
            try:
                parsed: dict[str, float | str] = {
                    "method": method,
                    "eeg_snr_db": float(row["eeg_snr_db"]),
                    "meg_snr_db": float(row["meg_snr_db"]),
                    "auc_tie_corrected": float(row["auc_tie_corrected"]),
                }
                if scenario_table:
                    parsed["scenario"] = row["scenario"].strip()
                for name in (
                    "auc",
                    "rmse",
                    "deep_sensitivity",
                    "deep_specificity",
                    "deep_balanced_accuracy",
                    *(metric for metric, _ in DISTANCE_METRICS),
                ):
                    parsed[name] = float(row.get(name, "nan"))
                for name in ("surface_auc_tie_corrected", "deep_auc_tie_corrected"):
                    parsed[name] = float(row.get(name) or "nan")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid numeric value") from exc
            if not np.isfinite(float(parsed["auc_tie_corrected"])):
                raise ValueError(f"{path}:{line}: An_auc is not finite")
            rows.append(parsed)
    return rows


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _summarize_rows(
    path: Path, penalty_mm: float
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    """Build the two plotting summaries when a runner exported only rows.csv."""
    raw_metrics = (
        "auc",
        "auc_tie_corrected",
        "rmse",
        "surface_sd_mm",
        "surface_dle_mm",
        "deep_sd_mm",
        "deep_dle_mm",
    )
    optional_metrics = ("surface_auc_tie_corrected", "deep_auc_tie_corrected")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "method",
            "eeg_snr_db",
            "meg_snr_db",
            "scenario",
            "status",
            "has_surface_true",
            "has_deep_true",
            "deep_detected",
            *raw_metrics,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        raw_rows = []
        for line, row in enumerate(reader, 2):
            method = _canonical_method(row["method"])
            if method not in METHODS:
                continue
            if row["status"] != "ok":
                raise ValueError(f"{path}:{line}: non-ok row for {method}")
            try:
                parsed = {
                    "method": method,
                    "eeg_snr_db": int(row["eeg_snr_db"]),
                    "meg_snr_db": int(row["meg_snr_db"]),
                    "scenario": row["scenario"].strip(),
                    "has_surface_true": int(float(row["has_surface_true"])),
                    "has_deep_true": int(float(row["has_deep_true"])),
                    "deep_detected": int(float(row["deep_detected"])),
                    **{name: float(row[name]) for name in raw_metrics},
                    **{name: float(row.get(name) or "nan") for name in optional_metrics},
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid numeric value") from exc
            raw_rows.append(parsed)

    groups: dict[tuple[str, int, int, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        groups[
            (
                str(row["method"]),
                int(row["eeg_snr_db"]),
                int(row["meg_snr_db"]),
                str(row["scenario"]),
            )
        ].append(row)
    scenario_rows = []
    for (method, eeg, meg, scenario), rows in sorted(groups.items()):
        summary: dict[str, float | str] = {
            "method": method,
            "eeg_snr_db": float(eeg),
            "meg_snr_db": float(meg),
            "scenario": scenario,
        }
        summary.update(
            {
                name: _finite_mean([float(row[name]) for row in rows])
                for name in (*raw_metrics, *optional_metrics)
            }
        )
        for layer in ("surface", "deep"):
            expected = [row for row in rows if int(row[f"has_{layer}_true"]) == 1]
            for short_name in ("sd_mm", "dle_mm"):
                name = f"{layer}_{short_name}"
                summary[f"{name}_penalized"] = (
                    float(
                        np.mean(
                            [
                                float(row[name])
                                if np.isfinite(float(row[name]))
                                and not (
                                    layer == "deep"
                                    and int(row["deep_detected"]) != 1
                                )
                                else penalty_mm
                                for row in expected
                            ]
                        )
                    )
                    if expected
                    else float("nan")
                )
        scenario_rows.append(summary)

    pair_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in scenario_rows:
        pair_groups[
            (str(row["method"]), int(row["eeg_snr_db"]), int(row["meg_snr_db"]))
        ].append(row)
    macro_rows = []
    summary_metrics = (
        "auc",
        "auc_tie_corrected",
        *optional_metrics,
        "rmse",
        *(name for name, _ in DISTANCE_METRICS),
    )
    for (method, eeg, meg), rows in sorted(pair_groups.items()):
        macro_rows.append(
            {
                "method": method,
                "eeg_snr_db": float(eeg),
                "meg_snr_db": float(meg),
                **{
                    name: _finite_mean([float(row[name]) for row in rows])
                    for name in summary_metrics
                },
            }
        )
    return macro_rows, scenario_rows


def load_results(
    input_dirs: Sequence[Path], penalty_mm: float = DEFAULT_MISS_PENALTY_MM
) -> tuple[dict[str, list[dict]], dict[str, dict[str, list[dict]]]]:
    """Load and validate complete 7x7 summaries from one or two result folders."""
    if not 1 <= len(input_dirs) <= 2:
        raise ValueError("provide one combined result directory or two method directories")
    macro = {method: [] for method in METHODS}
    scenarios = {scenario: {method: [] for method in METHODS} for scenario in SCENARIOS}
    for directory in map(Path, input_dirs):
        macro_path = directory / MACRO_NAME
        scenario_path = directory / SCENARIO_NAME
        rows_path = directory / ROWS_NAME
        if macro_path.is_file() and scenario_path.is_file():
            loaded_macro = _read(macro_path, False)
            loaded_scenarios = _read(scenario_path, True)
        elif rows_path.is_file():
            loaded_macro, loaded_scenarios = _summarize_rows(rows_path, penalty_mm)
        else:
            raise FileNotFoundError(
                f"{directory}: expected both summaries or {ROWS_NAME}"
            )
        for row in loaded_macro:
            macro[str(row["method"])].append(row)
        for row in loaded_scenarios:
            scenario = str(row["scenario"])
            if scenario in scenarios:
                scenarios[scenario][str(row["method"])].append(row)

    expected = {(eeg, meg) for eeg in SNR_LEVELS for meg in SNR_LEVELS}
    for method in METHODS:
        tables = [macro[method], *(scenarios[scenario][method] for scenario in SCENARIOS)]
        for label, rows in zip(("macro", *SCENARIOS), tables, strict=True):
            pairs = {
                (int(row["eeg_snr_db"]), int(row["meg_snr_db"])) for row in rows
            }
            if len(rows) != 49 or pairs != expected:
                raise ValueError(
                    f"{method}/{label}: expected one row for each of 49 SNR pairs; "
                    f"found {len(rows)} rows and {len(pairs)} pairs"
                )
            rows.sort(key=lambda row: (row["eeg_snr_db"], row["meg_snr_db"]))
    return macro, scenarios


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "white",
            "grid.color": "#D7D7D7",
            "savefig.dpi": 260,
        }
    )


def _matrix(rows: list[dict], metric: str = "auc_tie_corrected") -> np.ndarray:
    positions = {snr: index for index, snr in enumerate(SNR_LEVELS)}
    matrix = np.full((7, 7), np.nan)
    for row in rows:
        matrix[
            positions[int(row["eeg_snr_db"])], positions[int(row["meg_snr_db"])]
        ] = float(row[metric])
    return matrix


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def write_comparison_table(macro: dict[str, list[dict]], path: Path) -> Path:
    fields = (
        "method",
        "snr_pair_count",
        "primary_metric",
        "an_auc_mean",
        "an_auc_sd",
        "an_auc_min",
        "an_auc_max",
        "an_auc_ge_0_90_pairs",
        "surface_auc_tie_corrected_mean",
        "deep_auc_tie_corrected_mean",
        "auc_mean",
        "rmse_mean",
        *(f"{metric}_mean" for metric, _ in DISTANCE_METRICS),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            rows = macro[method]
            an_auc = np.asarray([row["auc_tie_corrected"] for row in rows], dtype=float)
            values = {
                "method": method,
                "snr_pair_count": len(rows),
                "primary_metric": "An_auc (auc_tie_corrected)",
                "an_auc_mean": an_auc.mean(),
                "an_auc_sd": an_auc.std(ddof=1),
                "an_auc_min": an_auc.min(),
                "an_auc_max": an_auc.max(),
                "an_auc_ge_0_90_pairs": int(np.count_nonzero(an_auc >= 0.9)),
                "surface_auc_tie_corrected_mean": _finite_mean(
                    [row.get("surface_auc_tie_corrected", np.nan) for row in rows]
                ),
                "deep_auc_tie_corrected_mean": _finite_mean(
                    [row.get("deep_auc_tie_corrected", np.nan) for row in rows]
                ),
                "auc_mean": np.nanmean([row["auc"] for row in rows]),
                "rmse_mean": np.nanmean([row["rmse"] for row in rows]),
            }
            values.update(
                {
                    f"{metric}_mean": np.nanmean([row[metric] for row in rows])
                    for metric, _ in DISTANCE_METRICS
                }
            )
            writer.writerow(values)
    return path


def write_scenario_table(
    scenarios: dict[str, dict[str, list[dict]]], path: Path
) -> Path:
    fields = (
        "scenario",
        "method",
        "snr_pair_count",
        "an_auc_mean",
        "an_auc_sd",
        "an_auc_min",
        "an_auc_max",
        "an_auc_ge_0_90_pairs",
        "surface_auc_tie_corrected_mean",
        "deep_auc_tie_corrected_mean",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scenario in SCENARIOS:
            for method in METHODS:
                values = np.asarray(
                    [row["auc_tie_corrected"] for row in scenarios[scenario][method]],
                    dtype=float,
                )
                writer.writerow(
                    {
                        "scenario": scenario,
                        "method": method,
                        "snr_pair_count": values.size,
                        "an_auc_mean": values.mean(),
                        "an_auc_sd": values.std(ddof=1),
                        "an_auc_min": values.min(),
                        "an_auc_max": values.max(),
                        "an_auc_ge_0_90_pairs": int(np.count_nonzero(values >= 0.9)),
                        "surface_auc_tie_corrected_mean": _finite_mean(
                            [
                                row.get("surface_auc_tie_corrected", np.nan)
                                for row in scenarios[scenario][method]
                            ]
                        ),
                        "deep_auc_tie_corrected_mean": _finite_mean(
                            [
                                row.get("deep_auc_tie_corrected", np.nan)
                                for row in scenarios[scenario][method]
                            ]
                        ),
                    }
                )
    return path


def write_deep_detection_table(macro: dict[str, list[dict]], path: Path) -> Path:
    fields = (
        "method",
        "deep_sensitivity_mean",
        "deep_specificity_mean",
        "deep_balanced_accuracy_mean",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            writer.writerow(
                {
                    "method": method,
                    **{
                        f"{name}_mean": _finite_mean(
                            [float(row.get(name, np.nan)) for row in macro[method]]
                        )
                        for name in (
                            "deep_sensitivity",
                            "deep_specificity",
                            "deep_balanced_accuracy",
                        )
                    },
                }
            )
    return path


def write_paired_statistics(macro: dict[str, list[dict]], path: Path) -> Path:
    reference = np.asarray(
        [row["auc_tie_corrected"] for row in macro["OASTER-ERP"]], dtype=float
    )
    rng = np.random.default_rng(20260918)
    rows = []
    for method in METHODS[1:]:
        comparator = np.asarray(
            [row["auc_tie_corrected"] for row in macro[method]], dtype=float
        )
        difference = reference - comparator
        bootstrap = difference[
            rng.integers(0, difference.size, size=(10_000, difference.size))
        ].mean(axis=1)
        statistic, p_value = wilcoxon(difference, alternative="two-sided", method="auto")
        rows.append(
            {
                "comparison": f"OASTER-ERP - {method}",
                "analysis_unit": "SNR pair; descriptive, not independent subjects",
                "snr_pair_count": difference.size,
                "mean_difference": difference.mean(),
                "mean_difference_ci95_low": np.quantile(bootstrap, 0.025),
                "mean_difference_ci95_high": np.quantile(bootstrap, 0.975),
                "median_difference": np.median(difference),
                "cohen_dz": difference.mean() / difference.std(ddof=1),
                "wins": int(np.count_nonzero(difference > 0.0)),
                "ties": int(np.count_nonzero(difference == 0.0)),
                "losses": int(np.count_nonzero(difference < 0.0)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
            }
        )
    order = np.argsort([row["wilcoxon_p"] for row in rows])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(rows) - rank) * rows[index]["wilcoxon_p"])
        rows[index]["holm_adjusted_p"] = min(1.0, running)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_configuration_cluster_statistics(
    rows_paths: Sequence[Path], path: Path
) -> Path:
    """Pair methods by simulated source configuration, averaging its 49 SNR cells."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for rows_path in rows_paths:
        with rows_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "configuration_id",
                "scenario",
                "method",
                "status",
                "auc_tie_corrected",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{rows_path}: missing columns {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                method = _canonical_method(row["method"])
                if method not in METHODS:
                    continue
                if row["status"] != "ok":
                    raise ValueError(f"{rows_path}:{line}: non-ok row for {method}")
                try:
                    value = float(row["auc_tie_corrected"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{rows_path}:{line}: invalid An_auc") from exc
                if not np.isfinite(value):
                    raise ValueError(f"{rows_path}:{line}: An_auc is not finite")
                grouped[(row["scenario"].strip(), method, row["configuration_id"])].append(
                    value
                )

    rows = []
    for scenario in ("all", *SCENARIOS):
        means: dict[str, dict[str, float]] = {}
        repetitions: list[int] = []
        for method in METHODS:
            selected = {
                configuration: values
                for (row_scenario, row_method, configuration), values in grouped.items()
                if row_method == method
                and (scenario == "all" or row_scenario == scenario)
            }
            means[method] = {
                configuration: float(np.mean(values))
                for configuration, values in selected.items()
            }
            repetitions.extend(map(len, selected.values()))
        reference_ids = set(means["OASTER-ERP"])
        if not reference_ids:
            raise ValueError(f"no OASTER rows found for scenario {scenario}")
        if min(repetitions) != 49 or max(repetitions) != 49:
            raise ValueError(
                f"{scenario}: expected 49 SNR rows per method/configuration; "
                f"found {min(repetitions)}..{max(repetitions)}"
            )
        scope_rows = []
        rng = np.random.default_rng(20260918)
        for method in METHODS[1:]:
            if set(means[method]) != reference_ids:
                raise ValueError(f"{scenario}/{method}: configuration IDs do not match OASTER")
            difference = np.asarray(
                [means["OASTER-ERP"][key] - means[method][key] for key in sorted(reference_ids)],
                dtype=float,
            )
            bootstrap = difference[
                rng.integers(0, difference.size, size=(10_000, difference.size))
            ].mean(axis=1)
            if np.all(difference == 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(
                    difference, alternative="two-sided", method="auto"
                )
            standard_deviation = difference.std(ddof=1)
            scope_rows.append(
                {
                    "scenario": scenario,
                    "comparison": f"OASTER-ERP - {method}",
                    "analysis_unit": "source configuration (49 SNR cells averaged)",
                    "configuration_count": difference.size,
                    "mean_difference": difference.mean(),
                    "mean_difference_ci95_low": np.quantile(bootstrap, 0.025),
                    "mean_difference_ci95_high": np.quantile(bootstrap, 0.975),
                    "median_difference": np.median(difference),
                    "cohen_dz": (
                        difference.mean() / standard_deviation
                        if standard_deviation > 0.0
                        else float("nan")
                    ),
                    "wins": int(np.count_nonzero(difference > 0.0)),
                    "ties": int(np.count_nonzero(difference == 0.0)),
                    "losses": int(np.count_nonzero(difference < 0.0)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
        order = np.argsort([row["wilcoxon_p"] for row in scope_rows])
        running = 0.0
        for rank, index in enumerate(order):
            running = max(
                running,
                (len(scope_rows) - rank) * scope_rows[index]["wilcoxon_p"],
            )
            scope_rows[index]["holm_adjusted_p"] = min(1.0, running)
        rows.extend(scope_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_omnibus_test(macro: dict[str, list[dict]], path: Path) -> Path:
    statistic, p_value = friedmanchisquare(
        *(
            [row["auc_tie_corrected"] for row in macro[method]]
            for method in METHODS
        )
    )
    fields = ("test", "unit", "method_count", "snr_pair_count", "statistic", "p")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "test": "Friedman",
                "unit": "SNR pair; descriptive, not subject-level inference",
                "method_count": len(METHODS),
                "snr_pair_count": len(macro[METHODS[0]]),
                "statistic": statistic,
                "p": p_value,
            }
        )
    return path


def plot_heatmaps(
    rows_by_method: dict[str, list[dict]], path: Path, title: str
) -> Path:
    figure, axes = plt.subplots(
        2, 4, figsize=(15.4, 7.8), sharex=True, sharey=True, layout="constrained"
    )
    image = None
    for axis, method in zip(axes.flat, METHODS, strict=True):
        matrix = _matrix(rows_by_method[method])
        image = axis.imshow(
            matrix, origin="lower", cmap="cividis", vmin=0.5, vmax=1.0, aspect="equal"
        )
        if matrix.min() < 0.9 < matrix.max():
            axis.contour(matrix, levels=[0.9], colors="white", linewidths=0.9)
        for row in range(7):
            for column in range(7):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white" if value < 0.77 else "#111111",
                )
        axis.set_title(f"{DISPLAY.get(method, method)} · mean {matrix.mean():.3f}")
        axis.set_xticks(range(7), SNR_LEVELS)
        axis.set_yticks(range(7), SNR_LEVELS)
        axis.tick_params(length=0, labelsize=7)
    for axis in axes[-1, :]:
        axis.set_xlabel("MEG SNR (dB)")
    for axis in axes[:, 0]:
        axis.set_ylabel("EEG SNR (dB)")
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.78, label="An_auc（白线 / white contour = 0.90）")
    figure.suptitle(title, fontsize=14, fontweight="semibold")
    return _save(figure, path)


def plot_robustness(macro: dict[str, list[dict]], path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), sharey=True, layout="constrained")
    for axis, varying, title in (
        (axes[0], "eeg_snr_db", "EEG SNR变化（对MEG取平均） / Vary EEG"),
        (axes[1], "meg_snr_db", "MEG SNR变化（对EEG取平均） / Vary MEG"),
    ):
        for method in METHODS:
            means = [
                np.mean(
                    [
                        row["auc_tie_corrected"]
                        for row in macro[method]
                        if int(row[varying]) == snr
                    ]
                )
                for snr in SNR_LEVELS
            ]
            axis.plot(
                SNR_LEVELS,
                means,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.5 if method == "OASTER-ERP" else 1.25,
                markersize=5,
                label=DISPLAY.get(method, method),
            )
        axis.axhline(0.9, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("SNR (dB)")
        axis.set_xticks(SNR_LEVELS)
        axis.set_ylim(0.5, 1.01)
        axis.grid()
        axis.set_axisbelow(True)
    axes[0].set_ylabel("平均 An_auc / Mean An_auc")
    figure.suptitle("八种方法的SNR鲁棒性 / SNR robustness", fontsize=14, fontweight="semibold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    return _save(figure, path)


def plot_layer_distances(macro: dict[str, list[dict]], path: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(13.3, 9.0), sharey=True, layout="constrained")
    for axis, (metric, title) in zip(axes.flat, DISTANCE_METRICS, strict=True):
        values = [np.asarray([row[metric] for row in macro[method]], dtype=float) for method in METHODS]
        boxes = axis.boxplot(
            values,
            orientation="horizontal",
            patch_artist=True,
            tick_labels=[DISPLAY.get(method, method) for method in METHODS],
            showfliers=False,
            widths=0.62,
            medianprops={"color": "white", "linewidth": 1.4},
        )
        for patch, method in zip(boxes["boxes"], METHODS, strict=True):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.84)
        axis.set_title(title)
        axis.set_xlabel("距离 / Distance (mm；49个SNR组合)")
        axis.grid(axis="x")
        axis.set_axisbelow(True)
    axes[0, 0].invert_yaxis()
    figure.suptitle(
        "表层与深层定位误差（漏检惩罚后） / Layer-specific penalized errors",
        fontsize=14,
        fontweight="semibold",
    )
    return _save(figure, path)


def plot_layer_auc(
    scenarios: dict[str, dict[str, list[dict]]], path: Path
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), sharey=True, layout="constrained")
    x = np.arange(len(METHODS), dtype=float)
    scenario_colors = ("#4477AA", "#EE6677", "#228833", "#AA3377")
    for axis, layer, title in (
        (axes[0], "surface", "表层 AUC / Surface AUC"),
        (axes[1], "deep", "深层 AUC / Deep AUC"),
    ):
        available = []
        for scenario, color in zip(SCENARIOS, scenario_colors, strict=True):
            arrays = [
                np.asarray(
                    [
                        row.get(f"{layer}_auc_tie_corrected", np.nan)
                        for row in scenarios[scenario][method]
                    ],
                    dtype=float,
                )
                for method in METHODS
            ]
            if not any(np.isfinite(values).any() for values in arrays):
                continue
            means = [_finite_mean(values) for values in arrays]
            errors = [
                float(values[np.isfinite(values)].std(ddof=1))
                if np.isfinite(values).sum() > 1
                else 0.0
                for values in arrays
            ]
            offset = 0.14 * (len(available) - 1)
            axis.errorbar(
                x + offset,
                means,
                yerr=errors,
                fmt="o",
                color=color,
                capsize=2.5,
                label=SCENARIO_TITLES[scenario],
            )
            available.append(scenario)
        if not available:
            axis.text(
                0.5,
                0.5,
                "旧结果无分层 AUC / Layer AUC unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.axhline(0.9, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(x, [DISPLAY.get(method, method) for method in METHODS], rotation=22, ha="right")
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        if available:
            axis.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("An_auc（点=49个SNR格等权均值；误差棒=格间SD）")
    figure.suptitle(
        "分层 An_auc：全局 An_auc 不能替代深层 AUC / Global An_auc does not replace deep AUC",
        fontsize=14,
        fontweight="semibold",
    )
    return _save(figure, path)


def generate(
    input_dirs: Sequence[Path],
    output: Path | None = None,
    penalty_mm: float = DEFAULT_MISS_PENALTY_MM,
) -> list[Path]:
    _style()
    inputs = tuple(map(Path, input_dirs))
    macro, scenarios = load_results(inputs, penalty_mm)
    output = Path(output) if output is not None else inputs[0] / "figures_erp_whole_head"
    products = [
        write_comparison_table(macro, output / "eight_method_comparison.csv"),
        write_scenario_table(scenarios, output / "scenario_method_comparison.csv"),
        write_deep_detection_table(macro, output / "deep_detection_comparison.csv"),
        write_paired_statistics(macro, output / "oaster_paired_statistics.csv"),
        write_omnibus_test(macro, output / "snr_grid_omnibus_test.csv"),
    ]
    rows_paths = [directory / ROWS_NAME for directory in inputs]
    if all(path.is_file() for path in rows_paths):
        products.append(
            write_configuration_cluster_statistics(
                rows_paths, output / "configuration_clustered_statistics.csv"
            )
        )
    products.append(
        plot_heatmaps(
            macro,
            output / "an_auc_all_49_pairs.png",
            "八种方法：全49组EEG × MEG SNR的An_auc / All 49 SNR pairs",
        )
    )
    products.extend(
        plot_heatmaps(
            scenarios[scenario],
            output / f"an_auc_{scenario}.png",
            f"{SCENARIO_TITLES[scenario]}：八种方法的An_auc",
        )
        for scenario in SCENARIOS
    )
    products.extend(
        (
            plot_robustness(macro, output / "an_auc_snr_robustness.png"),
            plot_layer_auc(scenarios, output / "layer_auc.png"),
            plot_layer_distances(macro, output / "layer_sd_dle.png"),
        )
    )
    return products


def main(argv: Sequence[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="one combined output directory, or OASTER and comparator output directories",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--miss-penalty-mm",
        type=float,
        default=DEFAULT_MISS_PENALTY_MM,
        help="rows.csv fallback penalty for a missed layer",
    )
    args = parser.parse_args(argv)
    if len(args.input_dirs) > 2:
        parser.error("at most two input directories are accepted")
    outputs = generate(args.input_dirs, args.output, args.miss_penalty_mm)
    print("\n".join(map(str, outputs)))
    return outputs


if __name__ == "__main__":
    main()
