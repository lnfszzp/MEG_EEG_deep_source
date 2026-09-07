"""Paired, SNR-blocked statistics for completed strict-benchmark methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

import plot_strict_metrics as reporting


ROOT = Path(__file__).resolve().parent
STRICT = ROOT / "results" / "strict_blind"
OUTPUT = STRICT / "statistics"
SEED = 20260907
BOOTSTRAP_RESAMPLES = 10_000
OASTER = "OASTER"
METHODS = reporting.METHODS
# direction: +1 means larger is better, -1 means smaller is better.
METRICS = (
    ("auc_tie_corrected", "An_auc", 1),
    ("auc", "Parcel AUC", 1),
    ("rmse", "Squared relative error (historical RMSE field)", -1),
    ("surface_sd_mm_penalized", "Surface penalized SD (mm)", -1),
    ("surface_dle_mm_penalized", "Surface penalized DLE (mm)", -1),
    ("deep_sd_mm_penalized", "Deep penalized SD (mm)", -1),
    ("deep_dle_mm_penalized", "Deep penalized DLE (mm)", -1),
    ("deep_sensitivity", "Deep sensitivity", 1),
    ("deep_specificity", "Deep specificity", 1),
    ("deep_balanced_accuracy", "Deep balanced accuracy", 1),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_blocks(
    root: Path = STRICT,
) -> tuple[
    dict[str, dict[tuple[int, int], dict[str, float]]], list[dict[str, str]]
]:
    results, availability = reporting.load_results_with_availability(root)
    metric_names = [field for field, _, _ in METRICS]
    blocks = {
        method: {
            (int(row["eeg_snr_db"]), int(row["meg_snr_db"])): {
                name: float(row[name]) for name in metric_names
            }
            for row in rows
        }
        for method, rows in results.items()
    }
    if OASTER not in blocks:
        raise ValueError("a complete OASTER summary is required for paired comparisons")
    if len(blocks) < 2:
        raise ValueError("at least two complete methods are required")
    return blocks, availability


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(p_values) - position) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[~np.isclose(differences, 0.0, rtol=1e-10, atol=1e-12)]
    if not nonzero.size:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _raw_path(summary: Path) -> tuple[Path | None, str | None]:
    rows = summary.parent / "rows.csv"
    if rows.is_file():
        return rows, None
    metadata = summary.parent / "metadata.json"
    if not metadata.is_file():
        return None, None
    values = json.loads(metadata.read_text(encoding="utf-8-sig"))
    source = Path(values["source_scores"]) if values.get("source_scores") else None
    return source, values.get("source_scores_sha256")


def _validate_case_inputs(
    root: Path, availability: list[dict[str, str]]
) -> dict[str, object]:
    manifest = root / "manifest.json"
    expected_cases = len(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.is_file() else None
    summaries = {
        Path(row["source"])
        for row in availability
        if row["status"] == "complete" and row["source"]
    }
    raw_files: dict[str, object] = {}
    case_sets: list[set[str]] = []
    missing: list[str] = []
    for summary in sorted(summaries):
        raw, expected_hash = _raw_path(summary)
        if raw is None or not raw.is_file():
            missing.append(str(raw or summary.parent / "rows.csv"))
            continue
        if expected_hash and _sha256(raw) != expected_hash:
            raise ValueError(f"preserved raw-score hash mismatch: {raw}")
        by_method: dict[str, set[str]] = defaultdict(set)
        bad_status = 0
        row_count = 0
        for row in _read_csv(raw):
            row_count += 1
            bad_status += row.get("status", "ok") != "ok"
            method = row.get("method", summary.parent.name)
            case_id = row["case_id"]
            if case_id in by_method[method]:
                raise ValueError(f"duplicate case/method in {raw}: {case_id} {method}")
            by_method[method].add(case_id)
        if bad_status:
            raise ValueError(f"raw result contains failed cases: {raw}")
        if expected_cases is not None and any(len(ids) != expected_cases for ids in by_method.values()):
            raise ValueError(f"raw result is incomplete for the manifest: {raw}")
        if len({frozenset(ids) for ids in by_method.values()}) != 1:
            raise ValueError(f"method case sets do not align within {raw}")
        case_sets.append(next(iter(by_method.values())))
        raw_files[str(raw)] = {
            "sha256": _sha256(raw),
            "row_count": row_count,
            "methods": sorted(by_method),
            "status_errors": bad_status,
        }
    aligned = not case_sets or len({frozenset(ids) for ids in case_sets}) == 1
    if not aligned:
        raise ValueError("completed methods do not contain identical case IDs")
    return {
        "performed": bool(raw_files),
        "case_ids_aligned": aligned,
        "expected_case_count": expected_cases,
        "files": raw_files,
        "missing_raw_inputs": missing,
    }


def analyze(output: Path | None = None, *, root: Path = STRICT) -> Path:
    root = Path(root)
    output = Path(output) if output is not None else root / "statistics"
    blocks, availability = _load_blocks(root)
    methods = tuple(blocks)
    raw_validation = _validate_case_inputs(root, availability)
    pairs = sorted(blocks[OASTER])
    rng = np.random.default_rng(SEED)
    descriptive: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []
    by_pair: list[dict[str, object]] = []
    omnibus: list[dict[str, object]] = []
    method_ranks: list[dict[str, object]] = []

    for metric, label, direction in METRICS:
        matrix = np.asarray(
            [[blocks[method][pair][metric] for method in methods] for pair in pairs], dtype=float
        )
        for method_index, method in enumerate(methods):
            values = matrix[:, method_index]
            ci_low, ci_high = _bootstrap_mean_ci(values, rng)
            descriptive.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "direction": "higher" if direction > 0 else "lower",
                    "method": method,
                    "n_snr_pairs": len(values),
                    "mean": float(values.mean()),
                    "sd_across_snr_pairs": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "bootstrap_mean_ci_low": ci_low,
                    "bootstrap_mean_ci_high": ci_high,
                }
            )

        benefit = direction * matrix
        if len(methods) >= 3:
            friedman = stats.friedmanchisquare(
                *(benefit[:, index] for index in range(len(methods)))
            )
            omnibus.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "n_snr_pairs": len(pairs),
                    "method_count": len(methods),
                    "friedman_chi_square": float(friedman.statistic),
                    "degrees_of_freedom": len(methods) - 1,
                    "p_value": float(friedman.pvalue),
                    "kendalls_w": float(
                        friedman.statistic / (len(pairs) * (len(methods) - 1))
                    ),
                }
            )
        ranks = stats.rankdata(-benefit, axis=1, method="average")
        for index, method in enumerate(methods):
            method_ranks.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "method": method,
                    "mean_rank_1_is_best": float(ranks[:, index].mean()),
                }
            )

        metric_rows: list[dict[str, object]] = []
        raw_p_values: list[float] = []
        for baseline_index, baseline in enumerate(methods[1:], start=1):
            differences = benefit[:, 0] - benefit[:, baseline_index]
            ci_low, ci_high = _bootstrap_mean_ci(differences, rng)
            ties = np.isclose(differences, 0.0, rtol=1e-10, atol=1e-12)
            tested = differences.copy()
            tested[ties] = 0.0
            wins = int(np.sum((differences > 0) & ~ties))
            losses = int(np.sum((differences < 0) & ~ties))
            tie_count = int(ties.sum())
            if np.all(ties):
                p_value = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p_value = float(stats.wilcoxon(tested, zero_method="wilcox").pvalue)
            raw_p_values.append(p_value)
            metric_rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "direction": "positive difference favors OASTER",
                    "baseline": baseline,
                    "n_snr_pairs": len(pairs),
                    "oaster_mean": float(matrix[:, 0].mean()),
                    "baseline_mean": float(matrix[:, baseline_index].mean()),
                    "mean_oriented_difference": float(differences.mean()),
                    "median_oriented_difference": float(np.median(differences)),
                    "bootstrap_difference_ci_low": ci_low,
                    "bootstrap_difference_ci_high": ci_high,
                    "oaster_wins": wins,
                    "ties": tie_count,
                    "oaster_losses": losses,
                    "win_rate_excluding_ties": float(wins / (wins + losses))
                    if wins + losses
                    else math.nan,
                    "common_language_win_probability": float(
                        (wins + 0.5 * tie_count) / len(pairs)
                    ),
                    "wilcoxon_p_value": p_value,
                    "holm_p_value_within_metric": math.nan,
                    "paired_rank_biserial": _rank_biserial(tested),
                }
            )
            for pair, a, b, difference, tie in zip(
                pairs, matrix[:, 0], matrix[:, baseline_index], differences, ties, strict=True
            ):
                by_pair.append(
                    {
                        "metric": metric,
                        "metric_label": label,
                        "baseline": baseline,
                        "eeg_snr_db": pair[0],
                        "meg_snr_db": pair[1],
                        "oaster_value": float(a),
                        "baseline_value": float(b),
                        "oriented_difference_positive_favors_oaster": float(difference),
                        "outcome": "tie" if tie else ("win" if difference > 0 else "loss"),
                    }
                )
        for row, adjusted in zip(metric_rows, _holm(raw_p_values), strict=True):
            row["holm_p_value_within_metric"] = adjusted
        pairwise.extend(metric_rows)

    if omnibus:
        for row, adjusted in zip(
            omnibus, _holm([float(row["p_value"]) for row in omnibus]), strict=True
        ):
            row["holm_p_value_across_metrics"] = adjusted

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "summary_method_descriptive.csv", descriptive)
    _write_csv(output / "summary_oaster_pairwise.csv", pairwise)
    _write_csv(output / "summary_snr_pair_differences.csv", by_pair)
    _write_csv(output / "summary_omnibus.csv", omnibus)
    _write_csv(output / "summary_method_ranks.csv", method_ranks)
    reporting.write_availability(availability, output / "method_availability.csv")

    source_summaries = sorted(
        {Path(row["source"]) for row in availability if row["status"] == "complete"}
    )
    metadata = {
        "results_root": str(root.resolve()),
        "analysis_unit": "49 paired EEG×MEG SNR cells; each value is a scenario macro-average",
        "methods_included": list(methods),
        "method_availability": availability,
        "metrics": [field for field, _, _ in METRICS],
        "bootstrap": {
            "type": "paired nonparametric percentile bootstrap over SNR cells",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": SEED,
            "confidence_level": 0.95,
        },
        "tests": {
            "omnibus": "Friedman test with Kendall's W; Holm correction across endpoints",
            "pairwise": (
                "two-sided paired Wilcoxon signed-rank; Holm correction over available "
                "OASTER contrasts within each endpoint"
            ),
            "effect_size": "paired rank-biserial correlation; positive favors OASTER",
        },
        "scope_warning": "The 49 SNR cells are a fixed experimental grid, not an iid population sample.",
        "localization_metrics": "Only miss-penalized layer SD/DLE enter inference.",
        "source_summaries": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_summaries
        ],
        "raw_case_validation": raw_validation,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", descriptive, pairwise, omnibus, methods, availability)
    return output


def _write_report(
    path: Path,
    descriptive: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    omnibus: list[dict[str, object]],
    methods: tuple[str, ...],
    availability: list[dict[str, str]],
) -> None:
    lookup = {(row["metric"], row["method"]): row for row in descriptive}
    primary = [row for row in pairwise if row["metric"] == "auc_tie_corrected"]
    missing = [row["method"] for row in availability if row["status"] != "complete"]
    lines = [
        "# 严格基准统计分析",
        "",
        f"统计区组为 49 个 EEG×MEG SNR 组合；本次仅纳入 {len(methods)} 个完整方法："
        + "、".join(methods)
        + "。",
        "",
        "## 方法可用性",
        "",
        "| 方法 | 状态 | 纳入排名/显著性 |",
        "|---|---:|---:|",
    ]
    for row in availability:
        lines.append(
            f"| {row['method']} | {row['status']} | {row['included_in_comparison']} |"
        )
    if "SISSES" in missing:
        lines.extend(["", "SISSES 为 **N/A**，未纳入排名、Friedman 或 Wilcoxon 检验。"])
    lines.extend(
        [
            "",
            "## 主指标：An_auc",
            "",
            "| 对比方法 | OASTER 均值 | 方法均值 | OASTER 优势 | 配对 bootstrap 95% CI | 胜/平/负 | Holm p | 配对秩二列 r |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary:
        lines.append(
            f"| {row['baseline']} | {row['oaster_mean']:.6f} | {row['baseline_mean']:.6f} | "
            f"{row['mean_oriented_difference']:+.6f} | "
            f"[{row['bootstrap_difference_ci_low']:+.6f}, {row['bootstrap_difference_ci_high']:+.6f}] | "
            f"{row['oaster_wins']}/{row['ties']}/{row['oaster_losses']} | "
            f"{row['holm_p_value_within_metric']:.3g} | {row['paired_rank_biserial']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "正的定向差值表示 OASTER 更好；检验和 Holm 校正只使用上表中的可用方法。",
            "",
            f"## {len(methods)} 方法总体检验",
            "",
            "| 指标 | Friedman χ² | 自由度 | Holm p（跨指标） | Kendall's W |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in omnibus:
        lines.append(
            f"| {row['metric_label']} | {row['friedman_chi_square']:.3f} | "
            f"{row['degrees_of_freedom']} | {row['holm_p_value_across_metrics']:.3g} | "
            f"{row['kendalls_w']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 关键描述统计",
            "",
            "| 方法 | An_auc 均值 ± SNR 间 SD | 中位数 [IQR] | 95% bootstrap CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in methods:
        row = lookup[("auc_tie_corrected", method)]
        lines.append(
            f"| {method} | {row['mean']:.6f} ± {row['sd_across_snr_pairs']:.6f} | "
            f"{row['median']:.6f} [{row['q25']:.6f}, {row['q75']:.6f}] | "
            f"[{row['bootstrap_mean_ci_low']:.6f}, {row['bootstrap_mean_ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "- 49 个 SNR 单元是预设实验网格，不是独立人群样本。",
            "- 推断只使用 miss-penalized 的表层/深层 SD、DLE。",
            "- `rmse` 沿用历史字段名，实际为平方相对 Frobenius 误差。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STRICT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.output, root=args.root)
    print(result)
    return result


if __name__ == "__main__":
    main()
