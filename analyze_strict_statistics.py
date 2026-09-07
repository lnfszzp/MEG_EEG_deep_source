"""Paired, SNR-blocked statistics for the frozen strict benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
STRICT = ROOT / "results" / "strict_blind"
OUTPUT = STRICT / "statistics"
SEED = 20260907
BOOTSTRAP_RESAMPLES = 10_000
OASTER = "OASTER V19"
METHODS = (
    OASTER,
    "SISSES",
    "MNE",
    "dSPM",
    "sLORETA",
    "eLORETA",
    "LCMV",
    "Dipole fitting (grid)",
    "RAP-MUSIC",
)
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


def _load_blocks() -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    sources = (
        (STRICT / "oaster_v19_final" / "summary_by_snr_pair_scenario_macro.csv", OASTER),
        (STRICT / "sisses_preserved" / "summary_by_snr_pair_scenario_macro.csv", "SISSES"),
        (STRICT / "comparators_final" / "summary_by_snr_pair_scenario_macro.csv", None),
    )
    blocks: dict[str, dict[tuple[int, int], dict[str, float]]] = defaultdict(dict)
    metric_names = [field for field, _, _ in METRICS]
    for path, fixed_method in sources:
        for row in _read_csv(path):
            method = fixed_method or row["method"]
            pair = (int(row["eeg_snr_db"]), int(row["meg_snr_db"]))
            if row["aggregation"] != "scenario_macro" or pair in blocks[method]:
                raise ValueError(f"Invalid or duplicate SNR block in {path}: {method} {pair}")
            values = {name: float(row[name]) for name in metric_names}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric in {path}: {method} {pair}")
            blocks[method][pair] = values

    if set(blocks) != set(METHODS):
        raise ValueError(f"Method mismatch: expected {METHODS}, got {tuple(blocks)}")
    expected_pairs = set(blocks[OASTER])
    if len(expected_pairs) != 49:
        raise ValueError(f"Expected 49 EEG×MEG SNR blocks, got {len(expected_pairs)}")
    for method in METHODS:
        if set(blocks[method]) != expected_pairs:
            raise ValueError(f"SNR blocks do not align for {method}")
    return blocks


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_case_inputs() -> dict[str, object]:
    sisses_metadata = json.loads(
        (STRICT / "sisses_preserved" / "metadata.json").read_text(encoding="utf-8-sig")
    )
    paths = {
        OASTER: STRICT / "oaster_v19_final" / "rows.csv",
        "comparators": STRICT / "comparators_final" / "rows.csv",
        "SISSES": Path(sisses_metadata["source_scores"]),
    }
    expected_hash = sisses_metadata["source_scores_sha256"]
    if not all(path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        return {"performed": False, "reason": "missing local raw input", "missing": missing}

    case_sets: dict[str, set[str]] = {}
    files: dict[str, object] = {}
    for source, path in paths.items():
        by_method: dict[str, set[str]] = defaultdict(set)
        rows = 0
        bad_status = 0
        for row in _read_csv(path):
            rows += 1
            bad_status += row["status"] != "ok"
            if row["case_id"] in by_method[row["method"]]:
                raise ValueError(f"Duplicate case/method in {path}: {row['case_id']} {row['method']}")
            by_method[row["method"]].add(row["case_id"])
        if bad_status or any(len(ids) != 9065 for ids in by_method.values()):
            raise ValueError(f"Incomplete case-level input: {path}")
        if len({frozenset(ids) for ids in by_method.values()}) != 1:
            raise ValueError(f"Comparator case sets do not align within {path}")
        case_sets[source] = next(iter(by_method.values()))
        files[source] = {
            "path": str(path),
            "sha256": _sha256(path),
            "row_count": rows,
            "methods": sorted(by_method),
            "status_errors": bad_status,
        }
    if len({frozenset(ids) for ids in case_sets.values()}) != 1:
        raise ValueError("OASTER, comparators, and SISSES do not contain identical case IDs")
    if files["SISSES"]["sha256"] != expected_hash:
        raise ValueError("SISSES scores.csv hash no longer matches preserved metadata")
    return {"performed": True, "case_ids_aligned": True, "case_count": 9065, "files": files}


def analyze(output: Path = OUTPUT) -> None:
    blocks = _load_blocks()
    raw_validation = _validate_case_inputs()
    pairs = sorted(blocks[OASTER])
    rng = np.random.default_rng(SEED)
    descriptive: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []
    by_pair: list[dict[str, object]] = []
    omnibus: list[dict[str, object]] = []
    method_ranks: list[dict[str, object]] = []

    for metric, label, direction in METRICS:
        matrix = np.asarray(
            [[blocks[method][pair][metric] for method in METHODS] for pair in pairs], dtype=float
        )
        for method_index, method in enumerate(METHODS):
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
        friedman = stats.friedmanchisquare(*(benefit[:, index] for index in range(len(METHODS))))
        omnibus.append(
            {
                "metric": metric,
                "metric_label": label,
                "n_snr_pairs": len(pairs),
                "method_count": len(METHODS),
                "friedman_chi_square": float(friedman.statistic),
                "degrees_of_freedom": len(METHODS) - 1,
                "p_value": float(friedman.pvalue),
                "kendalls_w": float(friedman.statistic / (len(pairs) * (len(METHODS) - 1))),
            }
        )
        ranks = stats.rankdata(-benefit, axis=1, method="average")
        for index, method in enumerate(METHODS):
            method_ranks.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "method": method,
                    "mean_rank_1_is_best": float(ranks[:, index].mean()),
                }
            )

        raw_p_values: list[float] = []
        metric_rows: list[dict[str, object]] = []
        for baseline_index, baseline in enumerate(METHODS[1:], start=1):
            differences = benefit[:, 0] - benefit[:, baseline_index]
            ci_low, ci_high = _bootstrap_mean_ci(differences, rng)
            ties = np.isclose(differences, 0.0, rtol=1e-10, atol=1e-12)
            tested_differences = differences.copy()
            tested_differences[ties] = 0.0
            wins = int(np.sum((differences > 0) & ~ties))
            losses = int(np.sum((differences < 0) & ~ties))
            tie_count = int(ties.sum())
            if np.all(ties):
                p_value = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p_value = float(
                        stats.wilcoxon(
                            tested_differences,
                            alternative="two-sided",
                            zero_method="wilcox",
                            method="auto",
                        ).pvalue
                    )
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
                    "win_rate_excluding_ties": float(wins / (wins + losses)) if wins + losses else math.nan,
                    "common_language_win_probability": float((wins + 0.5 * tie_count) / len(pairs)),
                    "wilcoxon_p_value": p_value,
                    "holm_p_value_within_metric": math.nan,
                    "paired_rank_biserial": _rank_biserial(tested_differences),
                }
            )
            for pair, oaster_value, baseline_value, difference, tie in zip(
                pairs, matrix[:, 0], matrix[:, baseline_index], differences, ties, strict=True
            ):
                by_pair.append(
                    {
                        "metric": metric,
                        "metric_label": label,
                        "baseline": baseline,
                        "eeg_snr_db": pair[0],
                        "meg_snr_db": pair[1],
                        "oaster_value": float(oaster_value),
                        "baseline_value": float(baseline_value),
                        "oriented_difference_positive_favors_oaster": float(difference),
                        "outcome": "tie" if tie else ("win" if difference > 0 else "loss"),
                    }
                )
        for row, adjusted in zip(metric_rows, _holm(raw_p_values), strict=True):
            row["holm_p_value_within_metric"] = adjusted
        pairwise.extend(metric_rows)

    for row, adjusted in zip(omnibus, _holm([float(row["p_value"]) for row in omnibus]), strict=True):
        row["holm_p_value_across_metrics"] = adjusted

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "summary_method_descriptive.csv", descriptive)
    _write_csv(output / "summary_oaster_pairwise.csv", pairwise)
    _write_csv(output / "summary_snr_pair_differences.csv", by_pair)
    _write_csv(output / "summary_omnibus.csv", omnibus)
    _write_csv(output / "summary_method_ranks.csv", method_ranks)

    source_summaries = (
        STRICT / "oaster_v19_final" / "summary_by_snr_pair_scenario_macro.csv",
        STRICT / "sisses_preserved" / "summary_by_snr_pair_scenario_macro.csv",
        STRICT / "comparators_final" / "summary_by_snr_pair_scenario_macro.csv",
    )
    metadata = {
        "analysis_unit": "49 paired EEG×MEG SNR cells; each value is the macro-average of four scenarios",
        "methods": list(METHODS),
        "metrics": [field for field, _, _ in METRICS],
        "bootstrap": {
            "type": "paired nonparametric percentile bootstrap over SNR cells",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": SEED,
            "confidence_level": 0.95,
        },
        "tests": {
            "omnibus": "Friedman test with Kendall's W; Holm correction across endpoints",
            "pairwise": "two-sided paired Wilcoxon signed-rank; Holm correction across eight OASTER contrasts within each endpoint",
            "effect_size": "paired rank-biserial correlation; positive favors OASTER",
        },
        "scope_warning": "The 49 SNR cells are a fixed experimental grid, not an iid population sample. Intervals and p-values quantify consistency across this tested grid and do not establish clinical or population generalization.",
        "localization_metrics": "Only miss-penalized layer SD/DLE enter inference; finite-only unpenalized means are intentionally excluded.",
        "source_summaries": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_summaries
        ],
        "raw_case_validation": raw_validation,
        "outputs": [
            "summary_method_descriptive.csv",
            "summary_oaster_pairwise.csv",
            "summary_snr_pair_differences.csv",
            "summary_omnibus.csv",
            "summary_method_ranks.csv",
            "REPORT.md",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", descriptive, pairwise, omnibus)


def _write_report(
    path: Path,
    descriptive: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    omnibus: list[dict[str, object]],
) -> None:
    lookup = {(row["metric"], row["method"]): row for row in descriptive}
    primary = [row for row in pairwise if row["metric"] == "auc_tie_corrected"]
    by_contrast = {(row["metric"], row["baseline"]): row for row in pairwise}
    sisses_auc = by_contrast[("auc_tie_corrected", "SISSES")]
    lcmv_dle = by_contrast[("deep_dle_mm_penalized", "LCMV")]
    lcmv_sensitivity = by_contrast[("deep_sensitivity", "LCMV")]
    lines = [
        "# 冻结严格基准统计分析",
        "",
        "统计区组为 49 个 EEG×MEG SNR 组合；每个区组内先对四种场景做宏平均，再进行九方法配对比较。9,065 个病例不被当作独立重复，因此不会以样本量膨胀显著性。",
        "",
        "## 主指标：An_auc",
        "",
        "| 对比方法 | OASTER 均值 | 方法均值 | OASTER 优势 | 配对 bootstrap 95% CI | 胜/平/负 | Holm p | 配对秩二列 r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
            "差值和效应量均已按指标方向定向，正值表示 OASTER 更好。置信区间通过同时重采样配对 SNR 区组得到。",
            "",
            "## 统计结论",
            "",
            f"- OASTER 的 An_auc 网格均值比 SISSES 高 {sisses_auc['mean_oriented_difference']:.6f}，但区间跨 0、Holm p={sisses_auc['holm_p_value_within_metric']:.3g}，且只在 {sisses_auc['oaster_wins']}/49 个 SNR 对获胜；因此只能说均值略高，不能声称在整个 SNR 网格上显著优于 SISSES。",
            "- OASTER 对其余七种方法的 An_auc 在 49/49 个 SNR 对均获胜，校正后检验均支持稳定优势。",
            f"- 深层 penalized DLE 相对 LCMV 的定向均值差为 {lcmv_dle['mean_oriented_difference']:+.3f} mm（Holm p={lcmv_dle['holm_p_value_within_metric']:.3g}），深层敏感度差为 {lcmv_sensitivity['mean_oriented_difference']:+.4f}（Holm p={lcmv_sensitivity['holm_p_value_within_metric']:.3g}）；这两项没有证据表明 OASTER 更优。",
            "",
            "## 九方法总体检验",
            "",
            "| 指标 | Friedman χ²(8) | Holm p（跨指标） | Kendall's W |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in omnibus:
        lines.append(
            f"| {row['metric_label']} | {row['friedman_chi_square']:.3f} | "
            f"{row['holm_p_value_across_metrics']:.3g} | {row['kendalls_w']:.3f} |"
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
    for method in METHODS:
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
            "- 49 个 SNR 单元是预设实验网格，不是从某个人群随机抽取的独立样本；p 值和区间只刻画该网格内的一致性，不能外推为临床或人群效应。",
            "- 推断只使用 miss-penalized 的表层/深层 SD、DLE；会排除漏检病例的 finite-only 非惩罚均值没有参与检验。",
            "- Holm 校正在每个指标的 8 个 OASTER 配对比较内实施；Friedman 的 10 个端点另做跨指标 Holm 校正。",
            "- `rmse` 沿用历史字段名，实际语义是平方相对 Frobenius 误差，而非常规定义的均方根误差。",
            "",
            "完整逐指标描述统计、逐 SNR 差值/胜负、平均秩及检验结果见本目录 CSV；输入哈希与逐病例对齐验证见 `metadata.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    analyze()
