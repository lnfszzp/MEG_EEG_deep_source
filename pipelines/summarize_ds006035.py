"""Summarize ds006035 results with the subject, not the run, as the unit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "real_data" / "ds006035"
SUBJECTS = ("sm04", "sm06", "sm07", "sm09", "sm12")
DEVELOPMENT_SUBJECT = "sm09"
VALIDATION_SUBJECTS = tuple(subject for subject in SUBJECTS if subject != DEVELOPMENT_SUBJECT)
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
MARKERS = dict(zip(METHODS, ("s", "D", "^", "v", "P", "X")))
METRICS = {
    "n20_left_s1_enrichment_median": ("N20 left-S1 enrichment", True),
    "n20_peak_euclidean_distance_to_left_s1_mm_median": ("N20 peak distance (mm)", False),
    "p30_left_s1_enrichment_median": ("P30 left-S1 enrichment", True),
    "p30_peak_euclidean_distance_to_left_s1_mm_median": ("P30 peak distance (mm)", False),
    "n20_run_map_spearman_mean": ("N20 run-map Spearman", True),
    "p30_run_map_spearman_mean": ("P30 run-map Spearman", True),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_subject_rows(
    root: Path,
) -> tuple[
    list[dict[str, object]], dict[str, list[int]], list[dict[str, object]],
    list[dict[str, str]],
]:
    rows: list[dict[str, object]] = []
    trials: dict[str, list[int]] = {}
    locks: list[dict[str, object]] = []
    all_run_rows: list[dict[str, str]] = []
    for subject in SUBJECTS:
        directory = root / f"sub-{subject}_final"
        comparison = _read_csv(directory / "method_comparison.csv")
        run_rows = _read_csv(directory / "run_metrics.csv")
        all_run_rows.extend({**row, "subject": subject} for row in run_rows)
        trials[subject] = [
            int(row["epochs"]) for row in run_rows if row["method"] == "OASTER ERP Joint"
        ]
        by_method = {row["method"]: row for row in comparison}
        missing = set(METHODS) - set(by_method)
        if missing:
            raise ValueError(f"{subject}: missing methods {sorted(missing)}")
        for method in METHODS:
            row: dict[str, object] = {
                "subject": subject,
                "cohort": "development" if subject == DEVELOPMENT_SUBJECT else "locked_validation",
                "method": method,
            }
            row.update({field: float(by_method[method][field]) for field in METRICS})
            rows.append(row)
        with np.load(directory / "source_maps.npz") as archive:
            for method in METHODS:
                key = method.lower().replace(" ", "_").replace("-", "_")
                n20_peak = np.argmax(archive[f"n20_{key}"], axis=1)
                p30_peak = np.argmax(archive[f"p30_{key}"], axis=1)
                locks.append({
                    "subject": subject,
                    "method": method,
                    "same_n20_p30_peak_runs": int(np.sum(n20_peak == p30_peak)),
                    "runs": len(n20_peak),
                })
    return rows, trials, locks, all_run_rows


def _bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(10_000, len(values)), replace=True)
    return tuple(np.percentile(np.median(draws, axis=1), (2.5, 97.5)))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = np.random.default_rng(6035)
    summary: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        for field in METRICS:
            values = np.asarray([row[field] for row in selected], dtype=float)
            ci_low, ci_high = _bootstrap_median_ci(values, rng)
            summary.append({
                "method": method,
                "metric": field,
                "subjects": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": np.median(values),
                "q1": np.percentile(values, 25),
                "q3": np.percentile(values, 75),
                "bootstrap_median_ci_low": ci_low,
                "bootstrap_median_ci_high": ci_high,
            })
    return summary


def paired_tests(
    rows: list[dict[str, object]], subjects=SUBJECTS
) -> list[dict[str, object]]:
    lookup = {(row["subject"], row["method"]): row for row in rows}
    tests: list[dict[str, object]] = []
    for field, (_, higher_is_better) in METRICS.items():
        joint = np.asarray([lookup[(subject, "OASTER ERP Joint")][field] for subject in subjects])
        dspm = np.asarray([lookup[(subject, "dSPM Joint")][field] for subject in subjects])
        delta = (joint - dspm) * (1 if higher_is_better else -1)
        non_ties = delta[~np.isclose(delta, 0.0)]
        wins = int(np.sum(non_ties > 0))
        p_value = binomtest(wins, len(non_ties), 0.5).pvalue if len(non_ties) else 1.0
        tests.append({
            "metric": field,
            "comparison": "OASTER ERP Joint vs dSPM Joint",
            "oaster_wins": wins,
            "non_ties": len(non_ties),
            "median_advantage_for_oaster": np.median(delta),
            "exact_two_sided_sign_p": p_value,
        })
    return tests


def negative_transfer(
    rows: list[dict[str, object]], field: str, subjects=SUBJECTS
) -> list[str]:
    lookup = {(row["subject"], row["method"]): row for row in rows}
    return [
        subject for subject in subjects
        if lookup[(subject, "OASTER Joint")][field]
        < max(lookup[(subject, "OASTER EEG")][field], lookup[(subject, "OASTER MAG")][field])
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_subjects(
    rows: list[dict[str, object]], output: Path, subjects=SUBJECTS
) -> None:
    lookup = {(row["subject"], row["method"]): row for row in rows}
    fields = tuple(list(METRICS)[:4])
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(METHODS))
    for ax, field in zip(axes.flat, fields):
        title, _ = METRICS[field]
        matrix = np.asarray([
            [lookup[(subject, method)][field] for method in METHODS] for subject in subjects
        ], dtype=float)
        for subject, values in zip(subjects, matrix):
            ax.plot(x, values, color="#B4B4B4", alpha=0.65, linewidth=1, zorder=1)
            for index, (method, value) in enumerate(zip(METHODS, values)):
                ax.scatter(index, value, s=42, marker=MARKERS[method], color=COLORS[method], edgecolor="white",
                           linewidth=0.7, zorder=2)
        medians = np.median(matrix, axis=0)
        q1, q3 = np.percentile(matrix, (25, 75), axis=0)
        ax.errorbar(x, medians, yerr=(medians - q1, q3 - medians), fmt="D",
                    color="black", markersize=5, capsize=4, linewidth=1.5, zorder=3)
        if "enrichment" in field:
            ax.axhline(1, color="#555555", linestyle="--", linewidth=1, label="uniform-source null = 1")
            ax.set_yscale("symlog", linthresh=0.02)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, loc="best")
        ax.set_title(title)
        ax.set_xticks(x, METHODS, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
    figure.suptitle("ds006035 locked validation (n=4; diamond = median, bars = IQR)")
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    tests: list[dict[str, object]],
    trials: dict[str, list[int]],
    locks: list[dict[str, object]],
    run_rows: list[dict[str, str]],
) -> None:
    grouped = {(row["method"], row["metric"]): row for row in summary}
    test_lookup = {row["metric"]: row for row in tests}
    subject_lookup = {(row["subject"], row["method"]): row for row in rows}
    n20_negative = negative_transfer(
        rows, "n20_left_s1_enrichment_median", VALIDATION_SUBJECTS
    )
    p30_negative = negative_transfer(
        rows, "p30_left_s1_enrichment_median", VALIDATION_SUBJECTS
    )
    locked = {
        method: sum(
            row["same_n20_p30_peak_runs"]
            for row in locks
            if row["method"] == method and row["subject"] in VALIDATION_SUBJECTS
        )
        for method in METHODS
    }

    def above_area_null(method: str, field: str) -> int:
        return sum(
            subject_lookup[(subject, method)][field] > 1
            for subject in VALIDATION_SUBJECTS
        )

    def hit_counts(method: str, prefix: str, distance_limit: float) -> tuple[int, int]:
        selected = [
            row for row in run_rows
            if row["subject"] in VALIDATION_SUBJECTS and row["method"] == method
        ]
        hits = sum(
            float(row[f"{prefix}_left_s1_enrichment"]) > 1.0
            and float(row[f"{prefix}_peak_euclidean_distance_to_left_s1_mm"])
            <= distance_limit
            for row in selected
        )
        return hits, len(selected)

    def value(method: str, field: str) -> str:
        row = grouped[(method, field)]
        return f"{row['median']:.3g} [{row['q1']:.3g}, {row['q3']:.3g}]"

    lines = [
        "# ds006035 同步 EEG-MEG 初步真实数据验证",
        "",
        f"sm09 仅用于开发；锁定验证为其余 4 名受试者、12 个 run，保留 {sum(sum(trials[subject]) for subject in VALIDATION_SUBJECTS)}/480 个触觉试次。统计单位是受试者；每名受试者先在 run 内完成定位，再取其 3 个 run 的中位数。下表只包含锁定验证。",
        "",
        "## 核心结果",
        "",
        "| 方法 | N20 左 S1 富集，中位数 [IQR] | N20 峰距 mm | N20 run-map ρ | P30 左 S1 富集 | P30 峰距 mm | P30 run-map ρ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {value(method, 'n20_left_s1_enrichment_median')} | "
            f"{value(method, 'n20_peak_euclidean_distance_to_left_s1_mm_median')} | "
            f"{value(method, 'n20_run_map_spearman_mean')} | "
            f"{value(method, 'p30_left_s1_enrichment_median')} | "
            f"{value(method, 'p30_peak_euclidean_distance_to_left_s1_mm_median')} | "
            f"{value(method, 'p30_run_map_spearman_mean')} |"
        )
    n20_test = test_lookup["n20_left_s1_enrichment_median"]
    p30_test = test_lookup["p30_left_s1_enrichment_median"]
    n20_hits = {method: hit_counts(method, "n20", 20.0) for method in METHODS}
    p30_hits = {method: hit_counts(method, "p30", 10.0) for method in METHODS}
    lines += [
        "",
        "## 结论",
        "",
        f"- N20 富集上，OASTER ERP Joint 相对 dSPM 的受试者胜场为 {n20_test['oaster_wins']}/{n20_test['non_ties']}，精确双侧符号检验 p={n20_test['exact_two_sided_sign_p']:.4f}。",
        f"- P30 富集上，OASTER ERP Joint 相对 dSPM 的受试者胜场为 {p30_test['oaster_wins']}/{p30_test['non_ties']}，精确双侧符号检验 p={p30_test['exact_two_sided_sign_p']:.4f}。",
        f"- N20 左 S1 富集超过均匀源点零假设（>1）的受试者数：OASTER ERP Joint {above_area_null('OASTER ERP Joint', 'n20_left_s1_enrichment_median')}/4，dSPM {above_area_null('dSPM Joint', 'n20_left_s1_enrichment_median')}/4。当前迁移版没有稳定解决 N20。",
        f"- 按预先固定的‘富集>1且峰距达标’口径，N20 run命中为 ERP {n20_hits['OASTER ERP Joint'][0]}/{n20_hits['OASTER ERP Joint'][1]}、原频谱OASTER {n20_hits['OASTER Joint'][0]}/{n20_hits['OASTER Joint'][1]}、dSPM {n20_hits['dSPM Joint'][0]}/{n20_hits['dSPM Joint'][1]}、eLORETA {n20_hits['eLORETA Joint'][0]}/{n20_hits['eLORETA Joint'][1]}；P30依次为 {p30_hits['OASTER ERP Joint'][0]}/{p30_hits['OASTER ERP Joint'][1]}、{p30_hits['OASTER Joint'][0]}/{p30_hits['OASTER Joint'][1]}、{p30_hits['dSPM Joint'][0]}/{p30_hits['dSPM Joint'][1]}、{p30_hits['eLORETA Joint'][0]}/{p30_hits['eLORETA Joint'][1]}。",
        f"- 原频谱联合 OASTER 的 N20 富集低于同一受试者最佳单模态：{len(n20_negative)}/4（{', '.join(n20_negative) or '无'}）；P30 为 {len(p30_negative)}/4（{', '.join(p30_negative) or '无'}）。这是模态失配/负迁移的直接迹象。",
        f"- N20 与 P30 的峰顶点完全相同：OASTER ERP Joint {locked['OASTER ERP Joint']}/12、原 OASTER Joint {locked['OASTER Joint']}/12、EEG {locked['OASTER EEG']}/12、MAG {locked['OASTER MAG']}/12；dSPM {locked['dSPM Joint']}/12、eLORETA {locked['eLORETA Joint']}/12。",
        "- OASTER ERP 对 N20、P30 分别进行多尺度时域基提取、EBIC 空间选择和独立的完整有符号时域回归；原频谱 OASTER 仍保留用于振荡/诱发功率问题。",
        "- ERP 分支每个窗口限定一个主模板；富集恰为 37 的结果是单个模板完全落入左 S1 的离散饱和值，不能按连续精度或泛化证据解释。",
        "",
        "## 解释边界",
        "",
        "- 数据没有已知真实源，因此不计算 AUC、DLE；此处只报告左 S1 解剖富集、峰到左 S1 的欧氏距离和跨 run 一致性。",
        "- 峰距为 0 只表示估计峰落在左 S1 ROI 内，不是真实源误差为 0。",
        "- 数据包含个体 T1 MRI，但没有现成 FreeSurfer/BEM/trans；本次快速试跑尚未重建个体头模，而是使用刚性配准的 MNE sample 模板，只能作为算法筛查。",
        "- 本轮逆解只使用模板皮层表面源空间，没有加入丘脑等深部体积源；这是公开数据的皮层体感响应试跑，不等同于此前的表层/深层仿真矩阵。",
        "- MEG 分支只使用 102 个 magnetometer，未纳入 204 个 planar gradiometer；结论不能外推为全 MEG 通道组合的最终表现。",
        "- 锁定验证 n=4，精确符号检验的最小双侧 p 值也只有 0.125；bootstrap 区间和检验均为探索性，不把 run 当独立样本。",
        "- 当首个空间模板没有改善 EBIC 时，当前实现仍会在已检测到传感器时间成分的窗口保留一个最强模板；诊断中的正 EBIC delta 必须解释为强制模板，而非 EBIC 支持。",
        "- top-5% Dice 已改为只在严格正支持内取前5%，避免稀疏图的并列零值虚高；本轮主判断仍以富集、峰距和 Spearman 为主。",
        "- dSPM/eLORETA 为 MNE 官方 inverse API；OASTER ERP 为迁移了多尺度时间基、EBIC空间选择和深源条件补救的时域版本；OASTER Joint 为从 PPT/早期代码还原的频谱版本。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _arguments()
    output = args.output or args.results / "group_final"
    output.mkdir(parents=True, exist_ok=True)
    rows, trials, locks, run_rows = load_subject_rows(args.results)
    locked_rows = [row for row in rows if row["subject"] in VALIDATION_SUBJECTS]
    summary = summarize(locked_rows)
    tests = paired_tests(locked_rows, VALIDATION_SUBJECTS)
    _write_csv(output / "subject_method_metrics.csv", rows)
    _write_csv(output / "group_method_comparison.csv", summary)
    _write_csv(output / "paired_sign_tests.csv", tests)
    _write_csv(output / "temporal_support_lock.csv", locks)
    plot_subjects(locked_rows, output / "group_method_metrics.png", VALIDATION_SUBJECTS)
    write_report(
        output / "GROUP_REPORT.md", rows, summary, tests, trials, locks, run_rows
    )
    print(f"Wrote subject-level summary to {output}")


if __name__ == "__main__":
    main()
