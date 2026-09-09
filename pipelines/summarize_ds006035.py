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
METHODS = (
    "OASTER Joint", "OASTER EEG", "OASTER MAG", "dSPM Joint", "eLORETA Joint",
)
COLORS = {
    "OASTER Joint": "#0072B2",
    "OASTER EEG": "#009E73",
    "OASTER MAG": "#E69F00",
    "dSPM Joint": "#CC79A7",
    "eLORETA Joint": "#D55E00",
}
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
) -> tuple[list[dict[str, object]], dict[str, list[int]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    trials: dict[str, list[int]] = {}
    locks: list[dict[str, object]] = []
    for subject in SUBJECTS:
        directory = root / f"sub-{subject}_final"
        comparison = _read_csv(directory / "method_comparison.csv")
        run_rows = _read_csv(directory / "run_metrics.csv")
        trials[subject] = [
            int(row["epochs"]) for row in run_rows if row["method"] == "OASTER Joint"
        ]
        by_method = {row["method"]: row for row in comparison}
        missing = set(METHODS) - set(by_method)
        if missing:
            raise ValueError(f"{subject}: missing methods {sorted(missing)}")
        for method in METHODS:
            row: dict[str, object] = {"subject": subject, "method": method}
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
    return rows, trials, locks


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


def paired_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {(row["subject"], row["method"]): row for row in rows}
    tests: list[dict[str, object]] = []
    for field, (_, higher_is_better) in METRICS.items():
        joint = np.asarray([lookup[(subject, "OASTER Joint")][field] for subject in SUBJECTS])
        dspm = np.asarray([lookup[(subject, "dSPM Joint")][field] for subject in SUBJECTS])
        delta = (joint - dspm) * (1 if higher_is_better else -1)
        non_ties = delta[~np.isclose(delta, 0.0)]
        wins = int(np.sum(non_ties > 0))
        p_value = binomtest(wins, len(non_ties), 0.5).pvalue if len(non_ties) else 1.0
        tests.append({
            "metric": field,
            "comparison": "OASTER Joint vs dSPM Joint",
            "oaster_wins": wins,
            "non_ties": len(non_ties),
            "median_advantage_for_oaster": np.median(delta),
            "exact_two_sided_sign_p": p_value,
        })
    return tests


def negative_transfer(rows: list[dict[str, object]], field: str) -> list[str]:
    lookup = {(row["subject"], row["method"]): row for row in rows}
    return [
        subject for subject in SUBJECTS
        if lookup[(subject, "OASTER Joint")][field]
        < max(lookup[(subject, "OASTER EEG")][field], lookup[(subject, "OASTER MAG")][field])
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_subjects(rows: list[dict[str, object]], output: Path) -> None:
    lookup = {(row["subject"], row["method"]): row for row in rows}
    fields = tuple(list(METRICS)[:4])
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(METHODS))
    for ax, field in zip(axes.flat, fields):
        title, _ = METRICS[field]
        matrix = np.asarray([
            [lookup[(subject, method)][field] for method in METHODS] for subject in SUBJECTS
        ], dtype=float)
        for subject, values in zip(SUBJECTS, matrix):
            ax.plot(x, values, color="#B4B4B4", alpha=0.65, linewidth=1, zorder=1)
            for index, (method, value) in enumerate(zip(METHODS, values)):
                ax.scatter(index, value, s=42, color=COLORS[method], edgecolor="white",
                           linewidth=0.7, zorder=2)
        medians = np.median(matrix, axis=0)
        q1, q3 = np.percentile(matrix, (25, 75), axis=0)
        ax.errorbar(x, medians, yerr=(medians - q1, q3 - medians), fmt="D",
                    color="black", markersize=5, capsize=4, linewidth=1.5, zorder=3)
        if "enrichment" in field:
            ax.axhline(1, color="#555555", linestyle="--", linewidth=1, label="area null = 1")
            ax.set_yscale("symlog", linthresh=0.02)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, loc="best")
        ax.set_title(title)
        ax.set_xticks(x, METHODS, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
    figure.suptitle("ds006035 somatomotor: subject-level results (diamond = median, bars = IQR)")
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    tests: list[dict[str, object]],
    trials: dict[str, list[int]],
    locks: list[dict[str, object]],
) -> None:
    grouped = {(row["method"], row["metric"]): row for row in summary}
    test_lookup = {row["metric"]: row for row in tests}
    subject_lookup = {(row["subject"], row["method"]): row for row in rows}
    n20_negative = negative_transfer(rows, "n20_left_s1_enrichment_median")
    p30_negative = negative_transfer(rows, "p30_left_s1_enrichment_median")
    locked = {
        method: sum(row["same_n20_p30_peak_runs"] for row in locks if row["method"] == method)
        for method in METHODS
    }

    def above_area_null(method: str, field: str) -> int:
        return sum(subject_lookup[(subject, method)][field] > 1 for subject in SUBJECTS)

    def value(method: str, field: str) -> str:
        row = grouped[(method, field)]
        return f"{row['median']:.3g} [{row['q1']:.3g}, {row['q3']:.3g}]"

    lines = [
        "# ds006035 同步 EEG-MEG 初步真实数据验证",
        "",
        f"共 5 名受试者、15 个 run，保留 {sum(map(sum, trials.values()))}/600 个触觉试次。统计单位是受试者；每名受试者先在 run 内完成定位，再取其 3 个 run 的中位数。",
        "",
        "## 核心结果",
        "",
        "| 方法 | N20 左 S1 富集，中位数 [IQR] | N20 峰距 mm | P30 左 S1 富集 | P30 峰距 mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {value(method, 'n20_left_s1_enrichment_median')} | "
            f"{value(method, 'n20_peak_euclidean_distance_to_left_s1_mm_median')} | "
            f"{value(method, 'p30_left_s1_enrichment_median')} | "
            f"{value(method, 'p30_peak_euclidean_distance_to_left_s1_mm_median')} |"
        )
    n20_test = test_lookup["n20_left_s1_enrichment_median"]
    p30_test = test_lookup["p30_left_s1_enrichment_median"]
    lines += [
        "",
        "## 结论",
        "",
        f"- N20 富集上，OASTER Joint 相对 dSPM 的受试者胜场为 {n20_test['oaster_wins']}/{n20_test['non_ties']}，精确双侧符号检验 p={n20_test['exact_two_sided_sign_p']:.4f}。",
        f"- P30 富集上，OASTER Joint 相对 dSPM 的受试者胜场为 {p30_test['oaster_wins']}/{p30_test['non_ties']}，精确双侧符号检验 p={p30_test['exact_two_sided_sign_p']:.4f}。",
        f"- N20 左 S1 富集超过面积零假设（>1）的受试者数：OASTER Joint {above_area_null('OASTER Joint', 'n20_left_s1_enrichment_median')}/5，dSPM {above_area_null('dSPM Joint', 'n20_left_s1_enrichment_median')}/5。",
        f"- 联合 OASTER 的 N20 富集低于同一受试者最佳单模态：{len(n20_negative)}/5（{', '.join(n20_negative) or '无'}）；P30 为 {len(p30_negative)}/5（{', '.join(p30_negative) or '无'}）。这是模态失配/负迁移的直接迹象。",
        f"- N20 与 P30 的峰顶点完全相同：OASTER Joint {locked['OASTER Joint']}/15、EEG {locked['OASTER EEG']}/15、MAG {locked['OASTER MAG']}/15；dSPM {locked['dSPM Joint']}/15、eLORETA {locked['eLORETA Joint']}/15。OASTER 的稀疏支持时间特异性不足。",
        "- 当前还原版 OASTER 没有通过真实瞬态体感诱发响应验证；现阶段不能声称优于标准逆解，也不应为了达到预设结果而调参。",
        "",
        "## 解释边界",
        "",
        "- 数据没有已知真实源，因此不计算 AUC、DLE；此处只报告左 S1 解剖富集、峰到左 S1 的欧氏距离和跨 run 一致性。",
        "- 峰距为 0 只表示估计峰落在左 S1 ROI 内，不是真实源误差为 0。",
        "- 数据包含个体 T1 MRI，但没有现成 FreeSurfer/BEM/trans；本次快速试跑尚未重建个体头模，而是使用刚性配准的 MNE sample 模板，只能作为算法筛查。",
        "- 本轮逆解只使用模板皮层表面源空间，没有加入丘脑等深部体积源；这是公开数据的皮层体感响应试跑，不等同于此前的表层/深层仿真矩阵。",
        "- MEG 分支只使用 102 个 magnetometer，未纳入 204 个 planar gradiometer；结论不能外推为全 MEG 通道组合的最终表现。",
        "- n=5，bootstrap 区间和符号检验均为探索性；不把 run 当独立样本。",
        "- dSPM/eLORETA 为 MNE 官方 inverse API；OASTER 为从 PPT/早期代码还原的当前版本。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _arguments()
    output = args.output or args.results / "group_final"
    output.mkdir(parents=True, exist_ok=True)
    rows, trials, locks = load_subject_rows(args.results)
    summary = summarize(rows)
    tests = paired_tests(rows)
    _write_csv(output / "subject_method_metrics.csv", rows)
    _write_csv(output / "group_method_comparison.csv", summary)
    _write_csv(output / "paired_sign_tests.csv", tests)
    _write_csv(output / "temporal_support_lock.csv", locks)
    plot_subjects(rows, output / "group_method_metrics.png")
    write_report(output / "GROUP_REPORT.md", rows, summary, tests, trials, locks)
    print(f"Wrote subject-level summary to {output}")


if __name__ == "__main__":
    main()
