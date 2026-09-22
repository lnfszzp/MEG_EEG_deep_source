"""Plot the 20-case development pilot; SNR repeats are not independent subjects."""

# %% 输入：只读取已完成的五 SNR 小样本，不启动仿真。
import argparse
import csv
from pathlib import Path

import numpy as np
import plot_erp_whole_head_results as style

plt = style.plt
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=root / "results/erp_whole_head/adaptive_v4/pilot_five_snr_all_methods")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
output = args.output or args.input / "figures_pilot"
methods = ("OASTER-ERP-v4", *style.METHODS[1:])
pairs = ((-10, -10), (-10, 20), (5, 5), (20, -10), (20, 20))
scenario_names = ("Surface only", "Deep only", "Deep + surface", "Deep + 2 surfaces")
tables = {}
for name in ("rows", "summary_by_snr_pair_scenario_macro", "summary_by_snr_scenario"):
    with (args.input / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
        tables[name] = list(csv.DictReader(stream))
rows = tables["rows"]
macro = tables["summary_by_snr_pair_scenario_macro"]
scenarios = tables["summary_by_snr_scenario"]
assert len(rows) == 20 * 8 and all(row["status"] == "ok" for row in rows)
assert len({row["case_id"] for row in rows}) == 20
assert len({row["configuration_id"] for row in rows}) == 4
assert {(int(row["eeg_snr_db"]), int(row["meg_snr_db"])) for row in rows} == set(pairs)
assert {row["method"] for row in rows} == set(methods)
assert {row["scenario"] for row in rows} == set(style.SCENARIOS)
assert len({(row["case_id"], row["method"]) for row in rows}) == 160
assert len(macro) == 40 and len(scenarios) == 160
assert len({(row["method"], row["eeg_snr_db"], row["meg_snr_db"]) for row in macro}) == 40
assert len({(row["method"], row["scenario"], row["eeg_snr_db"], row["meg_snr_db"]) for row in scenarios}) == 160
output.mkdir(parents=True, exist_ok=True)
style._style()
colors = [style.COLORS[style._canonical_method(method)] for method in methods]
labels = [style.DISPLAY.get(method, method) for method in methods]
title = "20-case development pilot | 4 configurations | five SNR pairs"


# %% 逐方法指标表。均值先对场景等权，再对五个 SNR 格等权；无独立样本统计。
metric_names = (
    "auc_tie_corrected", "surface_auc_tie_corrected", "deep_auc_tie_corrected",
    "surface_sd_mm_penalized", "surface_dle_mm_penalized",
    "deep_sd_mm_penalized", "deep_dle_mm_penalized",
    "deep_sensitivity", "deep_specificity", "deep_balanced_accuracy",
)
comparison = []
for method in methods:
    selected = [row for row in macro if row["method"] == method]
    assert len(selected) == 5
    record = {"method": method, "case_count": 20, "configuration_count": 4, "snr_pair_count": 5}
    for metric in metric_names:
        record[f"{metric}_mean"] = style._finite_mean([float(row[metric]) for row in selected])
    values = np.array([float(row["auc_tie_corrected"]) for row in selected])
    record.update(an_auc_min_snr_cell=float(values.min()), an_auc_max_snr_cell=float(values.max()))
    comparison.append(record)
with (output / "eight_method_pilot_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
    writer.writeheader()
    writer.writerows(comparison)


# %% 总体/表深层 AUC、表深层 SD/DLE 和深源检出的并列点图。
panels = (
    (("auc_tie_corrected", "Overall", "^"), ("surface_auc_tie_corrected", "Surface", "o"),
     ("deep_auc_tie_corrected", "Deep", "s")),
    (("surface_sd_mm_penalized", "SD", "o"), ("surface_dle_mm_penalized", "DLE", "s")),
    (("deep_sd_mm_penalized", "SD", "o"), ("deep_dle_mm_penalized", "DLE", "s")),
    (("deep_sensitivity", "Sensitivity", "o"), ("deep_specificity", "Specificity", "s")),
)
figure, axes = plt.subplots(1, 4, figsize=(18, 6.5), sharey=True)
for axis, panel, panel_title in zip(axes, panels, ("An_auc", "Surface distances", "Deep distances", "Deep detection")):
    for offset, (metric, label, marker) in zip(np.linspace(-0.18, 0.18, len(panel)), panel):
        values = [record[f"{metric}_mean"] for record in comparison]
        axis.scatter(values, np.arange(8) + offset, c=colors, marker=marker, s=58,
                     edgecolors="white", linewidths=0.6, zorder=3)
        axis.scatter([], [], color="#555555", marker=marker, label=label, s=42)
    axis.set_title(panel_title, pad=12)
    axis.set_yticks(np.arange(8), labels)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=len(panel))
    axis.set_ylim(7.5, -0.5)
    axis.spines[["top", "right"]].set_visible(False)
    if "distances" in panel_title:
        axis.set_xlabel("Miss-penalized distance (mm); lower is better")
        axis.set_xlim(left=0)
    else:
        axis.set_xlabel("Score; higher is better")
        axis.set_xlim(-0.03, 1.03)
figure.suptitle(title, fontsize=15)
figure.text(0.5, 0.02, "Descriptive means only; repeated SNRs are not independent subjects. Misses retain the frozen distance penalty.", ha="center", fontsize=9)
figure.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.23, wspace=0.24)
figure.savefig(output / "eight_method_pilot_metrics.png", dpi=180, facecolor="white")
plt.close(figure)


# %% 每种方法四场景 × 五 SNR 的 An_auc 热图，保留每个小样本格的实际值。
figure, axes = plt.subplots(2, 4, figsize=(17, 8.2), layout="constrained")
for axis, method, label in zip(axes.flat, methods, labels):
    lookup = {
        (row["scenario"], int(row["eeg_snr_db"]), int(row["meg_snr_db"])): float(row["auc_tie_corrected"])
        for row in scenarios if row["method"] == method
    }
    matrix = np.array([[lookup[(scenario, *pair)] for pair in pairs] for scenario in style.SCENARIOS])
    heat = axis.imshow(matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    axis.set_title(label)
    axis.set_xticks(range(5), [f"{eeg:+d}/{meg:+d}" for eeg, meg in pairs], rotation=35)
    axis.set_yticks(range(4), scenario_names)
    axis.set_xlabel("EEG / MEG SNR (dB)")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                  color="white" if value < 0.5 else "#111111", fontsize=9)
figure.colorbar(heat, ax=list(axes.flat), shrink=0.85, label="An_auc (0–1)")
figure.suptitle(title + "\nOne configuration per scenario, repeated across SNR; exploratory results", fontsize=14)
figure.savefig(output / "four_scenarios_five_snr_pilot.png", dpi=180, facecolor="white")
plt.close(figure)
print("Saved pilot comparison table and two figures:", output)
