"""# %% 重画balanced20完整盲定位结果，并补充分SNR统计。"""

# %% 读取已经完成的定位结果；不重算、不改任何病例决定。
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/blind_gate_combined40_localization_excess_balanced_full20"
summary_path = output / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["complete"] and summary["case_count"] == 20
with (output / "metrics_table.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 20
numeric = ("eeg_snr_db", "meg_snr_db", "predictive_score", "development_threshold",
           "deep_present_decision", "has_deep_true_posthoc", "family_correct_posthoc",
           "conditional_local_AUC", "surface_An_auc", "deep_An_auc", "surface_SD_mm",
           "surface_DLE_mm", "deep_DLE_mm", "active_count")
for row in rows:
    for name in numeric:
        row[name] = float(row[name])

# %% 分SNR统计，DLE只在存在表层真值时汇总。
pairs = [(-10, -10), (-10, 20), (20, -10), (5, 5)]
pair_labels = ["−10/−10", "−10/+20", "+20/−10", "+5/+5"]
snr_rows = []
for pair, label in zip(pairs, pair_labels):
    subset = [row for row in rows if (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    surface = [row for row in subset if np.isfinite(row["surface_DLE_mm"])]
    deep = [row for row in subset if row["has_deep_true_posthoc"]]
    snr_rows.append({"eeg_meg_snr_db": label, "case_count": len(subset),
        "minimum_local_AUC": min(row["conditional_local_AUC"] for row in subset),
        "mean_local_AUC": float(np.mean([row["conditional_local_AUC"] for row in subset])),
        "mean_surface_SD_mm": float(np.mean([row["surface_SD_mm"] for row in surface])),
        "mean_surface_DLE_mm": float(np.mean([row["surface_DLE_mm"] for row in surface])),
        "maximum_surface_DLE_mm": max(row["surface_DLE_mm"] for row in surface),
        "maximum_deep_DLE_mm": max(row["deep_DLE_mm"] for row in deep)})
with (output / "snr_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=snr_rows[0].keys())
    writer.writeheader()
    writer.writerows(snr_rows)

# %% 门控、AUC、DLE和SD-DLE关系使用不同图形呈现，纵轴不裁掉大DLE。
figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)
for index, pair in enumerate(pairs):
    subset = [row for row in rows if (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    for deep, marker, color, label in ((0, "s", "#007C83", "No deep source"),
                                        (1, "D", "#D55E00", "Deep source present")):
        values = [row["predictive_score"] for row in subset if row["has_deep_true_posthoc"] == deep]
        axes[0, 0].scatter(index + np.linspace(-.10, .10, len(values)), values,
                           marker=marker, color=color, s=58, edgecolor="white", linewidth=.7,
                           label=label if index == 0 else None, zorder=3)
axes[0, 0].axhline(summary["development_threshold"], color="#C44E52",
                    linestyle=":", linewidth=1.8, label="Development threshold")
axes[0, 0].set(yscale="log", xticks=np.arange(4), xticklabels=pair_labels,
               ylabel="Excess-loss fraction U", title="Independent blind-family gate")
axes[0, 0].legend(frameon=False, fontsize=9)

auc_groups = [[row["conditional_local_AUC"] for row in rows
               if (row["eeg_snr_db"], row["meg_snr_db"]) == pair] for pair in pairs]
axes[0, 1].boxplot(auc_groups, tick_labels=pair_labels, widths=.55, patch_artist=True,
                   boxprops={"facecolor": "#B8DEE0", "edgecolor": "#007C83"},
                   medianprops={"color": "#D55E00", "linewidth": 2})
for index, values in enumerate(auc_groups, 1):
    axes[0, 1].scatter(index + np.linspace(-.08, .08, len(values)), values,
                       color="#007C83", s=28, zorder=3)
axes[0, 1].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.6)
axes[0, 1].set(ylabel="Local An_cal_AUC", ylim=(.88, 1.01), title="Localization discrimination")

for index, pair in enumerate(pairs):
    values = [row["surface_DLE_mm"] for row in rows
              if (row["eeg_snr_db"], row["meg_snr_db"]) == pair
              and np.isfinite(row["surface_DLE_mm"])]
    axes[1, 0].scatter(index + np.linspace(-.08, .08, len(values)), values,
                       color="#007C83", marker="s", s=48)
axes[1, 0].set(xticks=np.arange(4), xticklabels=pair_labels, ylabel="Surface DLE (mm)",
               ylim=(0, 1.12 * max(row["surface_DLE_mm"] for row in rows
                                   if np.isfinite(row["surface_DLE_mm"]))),
               title="Surface peak error (uncropped)")

surface_rows = [row for row in rows if np.isfinite(row["surface_DLE_mm"])]
colors = ["#C44E52" if (row["eeg_snr_db"], row["meg_snr_db"]) == (20, -10)
          else "#007C83" for row in surface_rows]
axes[1, 1].scatter([row["surface_SD_mm"] for row in surface_rows],
                   [row["surface_DLE_mm"] for row in surface_rows], color=colors, s=48)
axes[1, 1].set(xlabel="Surface SD (mm)", ylabel="Surface DLE (mm)",
               title="Spread versus peak error")
figure.suptitle("Balanced20 blind gate + combined40 localization (development)", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

# %% 修正摘要措辞并记录未裁剪的最差定位误差。
summary["decision_statistic"] = "independent confirmation excess-loss fraction"
summary["threshold_status"] = "chosen inside the labeled development gap; must be replaced by frozen null calibration"
summary["localization_script_sha256"] = hashlib.sha256(
    (root / "作者风格版/48-独立门控后40试次盲定位.py").read_bytes()).hexdigest()
summary["minimum_local_AUC"] = min(row["conditional_local_AUC"] for row in rows)
summary["maximum_surface_DLE_mm"] = max(row["surface_DLE_mm"] for row in surface_rows)
summary["maximum_surface_SD_mm"] = max(row["surface_SD_mm"] for row in surface_rows)
summary["snr_summary"] = snr_rows
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = ["# balanced20完整盲定位（开发闭环）", "",
    "train20拟合H0/H1，独立confirmation20以U盲选family，随后combined40重定位。真值只用于本报告的后验评价。", "",
    "| EEG/MEG SNR | 最低/平均局部AUC | 平均/最大表层DLE mm | 平均表层SD mm | 最大深层DLE mm |",
    "|---|---:|---:|---:|---:|"]
for row in snr_rows:
    lines.append(f"| {row['eeg_meg_snr_db']} | {row['minimum_local_AUC']:.3f}/{row['mean_local_AUC']:.3f} | "
        f"{row['mean_surface_DLE_mm']:.2f}/{row['maximum_surface_DLE_mm']:.2f} | "
        f"{row['mean_surface_SD_mm']:.2f} | {row['maximum_deep_DLE_mm']:.2f} |")
lines += ["", "盲选正确20/20；纯表层误报0/8；深源检出12/12；所选family全部收敛；所有局部AUC≥0.90；所有深源DLE=0。",
    f"当前最差局部AUC={summary['minimum_local_AUC']:.3f}，最大表层DLE={summary['maximum_surface_DLE_mm']:.2f} mm。+20/−10的表层DLE系统性偏高，仍需改进定位后处理。",
    "0.12看过开发标签，正式校准/验证尚未消费，accepted=false。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"minimum_local_AUC": summary["minimum_local_AUC"],
                  "maximum_surface_DLE_mm": summary["maximum_surface_DLE_mm"],
                  "snr_summary": snr_rows}, ensure_ascii=True, indent=2))
