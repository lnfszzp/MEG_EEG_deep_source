"""# %% 汇总固定 DCT 下逐模源、逐模边和完整 SISSES 的开发结果。"""

# %% 路径和不覆盖规则；只读取两个预先指定的 -10/-10 dB 难例。
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/sisses_source_edge_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧汇总：{output}")
output.mkdir(parents=True)

directories = {
    "Convex group": "dev_component_balanced_convex_admm_mrf080_m10_m10_case{case}",
    "Per-mode edge": "dev_component_balanced_sissesedge_admm_mrf080_m10_m10_case{case}",
    "Per-mode source": "dev_component_balanced_sissessource_admm_mrf080_m10_m10_case{case}",
    "Full SISSES": "dev_component_balanced_sissesfull_admm_mrf080_m10_m10_case{case}",
}

# %% case01 用 H0，case03 用 H1；同时保留纯表层病例 H1 的深源误报诊断。
records = []
for method, pattern in directories.items():
    for case, selected in (("01", "v6-surface-only"), ("03", "v6-full-ungated")):
        directory = base / pattern.format(case=case)
        rows = list(csv.DictReader((directory / "rows.csv").open(encoding="utf-8-sig")))
        row = next(item for item in rows if item["method"] == selected)
        negative_h1 = (next(item for item in rows if item["method"] == "v6-full-ungated")
                       if case == "01" else None)
        evidence = list(csv.DictReader((directory / "evidence.csv").open(encoding="utf-8-sig")))
        completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        records.append({"method": method, "case": case, "scenario": row["scenario"],
            "selected_family": selected, "local_An_cal_AUC": float(row["auc"]),
            "An_auc": float(row["auc_tie_corrected"]),
            "surface_An_auc": float(row["surface_auc_tie_corrected"]),
            "deep_An_auc": float(row["deep_auc_tie_corrected"]),
            "surface_SD_mm": float(row["surface_sd_mm"]),
            "surface_DLE_mm": float(row["surface_dle_mm"]),
            "deep_DLE_mm": float(row["deep_dle_mm"]),
            "deep_detected": int(row["deep_detected"]),
            "negative_H1_deep_false_positive": (int(negative_h1["deep_false_positive"])
                                                  if negative_h1 is not None else 0),
            "held_out_score": float(evidence[0]["score"]),
            "all_converged": int(completion["all_converged"]),
            "wall_seconds": float(completion["wall_seconds"])})

with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# %% 固定验收：两个局部 AUC、纯表层不报深源、深源定位误差和收敛同时检查。
full = [row for row in records if row["method"] == "Full SISSES"]
summary = {"complete": True, "phase": "development", "snr_eeg_meg_db": [-10, -10],
    "cases": [1, 3], "primary_metric": "canonical tied-rank An_auc",
    "local_metric": "parcel-local An_cal_AUC", "local_AUC_gate": .9,
    "full_SISSES_local_gate_passed": all(row["local_An_cal_AUC"] >= .9 for row in full),
    "full_SISSES_all_converged": all(row["all_converged"] for row in full),
    "full_SISSES_case01_ungated_deep_false_positive": full[0]["negative_H1_deep_false_positive"],
    "full_SISSES_case03_deep_detected": full[1]["deep_detected"],
    "full_SISSES_case03_deep_DLE_mm": full[1]["deep_DLE_mm"],
    "full_SISSES_wall_seconds": [row["wall_seconds"] for row in full],
    "accepted": False,
    "reason": "Local AUC stayed below 0.90; the pure-surface H1 reported deep activity and the true deep peak was 20 mm away."}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 三联图：全局排序、局部范围和计算代价分开，避免一个高 An_auc 掩盖失败。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = ["#4C78A8", "#F28E2B", "#59A14F", "#B07AA1"]
figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
x = np.arange(2)
for color, method in zip(colors, directories):
    pair = [row for row in records if row["method"] == method]
    axes[0].plot(x, [row["An_auc"] for row in pair], "o--", color=color,
                 linewidth=1.5, alpha=.8)
    axes[0].plot(x, [row["local_An_cal_AUC"] for row in pair], "o-", color=color,
                 linewidth=2.3, label=method)
axes[0].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[0].set(title="Global (dashed) vs local AUC", ylabel="AUC", xticks=x,
            xticklabels=["S+S (H0)", "S+D (H1)"], ylim=(.45, 1.02))
axes[0].legend(frameon=False, fontsize=8)

width = .18
for index, (color, method) in enumerate(zip(colors, directories)):
    pair = [row for row in records if row["method"] == method]
    axes[1].bar(x + (index - 1.5) * width, [row["surface_DLE_mm"] for row in pair],
                width, color=color, label=method)
axes[1].set(title="Surface localization error", ylabel="DLE (mm)", xticks=x,
            xticklabels=["S+S", "S+D"])

for color, method in zip(colors, directories):
    pair = [row for row in records if row["method"] == method]
    axes[2].plot(x, [row["wall_seconds"] / 60 for row in pair], "o-", color=color,
                 linewidth=2, label=method)
axes[2].set(title="Runtime cost", ylabel="Minutes / case", xticks=x,
            xticklabels=["S+S", "S+D"], yscale="log")
figure.suptitle("Fixed-DCT development diagnosis at EEG/MEG = -10/-10 dB", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

# %% 论文式表格和明确拒绝结论。
lines = ["# 固定 DCT 下 SISSES 逐模重加权结果", "",
    "实线为局部 `An_cal_AUC`，虚线为全局 `An_auc`。全局排序较高不能替代活动区范围、DLE 和深源误报。", "",
    "| 方法 | case | 局部 AUC | An_auc | 表层 SD/DLE mm | 深层 DLE mm | 深层检出 | 纯表层 H1 误报 | 耗时 min |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for row in records:
    deep_dle = "—" if not np.isfinite(row["deep_DLE_mm"]) else f"{row['deep_DLE_mm']:.2f}"
    lines.append(f"| {row['method']} | {row['case']} | {row['local_An_cal_AUC']:.3f} | "
        f"{row['An_auc']:.3f} | {row['surface_SD_mm']:.2f}/{row['surface_DLE_mm']:.2f} | "
        f"{deep_dle} | {row['deep_detected']} | {row['negative_H1_deep_false_positive']} | "
        f"{row['wall_seconds'] / 60:.1f} |")
lines += ["", "## 结论", "",
    "完整 SISSES 在两个难例的局部 AUC 分别为 0.582 和 0.830，未通过 0.90 门槛；case03 深源峰仍偏离 20 mm。",
    "case01 的未门控 H1 仍产生深源误报，且完整组合每例约需 21–22 分钟，因此拒绝该版本。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
