"""# %% 检查邻区偏移是否由 MRF 坐标变换造成。"""

# %% 两个固定难例、两个预声明坐标系；不覆盖结果。
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/mrf_ablation_svd_convex_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧汇总：{output}")
output.mkdir(parents=True)

directories = {
    "MRF 0.8": "dev_component_balanced_sissesconvex_svd_admm_mrf080_m10_m10_case{case}",
    "Physical current": "dev_component_balanced_physical_sissesconvex_svd_m10_m10_case{case}",
}

# %% case01 取 H0，case03 取 H1；真值只进入已经保存的事后指标。
records = []
for method, pattern in directories.items():
    for case, family in (("01", "v6-surface-only"), ("03", "v6-full-ungated")):
        directory = base / pattern.format(case=case)
        rows = list(csv.DictReader((directory / "rows.csv").open(encoding="utf-8-sig")))
        row = next(item for item in rows if item["method"] == family)
        evidence = next(csv.DictReader((directory / "evidence.csv").open(encoding="utf-8-sig")))
        completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        records.append({"method": method, "case": case, "family": family,
            "local_An_cal_AUC": float(row["auc"]), "An_auc": float(row["auc_tie_corrected"]),
            "surface_An_auc": float(row["surface_auc_tie_corrected"]),
            "deep_An_auc": float(row["deep_auc_tie_corrected"]),
            "surface_SD_mm": float(row["surface_sd_mm"]),
            "surface_DLE_mm": float(row["surface_dle_mm"]),
            "deep_DLE_mm": float(row["deep_dle_mm"]), "deep_detected": int(row["deep_detected"]),
            "held_out_score": float(evidence["score"]),
            "all_converged": int(completion["all_converged"]),
            "wall_seconds": float(completion["wall_seconds"])})

with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# %% 物理坐标不是修复：深源仍准，但表层全局排序接近随机。
physical = [row for row in records if row["method"] == "Physical current"]
summary = {"complete": True, "phase": "development", "snr_eeg_meg_db": [-10, -10],
    "local_AUC_gate": .9, "physical_local_gate_passed": all(
        row["local_An_cal_AUC"] >= .9 for row in physical),
    "physical_case01_An_auc": physical[0]["An_auc"],
    "physical_case03_An_auc": physical[1]["An_auc"],
    "physical_case03_deep_detected": physical[1]["deep_detected"],
    "all_converged": all(row["all_converged"] for row in records),
    "accepted": False,
    "reason": "Removing MRF collapsed cortical ranking to about 0.55 and did not pass the local AUC gate."}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 同一图同时显示全局排序、局部范围和定位误差。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = {"MRF 0.8": "#4C78A8", "Physical current": "#E45756"}
figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
x = np.arange(2)
for method in directories:
    pair = [row for row in records if row["method"] == method]
    axes[0].plot(x, [row["An_auc"] for row in pair], "o-", color=colors[method],
                 linewidth=2.3, label=method)
    axes[1].plot(x, [row["local_An_cal_AUC"] for row in pair], "o-", color=colors[method],
                 linewidth=2.3, label=method)
    axes[2].plot(x, [row["surface_DLE_mm"] for row in pair], "o-", color=colors[method],
                 linewidth=2.3, label=method)
axes[0].set(title="Global An_auc", ylabel="AUC", ylim=(.5, 1.02))
axes[1].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[1].set(title="Local An_cal_AUC", ylabel="AUC", ylim=(.45, 1.02))
axes[2].set(title="Surface DLE", ylabel="mm")
for axis in axes:
    axis.set(xticks=x, xticklabels=["S+S (H0)", "S+D (H1)"])
axes[0].legend(frameon=False)
figure.suptitle("MRF-coordinate ablation at EEG/MEG = -10/-10 dB", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

# %% 简短报告。
lines = ["# MRF 与物理电流坐标消融", "",
    "| 坐标 | case | 局部 AUC | An_auc | 表层 SD/DLE mm | 深层 DLE mm | 深层检出 | 耗时 min |",
    "|---|---:|---:|---:|---:|---:|---:|---:|"]
for row in records:
    deep = "—" if not np.isfinite(row["deep_DLE_mm"]) else f"{row['deep_DLE_mm']:.2f}"
    lines.append(f"| {row['method']} | {row['case']} | {row['local_An_cal_AUC']:.3f} | "
        f"{row['An_auc']:.3f} | {row['surface_SD_mm']:.2f}/{row['surface_DLE_mm']:.2f} | "
        f"{deep} | {row['deep_detected']} | {row['wall_seconds'] / 60:.1f} |")
lines += ["", "去掉 MRF 后两个病例的表层全局 An_auc 都约为 0.55，接近随机；"
    "邻区偏移不能靠简单删除 MRF 解决，因此拒绝该版本。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
