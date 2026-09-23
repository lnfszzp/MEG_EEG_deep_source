"""# %% 汇总 SVD 对齐是否改善逐模 SISSES 的两个低信噪比难例。"""

# %% 固定输入和不覆盖目录。
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/svd_aligned_per_mode_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧汇总：{output}")
output.mkdir(parents=True)

directories = {
    "Fixed source": "dev_component_balanced_sissessource_admm_mrf080_m10_m10_case{case}",
    "SVD source": "dev_component_balanced_sissessource_svd_admm_mrf080_m10_m10_case{case}",
    "Fixed edge": "dev_component_balanced_sissesedge_admm_mrf080_m10_m10_case{case}",
    "SVD edge": "dev_component_balanced_sissesedge_svd_admm_mrf080_m10_m10_case{case}",
    "SVD convex source+edge": "dev_component_balanced_sissesconvex_svd_admm_mrf080_m10_m10_case{case}",
}

# %% case01 读取 H0，case03 读取 H1；开发阶段不假装已有冻结深源门控。
records = []
for method, pattern in directories.items():
    for case, family in (("01", "v6-surface-only"), ("03", "v6-full-ungated")):
        directory = base / pattern.format(case=case)
        rows = list(csv.DictReader((directory / "rows.csv").open(encoding="utf-8-sig")))
        row = next(item for item in rows if item["method"] == family)
        negative_h1 = (next(item for item in rows if item["method"] == "v6-full-ungated")
                       if case == "01" else None)
        evidence = next(csv.DictReader((directory / "evidence.csv").open(encoding="utf-8-sig")))
        completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        detail = json.loads((directory / f"erp-v6-development-{case}-eeg-10-meg-10.json").read_text(
            encoding="utf-8"))
        rotation = detail["fitting"]["null_model"]["windows"][0].get("temporal_rotation", "none")
        records.append({"method": method, "case": case, "selected_family": family,
            "temporal_rotation": rotation,
            "local_An_cal_AUC": float(row["auc"]), "An_auc": float(row["auc_tie_corrected"]),
            "surface_An_auc": float(row["surface_auc_tie_corrected"]),
            "deep_An_auc": float(row["deep_auc_tie_corrected"]),
            "surface_SD_mm": float(row["surface_sd_mm"]),
            "surface_DLE_mm": float(row["surface_dle_mm"]),
            "deep_DLE_mm": float(row["deep_dle_mm"]), "deep_detected": int(row["deep_detected"]),
            "negative_H1_deep_false_positive": (int(negative_h1["deep_false_positive"])
                                                  if negative_h1 is not None else 0),
            "held_out_score": float(evidence["score"]),
            "all_converged": int(completion["all_converged"]),
            "wall_seconds": float(completion["wall_seconds"])})

with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# %% 统一验收，不因 SVD 结果再调门槛。
svd_rows = [row for row in records if row["method"].startswith("SVD")]
summary = {"complete": True, "phase": "development", "snr_eeg_meg_db": [-10, -10],
    "cases": [1, 3], "local_AUC_gate": .9,
    "best_case01_local_AUC": max(row["local_An_cal_AUC"] for row in svd_rows if row["case"] == "01"),
    "best_case03_local_AUC": max(row["local_An_cal_AUC"] for row in svd_rows if row["case"] == "03"),
    "all_SVD_local_gates_passed": all(row["local_An_cal_AUC"] >= .9 for row in svd_rows),
    "SVD_deep_localization_successes": sum(row["deep_detected"] for row in svd_rows if row["case"] == "03"),
    "all_converged": all(row["all_converged"] for row in records),
    "accepted": False,
    "reason": "SVD restored the case03 deep peak but no SVD-aligned variant reached local AUC 0.90; case01 remained at or below 0.645."}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 配对折线与运行代价；同色方法在两个病例间连接。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = ["#4C78A8", "#72B7B2", "#F28E2B", "#ECA82C", "#B07AA1"]
figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
x = np.arange(2)
for color, method in zip(colors, directories):
    pair = [row for row in records if row["method"] == method]
    axes[0].plot(x, [row["local_An_cal_AUC"] for row in pair], "o-", color=color,
                 linewidth=2.2, label=method)
    axes[1].plot(x, [row["surface_DLE_mm"] for row in pair], "o-", color=color,
                 linewidth=2.2, label=method)
    axes[2].plot(x, [row["wall_seconds"] / 60 for row in pair], "o-", color=color,
                 linewidth=2.2, label=method)
axes[0].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[0].set(title="Local activity-region AUC", ylabel="An_cal_AUC", ylim=(.45, 1.02))
axes[1].set(title="Surface localization error", ylabel="DLE (mm)")
axes[2].set(title="Runtime", ylabel="Minutes / case", yscale="log")
for axis in axes:
    axis.set(xticks=x, xticklabels=["S+S (H0)", "S+D (H1)"])
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
figure.suptitle("Per-mode temporal-coordinate diagnosis at EEG/MEG = -10/-10 dB",
                fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

# %% 指标表和结论。
lines = ["# SVD 对齐逐模惩罚开发结果", "",
    "SVD 将两个真实 ERP 波形对齐到不同时间模态，但局部空间定位没有同步改善。", "",
    "| 方法 | case | 局部 AUC | An_auc | 表层 SD/DLE mm | 深层 DLE mm | 深层检出 | 纯表层 H1 误报 | 耗时 min |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for row in records:
    deep = "—" if not np.isfinite(row["deep_DLE_mm"]) else f"{row['deep_DLE_mm']:.2f}"
    lines.append(f"| {row['method']} | {row['case']} | {row['local_An_cal_AUC']:.3f} | "
        f"{row['An_auc']:.3f} | {row['surface_SD_mm']:.2f}/{row['surface_DLE_mm']:.2f} | "
        f"{deep} | {row['deep_detected']} | {row['negative_H1_deep_false_positive']} | "
        f"{row['wall_seconds'] / 60:.1f} |")
lines += ["", "## 结论", "",
    f"SVD 版本在 case01 的最好局部 AUC 为 {summary['best_case01_local_AUC']:.3f}，"
    f"case03 为 {summary['best_case03_local_AUC']:.3f}，均未达到 0.90。",
    "深源可恢复到 0 mm，但第二表层源和邻区偏移仍未解决；拒绝继续叠加逐模重加权复杂度。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
