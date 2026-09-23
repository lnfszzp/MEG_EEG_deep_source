"""# %% 汇总 SISSES 式逐时间基边界；未达预设门槛即拒绝。"""

# %% 读取同一观测、同一层级家族的两种边界正则结果。
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/sisses_per_mode_edge_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧汇总：{output}")
output.mkdir(parents=True)

records = []
for number, method in (("01", "v6-surface-only"), ("03", "v6-full-ungated")):
    case_id = f"erp-v6-development-{number}-eeg-10-meg-10"
    directories = {
        "convex group edge": base / f"dev_component_balanced_convex_admm_mrf080_m10_m10_case{number}",
        "SISSES per-mode edge": base / f"dev_component_balanced_sissesedge_admm_mrf080_m10_m10_case{number}",
    }
    for variant, directory in directories.items():
        with (directory / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
            found = [row for row in csv.DictReader(stream)
                     if row["case_id"] == case_id and row["method"] == method]
        with (directory / "evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
            evidence = list(csv.DictReader(stream))
        assert len(found) == len(evidence) == 1
        row = found[0]
        records.append({"case": number, "scenario": row["scenario"], "variant": variant,
            "An_cal_AUC": float(row["auc"]), "An_auc": float(row["auc_tie_corrected"]),
            "surface_An_auc": float(row["surface_auc_tie_corrected"]),
            "deep_An_auc": float(row["deep_auc_tie_corrected"]),
            "surface_SD_mm": float(row["surface_sd_mm"]), "surface_DLE_mm": float(row["surface_dle_mm"]),
            "deep_DLE_mm": float(row["deep_dle_mm"]), "deep_detected": int(row["deep_detected"]),
            "deep_false_positive": int(row["deep_false_positive"]),
            "all_converged": int(evidence[0]["null_converged"]) * int(evidence[0]["full_converged"])})

with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# %% 预声明局部 AUC 0.90 门槛；不因相对改善而改门槛。
per_mode = [row for row in records if row["variant"] == "SISSES per-mode edge"]
summary = {"complete": True, "phase": "development", "snr_eeg_meg_db": [-10, -10],
    "cases": [1, 3], "calibration_or_validation_read": False,
    "solver_settings": {"solver_kind": "admm", "mrf_strength": .8,
        "surface_reweight_floor": 1., "deep_reweight_floor": 1.,
        "edge_penalty_mode": "elementwise"},
    "local_An_cal_AUC_gate": .9,
    "passes_local_gate": all(row["An_cal_AUC"] >= .9 for row in per_mode),
    "all_converged": all(row["all_converged"] for row in per_mode),
    "accepted": False,
    "reason": "Per-mode edge reweighting improves both cases but reaches only 0.644 and 0.851 local An_cal_AUC, below the frozen 0.90 gate."}
(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 不是柱状图：配对折线同时显示局部范围、全局排序和定位误差。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
x = np.arange(2)
labels = ["Case 01\nS+S", "Case 03\nS+D"]
colors = {"convex group edge": "#7A8C99", "SISSES per-mode edge": "#007C83"}
for variant in colors:
    chosen = [row for number in ("01", "03") for row in records
              if row["case"] == number and row["variant"] == variant]
    axes[0].plot(x, [row["An_cal_AUC"] for row in chosen], "o-", linewidth=2.2,
                 color=colors[variant], label=variant)
    axes[1].plot(x, [row["surface_DLE_mm"] for row in chosen], "o-", linewidth=2.2,
                 color=colors[variant], label=variant)
axes[0].axhline(.9, color="#D55E00", linestyle="--", linewidth=1.2, label="0.90 target")
axes[0].set(title="Local activity-extent AUC", ylabel="An_cal_AUC", ylim=(.5, .93),
            xticks=x, xticklabels=labels)
axes[1].set(title="Cortical peak localization", ylabel="DLE (mm)", ylim=(0, 14),
            xticks=x, xticklabels=labels)
for axis in axes:
    axis.grid(axis="y", color="#E9ECEF", linewidth=.8)
axes[0].legend(frameon=False, fontsize=9)
figure.savefig(output / "per_mode_edge_comparison.png", dpi=240, bbox_inches="tight", facecolor="white")
plt.close(figure)

lines = ["# SISSES 式逐时间基边界开发结果（拒绝）", "",
    "两个困难病例均改善，但局部 `An_cal_AUC` 未达到预先固定的 0.90 门槛；不继续针对这两个病例扫参数。", "",
    "| case | 旧局部 AUC | 新局部 AUC | 新 An_auc | 表层 SD mm | 表层 DLE mm | 深层 DLE mm |",
    "|---:|---:|---:|---:|---:|---:|---:|" ]
for number in ("01", "03"):
    old = next(row for row in records if row["case"] == number and row["variant"] == "convex group edge")
    new = next(row for row in records if row["case"] == number and row["variant"] == "SISSES per-mode edge")
    deep = "—" if not np.isfinite(new["deep_DLE_mm"]) else f"{new['deep_DLE_mm']:.2f}"
    lines.append(f"| {number} | {old['An_cal_AUC']:.3f} | {new['An_cal_AUC']:.3f} | "
        f"{new['An_auc']:.3f} | {new['surface_SD_mm']:.2f} | {new['surface_DLE_mm']:.2f} | {deep} |")
lines += ["", "逐时间基边界说明 SISSES 的细粒度边缘权重有价值，但它不能单独找回全部漏检源。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
