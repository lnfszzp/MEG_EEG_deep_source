"""#%% 汇总固定 elastic-net 开发诊断；只读已完成结果，不读取校准或验证。"""

# %% 路径和四个预先运行的设置。
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base_a = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases00_01"
base_b = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04"
directories = {
    ("baseline", "01"): base_a,
    ("baseline", "04"): base_b,
    ("ridge .05", "01"): root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_elastic005_admm_mrf080_m10_m10_case01",
    ("ridge .05", "04"): root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_elastic005_admm_mrf080_m10_m10_case04",
    ("ridge .5", "01"): root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_elastic050_admm_mrf080_m10_m10_case01",
    ("ridge .5", "04"): root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_elastic050_admm_mrf080_m10_m10_case04",
    ("lambda .5 + ridge .05", "01"): root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_elastic005_nm050_admm_mrf080_m10_m10_case01",
    ("lambda .5 + ridge .05", "04"): root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_elastic005_nm050_admm_mrf080_m10_m10_case04",
}
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/elastic_net_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧汇总：{output}")
output.mkdir(parents=True)

# %% 逐设置读取同一病例的 H0/H1 与独立确认分数。
records = []
for (variant, number), directory in directories.items():
    case_id = f"erp-v6-development-{number}-eeg-10-meg-10"
    with (directory / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
        case_rows = [row for row in csv.DictReader(stream) if row["case_id"] == case_id and row["method"].startswith("v6-")]
    with (directory / "evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        checks = [row for row in csv.DictReader(stream) if row["case_id"] == case_id]
    assert len(case_rows) == 2 and len(checks) == 1
    check = checks[0]
    for row in case_rows:
        family = "H0" if row["method"] == "v6-surface-only" else "H1"
        record = {"case": number, "variant": variant, "family": family,
            "An_cal_AUC": float(row["auc"]), "An_auc": float(row["auc_tie_corrected"]),
            "surface_An_auc": float(row["surface_auc_tie_corrected"]),
            "deep_An_auc": float(row["deep_auc_tie_corrected"]),
            "surface_SD_mm": float(row["surface_sd_mm"]), "deep_SD_mm": float(row["deep_sd_mm"]),
            "surface_DLE_mm": float(row["surface_dle_mm"]), "deep_DLE_mm": float(row["deep_dle_mm"]),
            "deep_score": float(row["deep_score"]), "deep_false_positive": int(row["deep_false_positive"]),
            "deep_detected": int(row["deep_detected"]), "confirmation_score": float(check["score"]),
            "converged": int(check["null_converged"] if family == "H0" else check["full_converged"])}
        for key, value in list(record.items()):
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
        records.append(record)

with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# %% 正确家族下比较空间定位；深源判别仍是另一道尚未完成的门槛。
variants = ["baseline", "ridge .05", "ridge .5", "lambda .5 + ridge .05"]
selected = {(row["case"], row["variant"]): row for row in records
            if (row["case"] == "01" and row["family"] == "H0")
            or (row["case"] == "04" and row["family"] == "H1")}
assert len(selected) == 8
gates = []
for variant in variants:
    case01, case04 = selected[("01", variant)], selected[("04", variant)]
    gates.append({"variant": variant,
        "case01_An_cal_AUC": case01["An_cal_AUC"], "case01_An_auc": case01["An_auc"],
        "case04_An_cal_AUC": case04["An_cal_AUC"], "case04_An_auc": case04["An_auc"],
        "case01_H1_false_positive": next(row["deep_false_positive"] for row in records
            if row["case"] == "01" and row["variant"] == variant and row["family"] == "H1"),
        "case04_deep_DLE_mm": case04["deep_DLE_mm"],
        "all_converged": all(row["converged"] for row in records if row["variant"] == variant),
        "passes_spatial_AUC_090": min(case01["An_cal_AUC"], case01["An_auc"],
            case04["An_cal_AUC"], case04["An_auc"]) >= .9})

summary = {"complete": True, "phase": "development", "cases": [1, 4],
    "snr_eeg_meg_db": [-10, -10], "mrf_strength": .8,
    "calibration_or_validation_read": False, "accepted": False,
    "reason": "No fixed elastic-net setting reaches both AUC definitions >=0.90 in both difficult two-surface cases.",
    "settings": gates}
(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 论文式折线图：同一病例内同时显示局部与全局排序 AUC。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True, constrained_layout=True)
x = np.arange(len(variants))
labels = ["Baseline", "Ridge .05", "Ridge .5", "Lambda .5\n+ ridge .05"]
colors = {"An_cal_AUC": "#264653", "An_auc": "#E76F51"}
for axis, number, title in zip(axes, ("01", "04"),
                               ("Case 01: two cortical patches", "Case 04: deep + two cortical patches")):
    for metric, label in (("An_cal_AUC", "Local parcel AUC"), ("An_auc", "Tie-corrected An_auc")):
        values = [selected[(number, variant)][metric] for variant in variants]
        axis.plot(x, values, marker="o", linewidth=2.2, markersize=6, color=colors[metric], label=label)
        for index, value in enumerate(values):
            axis.text(index, value + .012, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color=colors[metric])
    axis.axhline(.9, color="#6C757D", linestyle="--", linewidth=1.3, label="0.90 target")
    axis.set_xticks(x, labels)
    axis.set_ylim(.60, 1.02)
    axis.set_title(title, fontweight="bold")
    axis.set_ylabel("AUC")
    axis.grid(axis="y", color="#D9D9D9", linewidth=.7, alpha=.8)
axes[1].legend(frameon=False, loc="lower left")
figure.savefig(output / "AUC_comparison.png", dpi=220, facecolor="white")
plt.close(figure)

# %% 明确拒绝，不能把 case01 的改善替代 case04 的失败。
lines = ["# Elastic-net 固定量级开发结果（拒绝）", "",
    "这是 EEG/MEG=-10/-10 dB 的两个最难双表层开发病例；未读取校准集或验证集。",
    "表中 case01 按 H0、case04 按 H1 报告条件定位；这不等于盲检已经解决。", "",
    "| 设置 | case01 An_cal | case01 An_auc | case04 An_cal | case04 An_auc | case01 H1深误报 | case04深层DLE | 全部收敛 |",
    "|---|---:|---:|---:|---:|---:|---:|:---:|"]
for row in gates:
    lines.append(f"| {row['variant']} | {row['case01_An_cal_AUC']:.4f} | {row['case01_An_auc']:.4f} | "
        f"{row['case04_An_cal_AUC']:.4f} | {row['case04_An_auc']:.4f} | {row['case01_H1_false_positive']} | "
        f"{'NA' if row['case04_deep_DLE_mm'] is None else f'{row['case04_deep_DLE_mm']:.2f}'} | "
        f"{'是' if row['all_converged'] else '否'} |")
lines += ["", "结论：`ridge_fraction=0.05` 修复了 case01 H1 的数值收敛，并把幅度型深误报从 1 压到 0，"
    "但 case04 的空间 AUC 没有改善；更强 ridge 造成表层过度收缩，低 lambda 与 ridge 联合又恢复深误报并使深层峰偏移。"
    "因此 elastic-net 作为数值稳定项有价值，但不能单独解决多表层源定位，本轮不并入最终空间模型。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
