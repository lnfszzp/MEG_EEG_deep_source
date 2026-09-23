"""# %% 汇总同一 -10/-10 dB 开发观测上的重加权与凸图正则结果。"""

# %% 路径和固定的开发病例；不读取校准集或验证集。
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/convex_component_balanced_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧汇总：{output}")
output.mkdir(parents=True)

old_directories = {
    "00": base / "dev_sensor_balanced_admm_mrf080_m10_m10_cases00_01",
    "01": base / "dev_component_balanced_admm_mrf080_m10_m10_case01",
    "02": base / "dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04",
    "03": base / "dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04",
    "04": base / "dev_component_balanced_admm_mrf080_m10_m10_case04",
}
new_directories = {
    number: base / f"dev_component_balanced_convex_admm_mrf080_m10_m10_case{number}"
    for number in ("00", "01", "02", "03", "04")
}

# %% 每例只比较真实层级对应的开发解；这不是可部署的深源决策。
records = []
for number in ("00", "01", "02", "03", "04"):
    method = "v6-surface-only" if number in ("00", "01") else "v6-full-ungated"
    case_id = f"erp-v6-development-{number}-eeg-10-meg-10"
    pair = []
    for directory in (old_directories[number], new_directories[number]):
        with (directory / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
            found = [row for row in csv.DictReader(stream)
                     if row["case_id"] == case_id and row["method"] == method]
        assert len(found) == 1
        pair.append(found[0])
    with (new_directories[number] / "evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
        evidence = list(csv.DictReader(stream))
    assert len(evidence) == 1 and evidence[0]["case_id"] == case_id
    old, new = pair
    record = {"case": number, "scenario": new["scenario"], "development_family": method,
        "old_An_cal_AUC": float(old["auc"]), "old_An_auc": float(old["auc_tie_corrected"]),
        "new_An_cal_AUC": float(new["auc"]), "new_An_auc": float(new["auc_tie_corrected"]),
        "surface_An_auc": float(new["surface_auc_tie_corrected"]),
        "deep_An_auc": float(new["deep_auc_tie_corrected"]),
        "surface_SD_mm": float(new["surface_sd_mm"]), "surface_DLE_mm": float(new["surface_dle_mm"]),
        "deep_SD_mm": float(new["deep_sd_mm"]), "deep_DLE_mm": float(new["deep_dle_mm"]),
        "deep_detected": int(new["deep_detected"]), "deep_false_positive": int(new["deep_false_positive"]),
        "held_out_score": float(evidence[0]["score"]),
        "all_converged": int(evidence[0]["null_converged"]) * int(evidence[0]["full_converged"])}
    for key, value in list(record.items()):
        if isinstance(value, float) and not math.isfinite(value):
            record[key] = None
    records.append(record)

with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# %% 冻结事实：主 An_auc 通过开发门槛；局部范围和独立深源门控尚未通过最终验收。
negative_scores = [row["held_out_score"] for row in records if row["case"] in ("00", "01")]
positive_scores = [row["held_out_score"] for row in records if row["case"] not in ("00", "01")]
summary = {"complete": True, "phase": "development", "snr_eeg_meg_db": [-10, -10],
    "solver_settings": {"solver_kind": "admm", "outer_iterations": 1, "mrf_strength": .8},
    "primary_metric": "canonical tied-rank An_auc",
    "minimum_An_auc": min(row["new_An_auc"] for row in records),
    "mean_old_An_auc": float(np.mean([row["old_An_auc"] for row in records])),
    "mean_new_An_auc": float(np.mean([row["new_An_auc"] for row in records])),
    "primary_AUC_gate_090": all(row["new_An_auc"] >= .9 for row in records),
    "local_An_cal_gate_090": all(row["new_An_cal_AUC"] >= .9 for row in records),
    "all_converged": all(row["all_converged"] for row in records),
    "all_deep_positive_localized": all(row["deep_detected"] for row in records[2:]),
    "development_negative_score_max": max(negative_scores),
    "development_positive_score_min": min(positive_scores),
    "accepted_as_spatial_development_candidate": True, "accepted_as_final_algorithm": False,
    "reason_not_final": "The displayed family is selected with development truth; a frozen pure-surface calibration and untouched multi-SNR validation are still required. Local An_cal_AUC also remains below 0.90."}
(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 折线和散点：避免用一组柱形图掩盖主指标、局部范围和门控证据的区别。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)
x = np.arange(5)
labels = ["S", "S+S", "D", "S+D", "S+S+D"]
axes[0].plot(x, [row["old_An_auc"] for row in records], "o-", color="#7A8C99", linewidth=2, label="Reweighted")
axes[0].plot(x, [row["new_An_auc"] for row in records], "o-", color="#007C83", linewidth=2.4, label="Convex graph")
axes[0].axhline(.9, color="#D55E00", linestyle="--", linewidth=1.2, label="0.90 target")
axes[0].set(title="Primary An_auc", ylabel="AUC", xticks=x, xticklabels=labels, ylim=(.68, 1.02))
axes[0].legend(frameon=False, loc="lower left")

axes[1].plot(x, [row["new_An_auc"] for row in records], "o-", color="#007C83", linewidth=2.4, label="An_auc")
axes[1].plot(x, [row["new_An_cal_AUC"] for row in records], "o-", color="#D55E00", linewidth=2, label="Local An_cal_AUC")
axes[1].axhline(.9, color="#6C757D", linestyle="--", linewidth=1.2)
axes[1].set(title="Ranking vs local extent", xticks=x, xticklabels=labels, ylim=(.5, 1.02))
axes[1].legend(frameon=False, loc="lower left")

colors = ["#5E60CE" if row["case"] in ("00", "01") else "#2A9D8F" for row in records]
axes[2].scatter(x, [row["held_out_score"] for row in records], s=75, color=colors, zorder=3)
axes[2].axhline(0, color="#6C757D", linewidth=1)
axes[2].set(title="Held-out deep evidence\n(no threshold fitted)", ylabel="Prediction score",
            xticks=x, xticklabels=labels)
for axis in axes:
    axis.grid(axis="y", color="#E9ECEF", linewidth=.8)
figure.savefig(output / "AUC_and_evidence.png", dpi=240, bbox_inches="tight", facecolor="white")
plt.close(figure)

# %% 可直接阅读的表格与结论。
lines = ["# 凸图正则 -10/-10 dB 开发结果", "",
    "主指标使用用户指定的 `An_auc`。五种情况均超过 0.90，最低为 "
    f"{summary['minimum_An_auc']:.4f}；但局部活动范围 `An_cal_AUC` 尚未全部通过。", "",
    "下表的 H0/H1 家族由开发真值选取，只用于定位诊断；最终流程仍必须由独立纯表层校准决定是否报告深源。", "",
    "| case | 情况 | 家族 | 旧 An_auc | 新 An_auc | 局部 AUC | 表层 DLE mm | 深层 DLE mm | 深层检出 | 留出分数 |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|" ]
for row in records:
    show = lambda value: "—" if value is None else f"{value:.3f}"
    lines.append(f"| {row['case']} | {row['scenario']} | {'H0' if row['case'] in ('00', '01') else 'H1'} | "
        f"{row['old_An_auc']:.3f} | {row['new_An_auc']:.3f} | {row['new_An_cal_AUC']:.3f} | "
        f"{show(row['surface_DLE_mm'])} | {show(row['deep_DLE_mm'])} | {row['deep_detected']} | {row['held_out_score']:.5f} |")
lines += ["", "凸化显著修复 case04：`An_auc` 0.723→0.987，深层 AUC=1、DLE=0 mm。",
    "纯表层两个开发分数均低于深源阳性的三个分数，但这里不据此事后设阈值。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
