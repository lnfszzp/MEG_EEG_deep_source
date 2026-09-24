"""# %% 汇总balanced20的无量纲盲门控；阈值仍是开发阈值，不替代正式校准。"""

# %% 固定输入和完整性检查。
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact, rankdata

root = Path(__file__).resolve().parents[1]
source = root / "results/erp_whole_head/adaptive_v6/dev_component_balanced_excess_retry5_full20"
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/excess_gate_balanced_full20"
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")
completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
assert completion["complete"] and completion["all_converged"] and completion["case_count"] == 20
assert metadata["phase"] == "development" and metadata["covariance"] == "trial"
assert metadata["score_kind"] == "excess"
assert metadata["solver_settings"] == {"solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 1, "max_inner_retries": 5}
for relative, expected in metadata["code_sha256"].items():
    path = root / relative
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
manifest = Path(metadata["manifest"])
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == metadata["manifest_sha256"]
cases = json.loads(manifest.read_text(encoding="utf-8"))
assert len(cases) == 20 and [case["case_id"] for case in cases] == metadata["case_ids"]
output.mkdir(parents=True)

# %% 0.12只来自开发间隙；决定后才读取真值做后验评价。
development_threshold = .12
rows = []
for case in cases:
    name = case["case_id"]
    saved = json.loads((source / (name + ".json")).read_text(encoding="utf-8"))
    score = float(saved["evidence"]["excess_fraction_score"])
    predicted = bool(score > development_threshold)
    has_deep = case.get("deep_index") is not None
    rows.append({"case_id": name, "scenario": case["scenario"],
        "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
        "score": score, "development_threshold": development_threshold,
        "deep_present_decision": int(predicted), "has_deep_true_posthoc": int(has_deep),
        "correct_posthoc": int(predicted == has_deep)})

y = np.array([row["has_deep_true_posthoc"] for row in rows])
scores = np.array([row["score"] for row in rows])
decisions = np.array([row["deep_present_decision"] for row in rows])
tp, fn = int(np.sum((y == 1) & (decisions == 1))), int(np.sum((y == 1) & (decisions == 0)))
tn, fp = int(np.sum((y == 0) & (decisions == 0))), int(np.sum((y == 0) & (decisions == 1)))
ranks = rankdata(scores)
n_positive, n_negative = int(y.sum()), int((1 - y).sum())
gate_auc = float((ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))
fisher_p = float(fisher_exact([[tp, fn], [fp, tn]], alternative="greater").pvalue)
summary = {"complete": True, "phase": "development", "case_count": len(rows),
    "all_models_converged": True, "score_kind": "excess",
    "score_formula": "(L0-L1)/max(L0-Enoise,0.05*Enoise)",
    "development_threshold": development_threshold,
    "threshold_status": "chosen between labeled development extrema; not calibrated or deployable",
    "runtime_decision_used_truth": False, "parameters_selected_during_development": True,
    "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
    "balanced_accuracy": .5 * (tp / (tp + fn) + tn / (tn + fp)),
    "gate_auc_posthoc": gate_auc, "fisher_exact_one_sided_p_posthoc": fisher_p,
    "max_surface_score": float(scores[y == 0].max()),
    "min_deep_score": float(scores[y == 1].min()),
    "development_margin": float(scores[y == 1].min() - scores[y == 0].max()),
    "calibration_consumed": False, "validation_consumed": False, "accepted": False}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with (output / "metrics_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# %% 分数图和每个SNR层的混淆结果。
pairs = [(-10, -10), (-10, 20), (20, -10), (5, 5)]
pair_labels = ["−10/−10", "−10/+20", "+20/−10", "+5/+5"]
figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
for index, pair in enumerate(pairs):
    subset = [row for row in rows if (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    for deep, marker, color, label in ((0, "s", "#007C83", "No deep source"),
                                        (1, "D", "#D55E00", "Deep source present")):
        values = [row["score"] for row in subset if row["has_deep_true_posthoc"] == deep]
        offsets = np.linspace(-.10, .10, len(values))
        axes[0].scatter(index + offsets, values, marker=marker, s=62, color=color,
                        edgecolor="white", linewidth=.7,
                        label=label if index == 0 else None, zorder=3)
axes[0].axhline(development_threshold, color="#C44E52", linestyle=":", linewidth=1.8,
                label="Development threshold")
axes[0].set(xticks=np.arange(4), xticklabels=pair_labels, yscale="log",
            xlabel="EEG/MEG SNR (dB)", ylabel="Excess-loss fraction U",
            title="Independent blind-gate score")
axes[0].legend(frameon=False, fontsize=9)
sensitivities, specificities = [], []
for pair in pairs:
    subset = [row for row in rows if (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    positives = [row for row in subset if row["has_deep_true_posthoc"]]
    negatives = [row for row in subset if not row["has_deep_true_posthoc"]]
    sensitivities.append(np.mean([row["deep_present_decision"] for row in positives]))
    specificities.append(np.mean([not row["deep_present_decision"] for row in negatives]))
axes[1].plot(pair_labels, sensitivities, "D-", color="#D55E00", linewidth=2,
             markersize=7, label="Sensitivity")
axes[1].plot(pair_labels, specificities, "s-", color="#007C83", linewidth=2,
             markersize=7, label="Specificity")
axes[1].set(xlabel="EEG/MEG SNR (dB)", ylabel="Rate", ylim=(0, 1.05),
            title="Post-hoc development classification")
axes[1].legend(frameon=False)
figure.suptitle("SNR-robust deep-source gate: balanced20 development", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

lines = ["# 四组SNR无量纲盲门控（balanced20开发集）", "",
    "运行时只使用train20拟合和独立confirmation20评分；真值仅在决定后计算本表。U=(L0−L1)/max(L0−Enoise, 0.05·Enoise)。", "",
    "| EEG/MEG SNR | 情况 | U | 盲判 | 后验正确 |", "|---|---|---:|---|---|"]
for row in rows:
    pair = f"{row['eeg_snr_db']:+d}/{row['meg_snr_db']:+d}"
    decision = "H1" if row["deep_present_decision"] else "H0"
    lines.append(f"| {pair} | {row['scenario']} | {row['score']:.5f} | {decision} | "
                 f"{'是' if row['correct_posthoc'] else '否'} |")
lines += ["", f"20/20严格收敛；TP/FN/TN/FP={tp}/{fn}/{tn}/{fp}；门控AUC={gate_auc:.3f}。",
    f"纯表层最大U={summary['max_surface_score']:.5f}；深源最小U={summary['min_deep_score']:.5f}；开发间隔={summary['development_margin']:.5f}。",
    "0.12看过本开发集标签，只能用于后续开发闭环；正式结论必须重新冻结同分布校准/验证，当前accepted=false。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
