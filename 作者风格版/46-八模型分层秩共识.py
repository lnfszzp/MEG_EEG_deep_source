"""# %% 八个已有逆解按层做秩乘积，共识压低各模型不一致的局部假峰。"""

# %% 固定输入；候选集合在开发期冻结，公式不读取源位置真值。
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/rank_product_consensus_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")
output.mkdir(parents=True)

directory_templates = [
    "dev_component_balanced_convex_admm_mrf080_m10_m10_case{}",
    "dev_component_balanced_sissessource_admm_mrf080_m10_m10_case{}",
    "dev_component_balanced_sissesedge_admm_mrf080_m10_m10_case{}",
    "dev_component_balanced_sissesfull_admm_mrf080_m10_m10_case{}",
    "dev_component_balanced_sissessource_svd_admm_mrf080_m10_m10_case{}",
    "dev_component_balanced_sissesedge_svd_admm_mrf080_m10_m10_case{}",
    "dev_component_balanced_sissesconvex_svd_admm_mrf080_m10_m10_case{}",
    "dev_component_balanced_physical_sissesconvex_svd_m10_m10_case{}",
]
candidate_covariance = ["trial", "mean", "trial", "mean", "mean", "mean", "mean", "mean"]
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = shared["n_surf"]
rows, report_rows, robustness = [], [], []

# %% case01 看表层族，case03 看浅深族；两族都计算，选取只用于开发指标诊断。
for case, oracle_key in (("01", "null"), ("03", "full")):
    case_record = json.loads((base / f"protocol/development_component_balanced_m10_m10_case{case}.json").read_text(encoding="utf-8"))[0]
    observation = prepare_trial_covariance_case(shared, case_record, seed_root=2026092206)
    name = f"erp-v6-development-{case}-eeg-10-meg-10.npz"
    files = [base / template.format(case) / name for template in directory_templates]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError([str(path) for path in files if not path.is_file()])
    packages = [np.load(path) for path in files]
    metadata = [json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
                for path in files]
    assert [item["covariance"] for item in metadata] == candidate_covariance
    assert all(item["phase"] == "development" for item in metadata)
    truth, active, baseline = packages[0]["truth"], packages[0]["active"], packages[0]["baseline"]
    assert all(np.array_equal(package["truth"], truth) and
               np.array_equal(package["active"], active) and
               np.array_equal(package["baseline"], baseline) for package in packages[1:])
    assert np.allclose(truth, observation["truth"])
    groups = observation["groups"]

    consensus_by_family = {}
    for family in ("null", "full"):
        estimates = [np.asarray(package[family], dtype=float) for package in packages]
        ranks = []
        for estimate in estimates:
            amplitude = np.linalg.norm(estimate[:, active], axis=1)
            percentile = np.zeros_like(amplitude)
            for indices in (np.arange(n_surf), np.arange(n_surf, amplitude.size)):
                values = amplitude[indices]
                if np.any(values != 0):
                    percentile[indices] = rankdata(values, method="average") / len(indices)
            ranks.append(percentile)
        score = np.prod(ranks, axis=0) ** 2
        consensus = np.zeros_like(estimates[0])
        consensus[:, int(np.asarray(active).ravel()[0])] = score
        consensus_by_family[family] = consensus
        values = metrics.evaluate_estimate(
            consensus, truth, shared["vertices"], groups, n_surf, active,
            shared["auc_cortex"], baseline=baseline)
        rows.append({"case": case, "family": family, "method": "rank-product-consensus", **values})

        if family == oracle_key:
            single = [metrics.evaluate_estimate(
                estimate, truth, shared["vertices"], groups, n_surf, active,
                shared["auc_cortex"], baseline=baseline)["auc"] for estimate in estimates]
            report_rows.append({"case": case, "oracle_family": family,
                "best_single_local_AUC": float(max(single)),
                "consensus_local_AUC": float(values["auc"]),
                "An_auc": float(values["auc_tie_corrected"]),
                "surface_SD_mm": float(values["surface_sd_mm"]),
                "surface_DLE_mm": float(values["surface_dle_mm"]),
                "deep_DLE_mm": float(values["deep_dle_mm"]),
                "deep_detected": int(values["deep_detected"]),
                "deep_false_positive": int(values["deep_false_positive"]),
                "active_count": int(values["active_count"])})

            leave_one_out = []
            for omitted in range(len(estimates)):
                score_7 = np.prod([item for index, item in enumerate(ranks) if index != omitted], axis=0) ** 2
                estimate_7 = np.zeros_like(estimates[0])
                estimate_7[:, int(np.asarray(active).ravel()[0])] = score_7
                leave_one_out.append(metrics.evaluate_estimate(
                    estimate_7, truth, shared["vertices"], groups, n_surf, active,
                    shared["auc_cortex"], baseline=baseline)["auc"])
            robustness.append({"case": case, "leave_one_out_min_local_AUC": float(min(leave_one_out)),
                "leave_one_out_max_local_AUC": float(max(leave_one_out))})

    np.savez_compressed(output / name, truth=truth, active=active, baseline=baseline,
                        null=consensus_by_family["null"], full=consensus_by_family["full"])
    for package in packages:
        package.close()

archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
with (output / "summary_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=report_rows[0].keys())
    writer.writeheader()
    writer.writerows(report_rows)
with (output / "leave_one_out.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=robustness[0].keys())
    writer.writeheader()
    writer.writerows(robustness)

# %% 图与审慎结论：达到开发目标不等于独立验证通过。
summary = {"complete": True, "phase": "development", "candidate_count": 8,
    "formula": "per-layer average percentile ranks; product across candidates; square for support sharpening",
    "truth_used_in_consensus": False, "candidates_selected_during_development": True,
    "oracle_family_selection_for_diagnostic": True,
    "deployable_presence_decision_evaluated": False,
    "local_AUC_target": .9,
    "development_target_met": all(row["consensus_local_AUC"] >= .9 for row in report_rows),
    "case01_null_deep_false_positive": report_rows[0]["deep_false_positive"],
    "case03_full_deep_detected": report_rows[1]["deep_detected"],
    "independent_validation_passed": False, "accepted": False,
    "candidate_directories": directory_templates,
    "candidate_covariance": candidate_covariance,
    "heterogeneous_cached_artifacts_recomputed_at_current_commit": False}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

colors = ["#9C9C9C", "#007C83"]
figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
x = np.arange(2)
axes[0].bar(x - .18, [row["best_single_local_AUC"] for row in report_rows], .36,
            color=colors[0], label="best single (development)")
axes[0].bar(x + .18, [row["consensus_local_AUC"] for row in report_rows], .36,
            color=colors[1], label="rank consensus")
axes[0].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[0].set(title="Local An_cal_AUC", ylabel="AUC", ylim=(.45, 1.02))
axes[0].legend(frameon=False, fontsize=8)
axes[1].bar(x, [row["An_auc"] for row in report_rows], color=colors[1], width=.55)
axes[1].set(title="Global An_auc", ylabel="AUC", ylim=(.9, 1.0))
axes[2].bar(x, [row["surface_DLE_mm"] for row in report_rows], color=colors[1], width=.55)
axes[2].set(title="Surface DLE", ylabel="mm")
for axis in axes:
    axis.set(xticks=x, xticklabels=["S+S (oracle H0)", "S+D (oracle H1)"])
figure.suptitle("Eight-model layer-wise rank consensus (development only)", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

lines = ["# 八模型分层秩共识（仅开发诊断）", "",
    "每个模型先在表层/深层内转成百分位秩，再将八个秩逐点相乘；逆解和共识计算均不读取真值。平方仅锐化默认能量支持，不改变 AUC 排序。当前候选是 2 个 trial-covariance 与 6 个 mean-covariance 的开发期旧产物，尚未在同一提交上统一重跑。", "",
    "| case | oracle family | 最佳单模型局部 AUC | 共识局部 AUC | An_auc | 表层 SD/DLE mm | 深层 DLE mm | 深层检出 | 支持点数 |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
for row in report_rows:
    deep = "—" if not np.isfinite(row["deep_DLE_mm"]) else f"{row['deep_DLE_mm']:.2f}"
    lines.append(f"| {row['case']} | {row['oracle_family']} | {row['best_single_local_AUC']:.3f} | "
        f"{row['consensus_local_AUC']:.3f} | {row['An_auc']:.3f} | "
        f"{row['surface_SD_mm']:.2f}/{row['surface_DLE_mm']:.2f} | {deep} | "
        f"{row['deep_detected']} | {row['active_count']} |")
lines += ["", "留一候选稳定性：" + "；".join(
    f"case {row['case']} = {row['leave_one_out_min_local_AUC']:.3f}–{row['leave_one_out_max_local_AUC']:.3f}"
    for row in robustness) + "。",
    "开发病例达到 0.90，但候选集合在这些病例上形成，且 H0/H1 仍为 oracle 选择；因此 accepted=false，冻结门控和独立验证完成前不得写成最终性能。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
