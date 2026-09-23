"""# %% 用两模态匹配证据和图一致性修正凸图逆解的皮层局部排序。"""

# %% 固定开发输入与公式；真值只进入末尾指标，不进入定位图。
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/svd_graph_support_fusion_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")
output.mkdir(parents=True)
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = shared["n_surf"]
graph = sparse.csr_matrix(shared["adjacency"][:n_surf, :n_surf])
graph = graph.maximum(graph.T)
degree = np.asarray(graph.sum(axis=1)).ravel()
transition = sparse.diags(1 / np.maximum(degree, 1)) @ graph
positions = np.asarray(shared["vertices"], float)
rows, selected_rows = [], []

# %% 五种配置均用同一参数；H0/H1 仍按开发真值选族，只诊断定位能力。
for case in ("00", "01", "02", "03", "04"):
    case_record = json.loads((base / f"protocol/development_component_balanced_m10_m10_case{case}.json").read_text(encoding="utf-8"))[0]
    observation = prepare_trial_covariance_case(shared, case_record, seed_root=2026092206)
    name = f"erp-v6-development-{case}-eeg-10-meg-10.npz"
    source = base / f"dev_component_balanced_convex_admm_mrf080_m10_m10_case{case}" / name
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source) as package:
        assert np.allclose(package["truth"], observation["truth"])
        assert np.array_equal(package["active"], observation["active"])
        assert np.array_equal(package["baseline"], observation["baseline"])
        assert np.allclose(package["vertices"], positions)
        base_estimates = {family: np.asarray(package[family], float) for family in ("null", "full")}

    baseline, active = observation["baseline"], observation["active"]
    active_window = observation["active_windows"][0]
    weights = np.asarray(observation["channel_weights"], float)
    centered = observation["training"] - observation["training"][:, baseline].mean(axis=1, keepdims=True)
    sensor = weights[:, None] * centered
    gain = weights[:, None] * observation["gain"]
    basis = _smooth_temporal_basis(sensor, baseline, active_window)[0]
    left, singular, _ = np.linalg.svd(sensor @ basis.T, full_matrices=False)
    sensitivity = np.linalg.norm(gain, axis=0)
    sensitivity_floor = .1 * np.median(sensitivity[sensitivity > 0])
    matched = np.abs(gain.T @ left[:, :2])
    matched /= np.maximum(sensitivity[:, None], sensitivity_floor) ** .6
    matched /= np.maximum(matched[:n_surf].max(axis=0, keepdims=True), np.finfo(float).eps)
    mode_weight_sum = max(float(singular[:2].sum()), np.finfo(float).eps)
    mode_weights = singular[:2] / mode_weight_sum
    evidence = matched[:n_surf, :2] @ mode_weights
    evidence = transition @ (transition @ evidence)
    evidence /= max(float(evidence.max()), np.finfo(float).eps)

    refined = {}
    for family, estimate in base_estimates.items():
        amplitude = np.linalg.norm(estimate[:, active], axis=1)
        surface = amplitude[:n_surf] / max(float(amplitude[:n_surf].max()), np.finfo(float).eps)
        seeds = np.flatnonzero(surface >= .10)
        support_band = np.zeros(n_surf, bool)
        if seeds.size:
            support_band = cKDTree(positions[seeds]).query(positions[:n_surf])[0] <= .015
        score = .5 * surface + .5 * evidence * support_band
        score /= max(float(score.max()), np.finfo(float).eps)
        result = np.zeros_like(estimate)
        result[:n_surf] = score[:, None] * basis[0]
        deep = amplitude[n_surf:]
        if family == "full" and np.any(deep):
            deep = deep / deep.max()
            result[n_surf:] = deep[:, None] * basis[0]
        refined[family] = result
        before = metrics.evaluate_estimate(
            estimate, observation["truth"], positions, observation["groups"], n_surf,
            active, shared["auc_cortex"], baseline=baseline)
        after = metrics.evaluate_estimate(
            result, observation["truth"], positions, observation["groups"], n_surf,
            active, shared["auc_cortex"], baseline=baseline)
        rows.append({"case": case, "scenario": case_record["scenario"], "family": family,
            "base_local_AUC": before["auc"],
            "base_deep_false_positive": before["deep_false_positive"], **after})

    oracle_family = "full" if case_record["deep_index"] is not None else "null"
    chosen = next(row for row in rows if row["case"] == case and row["family"] == oracle_family)
    selected_rows.append({"case": case, "scenario": case_record["scenario"],
        "oracle_family": oracle_family, "base_local_AUC": float(chosen["base_local_AUC"]),
        "refined_local_AUC": float(chosen["auc"]),
        "surface_An_auc": float(chosen["surface_auc_tie_corrected"]),
        "deep_An_auc": float(chosen["deep_auc_tie_corrected"]),
        "surface_SD_mm": float(chosen["surface_sd_mm"]),
        "surface_DLE_mm": float(chosen["surface_dle_mm"]),
        "deep_DLE_mm": float(chosen["deep_dle_mm"]),
        "oracle_H1_deep_peak_within_10mm": int(np.isfinite(chosen["deep_peak_distance_mm"])
            and chosen["deep_peak_distance_mm"] <= 10 + 1e-6),
        "active_count": int(chosen["active_count"])})
    np.savez_compressed(output / name, truth=observation["truth"], active=active,
                        baseline=baseline, null=refined["null"], full=refined["full"])

archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
with (output / "selected_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=selected_rows[0].keys())
    writer.writeheader()
    writer.writerows(selected_rows)

# %% 结果只作为 localization score；门控和独立验证未完成时固定不验收。
summary = {"complete": True, "phase": "development", "case_count": 5,
    "input": "cached trial-covariance convex graph inverse artifacts listed by case",
    "fixed_parameters": {"svd_modes": 2, "depth_exponent": .6, "graph_walk_steps": 2,
        "mode_weighting": "training singular values", "sensitivity_floor": ".1 x median",
        "base_support_threshold": .10, "support_band_mm": 15, "mixing": .5,
        "layer_score_normalization": "separate surface/deep maxima after family selection"},
    "truth_used_in_localization": False, "parameters_selected_during_development": True,
    "oracle_family_selection_for_diagnostic": True,
    "deployable_presence_decision_evaluated": False,
    "local_AUC_target": .9,
    "all_development_local_AUC_at_least_0p9": all(row["refined_local_AUC"] >= .9 for row in selected_rows),
    "all_surface_DLE_finite_when_surface_true": all(np.isfinite(row["surface_DLE_mm"])
        for row in selected_rows if row["scenario"] != "deep_only"),
    "all_oracle_H1_deep_peaks_within_10mm": all(row["oracle_H1_deep_peak_within_10mm"]
        for row in selected_rows
        if row["scenario"] != "surface_only"),
    "pure_surface_raw_full_base_false_positives": sum(int(row["base_deep_false_positive"])
        for row in rows if row["scenario"] == "surface_only" and row["family"] == "full"),
    "independent_validation_passed": False, "accepted": False}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

labels = ["S", "S+S", "D", "S+D", "S+S+D"]
x = np.arange(5)
figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
axes[0].plot(x, [row["base_local_AUC"] for row in selected_rows], "o-",
             color="#9C9C9C", linewidth=2.2, label="convex graph")
axes[0].plot(x, [row["refined_local_AUC"] for row in selected_rows], "o-",
             color="#007C83", linewidth=2.2, label="adaptive evidence fusion")
axes[0].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[0].set(title="Local An_cal_AUC", ylabel="AUC", ylim=(.5, 1.02))
axes[0].legend(frameon=False, fontsize=8)
surface_auc = np.array([row["surface_An_auc"] for row in selected_rows])
deep_auc = np.array([row["deep_An_auc"] for row in selected_rows])
axes[1].plot(x[np.isfinite(surface_auc)], surface_auc[np.isfinite(surface_auc)], "o-",
             color="#007C83", linewidth=2.2, label="surface An_auc")
axes[1].scatter(x[np.isfinite(deep_auc)], deep_auc[np.isfinite(deep_auc)], marker="D",
                color="#D55E00", label="deep An_auc")
axes[1].set(title="Layerwise canonical An_auc", ylabel="AUC", ylim=(.5, 1.01))
axes[1].legend(frameon=False, fontsize=8)
finite_dle = np.array([np.isfinite(row["surface_DLE_mm"]) for row in selected_rows])
axes[2].scatter(x[finite_dle], [selected_rows[index]["surface_DLE_mm"]
                for index in np.flatnonzero(finite_dle)], color="#007C83", s=55)
axes[2].set(title="Surface DLE", ylabel="mm", ylim=(0, 15))
for index, row in enumerate(selected_rows):
    if not np.isfinite(row["surface_DLE_mm"]):
        label = "N/A" if row["scenario"] == "deep_only" else "FAIL"
        color = "#777777" if label == "N/A" else "#C44E52"
        axes[2].text(index, 1, label, ha="center", color=color, fontweight="bold")
for axis in axes:
    axis.set(xticks=x, xticklabels=labels)
figure.suptitle("SVD evidence + graph support (truth-selected family; development only)", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

lines = ["# SVD 图一致性自适应支持融合（仅开发诊断）", "",
    "固定使用前 2 个传感器时间模态、0.6 深度校正、按奇异值加权、2 步皮层随机游走，并只在原凸图 10% 峰值支持的 15 mm 邻域内融合；定位计算不读取真值。表层和深层各自转成层内归一的定位分数，只能解释为已知 family 后的条件定位，不能解释为源电流、全脑层间排序或 presence 证据。", "",
    "| case | 配置 | oracle family | 原/新条件局部 AUC | 表层/深层 An_auc | 表层 SD/DLE mm | 深层 DLE mm | H1深峰≤10mm |",
    "|---:|---|---|---:|---:|---:|---:|---:|"]
for row in selected_rows:
    surface_sd_text = "—" if not np.isfinite(row["surface_SD_mm"]) else f"{row['surface_SD_mm']:.2f}"
    surface_dle_text = "—" if not np.isfinite(row["surface_DLE_mm"]) else f"{row['surface_DLE_mm']:.2f}"
    deep = "—" if not np.isfinite(row["deep_DLE_mm"]) else f"{row['deep_DLE_mm']:.2f}"
    surface_auc_text = "—" if not np.isfinite(row["surface_An_auc"]) else f"{row['surface_An_auc']:.3f}"
    deep_auc_text = "—" if not np.isfinite(row["deep_An_auc"]) else f"{row['deep_An_auc']:.3f}"
    lines.append(f"| {row['case']} | {row['scenario']} | {row['oracle_family']} | "
        f"{row['base_local_AUC']:.3f}/{row['refined_local_AUC']:.3f} | "
        f"{surface_auc_text}/{deep_auc_text} | "
        f"{surface_sd_text}/{surface_dle_text} | {deep} | "
        f"{row['oracle_H1_deep_peak_within_10mm']} |")
target_text = "均超过" if summary["all_development_local_AUC_at_least_0p9"] else "并非全部超过"
failed_dle = [row["case"] for row in selected_rows
              if row["scenario"] != "deep_only" and not np.isfinite(row["surface_DLE_mm"])]
lines += ["", f"五种开发配置的局部 AUC {target_text} 0.90；表层 DLE 失败病例：{','.join(failed_dle) or '无'}。",
    f"未门控 full 基线在两个纯表层病例中误报 {summary['pure_surface_raw_full_base_false_positives']}/2。",
    "图表使用开发真值选择 H0/H1；独立 presence 门控和盲测尚未完成，因此 accepted=false。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
