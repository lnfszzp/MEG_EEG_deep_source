"""汇总 sensor-balanced case04 的固定 ridge_fraction=0.05 开发诊断。"""

# %% 固定输入；只读取同一开发病例的 ridge0 与 ridge0.05 结果。
import csv
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_partial_confirmation import score_partial_deep_presence
from candidates.oaster_predictive import score_predictive_models
import run_erp_whole_head_matrix as original

manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_sensor_balanced_m10_m10_case04.json"
baseline = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04"
elastic = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_elastic005_admm_mrf080_m10_m10_case04"
case = json.loads(manifest.read_text(encoding="utf-8"))[0]
assert case["configuration_number"] == 4 and case["eeg_snr_db"] == case["meg_snr_db"] == -10
elastic_metadata = json.loads((elastic / "metadata.json").read_text(encoding="utf-8"))
elastic_completion = json.loads((elastic / "completion.json").read_text(encoding="utf-8"))
baseline_metadata = json.loads((baseline / "metadata.json").read_text(encoding="utf-8"))
assert elastic_completion == {"complete": True, "case_count": 1, "all_converged": True,
                              "wall_seconds": elastic_completion["wall_seconds"]}
assert elastic_metadata["phase"] == baseline_metadata["phase"] == "development"
assert elastic_metadata["covariance"] == baseline_metadata["covariance"] == "trial"
assert elastic_metadata["solver_settings"] == {"solver_kind": "admm", "mrf_strength": .8, "ridge_fraction": .05}
assert baseline_metadata["solver_settings"] == {"solver_kind": "admm", "mrf_strength": .8}

# %% 用同一 confirmation 数据配对计算 prediction 与 partial；真值只用于事后指标。
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
observation = prepare_trial_covariance_case(shared, case, seed_root=2026092206)
metric_rows, evidence_rows = [], []
for label, directory, ridge_fraction in (("ridge0", baseline, 0.), ("ridge0.05", elastic, .05)):
    detail = json.loads((directory / (case["case_id"] + ".json")).read_text(encoding="utf-8"))
    convergence = {name: bool(detail["fitting"][name + "_model"]["windows"][0]["solver"]["converged"])
                   for name in ("null", "full")}
    with np.load(directory / (case["case_id"] + ".npz")) as saved:
        null, full = saved["null"].astype(float), saved["full"].astype(float)
        assert np.allclose(saved["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    prediction, _ = score_predictive_models(
        observation["confirmation"], observation["gain"], null, full,
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"])
    partial, partial_detail = score_partial_deep_presence(
        observation["confirmation"], observation["gain"], null, full, shared["n_surf"],
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"])
    evidence_rows.append({"setting": label, "ridge_fraction": ridge_fraction,
        "prediction_score": prediction, "partial_confirmation_score": partial,
        "partial_selected_deep_index": partial_detail["selected_deep_index"],
        "partial_conditional_gain_fraction": partial_detail["conditional_gain_fraction"],
        "null_converged": int(convergence["null"]), "full_converged": int(convergence["full"]),
        "all_converged": int(all(convergence.values()))})
    for hypothesis, estimate in (("H0_surface_only", null), ("H1_surface_deep", full)):
        values = metrics.evaluate_estimate(
            estimate, observation["truth"], shared["vertices"], observation["groups"],
            shared["n_surf"], observation["active"], shared["auc_cortex"],
            baseline=observation["baseline"])
        metric_rows.append({"setting": label, "ridge_fraction": ridge_fraction, "hypothesis": hypothesis,
            "An_cal_AUC": values["auc"], "An_auc": values["auc_tie_corrected"],
            "surface_An_auc": values["surface_auc_tie_corrected"], "deep_An_auc": values["deep_auc_tie_corrected"],
            "surface_SD_mm": values["surface_sd_mm"], "deep_SD_mm": values["deep_sd_mm"],
            "surface_DLE_mm": values["surface_dle_mm"], "deep_DLE_mm": values["deep_dle_mm"]})

for name, rows in (("paired_metrics.csv", metric_rows), ("paired_evidence.csv", evidence_rows)):
    with (elastic / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# %% 配对报告。不把单例开发分数解释为阈值或验证结论。
h1 = {row["setting"]: row for row in metric_rows if row["hypothesis"] == "H1_surface_deep"}
evidence = {row["setting"]: row for row in evidence_rows}
summary = {"complete": True, "phase": "development", "case_id": case["case_id"],
    "fixed_ridge_fraction": .05, "grid_search": False, "calibration_or_validation_read": False,
    "all_converged": all(row["all_converged"] for row in evidence_rows),
    "H1_An_cal_AUC_delta": h1["ridge0.05"]["An_cal_AUC"] - h1["ridge0"]["An_cal_AUC"],
    "H1_An_auc_delta": h1["ridge0.05"]["An_auc"] - h1["ridge0"]["An_auc"],
    "prediction_score_delta": evidence["ridge0.05"]["prediction_score"] - evidence["ridge0"]["prediction_score"],
    "partial_score_delta": evidence["ridge0.05"]["partial_confirmation_score"] - evidence["ridge0"]["partial_confirmation_score"]}
(elastic / "paired_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = ["# case04 固定 elastic-net 开发诊断", "",
    "只比较预先固定的 `ridge_fraction=0.05` 与同一病例 ridge0 基线；未做网格，未读取 calibration/validation。",
    "", "| 设置 | 假设 | 收敛 H0/H1 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) | prediction | partial |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for row in metric_rows:
    item = evidence[row["setting"]]
    show = lambda value: "NA" if not np.isfinite(value) else f"{value:.4f}"
    lines.append(f"| {row['setting']} | {row['hypothesis']} | {item['null_converged']}/{item['full_converged']} | "
        f"{show(row['An_cal_AUC'])} | {show(row['An_auc'])} | {show(row['surface_An_auc'])}/{show(row['deep_An_auc'])} | "
        f"{show(row['surface_SD_mm'])}/{show(row['deep_SD_mm'])} | {show(row['surface_DLE_mm'])}/{show(row['deep_DLE_mm'])} | "
        f"{show(item['prediction_score'])} | {show(item['partial_confirmation_score'])} |")
lines.extend(["", "## 配对结论", "",
    f"- H1 An_cal AUC 变化：{summary['H1_An_cal_AUC_delta']:+.6f}",
    f"- H1 An_auc 变化：{summary['H1_An_auc_delta']:+.6f}",
    f"- confirmation prediction 变化：{summary['prediction_score_delta']:+.6f}",
    f"- partial confirmation 变化：{summary['partial_score_delta']:+.6f}",
    "", "这是单例开发诊断，confirmation/partial 均未校准，不能作为盲检阈值或验证结论。"])
(elastic / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
