"""# %% 汇总一次性正式验证；只读既有结果，不运行病例、不改变任何决定。"""

# %% 1. 输入必须来自同一冻结执行锁，输出目录拒绝覆盖。
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from candidates.oaster_predictive import conformal_decision

if not __debug__:
    raise RuntimeError("正式验收禁止使用python -O；否则assert完整性检查会被关闭")
default_lock = root / "results/erp_whole_head/adaptive_v6/protocol/component_balanced_v3_execution_lock.json"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation-source", type=Path, required=True,
                    help="23号脚本的一次性validation输出目录")
parser.add_argument("--localization", type=Path, required=True,
                    help="48号脚本对同一validation输出完成的combined40定位目录")
parser.add_argument("--output", type=Path, required=True, help="新的正式验收目录，禁止覆盖")
parser.add_argument("--execution-lock", type=Path, default=default_lock,
                    help="运行正式校准前冻结的执行锁")
args = parser.parse_args()
source = args.validation_source.resolve()
localization = args.localization.resolve()
output = args.output.resolve()
lock_path = args.execution_lock.resolve()
if output.exists():
    raise FileExistsError(f"正式验收目录已存在，拒绝覆盖：{output}")
for path in (source, localization):
    if not path.is_dir():
        raise FileNotFoundError(path)
if not lock_path.is_file():
    raise FileNotFoundError(lock_path)

lock_bytes = lock_path.read_bytes()
lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
lock_sidecar = lock_path.with_suffix(lock_path.suffix + ".sha256")
assert lock_sidecar.read_text(encoding="ascii") == f"{lock_sha256}  {lock_path.name}\n", \
    "execution lock或其SHA256旁车不一致"
lock = json.loads(lock_bytes)
assert lock["protocol"] == "erp-v6-component-balanced-formal-v3-execution"
assert lock["status"] == "algorithm_frozen_calibration_allowed"

# %% 2. 固定门限不能由本次验证或七个对比方法改变。
required_thresholds = {
    "required_false_positives_each_snr": 0,
    "required_localized_deep_detections_of_12": 10,
    "all_selected_families_converged": True,
    "minimum_local_AUC": 0.90,
    "maximum_surface_DLE_mm": 15.0,
    "maximum_surface_SD_mm": 20.0,
    "maximum_deep_DLE_mm": 10.0,
    "existing_deep_acceptance_passed_required": True,
}
for name, expected in required_thresholds.items():
    assert lock["acceptance"][name] == expected, f"执行锁验收门限发生变化：{name}"
assert lock["algorithm"]["covariance"] == "trial"
assert lock["algorithm"]["score_kind"] == "excess"
assert lock["algorithm"]["excess_loss_denominator_floor_fraction"] == .05
assert lock["algorithm"]["solver_settings"] == {
    "solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 1, "max_inner_retries": 10}
expected_calibration_decision = {
    "accept_deep": "T > 0 AND p(T) <= 0.05",
    "alpha": .05,
    "calibration_score_count": 57,
    "claim_scope": "marginal engineering FPR for the equal-weight three-SNR mixture only; no per-SNR or population guarantee",
    "equivalent_boundary": "T must be strictly greater than the second-largest of all 57 calibration scores",
    "final_validation_requirement": "retain the predeclared zero false positives among four H0 cases in every SNR cell",
    "ideal_continuous_exchangeable_tail_bound": 2 / 58,
    "maximum_calibration_scores_greater_or_equal": 1,
    "missing_nonfinite_or_unconverged_score_action": "calibration_invalid_do_not_open_validation",
    "rank_formula": "p(T) = (1 + count(T_cal >= T)) / 58",
    "required_complete_finite_scores": 57,
    "strata_count": 1,
    "test_true_snr_used": False,
    "ties_count_against_acceptance": True,
}
assert lock["algorithm"]["calibration_decision"] == expected_calibration_decision
expected_localization_rule = {
    "svd_modes": 2, "sensitivity_floor_fraction": .1, "depth_exponent": .6,
    "graph_steps": 2, "support_seed_fraction": .1, "support_radius_m": .015,
    "amplitude_evidence_mix_rule": "0.5 * training_MEG / max(training_EEG, training_MEG)",
}
assert lock["algorithm"]["localization_rule"] == expected_localization_rule

# %% 3. 逐文件核对代码、manifest、phase、病例身份和执行锁绑定。
normalized_lock_hashes = {Path(name).as_posix(): value
                          for name, value in lock["code_sha256"].items()}
for relative, expected in normalized_lock_hashes.items():
    path = root / Path(relative)
    assert path.is_file(), f"冻结代码不存在：{relative}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"冻结后代码改变：{relative}"
self_name = Path(__file__).resolve().relative_to(root).as_posix()
script23 = "作者风格版/23-独立观测浅深检出仿真.py"
script48 = "作者风格版/48-独立门控后40试次盲定位.py"
assert self_name in normalized_lock_hashes and script23 in normalized_lock_hashes \
    and script48 in normalized_lock_hashes

source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
source_completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
location_summary = json.loads((localization / "summary.json").read_text(encoding="utf-8"))
assert source_metadata["phase"] == location_summary["phase"] == "validation"
assert source_metadata["execution_lock_sha256"] == lock_sha256
assert source_metadata["covariance"] == lock["algorithm"]["covariance"]
assert source_metadata["score_kind"] == lock["algorithm"]["score_kind"]
assert source_metadata["solver_settings"] == lock["algorithm"]["solver_settings"]
source_hashes = {Path(name).as_posix(): value
                 for name, value in source_metadata["code_sha256"].items()}
assert source_hashes == normalized_lock_hashes, "validation运行代码与execution lock不一致"
assert source_metadata["shared_fingerprint"] == lock["shared_fingerprint"]
assert source_metadata["environment"] == lock["environment"]
assert source_completion["complete"] and source_completion["all_converged"]
assert source_completion["case_count"] == location_summary["case_count"] == 24
assert location_summary["complete"] and not location_summary["accepted"]
assert location_summary["runtime_family_choice_used_truth"] is False
assert location_summary["development_threshold"] is None
assert location_summary["score_kind"] == lock["algorithm"]["score_kind"]
assert location_summary["solver_settings"] == lock["algorithm"]["solver_settings"]
assert location_summary["covariance"] == lock["algorithm"]["covariance"]
assert location_summary["localization_script_sha256"] == normalized_lock_hashes[script48]
for name, expected in expected_localization_rule.items():
    assert location_summary["localization_parameters"][name] == expected

calibration_path_value = source_metadata.get("calibration_path")
assert isinstance(calibration_path_value, str) and calibration_path_value
calibration_path = Path(calibration_path_value).resolve()
assert calibration_path.name == "frozen_calibration.json" and calibration_path.is_file(), \
    f"冻结校准不存在：{calibration_path}"
calibration_bytes = calibration_path.read_bytes()
calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
assert source_metadata.get("calibration_sha256") == calibration_sha256
calibration = json.loads(calibration_bytes)
calibration_lock = lock["manifests"]["calibration"]
calibration_manifest_path = (root / calibration_lock["path"]).resolve()
calibration_manifest_bytes = calibration_manifest_path.read_bytes()
calibration_manifest_sha256 = hashlib.sha256(calibration_manifest_bytes).hexdigest()
calibration_manifest_sidecar = calibration_manifest_path.with_suffix(calibration_manifest_path.suffix + ".sha256")
assert calibration_lock["path"] == \
    "results/erp_whole_head/adaptive_v6/protocol/calibration_component_balanced_v3_manifest.json"
assert calibration_manifest_sha256 == calibration_lock["sha256"] == calibration["manifest_sha256"] == \
    "bb9f1a7d949efa6ba91dc8aceff604edca79752ecfb507134d06e4ccf56421d2"
assert hashlib.sha256(calibration_manifest_sidecar.read_text(encoding="ascii").encode("ascii")).hexdigest() == \
    calibration_lock["sidecar_text_sha256"]
assert Path(calibration["manifest"]).resolve() == calibration_manifest_path
assert calibration_lock["seed_roots"] == {"fit": 2026092501, "check": 2026092502}
assert calibration["seed_root"] == 2026092501
calibration_cases = json.loads(calibration_manifest_bytes)
calibration_case_ids = [case["case_id"] for case in calibration_cases]
assert len(calibration_cases) == calibration_lock["case_count"] == 57
assert len(set(calibration_case_ids)) == 57 and calibration["case_ids"] == calibration_case_ids
assert calibration["phase"] == "calibration"
assert calibration["complete"] is True and calibration["all_converged"] is True
assert calibration["case_count"] == 57 and calibration["alpha"] == .05
null_scores = np.asarray(calibration["null_scores"], dtype=float)
assert null_scores.shape == (57,) and np.isfinite(null_scores).all()
calibration_hashes = {Path(name).as_posix(): value
                      for name, value in calibration["code_sha256"].items()}
assert calibration_hashes == normalized_lock_hashes
assert calibration["shared_fingerprint"] == lock["shared_fingerprint"]
assert calibration["environment"] == lock["environment"]
assert calibration["solver_settings"] == lock["algorithm"]["solver_settings"]
assert calibration["covariance"] == lock["algorithm"]["covariance"]
assert calibration["score_kind"] == lock["algorithm"]["score_kind"]
assert calibration["execution_lock_sha256"] == lock_sha256

manifest_lock = lock["manifests"]["validation"]
manifest_path = (root / manifest_lock["path"]).resolve()
manifest_bytes = manifest_path.read_bytes()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
manifest_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
assert manifest_sha256 == manifest_lock["sha256"] == source_metadata["manifest_sha256"]
assert hashlib.sha256(manifest_sidecar.read_text(encoding="ascii").encode("ascii")).hexdigest() == \
    manifest_lock["sidecar_text_sha256"]
assert Path(source_metadata["manifest"]).resolve() == manifest_path
assert source_metadata["seed_root"] == manifest_lock["seed_roots"]["fit"]
cases = json.loads(manifest_bytes)
case_ids = [case["case_id"] for case in cases]
assert len(cases) == manifest_lock["case_count"] == 24 and len(set(case_ids)) == 24
assert source_metadata["case_ids"] == case_ids

with (source / "evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
    evidence_rows = list(csv.DictReader(stream))
with (source / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    source_rows = list(csv.DictReader(stream))
with (localization / "metrics_table.csv").open(encoding="utf-8-sig", newline="") as stream:
    location_rows = list(csv.DictReader(stream))
assert [row["case_id"] for row in evidence_rows] == case_ids
assert [row["case_id"] for row in location_rows] == case_ids
case_by_id = {case["case_id"]: case for case in cases}
evidence_by_id = {row["case_id"]: row for row in evidence_rows}
for evidence in evidence_rows:
    score = float(evidence["score"])
    decision = conformal_decision(score, null_scores, alpha=.05)
    expected_p = float((1 + np.count_nonzero(null_scores >= score)) / 58)
    assert decision["n_calibration"] == 57 and decision["alpha"] == .05
    assert decision["p_value"] == expected_p
    assert np.isclose(float(evidence["p_value"]), expected_p, rtol=0, atol=1e-15)
    assert int(float(evidence["deep_present_decision"])) == int(decision["deep_present"])
for row in location_rows:
    case = case_by_id[row["case_id"]]
    evidence = evidence_by_id[row["case_id"]]
    assert row["scenario"] == case["scenario"]
    assert int(row["eeg_snr_db"]) == int(case["eeg_snr_db"])
    assert int(row["meg_snr_db"]) == int(case["meg_snr_db"])
    assert int(float(row["deep_present_decision"])) == int(float(evidence["deep_present_decision"]))
    assert np.isclose(float(row["predictive_score"]), float(evidence["score"]), rtol=5e-6, atol=3e-8)
    detail = json.loads((localization / f"{row['case_id']}.json").read_text(encoding="utf-8"))
    assert detail["case_id"] == row["case_id"] and detail["truth_used_by_decision"] is False
    assert detail["selected_family"] == row["selected_family"]
    assert bool(detail["deep_present_decision"]) == bool(int(float(row["deep_present_decision"])))

# %% 4. 复算23号脚本的既有深源验收；这里开始读取真值且只做最终评价。
from benchmark.deep_acceptance import evaluate_acceptance

source_acceptance = json.loads((source / "acceptance.json").read_text(encoding="utf-8"))
recomputed_source_acceptance = evaluate_acceptance(
    cases, source_rows, evidence_rows, float(source_acceptance["penalty_mm"]))
assert recomputed_source_acceptance["gates"] == source_acceptance["gates"]
assert recomputed_source_acceptance["by_snr"] == source_acceptance["by_snr"]
assert recomputed_source_acceptance["passed"] == source_acceptance["passed"]

snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
assert {(int(case["eeg_snr_db"]), int(case["meg_snr_db"])) for case in cases} == set(snr_pairs)
assert sum(case["deep_index"] is not None for case in cases) == 12
by_snr = {}
all_selected_converged = True
for eeg, meg in snr_pairs:
    selected_cases = [case for case in cases
                      if (int(case["eeg_snr_db"]), int(case["meg_snr_db"])) == (eeg, meg)]
    selected_rows = [row for row in location_rows
                     if (int(row["eeg_snr_db"]), int(row["meg_snr_db"])) == (eeg, meg)]
    negative_ids = {case["case_id"] for case in selected_cases if case["deep_index"] is None}
    positive_ids = {case["case_id"] for case in selected_cases if case["deep_index"] is not None}
    assert len(selected_cases) == 8 and len(negative_ids) == len(positive_ids) == 4
    false_positives = sum(int(float(row["deep_present_decision"]))
                          for row in selected_rows if row["case_id"] in negative_ids)
    localized = sum(int(float(row["deep_present_decision"]))
                    and math.isfinite(float(row["deep_DLE_mm"]))
                    and float(row["deep_DLE_mm"]) <= required_thresholds["maximum_deep_DLE_mm"]
                    for row in selected_rows if row["case_id"] in positive_ids)
    by_snr[f"{eeg:+d}/{meg:+d}"] = {
        "case_count": 8, "h0_count": 4, "h1_count": 4,
        "h0_false_positives": false_positives,
        "localized_deep_detections": localized,
        "minimum_local_AUC": min(float(row["conditional_local_AUC"]) for row in selected_rows),
        "maximum_surface_DLE_mm": max((float(row["surface_DLE_mm"]) for row in selected_rows
                                       if math.isfinite(float(row["surface_DLE_mm"]))), default=math.nan),
        "maximum_surface_SD_mm": max((float(row["surface_SD_mm"]) for row in selected_rows
                                      if math.isfinite(float(row["surface_SD_mm"]))), default=math.nan),
        "maximum_deep_DLE_mm": max((float(row["deep_DLE_mm"]) for row in selected_rows
                                    if math.isfinite(float(row["deep_DLE_mm"]))), default=math.nan),
    }

for row in location_rows:
    detail = json.loads((localization / f"{row['case_id']}.json").read_text(encoding="utf-8"))
    all_selected_converged &= bool(detail["convergence"][detail["selected_family"]])
assert bool(location_summary["all_selected_families_converged"]) == all_selected_converged
local_auc = [float(row["conditional_local_AUC"]) for row in location_rows]
surface_dle = [float(row["surface_DLE_mm"]) for row in location_rows
               if math.isfinite(float(row["surface_DLE_mm"]))]
surface_sd = [float(row["surface_SD_mm"]) for row in location_rows
              if math.isfinite(float(row["surface_SD_mm"]))]
deep_dle = [float(row["deep_DLE_mm"]) for row in location_rows
            if case_by_id[row["case_id"]]["deep_index"] is not None
            and math.isfinite(float(row["deep_DLE_mm"]))]
surface_truth_count = sum(bool(case["surface_centers"]) for case in cases)
assert surface_truth_count == 18
localized_deep_count = sum(item["localized_deep_detections"] for item in by_snr.values())
gates = {
    "source_acceptance_passed": bool(source_acceptance["passed"]),
    "h0_false_positives_each_snr_equal_zero": all(
        item["h0_false_positives"] == required_thresholds["required_false_positives_each_snr"]
        for item in by_snr.values()),
    "localized_deep_detections_at_least_10_of_12":
        localized_deep_count >= required_thresholds["required_localized_deep_detections_of_12"],
    "all_selected_families_converged":
        all_selected_converged == required_thresholds["all_selected_families_converged"],
    "all_local_AUC_at_least_0p90":
        len(local_auc) == 24 and all(math.isfinite(value) and value >= required_thresholds["minimum_local_AUC"]
                                    for value in local_auc),
    "maximum_surface_DLE_at_most_15mm":
        len(surface_dle) == surface_truth_count
        and max(surface_dle) <= required_thresholds["maximum_surface_DLE_mm"],
    "maximum_surface_SD_at_most_20mm":
        len(surface_sd) == surface_truth_count
        and max(surface_sd) <= required_thresholds["maximum_surface_SD_mm"],
    "maximum_deep_DLE_at_most_10mm":
        bool(deep_dle) and max(deep_dle) <= required_thresholds["maximum_deep_DLE_mm"],
}
passed = all(gates.values())

# %% 5. 七个固定对比方法分别汇总train20和combined40；它们不进入上面的验收门。
comparator_names = ("MNE", "dSPM", "sLORETA", "eLORETA", "LCMV",
                    "Dipole fitting (grid)", "RAP-MUSIC")
method_rows = []
for method in comparator_names:
    for data_use in ("train20", "combined40"):
        method_name = f"{method} [{data_use}]"
        selected = [row for row in source_rows if row["method"] == method_name]
        assert len(selected) == 24 and {row["case_id"] for row in selected} == set(case_ids)
        auc_values = [float(row["auc"]) for row in selected]
        surface_auc_values = [float(row["surface_auc_tie_corrected"]) for row in selected
                              if math.isfinite(float(row["surface_auc_tie_corrected"]))]
        deep_auc_values = [float(row["deep_auc_tie_corrected"]) for row in selected
                           if math.isfinite(float(row["deep_auc_tie_corrected"]))]
        selected_surface_sd = [float(row["surface_sd_mm"]) for row in selected
                               if math.isfinite(float(row["surface_sd_mm"]))]
        selected_surface_dle = [float(row["surface_dle_mm"]) for row in selected
                                if math.isfinite(float(row["surface_dle_mm"]))]
        selected_deep_dle = [float(row["deep_dle_mm"]) for row in selected
                             if math.isfinite(float(row["deep_dle_mm"]))]
        method_rows.append({
            "method": method, "data_use": data_use, "case_count": 24,
            "mean_local_AUC": float(np.mean(auc_values)),
            "minimum_local_AUC": min(auc_values),
            "mean_surface_An_auc": float(np.mean(surface_auc_values)),
            "mean_deep_An_auc": float(np.mean(deep_auc_values)),
            "mean_surface_SD_mm": float(np.mean(selected_surface_sd)),
            "maximum_surface_SD_mm": max(selected_surface_sd),
            "mean_surface_DLE_mm": float(np.mean(selected_surface_dle)),
            "maximum_surface_DLE_mm": max(selected_surface_dle),
            "deep_DLE_available_count": len(selected_deep_dle),
            "mean_deep_DLE_mm": float(np.mean(selected_deep_dle)) if selected_deep_dle else math.nan,
            "maximum_deep_DLE_mm": max(selected_deep_dle, default=math.nan),
            "localized_deep_detections_of_12": sum(int(float(row["deep_detected"])) for row in selected
                                                    if case_by_id[row["case_id"]]["deep_index"] is not None),
            "h0_false_positives_of_12": sum(int(float(row["deep_false_positive"])) for row in selected
                                             if case_by_id[row["case_id"]]["deep_index"] is None),
            "deep_decision_definition": "legacy deep score >=0.14 and peak within 10 mm",
            "used_for_formal_acceptance": False,
        })

oaster_surface_auc = [float(row["surface_An_auc"]) for row in location_rows
                      if math.isfinite(float(row["surface_An_auc"]))]
oaster_deep_auc = [float(row["deep_An_auc"]) for row in location_rows
                   if math.isfinite(float(row["deep_An_auc"]))]
method_rows.append({
    "method": "OASTER-ERP-v6", "data_use": "train20/check20 gate + combined40 localization",
    "case_count": 24, "mean_local_AUC": float(np.mean(local_auc)),
    "minimum_local_AUC": min(local_auc),
    "mean_surface_An_auc": float(np.mean(oaster_surface_auc)),
    "mean_deep_An_auc": float(np.mean(oaster_deep_auc)),
    "mean_surface_SD_mm": float(np.mean(surface_sd)), "maximum_surface_SD_mm": max(surface_sd),
    "mean_surface_DLE_mm": float(np.mean(surface_dle)), "maximum_surface_DLE_mm": max(surface_dle),
    "deep_DLE_available_count": len(deep_dle),
    "mean_deep_DLE_mm": float(np.mean(deep_dle)), "maximum_deep_DLE_mm": max(deep_dle),
    "localized_deep_detections_of_12": localized_deep_count,
    "h0_false_positives_of_12": sum(item["h0_false_positives"] for item in by_snr.values()),
    "deep_decision_definition": "frozen independent-confirmation gate and post-decision DLE <=10 mm",
    "used_for_formal_acceptance": True,
})

# %% 6. 写一次最终结论、对比表和不裁剪的清晰图；任何对比数值都不能回写gates。
output.mkdir(parents=True)
with (output / "method_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=method_rows[0].keys())
    writer.writeheader()
    writer.writerows(method_rows)

method_index = {name: index for index, name in enumerate(comparator_names)}
figure, axes = plt.subplots(1, 4, figsize=(18, 7.2), sharey=True, constrained_layout=True)
for data_use, marker, color, label in (("train20", "s", "#6F7782", "Comparator train20"),
                                        ("combined40", "D", "#007C83", "Comparator combined40")):
    selected = [row for row in method_rows if row["data_use"] == data_use]
    y = [method_index[row["method"]] for row in selected]
    for axis, field in zip(axes, ("minimum_local_AUC", "maximum_surface_DLE_mm",
                                  "localized_deep_detections_of_12", "h0_false_positives_of_12")):
        axis.scatter([row[field] for row in selected], y, marker=marker, color=color,
                     s=64, label=label if axis is axes[0] else None, zorder=3)
oaster = method_rows[-1]
for axis, field in zip(axes, ("minimum_local_AUC", "maximum_surface_DLE_mm",
                              "localized_deep_detections_of_12", "h0_false_positives_of_12")):
    axis.scatter([oaster[field]], [len(comparator_names)], marker="^", color="#D55E00",
                 edgecolor="white", linewidth=.8, s=100,
                 label="OASTER formal" if axis is axes[0] else None, zorder=4)
axes[0].axvline(required_thresholds["minimum_local_AUC"], color="#C44E52", linestyle=":")
axes[1].axvline(required_thresholds["maximum_surface_DLE_mm"], color="#C44E52", linestyle=":")
axes[2].axvline(required_thresholds["required_localized_deep_detections_of_12"],
                color="#C44E52", linestyle=":")
axes[3].axvline(required_thresholds["required_false_positives_each_snr"] * 3,
                color="#C44E52", linestyle=":")
axes[0].set(xlabel="Minimum local An_cal_AUC", title="Worst-case localization discrimination")
axes[1].set(xlabel="Maximum surface DLE (mm)", title="Worst-case surface localization error")
axes[2].set(xlabel="Localized deep detections / 12", title="Deep sensitivity")
axes[3].set(xlabel="H0 false positives / 12", title="Deep false positives")
axes[0].set_yticks(np.arange(len(comparator_names) + 1), [*comparator_names, "OASTER-ERP-v6"])
axes[0].invert_yaxis()
axes[0].legend(frameon=False, loc="lower right", fontsize=9)
for axis in axes:
    axis.grid(axis="x", color="#E4E7EB", linewidth=.8)
figure.suptitle(f"Formal blind validation: {'PASS' if passed else 'FAIL'}\n"
                 "Comparators are descriptive only; train20 and combined40 are kept separate",
                 fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

acceptance = {
    "passed": passed, "phase": "validation", "case_count": 24,
    "execution_lock": str(lock_path), "execution_lock_sha256": lock_sha256,
    "calibration": str(calibration_path), "calibration_sha256": calibration_sha256,
    "manifest": str(manifest_path), "manifest_sha256": manifest_sha256,
    "validation_source": str(source), "localization": str(localization),
    "truth_used_only_for_final_evaluation": True,
    "comparison_methods_used_for_acceptance": False,
    "thresholds": required_thresholds, "gates": gates, "by_snr": by_snr,
    "metrics": {
        "localized_deep_detections_of_12": localized_deep_count,
        "minimum_local_AUC": min(local_auc),
        "maximum_surface_DLE_mm": max(surface_dle),
        "maximum_surface_SD_mm": max(surface_sd),
        "maximum_deep_DLE_mm": max(deep_dle),
    },
    "source_acceptance": source_acceptance,
    "artifacts": {"method_table": "method_table.csv", "figure": "comparison.png",
                  "report": "REPORT.md"},
}
(output / "acceptance.json").write_text(
    json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

report = ["# 完整盲定位正式验收", "",
          f"结论：**{'通过' if passed else '未通过'}**。本报告只读取冻结后生成的结果；真值只用于最终评价，七个对比方法不参与调参或通过判定。", "",
          "## 冻结绑定", "",
          f"- execution lock SHA256：`{lock_sha256}`",
          f"- frozen calibration SHA256：`{calibration_sha256}`",
          f"- validation manifest SHA256：`{manifest_sha256}`",
          "- 决策：train20拟合H0/H1，独立check20门控，决定后combined40定位。", "",
          "## 硬性验收", "",
          "| 条件 | 结果 |", "|---|---:|"]
for name, value in gates.items():
    report.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
report += ["", "| EEG/MEG SNR | H0误报/4 | 深源定位检出/4 | 最低AUC | 最大表层DLE/SD mm | 最大深层DLE mm |",
           "|---|---:|---:|---:|---:|---:|"]
for label, values in by_snr.items():
    report.append(f"| {label} | {values['h0_false_positives']}/4 | "
                  f"{values['localized_deep_detections']}/4 | {values['minimum_local_AUC']:.3f} | "
                  f"{values['maximum_surface_DLE_mm']:.2f}/{values['maximum_surface_SD_mm']:.2f} | "
                  f"{values['maximum_deep_DLE_mm']:.2f} |")
report += ["", "## 方法对比（不改变验收）", "",
           "| 方法 | 数据 | 平均/最低AUC | 平均/最大表层DLE mm | 深源检出/12 | H0误报/12 |",
           "|---|---|---:|---:|---:|---:|"]
for row in method_rows:
    report.append(f"| {row['method']} | {row['data_use']} | "
                  f"{row['mean_local_AUC']:.3f}/{row['minimum_local_AUC']:.3f} | "
                  f"{row['mean_surface_DLE_mm']:.2f}/{row['maximum_surface_DLE_mm']:.2f} | "
                  f"{row['localized_deep_detections_of_12']}/12 | "
                  f"{row['h0_false_positives_of_12']}/12 |")
report += ["", "对比方法的深源检出沿用0.14分数阈值和10 mm半径；OASTER使用冻结的独立确认门控，因此表中已明确分别标注，不能把两者当作同一决策器。", ""]
(output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
print(json.dumps({"passed": passed, "gates": gates, "metrics": acceptance["metrics"]},
                 ensure_ascii=False, indent=2))
