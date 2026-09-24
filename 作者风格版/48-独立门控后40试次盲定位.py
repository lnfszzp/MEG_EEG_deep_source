"""# %% 开发闭环：train20/check20盲选H0/H1，决定后合并40试次定位。"""

# %% 固定算法，输入目录和开发阈值可切换；正式阈值必须来自纯表层校准。
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
from candidates.oaster_predictive import fit_predictive_models, score_predictive_models
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

base = root / "results/erp_whole_head/adaptive_v6"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path,
    default=base / "dev_component_balanced_convex_current_m10_m10")
parser.add_argument("--output", type=Path,
    default=base / "development_diagnosis/blind_gate_combined40_localization_m10_m10")
parser.add_argument("--development-threshold", type=float)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
development_threshold = (None if args.development_threshold is None
                         else float(args.development_threshold))
if development_threshold is not None and not np.isfinite(development_threshold):
    raise ValueError("development threshold must be finite")
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")
completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
assert completion["complete"] and completion["all_converged"]
assert completion["case_count"] == len(metadata["case_ids"])
phase = metadata["phase"]
if phase == "development" and development_threshold is None:
    raise ValueError("开发阶段必须明确给出 --development-threshold")
if phase == "validation" and development_threshold is not None:
    raise ValueError("正式验证必须使用源目录内的冻结校准决定，不能再给开发阈值")
if phase not in ("development", "validation"):
    raise ValueError("这里只处理development或validation结果")
assert metadata["covariance"] == "trial"
assert metadata["solver_settings"]["solver_kind"] == "admm"
assert metadata["solver_settings"]["mrf_strength"] == .8
assert metadata["solver_settings"]["outer_iterations"] == 1
assert metadata.get("score_kind", "noise") in ("noise", "excess")
for relative, expected in metadata["code_sha256"].items():
    path = root / relative
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected

manifest = Path(metadata["manifest"])
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == metadata["manifest_sha256"]
cases = json.loads(manifest.read_text(encoding="utf-8"))
assert len(cases) == completion["case_count"]
assert [case["case_number"] for case in cases] == list(range(len(cases)))
assert [case["case_id"] for case in cases] == metadata["case_ids"]
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
assert metadata["shared_fingerprint"] == original._shared_fingerprint(shared)
output.mkdir(parents=True)
n_surf = shared["n_surf"]
positions = np.asarray(shared["vertices"], float)
graph = sparse.csr_matrix(shared["adjacency"][:n_surf, :n_surf])
graph = graph.maximum(graph.T)
degree = np.asarray(graph.sum(axis=1)).ravel()
transition = sparse.diags(1 / np.maximum(degree, 1)) @ graph
rows, details = [], []
started = time.perf_counter()

# %% 决策不读真值；确认数据只在打分后才进入combined40重定位。
for case in cases:
    tick = time.perf_counter()
    observation = prepare_trial_covariance_case(shared, case, seed_root=metadata["seed_root"])
    name = case["case_id"]
    with np.load(source / (name + ".npz")) as package:
        train_null = np.asarray(package["null"], float)
        train_full = np.asarray(package["full"], float)
    recomputed_noise_score, score_info = score_predictive_models(
        observation["confirmation"], observation["gain"], train_null, train_full,
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"])
    saved = json.loads((source / (name + ".json")).read_text(encoding="utf-8"))
    saved_noise_score = (saved["evidence"]["loss_improvement"]
                         / saved["evidence"]["expected_response_noise_energy"])
    score_kind = metadata.get("score_kind", "noise")
    predictive_score = (saved_noise_score if score_kind == "noise"
                        else saved["evidence"]["excess_fraction_score"])
    recomputed_score = (recomputed_noise_score if score_kind == "noise"
                        else score_info["excess_fraction_score"])
    assert np.isclose(recomputed_score, predictive_score, rtol=5e-6, atol=3e-8)
    if phase == "development":
        assert (recomputed_score > development_threshold) == (predictive_score > development_threshold)
        deep_present = bool(predictive_score > development_threshold)
        decision_p_value = np.nan
        decision_rule = "labeled development threshold; not deployable"
    else:
        if not isinstance(saved.get("decision"), dict):
            raise ValueError(f"{name}: 验证结果缺少冻结校准决定")
        deep_present = bool(saved["decision"]["deep_present"])
        decision_p_value = float(saved["decision"]["p_value"])
        decision_rule = saved["decision"]["rule"]

    combined = (observation["training"] + observation["confirmation"]) / 2
    combined_null, combined_full, fitting = fit_predictive_models(
        combined, observation["gain"], n_surf, adjacency=shared["adjacency"],
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"], solver_settings=metadata["solver_settings"])
    convergence = {family: bool(fitting[family + "_model"]["windows"][0]["solver"]["converged"])
                   for family in ("null", "full")}
    selected_family = "full" if deep_present else "null"
    assert convergence[selected_family]
    selected = combined_full if deep_present else combined_null

    baseline, active = observation["baseline"], observation["active"]
    weights = np.asarray(observation["channel_weights"], float)
    retained = np.asarray(observation["metadata"]["retained_channels"], int)
    if retained.shape != (2,) or np.any(retained <= 0) or retained.sum() != weights.size:
        raise ValueError("retained_channels必须依次对应EEG、MEG，且总数等于channel_weights")
    modality_weights = np.array([
        np.median(weights[:retained[0]]), np.median(weights[retained[0]:])], float)
    if (not np.isfinite(modality_weights).all() or np.any(modality_weights < 0)
            or modality_weights.max() <= 0):
        raise ValueError("训练模态权重必须有限、非负且不能同时为零")
    np.testing.assert_allclose(modality_weights, observation["metadata"]["modality_weights"])
    eeg_weight, meg_weight = map(float, modality_weights)
    evidence_mix = .5 * meg_weight / float(modality_weights.max())
    assert 0 <= evidence_mix <= .5
    centered = combined - combined[:, baseline].mean(axis=1, keepdims=True)
    sensor = weights[:, None] * centered
    gain = weights[:, None] * observation["gain"]
    basis = _smooth_temporal_basis(sensor, baseline, observation["active_windows"][0])[0]
    left, singular, _ = np.linalg.svd(sensor @ basis.T, full_matrices=False)
    sensitivity = np.linalg.norm(gain, axis=0)
    sensitivity_floor = .1 * np.median(sensitivity[sensitivity > 0])
    matched = np.abs(gain.T @ left[:, :2])
    matched /= np.maximum(sensitivity[:, None], sensitivity_floor) ** .6
    matched /= np.maximum(matched[:n_surf].max(axis=0, keepdims=True), np.finfo(float).eps)
    mode_weights = singular[:2] / max(float(singular[:2].sum()), np.finfo(float).eps)
    evidence = transition @ (transition @ (matched[:n_surf] @ mode_weights))
    evidence /= max(float(evidence.max()), np.finfo(float).eps)

    amplitude = np.linalg.norm(selected[:, active], axis=1)
    surface = amplitude[:n_surf] / max(float(amplitude[:n_surf].max()), np.finfo(float).eps)
    seeds = np.flatnonzero(surface >= .10)
    support_band = (cKDTree(positions[seeds]).query(positions[:n_surf])[0] <= .015
                    if seeds.size else np.zeros(n_surf, bool))
    surface_score = (1 - evidence_mix) * surface + evidence_mix * evidence * support_band
    surface_score /= max(float(surface_score.max()), np.finfo(float).eps)
    blind = np.zeros_like(selected)
    blind[:n_surf] = surface_score[:, None] * basis[0]
    deep = amplitude[n_surf:]
    if deep_present and np.any(deep):
        blind[n_surf:] = (deep / deep.max())[:, None] * basis[0]

    # 盲估计已经固定后才读取真值，只做输入完整性检查和事后评价。
    with np.load(source / (name + ".npz")) as package:
        assert np.allclose(package["truth"], observation["truth"])
        assert np.array_equal(package["active"], observation["active"])
        assert np.array_equal(package["baseline"], observation["baseline"])

    values = metrics.evaluate_estimate(
        blind, observation["truth"], positions, observation["groups"], n_surf,
        active, shared["auc_cortex"], baseline=baseline)
    has_deep = case.get("deep_index") is not None
    rows.append({"case_id": name, "scenario": case["scenario"],
        "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
        "score_kind": score_kind, "predictive_score": predictive_score,
        "development_threshold": np.nan if development_threshold is None else development_threshold,
        "decision_p_value": decision_p_value, "decision_rule": decision_rule,
        "training_eeg_weight": eeg_weight, "training_meg_weight": meg_weight,
        "amplitude_evidence_mix": evidence_mix,
        "deep_present_decision": int(deep_present), "has_deep_true_posthoc": int(has_deep),
        "family_correct_posthoc": int(deep_present == has_deep),
        "selected_family": selected_family, "conditional_local_AUC": values["auc"],
        "surface_An_auc": values["surface_auc_tie_corrected"],
        "deep_An_auc": values["deep_auc_tie_corrected"],
        "surface_SD_mm": values["surface_sd_mm"], "surface_DLE_mm": values["surface_dle_mm"],
        "deep_DLE_mm": values["deep_dle_mm"], "active_count": values["active_count"]})
    detail = {"case_id": name, "complete": True, "truth_used_by_decision": False,
        "truth_used_only_for_posthoc_metrics": True, "predictive_score": predictive_score,
        "score_info": score_info,
        "development_threshold": development_threshold,
        "decision_p_value": decision_p_value, "decision_rule": decision_rule,
        "training_modality_weights": {"eeg": eeg_weight, "meg": meg_weight},
        "amplitude_evidence_mix": evidence_mix,
        "threshold_status": ("development mid-gap only; not frozen and not deployable"
            if phase == "development" else "frozen null-calibration decision copied from validation source"),
        "deep_present_decision": deep_present, "selected_family": selected_family,
        "post_decision_localization": "combined40 refit plus two-mode graph-support score",
        "only_selected_family_convergence_required": True,
        "convergence": convergence, "fitting": fitting,
        "elapsed_seconds": time.perf_counter() - tick}
    details.append(detail)
    (output / (name + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / (name + ".npz"), truth=observation["truth"].astype(np.float32),
                        estimate=blind.astype(np.float32), vertices=positions,
                        active=active, baseline=baseline)
    archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
    print(name, f"{score_kind}={predictive_score:.6f}", selected_family,
          f"AUC={values['auc']:.3f}", f"DLE={values['surface_dle_mm']}", flush=True)

# %% 汇总：开发闭环通过仍不等于校准/盲测通过。
summary = {"complete": True, "phase": phase, "case_count": len(rows),
    "solver_settings": metadata["solver_settings"], "covariance": "trial",
    "decision_statistic": ("independent confirmation excess-loss fraction"
        if metadata.get("score_kind") == "excess"
        else "independent confirmation noise-normalized loss improvement"),
    "score_kind": metadata.get("score_kind", "noise"),
    "development_threshold": development_threshold,
    "threshold_status": ("chosen inside the labeled development gap; must be replaced by frozen null calibration"
        if phase == "development" else "frozen null-calibration decisions loaded from untouched validation output"),
    "runtime_family_choice_used_truth": False,
    "parameters_selected_during_development": True,
    "all_selected_families_converged": all(
        detail["convergence"][detail["selected_family"]] for detail in details),
    "localization_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "localization_parameters": {"svd_modes": 2, "sensitivity_floor_fraction": .1,
        "depth_exponent": .6, "graph_steps": 2, "support_seed_fraction": .1,
        "support_radius_m": .015,
        "amplitude_evidence_mix_rule": "0.5 * training_MEG / max(training_EEG, training_MEG)",
        "amplitude_evidence_mix_range": [min(row["amplitude_evidence_mix"] for row in rows),
                                         max(row["amplitude_evidence_mix"] for row in rows)]},
    "family_correct_posthoc_count": sum(row["family_correct_posthoc"] for row in rows),
    "pure_surface_false_positive_count": sum(row["deep_present_decision"] for row in rows
        if not row["has_deep_true_posthoc"]),
    "deep_true_detected_count": sum(row["deep_present_decision"] for row in rows
        if row["has_deep_true_posthoc"]),
    "all_conditional_local_AUC_at_least_0p9": all(row["conditional_local_AUC"] >= .9 for row in rows),
    "all_surface_DLE_finite_when_surface_true": all(np.isfinite(row["surface_DLE_mm"])
        for row in rows if row["scenario"] != "deep_only"),
    "all_deep_DLE_zero_when_deep_true": all(row["deep_DLE_mm"] == 0
        for row in rows if row["has_deep_true_posthoc"]),
    "calibration_consumed": phase == "validation", "validation_consumed": phase == "validation",
    "accepted": False, "wall_seconds": time.perf_counter() - started}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

scenario_labels = {"surface_only": "S", "deep_only": "D",
    "deep_plus_surface": "S+D", "deep_plus_two_surface": "S+S+D"}
labels = [f"{scenario_labels[row['scenario']]}\n{row['eeg_snr_db']:+d}/{row['meg_snr_db']:+d}"
          for row in rows]
x = np.arange(len(rows))
figure, axes = plt.subplots(1, 3, figsize=(max(14.5, .72 * len(rows)), 5.2), constrained_layout=True)
colors = ["#D55E00" if row["deep_present_decision"] else "#007C83" for row in rows]
axes[0].scatter(x, [row["predictive_score"] for row in rows], color=colors, marker="s", s=48)
if development_threshold is not None:
    axes[0].axhline(development_threshold, color="#C44E52", linestyle=":", linewidth=1.7)
axes[0].set(title=f"Independent confirmation gate ({summary['score_kind']})", ylabel="Gate score")
if min(row["predictive_score"] for row in rows) > 0:
    axes[0].set_yscale("log")
axes[1].plot(x, [row["conditional_local_AUC"] for row in rows], "o-",
             color="#007C83", linewidth=2.2)
axes[1].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[1].set(title="Conditional local An_cal_AUC", ylabel="AUC", ylim=(.5, 1.02))
finite = np.array([np.isfinite(row["surface_DLE_mm"]) for row in rows])
axes[2].scatter(x[finite], [rows[index]["surface_DLE_mm"] for index in np.flatnonzero(finite)],
                color="#007C83", s=55)
for index, row in enumerate(rows):
    if not np.isfinite(row["surface_DLE_mm"]):
        axes[2].text(index, 1, "N/A" if row["scenario"] == "deep_only" else "FAIL",
                     ha="center", color="#777777" if row["scenario"] == "deep_only" else "#C44E52")
finite_dle = [row["surface_DLE_mm"] for row in rows if np.isfinite(row["surface_DLE_mm"])]
axes[2].set(title="Surface DLE", ylabel="mm",
            ylim=(0, max(15, 1.12 * max(finite_dle, default=0))))
for axis in axes:
    axis.set(xticks=x, xticklabels=labels)
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
figure.suptitle(f"Blind family gate + post-decision 40-trial localization ({phase})", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

with (output / "metrics_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
decision_text = (f"{development_threshold:g}是开发阈值，只用于闭环检查，正式流程必须换成纯表层校准规则。"
    if phase == "development" else "H0/H1决定来自运行前冻结的纯表层校准；本脚本不接收或重调阈值。")
lines = [f"# 独立门控后40试次盲定位（{phase}）", "",
    f"H0/H1 由 check20 的独立{summary['score_kind']}分数选择，决定后才合并40试次重定位；决定过程不读取真值。{decision_text}", "",
    "| 配置 | 门控分数 | 盲选family | 条件局部AUC | 表层An_auc | 表层SD/DLE mm | 深层DLE mm |",
    "|---|---:|---|---:|---:|---:|---:|"]
for label, row in zip(labels, rows):
    label = label.replace("\n", " ")
    surface_sd = "—" if not np.isfinite(row["surface_SD_mm"]) else f"{row['surface_SD_mm']:.2f}"
    surface_dle = "—" if not np.isfinite(row["surface_DLE_mm"]) else f"{row['surface_DLE_mm']:.2f}"
    deep_dle = "—" if not np.isfinite(row["deep_DLE_mm"]) else f"{row['deep_DLE_mm']:.2f}"
    surface_auc = "—" if not np.isfinite(row["surface_An_auc"]) else f"{row['surface_An_auc']:.3f}"
    lines.append(f"| {label} | {row['predictive_score']:.5f} | {row['selected_family']} | "
        f"{row['conditional_local_AUC']:.3f} | {surface_auc} | {surface_sd}/{surface_dle} | {deep_dle} |")
surface_count = sum(not row["has_deep_true_posthoc"] for row in rows)
deep_count = len(rows) - surface_count
lines += ["", f"盲选正确：{summary['family_correct_posthoc_count']}/{len(rows)}；纯表层误报：{summary['pure_surface_false_positive_count']}/{surface_count}；深源检出：{summary['deep_true_detected_count']}/{deep_count}。",
    f"局部AUC全部≥0.90：{summary['all_conditional_local_AUC_at_least_0p9']}；表层DLE全部有限：{summary['all_surface_DLE_finite_when_surface_true']}。",
    ("尚未消费正式 calibration/validation；accepted=false。" if phase == "development"
     else "正式验收结论由冻结的验收汇总规则给出；本定位步骤不改病例决定。"), ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
