"""# %% 新位置开发集：冻结门控，只比较三个最小定位后处理候选。"""

# %% 1. 输入只允许 development；旧正式 validation 不参与候选选择。
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
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--development-threshold", type=float, required=True)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
development_threshold = float(args.development_threshold)
if not np.isfinite(development_threshold):
    raise ValueError("development threshold must be finite")
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")

completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
if metadata["phase"] != "development" or metadata["covariance"] != "trial" or \
        metadata.get("score_kind") != "excess":
    raise ValueError("本脚本只处理 trial covariance + excess score 的 development 输出")
if metadata["solver_settings"] != {
        "solver_kind": "admm", "mrf_strength": .8,
        "outer_iterations": 1, "max_inner_retries": 20}:
    raise ValueError("定位候选必须共享冻结的 ADMM 设置")
if not completion["complete"] or not completion["all_converged"]:
    raise ValueError("上游 20+20 试次拟合尚未完整收敛")
for relative, expected in metadata["code_sha256"].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"上游代码已变化：{path}")

manifest = Path(metadata["manifest"])
if hashlib.sha256(manifest.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
    raise ValueError("development manifest 与上游记录不一致")
cases = json.loads(manifest.read_text(encoding="utf-8"))
if len(cases) != 15 or len({case["configuration_id"] for case in cases}) != 5 or \
        any(case.get("component_balance_revision") !=
            "development_localization_v5_new_positions_paired_snr" for case in cases):
    raise ValueError("这里只接受 v5 配对新位置开发面板")
if [case["case_id"] for case in cases] != metadata["case_ids"]:
    raise ValueError("病例顺序与上游输出不一致")

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
if metadata["shared_fingerprint"] != original._shared_fingerprint(shared):
    raise ValueError("共享 forward 已变化")
output.mkdir(parents=True)
n_surf = int(shared["n_surf"])
positions = np.asarray(shared["vertices"], float)
graph = sparse.csr_matrix(shared["adjacency"][:n_surf, :n_surf])
graph = graph.maximum(graph.T)
degree = np.asarray(graph.sum(axis=1)).ravel()
transition = sparse.diags(1 / np.maximum(degree, 1)) @ graph
candidate_names = ("current", "baseline_corrected", "residual_deep", "consensus_deep")
rows, details = [], []
started = time.perf_counter()


# %% 2. 每例上游只重拟合一次；四个结果共享同一个盲门控决定。
for case in cases:
    tick = time.perf_counter()
    name = case["case_id"]
    observation = prepare_trial_covariance_case(shared, case, seed_root=metadata["seed_root"])
    with np.load(source / (name + ".npz")) as package:
        train_null = np.asarray(package["null"], float)
        train_full = np.asarray(package["full"], float)
    _, score_info = score_predictive_models(
        observation["confirmation"], observation["gain"], train_null, train_full,
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"])
    recomputed_score = float(score_info["excess_fraction_score"])
    saved = json.loads((source / (name + ".json")).read_text(encoding="utf-8"))
    predictive_score = float(saved["evidence"]["excess_fraction_score"])
    if not np.isclose(recomputed_score, predictive_score, rtol=5e-6, atol=3e-8):
        raise ValueError(f"{name}: 独立门控分数无法复现")
    deep_present = bool(predictive_score > development_threshold)

    combined = (observation["training"] + observation["confirmation"]) / 2
    combined_null, combined_full, fitting = fit_predictive_models(
        combined, observation["gain"], n_surf, adjacency=shared["adjacency"],
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"],
        solver_settings=metadata["solver_settings"])
    convergence = {family: bool(fitting[family + "_model"]["windows"][0]["solver"]["converged"])
                   for family in ("null", "full")}
    selected_family = "full" if deep_present else "null"
    if not convergence[selected_family]:
        raise RuntimeError(f"{name}: 盲选 {selected_family} 未收敛")
    selected = combined_full if deep_present else combined_null

    baseline, active = observation["baseline"], observation["active"]
    if np.asarray(active).dtype == bool:
        raise ValueError("source_amplitude 的 active 必须是样本索引，不能是布尔 mask")
    weights = np.asarray(observation["channel_weights"], float)
    retained = np.asarray(observation["metadata"]["retained_channels"], int)
    if retained.shape != (2,) or np.any(retained <= 0) or retained.sum() != weights.size:
        raise ValueError("retained_channels 必须依次对应 EEG、MEG")
    modality_weights = np.array([
        np.median(weights[:retained[0]]), np.median(weights[retained[0]:])], float)
    if not np.isfinite(modality_weights).all() or np.any(modality_weights < 0) or \
            modality_weights.max() <= 0:
        raise ValueError("模态权重必须有限、非负且不能同时为零")
    np.testing.assert_allclose(modality_weights, observation["metadata"]["modality_weights"])
    evidence_mix = .5 * float(modality_weights[1] / modality_weights.max())

    centered = combined - combined[:, baseline].mean(axis=1, keepdims=True)
    sensor = weights[:, None] * centered
    gain = weights[:, None] * observation["gain"]
    basis = _smooth_temporal_basis(sensor, baseline, observation["active_windows"][0])[0]
    if basis.ndim != 2 or not basis.shape[0]:
        raise RuntimeError(f"{name}: 没有可用的 ERP 时间基")
    left, singular, _ = np.linalg.svd(sensor @ basis.T, full_matrices=False)
    sensitivity = np.linalg.norm(gain, axis=0)
    sensitivity_floor = .1 * np.median(sensitivity[sensitivity > 0])
    matched = np.abs(gain.T @ left[:, :2])
    matched /= np.maximum(sensitivity[:, None], sensitivity_floor) ** .6
    matched /= np.maximum(matched[:n_surf].max(axis=0, keepdims=True), np.finfo(float).eps)
    mode_weights = singular[:2] / max(float(singular[:2].sum()), np.finfo(float).eps)
    evidence = transition @ (transition @ (matched[:n_surf] @ mode_weights))
    evidence /= max(float(evidence.max()), np.finfo(float).eps)

    current_amplitude = np.linalg.norm(selected[:, active], axis=1)
    corrected_amplitude = metrics.source_amplitude(selected, active, baseline)
    surface_scores = {}
    corrected_surface_fallback = False
    for amplitude_name, amplitude in (("current", current_amplitude),
                                      ("corrected", corrected_amplitude)):
        surface = amplitude[:n_surf] / max(
            float(amplitude[:n_surf].max()), np.finfo(float).eps)
        seeds = np.flatnonzero(surface >= .10)
        support_band = (cKDTree(positions[seeds]).query(positions[:n_surf])[0] <= .015
                        if seeds.size else np.zeros(n_surf, bool))
        score = (1 - evidence_mix) * surface + evidence_mix * evidence * support_band
        score /= max(float(score.max()), np.finfo(float).eps)
        if amplitude_name == "corrected" and not np.any(score):
            score = surface_scores["current"].copy()
            corrected_surface_fallback = True
        surface_scores[amplitude_name] = score

    current_deep = current_amplitude[n_surf:].copy()
    corrected_deep = corrected_amplitude[n_surf:].copy()
    if current_deep.max(initial=0.0) > 0:
        current_deep /= current_deep.max()
    if corrected_deep.max(initial=0.0) > 0:
        corrected_deep /= corrected_deep.max()

    # B/C：只在 H1 内用 H0 拟合响应的低秩列空间做条件残差定位。
    residual_deep = corrected_deep.copy()
    additive_deep = corrected_deep.copy()
    consensus_deep = corrected_deep.copy()
    residual_fallback_reason = None
    consensus_fallback = False
    residual_rank = 0
    if deep_present:
        if not convergence["null"]:
            residual_fallback_reason = "combined40 null model did not converge"
        else:
            response = sensor @ basis.T
            fitted = gain @ (combined_null @ basis.T)
            if np.any(fitted):
                fitted_left, fitted_singular, _ = np.linalg.svd(fitted, full_matrices=False)
                tolerance = fitted_singular[0] * max(fitted.shape) * np.finfo(float).eps
                keep = fitted_singular > tolerance
                surface_space = fitted_left[:, keep]
            else:
                surface_space = np.zeros((gain.shape[0], 0))
            residual_rank = int(surface_space.shape[1])
            residual = response - fitted
            conditional_gain = gain[:, n_surf:].copy()
            if residual_rank:
                residual -= surface_space @ (surface_space.T @ residual)
                conditional_gain -= surface_space @ (surface_space.T @ conditional_gain)
            conditional_norm = np.sum(conditional_gain ** 2, axis=0)
            valid = conditional_norm > np.finfo(float).eps
            proposed = np.zeros(conditional_gain.shape[1])
            if np.any(valid) and np.any(residual):
                proposed[valid] = np.sqrt(np.sum(
                    (conditional_gain[:, valid].T @ residual) ** 2, axis=1)
                    / conditional_norm[valid])
            if proposed.max(initial=0.0) > 0:
                residual_deep = proposed / proposed.max()
                additive_deep = corrected_deep + residual_deep
                additive_deep /= max(float(additive_deep.max()), np.finfo(float).eps)
                proposed_consensus = np.sqrt(corrected_deep * residual_deep)
                if proposed_consensus.max(initial=0.0) > 0:
                    consensus_deep = proposed_consensus / proposed_consensus.max()
                else:
                    consensus_fallback = True
            else:
                residual_fallback_reason = "conditional residual score was empty"

    candidate_surface = {
        "current": surface_scores["current"],
        "baseline_corrected": surface_scores["corrected"],
        "residual_deep": surface_scores["corrected"],
        "consensus_deep": surface_scores["corrected"],
    }
    candidate_deep = {
        "current": current_deep,
        "baseline_corrected": corrected_deep,
        "residual_deep": additive_deep,
        "consensus_deep": consensus_deep,
    }
    candidate_fallback = {
        "current": False,
        "baseline_corrected": corrected_surface_fallback,
        "residual_deep": corrected_surface_fallback or residual_fallback_reason is not None,
        "consensus_deep": (corrected_surface_fallback or residual_fallback_reason is not None
                           or consensus_fallback),
    }
    candidate_maps = {}
    for candidate in candidate_names:
        spatial = np.zeros(selected.shape[0])
        spatial[:n_surf] = candidate_surface[candidate]
        if deep_present:
            spatial[n_surf:] = candidate_deep[candidate]
        candidate_maps[candidate] = spatial

    # 所有候选的盲结果固定后才读取真值，只做开发集事后比较。
    with np.load(source / (name + ".npz")) as package:
        if not np.allclose(package["truth"], observation["truth"]) or \
                not np.array_equal(package["active"], active) or \
                not np.array_equal(package["baseline"], baseline):
            raise ValueError(f"{name}: 上游保存数组与重建观测不一致")
    has_deep = case.get("deep_index") is not None
    family_correct = deep_present == has_deep
    for candidate in candidate_names:
        blind = candidate_maps[candidate][:, None] * basis[0]
        values = metrics.evaluate_estimate(
            blind, observation["truth"], positions, observation["groups"], n_surf,
            active, shared["auc_cortex"], baseline=baseline)
        rows.append({
            "case_id": name, "configuration_id": case["configuration_id"],
            "candidate": candidate, "scenario": case["scenario"],
            "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
            "predictive_score": predictive_score, "development_threshold": development_threshold,
            "deep_present_decision": int(deep_present),
            "has_deep_true_posthoc": int(has_deep),
            "family_correct_posthoc": int(family_correct),
            "selected_family": selected_family,
            "conditional_local_AUC": values["auc"],
            "surface_An_auc": values["surface_auc_tie_corrected"],
            "deep_An_auc": values["deep_auc_tie_corrected"],
            "surface_SD_mm": values["surface_sd_mm"],
            "surface_DLE_mm": values["surface_dle_mm"],
            "deep_DLE_mm": values["deep_dle_mm"],
            "deep_peak_distance_mm": values["deep_peak_distance_mm"],
            "candidate_fallback": int(candidate_fallback[candidate]),
            "active_count": values["active_count"],
        })
    np.savez_compressed(output / (name + ".npz"),
        truth=metrics.source_amplitude(observation["truth"], active, baseline).astype(np.float32),
        vertices=positions.astype(np.float32),
        **{candidate: candidate_maps[candidate].astype(np.float32)
           for candidate in candidate_names})
    detail = {"case_id": name, "complete": True,
        "truth_used_by_gate_or_candidate_maps": False,
        "truth_used_only_for_posthoc_metrics": True,
        "predictive_score": predictive_score, "score_info": score_info,
        "development_threshold": development_threshold,
        "deep_present_decision": deep_present, "selected_family": selected_family,
        "family_correct_posthoc": family_correct, "convergence": convergence,
        "corrected_surface_fallback": corrected_surface_fallback,
        "residual_fallback_reason": residual_fallback_reason,
        "consensus_fallback": consensus_fallback,
        "residual_surface_rank": residual_rank,
        "fitting": fitting, "elapsed_seconds": time.perf_counter() - tick}
    details.append(detail)
    (output / (name + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
    print(name, f"gate={predictive_score:.6f}", selected_family,
          " ".join(f"{candidate}={rows[-4 + index]['conditional_local_AUC']:.3f}"
                   for index, candidate in enumerate(candidate_names)), flush=True)


# %% 3. 先过共同门控，再按 A、B、C 的最小改动顺序选第一个全达标者。
common_gate_passed = all(detail["family_correct_posthoc"] for detail in details)
all_selected_converged = all(
    detail["convergence"][detail["selected_family"]] for detail in details)
candidate_summary = []
for candidate in candidate_names:
    selected_rows = [row for row in rows if row["candidate"] == candidate]
    surface_rows = [row for row in selected_rows if row["scenario"] != "deep_only"]
    deep_rows = [row for row in selected_rows if row["has_deep_true_posthoc"]]
    local_auc_pass = all(np.isfinite(row["conditional_local_AUC"]) and
                         row["conditional_local_AUC"] >= .9 for row in selected_rows)
    surface_finite = all(np.isfinite(row["surface_DLE_mm"]) for row in surface_rows)
    surface_dle_pass = surface_finite and all(row["surface_DLE_mm"] <= 15 for row in surface_rows)
    surface_sd_pass = all(np.isfinite(row["surface_SD_mm"]) and row["surface_SD_mm"] <= 20
                          for row in surface_rows)
    deep_distance_pass = all(np.isfinite(row["deep_peak_distance_mm"]) and
                             row["deep_peak_distance_mm"] <= 10 for row in deep_rows)
    fallback_count = sum(row["candidate_fallback"] for row in selected_rows)
    passed = bool(common_gate_passed and all_selected_converged and local_auc_pass and
                  surface_dle_pass and surface_sd_pass and deep_distance_pass and
                  fallback_count == 0)
    candidate_summary.append({
        "candidate": candidate,
        "mean_local_AUC": float(np.mean([row["conditional_local_AUC"] for row in selected_rows])),
        "min_local_AUC": float(np.min([row["conditional_local_AUC"] for row in selected_rows])),
        "max_surface_DLE_mm": float(max((row["surface_DLE_mm"] for row in surface_rows
                                         if np.isfinite(row["surface_DLE_mm"])), default=np.nan)),
        "max_surface_SD_mm": float(max((row["surface_SD_mm"] for row in surface_rows
                                        if np.isfinite(row["surface_SD_mm"])), default=np.nan)),
        "surface_DLE_finite_count": sum(np.isfinite(row["surface_DLE_mm"])
                                        for row in surface_rows),
        "surface_case_count": len(surface_rows),
        "deep_within_10mm_count": sum(np.isfinite(row["deep_peak_distance_mm"]) and
                                      row["deep_peak_distance_mm"] <= 10 for row in deep_rows),
        "deep_case_count": len(deep_rows),
        "fallback_count": fallback_count,
        "all_local_AUC_at_least_0p9": local_auc_pass,
        "all_surface_DLE_at_most_15mm": surface_dle_pass,
        "all_surface_SD_at_most_20mm": surface_sd_pass,
        "all_deep_peaks_within_10mm": deep_distance_pass,
        "development_screen_passed": passed,
    })

priority = ("baseline_corrected", "residual_deep", "consensus_deep", "current")
winner = next((candidate for candidate in priority if next(
    row["development_screen_passed"] for row in candidate_summary
    if row["candidate"] == candidate)), None)
summary = {
    "complete": True, "phase": "development", "accepted": False,
    "case_count": len(cases), "candidate_count": len(candidate_names),
    "development_threshold": development_threshold,
    "threshold_status": "fixed diagnostic carry-over; not a new formal calibration",
    "presence_gate_changed": False,
    "family_correct_count": sum(detail["family_correct_posthoc"] for detail in details),
    "common_gate_passed": common_gate_passed,
    "all_selected_families_converged": all_selected_converged,
    "candidate_priority": list(priority),
    "winner": winner,
    "winner_requires_new_blind_confirmation": winner is not None,
    "candidate_summary": candidate_summary,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "wall_seconds": time.perf_counter() - started,
}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
archive._atomic_csv(output / "candidate_summary.csv", candidate_summary,
                    candidate_summary[0].keys())
with (output / "metrics_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


# %% 4. 不用柱状图；逐例连线保留配对关系。
x = np.arange(len(cases))
colors = {"current": "#7A7A7A", "baseline_corrected": "#007C83",
          "residual_deep": "#D55E00", "consensus_deep": "#6A51A3"}
labels = {"current": "v4 current", "baseline_corrected": "A baseline-corrected",
          "residual_deep": "B residual deep", "consensus_deep": "C consensus deep"}
figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
for candidate in candidate_names:
    selected_rows = [row for row in rows if row["candidate"] == candidate]
    axes[0].plot(x, [row["conditional_local_AUC"] for row in selected_rows], "o-",
                 color=colors[candidate], linewidth=1.6, markersize=4, label=labels[candidate])
    axes[1].plot(x, [row["surface_DLE_mm"] for row in selected_rows], "o-",
                 color=colors[candidate], linewidth=1.6, markersize=4)
    axes[2].plot(x, [row["deep_peak_distance_mm"] for row in selected_rows], "o-",
                 color=colors[candidate], linewidth=1.6, markersize=4)
axes[0].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[1].axhline(15, color="#C44E52", linestyle=":", linewidth=1.5)
axes[2].axhline(10, color="#C44E52", linestyle=":", linewidth=1.5)
axes[0].set(title="Conditional local An_cal_AUC", ylabel="AUC", ylim=(.5, 1.02))
axes[1].set(title="Surface DLE", ylabel="mm")
axes[2].set(title="Deep peak distance", ylabel="mm")
case_labels = [f"{case['configuration_number']}\n{case['eeg_snr_db']:+d}/{case['meg_snr_db']:+d}"
               for case in cases]
for axis in axes:
    axis.set(xticks=x, xticklabels=case_labels, xlabel="geometry / EEG-MEG SNR (dB)")
    axis.grid(axis="y", color="#DDDDDD", linewidth=.7)
axes[0].legend(frameon=False, fontsize=8)
figure.suptitle("v5 paired new-position development: frozen gate, post-processing only",
                fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

lines = ["# v5 新位置定位后处理开发比较", "",
    f"presence 门控不变，15 例盲选正确 {summary['family_correct_count']}/15；"
    f"定位候选只在所有盲图固定后读取真值。开发胜者：{winner or '无'}。", "",
    "| 候选 | 平均/最低局部AUC | 表层DLE有限 | 最大表层DLE/SD mm | 深峰≤10 mm | fallback | 全门槛 |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for item in candidate_summary:
    lines.append(f"| {item['candidate']} | {item['mean_local_AUC']:.3f}/{item['min_local_AUC']:.3f} | "
        f"{item['surface_DLE_finite_count']}/{item['surface_case_count']} | "
        f"{item['max_surface_DLE_mm']:.2f}/{item['max_surface_SD_mm']:.2f} | "
        f"{item['deep_within_10mm_count']}/{item['deep_case_count']} | {item['fallback_count']} | "
        f"{item['development_screen_passed']} |")
lines += ["", "本目录是开发筛选，不构成正式验收；若有胜者，必须换全新位置做一次盲确认。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
