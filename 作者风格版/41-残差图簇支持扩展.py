"""# %% 用两半观测复现的表层残差图簇做一次固定降罚重拟合。"""

# %% 路径、固定开发病例和不覆盖规则；不读取校准集或验证集。
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
from candidates.oaster_predictive import fit_predictive_models
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

base = root / "results/erp_whole_head/adaptive_v6"
manifest = base / "protocol/development_component_balanced_m10_m10.json"
all_cases = json.loads(manifest.read_text(encoding="utf-8"))
cases = [case for case in all_cases if case["case_id"].split("-")[3] in ("01", "03")]
assert [case["case_id"].split("-")[3] for case in cases] == ["01", "03"]
output = base / "development_diagnosis/residual_cluster_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧开发结果：{output}")
output.mkdir(parents=True)

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = int(shared["n_surf"])
surface_graph = sparse.csr_matrix(shared["adjacency"][:n_surf, :n_surf])
surface_graph = surface_graph.maximum(surface_graph.T)
surface_graph.setdiag(0)
surface_graph.eliminate_zeros()
seed_root = 2026092206

# %% 每例先重建同一观测和既有凸 H0，再按冻结公式寻找一个复现残差簇。
rows, details, saved_arrays = [], [], {}
for case in cases:
    number = case["case_id"].split("-")[3]
    source = base / f"dev_component_balanced_convex_admm_mrf080_m10_m10_case{number}"
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    with np.load(source / (case["case_id"] + ".npz")) as arrays:
        initial_null = arrays["null"].astype(float)
        assert np.allclose(arrays["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    source_detail = json.loads((source / (case["case_id"] + ".json")).read_text(encoding="utf-8"))
    null_window = source_detail["fitting"]["null_model"]["windows"][0]
    assert null_window["solver"]["converged"]
    surface_lambda = float(null_window["source_lambda_surface"])

    baseline = observation["baseline"]
    active = observation["active_windows"][0]
    channel_weights = observation["channel_weights"]
    weighted_gain = channel_weights[:, None] * observation["gain"][:, :n_surf]
    sensitivity = np.linalg.norm(weighted_gain, axis=0)
    gain_scale = float(np.median(sensitivity[sensitivity > 0]))
    depth_weights = np.maximum(sensitivity / gain_scale, .1) ** .8
    design = weighted_gain / gain_scale
    degree = np.asarray(surface_graph.sum(axis=1)).ravel()
    transition = sparse.diags(1 / np.maximum(degree, 1)) @ surface_graph
    mrf_factor = splu((sparse.eye(n_surf) - .8 * transition).tocsc())
    design = mrf_factor.solve(design.T, trans="T").T

    basis, basis_info = _smooth_temporal_basis(observation["training"], baseline, active)
    source_modes = initial_null[:n_surf] @ basis.T
    prediction = weighted_gain @ source_modes
    gradients = []
    for half in ("training", "confirmation"):
        centered = observation[half] - observation[half][:, baseline].mean(axis=1, keepdims=True)
        response = (channel_weights[:, None] * centered) @ basis.T
        gradients.append(design.T @ (response - prediction))
    train_gradient, confirmation_gradient = gradients
    train_norm = np.linalg.norm(train_gradient, axis=1)
    denominator = depth_weights * surface_lambda
    train_ratio = train_norm / denominator
    train_direction = np.divide(train_gradient, train_norm[:, None],
        out=np.zeros_like(train_gradient), where=train_norm[:, None] > 0)
    confirmation_ratio = np.sum(confirmation_gradient * train_direction, axis=1) / denominator

    energy = np.sum(source_modes ** 2, axis=1)
    current_support = energy >= .1 * energy.max(initial=0)
    candidate_indices = np.flatnonzero((train_ratio > 1) & ~current_support)
    if candidate_indices.size:
        count, labels = csgraph.connected_components(
            surface_graph[candidate_indices][:, candidate_indices], directed=False)
    else:
        count, labels = 0, np.empty(0, dtype=int)
    clusters = []
    for label in range(count):
        indices = candidate_indices[labels == label]
        train_excess = float(np.maximum(train_ratio[indices] - 1, 0).sum())
        confirmation_excess = float(np.maximum(confirmation_ratio[indices] - 1, 0).sum())
        clusters.append({"indices": indices.tolist(), "size": int(indices.size),
            "train_ratio_mean": float(train_ratio[indices].mean()),
            "confirmation_ratio_mean": float(confirmation_ratio[indices].mean()),
            "train_excess_sum": train_excess, "confirmation_excess_sum": confirmation_excess,
            "replicated_score": min(train_excess, confirmation_excess),
            "eligible": bool(indices.size >= 2 and confirmation_ratio[indices].mean() > 1)})
    eligible = [cluster for cluster in clusters if cluster["eligible"]]
    selected = (max(eligible, key=lambda cluster: (cluster["replicated_score"],
                -min(cluster["indices"]))) if eligible else None)
    multipliers = np.ones(n_surf)
    if selected is not None:
        multipliers[selected["indices"]] = .5

    # %% 同一表层 multiplier 同时用于 H0/H1；深层 penalty 从不改变。
    solver_settings = {"solver_kind": "admm", "mrf_strength": .8,
        "outer_iterations": 1, "surface_penalty_multiplier": multipliers.tolist()}
    null, full, fitting = fit_predictive_models(
        observation["training"], observation["gain"], n_surf,
        adjacency=shared["adjacency"], baseline=baseline, active=active,
        channel_weights=channel_weights, solver_settings=solver_settings)
    convergence = {name: bool(fitting[name + "_model"]["windows"][0]["solver"]["converged"])
                   for name in ("null", "full")}
    for method, estimate in (("residual-cluster-surface-only", null),
                             ("residual-cluster-full-ungated", full)):
        values = metrics.evaluate_estimate(
            estimate, observation["truth"], shared["vertices"], observation["groups"],
            n_surf, observation["active"], shared["auc_cortex"], baseline=baseline)
        rows.append({"case_id": case["case_id"], "scenario": case["scenario"],
            "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
            "method": method, **values})
    detail = {"case_id": case["case_id"], "complete": True,
        "initial_source": str(source), "initial_family_for_cluster": "surface-only H0",
        "basis": basis_info, "surface_lambda": surface_lambda,
        "candidate_count": int(candidate_indices.size), "cluster_count": len(clusters),
        "eligible_cluster_count": len(eligible), "selected_cluster": selected,
        "selected_multiplier": .5 if selected is not None else 1.,
        "deep_penalty_changed": False, "truth_used_for_selection": False,
        "confirmation_used_for_cluster_selection": True,
        "confirmation_available_for_old_deep_score": False,
        "old_calibration_or_validation_reusable": False,
        "convergence": convergence, "fitting": fitting}
    details.append(detail)
    saved_arrays[case["case_id"]] = (observation, null, full)
    print(f"{case['case_id']}: candidates={len(candidate_indices)}, eligible={len(eligible)}, "
          f"selected={None if selected is None else selected['indices']}, converged={convergence}", flush=True)

# %% 全部完成后再统一保存，避免把半成品误当完整结果。
archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
for detail in details:
    case_id = detail["case_id"]
    observation, null, full = saved_arrays[case_id]
    (output / (case_id + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / (case_id + ".npz"), truth=observation["truth"].astype(np.float32),
        null=null.astype(np.float32), full=full.astype(np.float32), vertices=shared["vertices"],
        times=shared["times"], active=observation["active"], baseline=observation["baseline"])

selected_rows = []
for number, method in (("01", "residual-cluster-surface-only"),
                       ("03", "residual-cluster-full-ungated")):
    selected_rows.append(next(row for row in rows
        if row["case_id"].split("-")[3] == number and row["method"] == method))
summary = {"complete": True, "phase": "development", "cases": [1, 3],
    "snr_eeg_meg_db": [-10, -10], "calibration_or_validation_read": False,
    "confirmation_role": "cluster selection; old held-out deep score and calibration invalid",
    "local_An_cal_AUC_gate": .9,
    "passes_local_gate": all(row["auc"] >= .9 for row in selected_rows),
    "all_converged": all(all(detail["convergence"].values()) for detail in details),
    "case01_full_ungated_deep_false_positive": next(row["deep_false_positive"] for row in rows
        if row["case_id"].split("-")[3] == "01" and row["method"].endswith("full-ungated")),
    "case03_deep_detected": selected_rows[1]["deep_detected"],
    "case03_deep_DLE_mm": selected_rows[1]["deep_dle_mm"],
    "accepted": False,
    "reason": "Development candidate only; acceptance requires both local AUCs >=0.90 and a newly frozen calibration."}
(output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = ["# 两半复现残差图簇开发结果", "",
    "confirmation 已参与候选簇选择，因此旧的独立深源分数、校准和验证均不可复用。", "",
    "| case | 情况 | An_cal_AUC | An_auc | 表层 SD mm | 表层 DLE mm | 深层 DLE mm | 深层检出 |",
    "|---:|---|---:|---:|---:|---:|---:|---:|" ]
for number, row in zip(("01", "03"), selected_rows):
    deep = "—" if not np.isfinite(row["deep_dle_mm"]) else f"{row['deep_dle_mm']:.2f}"
    lines.append(f"| {number} | {row['scenario']} | {row['auc']:.3f} | {row['auc_tie_corrected']:.3f} | "
        f"{row['surface_sd_mm']:.2f} | {row['surface_dle_mm']:.2f} | {deep} | {row['deep_detected']} |")
lines += ["", "只有两例局部 AUC 都达到 0.90 才保留该候选；否则不再调簇阈值。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
