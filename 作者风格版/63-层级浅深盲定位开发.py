"""# %% 开发集层级定位：先判深源，再定位深源，最后条件拟合表层。"""

# %% 1. 输入。这里只接受完整保存 H0/H1 的 development 结果，不冒充正式验证。
import argparse
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
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import methods, metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import (
    _smooth_temporal_basis,
    reconstruct_evoked_oaster_v5_from_whitened,
)
from candidates.oaster_predictive import score_predictive_models
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--development-gate-threshold", type=float, required=True)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
gate_threshold = float(args.development_gate_threshold)
gate_modality_weights = np.array([.75, .25])
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")
if not np.isfinite(gate_threshold):
    raise ValueError("development gate threshold must be finite")

metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
expected_settings = {
    "solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 40, "max_inner_retries": 20,
    "epsilon_fraction": .05, "tolerance": .001,
    "outer_tolerance": .01, "surface_reweight_floor": .5,
    "deep_reweight_floor": .5, "edge_weight_floor": .5,
}
if metadata["phase"] != "development" or metadata["covariance"] != "trial" or \
        metadata.get("score_kind") != "conjunctive" or metadata["solver_settings"] != expected_settings:
    raise ValueError("只接受 trial covariance + conjunctive + outer40 bounded-MM development 输出")
if not completion["complete"] or not completion["all_converged"]:
    raise ValueError("上游 H0/H1 尚未全部收敛")
for relative, expected in metadata["code_sha256"].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"上游代码已变化：{path}")

manifest = Path(metadata["manifest"])
if hashlib.sha256(manifest.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
    raise ValueError("上游 manifest 已变化")
cases = json.loads(manifest.read_text(encoding="utf-8"))
if len(cases) != 15 or any(case.get("component_balance_revision") !=
        "development_localization_v5_new_positions_paired_snr" for case in cases):
    raise ValueError("这里只接受冻结的15例新位置开发面板")
if [case["case_id"] for case in cases] != metadata["case_ids"]:
    raise ValueError("manifest 病例顺序与上游结果不一致")

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
if metadata["shared_fingerprint"] != original._shared_fingerprint(shared):
    raise ValueError("共享 forward 已变化")
n_surf = int(shared["n_surf"])
vertices = np.asarray(shared["vertices"], float)
adjacency = shared["adjacency"]
surface_graph = sparse.csr_matrix(adjacency[:n_surf, :n_surf])
surface_graph = surface_graph.maximum(surface_graph.T)
surface_graph.setdiag(0)
surface_graph.eliminate_zeros()
surface_degree = np.asarray(surface_graph.sum(axis=1)).ravel()
surface_transition = sparse.diags(1 / np.maximum(surface_degree, 1)) @ surface_graph
surface_mrf = sparse.eye(n_surf) - .8 * surface_transition
surface_mrf_factor = splu(surface_mrf.tocsc())
output.mkdir(parents=True)
result_metadata = {
    "phase": "development", "formal_acceptance_allowed": False,
    "source": str(source),
    "source_metadata_sha256": hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest(),
    "manifest": str(manifest), "manifest_sha256": metadata["manifest_sha256"],
    "seed_root": metadata["seed_root"], "development_gate_threshold": gate_threshold,
    "gate_score_kind": "fixed_3_to_1_linear_fusion",
    "gate_modality_weights_EEG_MEG": gate_modality_weights.tolist(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "candidate selection after inspecting development failures; fresh calibration and validation required",
}
(output / "metadata.json").write_text(
    json.dumps(result_metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
rows = []
started = time.perf_counter()


# %% 2. 每例只用盲分数选择路径；真值在整张结果图固定后才用于指标。
for case in cases:
    tick = time.perf_counter()
    name = case["case_id"]
    observation = prepare_trial_covariance_case(shared, case, seed_root=metadata["seed_root"])
    baseline = np.asarray(observation["baseline"], bool)
    active_window = np.asarray(observation["active_windows"][0], bool)
    active_samples = np.asarray(observation["active"], int)
    gain = np.asarray(observation["gain"], float)
    weights = np.asarray(observation["channel_weights"], float)
    modality_sizes = tuple(int(value) for value in observation["metadata"]["retained_channels"])

    with np.load(source / (name + ".npz")) as package:
        null = np.asarray(package["null"], float)
        full = np.asarray(package["full"], float)

    saved = json.loads((source / (name + ".json")).read_text(encoding="utf-8"))
    convergence = {family: bool(saved["fitting"][family + "_model"]["windows"][0]["solver"]["converged"])
                   for family in ("null", "full")}
    if not all(convergence.values()):
        raise RuntimeError(f"{name}: H0/H1 未全部收敛")

    _, gate_info = score_predictive_models(
        observation["confirmation"], gain, null, full,
        baseline=baseline, active=active_window, channel_weights=weights,
        modality_sizes=modality_sizes)
    gate_modality_scores = np.asarray(gate_info["modality_noise_scores"], float)
    if gate_modality_scores.shape != (2,) or not np.all(np.isfinite(gate_modality_scores)):
        raise ValueError(f"{name}: 一级分数必须依次包含 EEG、MEG")
    np.testing.assert_allclose(
        gate_modality_scores, saved["evidence"]["modality_noise_scores"],
        rtol=5e-6, atol=3e-8, err_msg=f"{name}: 一级模态分数无法复现")
    gate_score = float(gate_modality_weights @ gate_modality_scores)
    deep_present = bool(gate_score > gate_threshold)

    selected_deep = None
    amplitude_deep = None
    loo_sum = None
    loo_by_modality = None
    loo_candidate_indices = None
    secondary_score = None
    secondary_modalities = None
    surface_solver_ok = True
    surface_solver = None
    final_surface_solver = None
    h0_innovation_indices = None

    if not deep_present:
        # H0 在稀疏创新坐标做 BIC 定位，再用 OASTER/SISSES 已有 R^-1 图扩散还原物理皮层。
        combined = (observation["training"] + observation["confirmation"]) / 2
        innovation_gain = surface_mrf_factor.solve(
            gain[:, :n_surf].T, trans="T").T
        innovation = methods.dipole_fit(combined, innovation_gain, active_samples)
        h0_innovation_indices = np.flatnonzero(
            np.linalg.norm(innovation[:, active_samples], axis=1) > 0)
        surface = surface_mrf_factor.solve(innovation)
        np.testing.assert_allclose(
            gain[:, :n_surf] @ surface, innovation_gain @ innovation,
            rtol=2e-10, atol=2e-10)
        selected = np.zeros_like(full)
        selected[:n_surf] = surface
        selected_family = "H0 combined40 MRF-innovation dipole"
    else:
        # 已确认 H1 后，位置用两模态 LOO 改善之和；presence 才使用严格的模态最小值。
        basis, _ = _smooth_temporal_basis(observation["training"], baseline, active_window)
        full_modes = full @ basis.T
        loo_candidate_indices = n_surf + np.flatnonzero(
            np.linalg.norm(full_modes[n_surf:], axis=1) > 0)
        if not loo_candidate_indices.size:
            raise RuntimeError(f"{name}: H1 没有活动深源候选")
        loo_by_modality = np.zeros((len(modality_sizes), len(loo_candidate_indices)))
        for local, deep_index in enumerate(loo_candidate_indices):
            without = full.copy()
            without[deep_index] = 0
            _, contribution = score_predictive_models(
                observation["confirmation"], gain, without, full,
                baseline=baseline, active=active_window, channel_weights=weights,
                modality_sizes=modality_sizes)
            loo_by_modality[:, local] = contribution["modality_noise_scores"]
        loo_sum = loo_by_modality.sum(axis=0)
        selected_deep = int(loo_candidate_indices[np.argmax(loo_sum)])
        full_amplitude = metrics.source_amplitude(full, active_samples, baseline)
        amplitude_deep = n_surf + int(np.argmax(full_amplitude[n_surf:]))

        # D：只在训练集、原训练权重坐标中解析拟合所选深源。
        centered = (observation["training"]
                    - observation["training"][:, baseline].mean(axis=1, keepdims=True))
        weighted_response = (weights[:, None] * centered) @ basis.T
        weighted_deep_gain = weights * gain[:, selected_deep]
        denominator = float(weighted_deep_gain @ weighted_deep_gain)
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError(f"{name}: 所选深源在训练坐标中没有灵敏度")
        deep_modes = (weighted_deep_gain @ weighted_response) / denominator
        deep = np.zeros_like(full)
        deep[selected_deep] = deep_modes @ basis
        np.testing.assert_allclose(deep[selected_deep] @ basis.T, deep_modes,
                                   rtol=1e-10, atol=1e-10)

        # S：固定 D 后，在未乘训练模态权重的白化坐标拟合表层。
        residual = observation["training"] - gain[:, [selected_deep]] @ deep[[selected_deep]]
        surface, surface_info = reconstruct_evoked_oaster_v5_from_whitened(
            residual, gain[:, :n_surf], n_surf, kernels=(),
            adjacency=adjacency[:n_surf, :n_surf], baseline=baseline,
            active_windows=(active_window,),
            window_channel_weights=(np.ones(gain.shape[0]),), require_one=False,
            edge_fraction=.5, noise_multiplier=1., mrf_strength=.8,
            calibration="layer", temporal_mode="smooth", solver_kind="admm",
            surface_reweight_floor=.5, deep_reweight_floor=.5,
            ridge_fraction=0., edge_penalty_mode="group",
            surface_penalty_multiplier=None, source_penalty_mode="group",
            deep_alias_penalty=False, outer_iterations=40, max_iter=2000,
            tolerance=.001, epsilon_fraction=.05, rho=1., outer_tolerance=.01,
            adaptive_rho=True, max_inner_retries=20, edge_weight_floor=.5)
        surface_solver = surface_info["windows"][0]["solver"]
        surface_solver_ok = bool(
            surface_solver["converged"] and surface_solver["outer_converged"]
            and surface_solver["final_inner_converged"]
            and surface_solver["final_stationarity_gap_relative"] <= .001 + 1e-12)
        if not surface_solver_ok:
            raise RuntimeError(f"{name}: 条件表层拟合未到固定点")
        surface_full = np.zeros_like(deep)
        surface_full[:n_surf] = surface
        mixed = deep + surface_full

        _, secondary = score_predictive_models(
            observation["confirmation"], gain, deep, mixed,
            baseline=baseline, active=active_window,
            channel_weights=np.ones(gain.shape[0]), modality_sizes=modality_sizes)
        secondary_score = float(secondary["conjunctive_modality_score"])
        secondary_modalities = [float(value) for value in secondary["modality_noise_scores"]]
        surface_present = secondary_score > 0

        # 训练/确认只负责冻结模型；最终幅度用全部40次平均重新拟合，不丢掉确认数据。
        combined = (observation["training"] + observation["confirmation"]) / 2
        final_basis, _ = _smooth_temporal_basis(combined, baseline, active_window)
        final_centered = combined - combined[:, baseline].mean(axis=1, keepdims=True)
        final_response = (weights[:, None] * final_centered) @ final_basis.T
        final_deep_modes = (weighted_deep_gain @ final_response) / denominator
        final_deep = np.zeros_like(full)
        final_deep[selected_deep] = final_deep_modes @ final_basis
        selected = final_deep
        if surface_present:
            final_residual = combined - gain[:, [selected_deep]] @ final_deep[[selected_deep]]
            final_surface, final_surface_info = reconstruct_evoked_oaster_v5_from_whitened(
                final_residual, gain[:, :n_surf], n_surf, kernels=(),
                adjacency=adjacency[:n_surf, :n_surf], baseline=baseline,
                active_windows=(active_window,),
                window_channel_weights=(np.ones(gain.shape[0]),), require_one=False,
                edge_fraction=.5, noise_multiplier=1., mrf_strength=.8,
                calibration="layer", temporal_mode="smooth", solver_kind="admm",
                surface_reweight_floor=.5, deep_reweight_floor=.5,
                ridge_fraction=0., edge_penalty_mode="group",
                surface_penalty_multiplier=None, source_penalty_mode="group",
                deep_alias_penalty=False, outer_iterations=40, max_iter=2000,
                tolerance=.001, epsilon_fraction=.05, rho=1., outer_tolerance=.01,
                adaptive_rho=True, max_inner_retries=20, edge_weight_floor=.5)
            final_surface_solver = final_surface_info["windows"][0]["solver"]
            final_surface_solver_ok = bool(
                final_surface_solver["converged"] and final_surface_solver["outer_converged"]
                and final_surface_solver["final_inner_converged"]
                and final_surface_solver["final_stationarity_gap_relative"] <= .001 + 1e-12)
            surface_solver_ok = surface_solver_ok and final_surface_solver_ok
            if not final_surface_solver_ok:
                raise RuntimeError(f"{name}: combined40 表层重拟合未到固定点")
            selected[:n_surf] = final_surface
        selected_family = ("H1 deep+surface combined40 refit" if surface_present
                           else "H1 deep-only combined40 refit")

    # 到这里盲图已经固定；下面真值只作一致性检查和开发集事后评价。
    with np.load(source / (name + ".npz")) as package:
        saved_truth = np.asarray(package["truth"], float)
    if not np.allclose(saved_truth, observation["truth"]):
        raise ValueError(f"{name}: 上游保存真值与重建观测不一致")
    values = metrics.evaluate_estimate(
        selected, observation["truth"], vertices, observation["groups"], n_surf,
        active_samples, shared["auc_cortex"], baseline=baseline)
    surface_groups = [np.asarray(group, int) for group in observation["groups"]
                      if np.asarray(group).size and np.all(np.asarray(group) < n_surf)]
    surface_component_dle = []
    surface_patch_hits = []
    if surface_groups:
        surface_energy = np.sum(selected[:n_surf, active_samples] ** 2, axis=1)
        truth_indices = np.concatenate(surface_groups)
        truth_owner = np.concatenate([
            np.full(len(group), index, int) for index, group in enumerate(surface_groups)])
        owner = truth_owner[cKDTree(vertices[truth_indices]).query(vertices[:n_surf])[1]]
        support = surface_energy > .1 * surface_energy.max(initial=0.)
        for index, group in enumerate(surface_groups):
            region = np.flatnonzero(owner == index)
            supported_region = region[support[region]]
            if not supported_region.size:
                surface_component_dle.append(None)
            else:
                peak = int(supported_region[np.argmax(surface_energy[supported_region])])
                center = vertices[group].mean(axis=0)
                surface_component_dle.append(float(np.linalg.norm(vertices[peak] - center) * 1000))
            surface_patch_hits.append(int(np.count_nonzero(support[group])))
    finite_component_dle = [value for value in surface_component_dle if value is not None]
    maximum_component_dle = (np.nan if len(finite_component_dle) != len(surface_groups)
                             else max(finite_component_dle, default=np.nan))
    row = {
        "case_id": name, "case_number": case["case_number"],
        "scenario": case["scenario"], "eeg_snr_db": case["eeg_snr_db"],
        "meg_snr_db": case["meg_snr_db"], "gate_score": gate_score,
        "gate_eeg_score": float(gate_modality_scores[0]),
        "gate_meg_score": float(gate_modality_scores[1]),
        "development_gate_threshold": gate_threshold,
        "deep_present_decision": int(deep_present), "selected_family": selected_family,
        "selected_deep_index": "" if selected_deep is None else selected_deep,
        "amplitude_deep_index": "" if amplitude_deep is None else amplitude_deep,
        "secondary_surface_score": np.nan if secondary_score is None else secondary_score,
        "h0_innovation_count": ("" if h0_innovation_indices is None
                                else len(h0_innovation_indices)),
        "surface_component_dle_max_mm": maximum_component_dle,
        "surface_patch_hit_count": int(sum(value > 0 for value in surface_patch_hits)),
        "surface_patch_count": len(surface_patch_hits),
        "surface_solver_ok": int(surface_solver_ok), **values,
    }
    rows.append(row)
    detail = {
        "case_id": name, "complete": True, "development_only": True,
        "truth_used_by_decisions": False, "truth_used_only_for_posthoc_metrics": True,
        "gate_score": gate_score, "gate_score_kind": "fixed_3_to_1_linear_fusion",
        "gate_modality_weights_EEG_MEG": gate_modality_weights.tolist(),
        "gate_modality_scores": gate_modality_scores.tolist(),
        "development_gate_threshold": gate_threshold,
        "deep_present_decision": deep_present, "selected_family": selected_family,
        "selected_deep_index": selected_deep, "amplitude_deep_index": amplitude_deep,
        "deep_loo_candidate_indices": (None if loo_candidate_indices is None
                                       else loo_candidate_indices.tolist()),
        "deep_loo_modality_scores": None if loo_by_modality is None else loo_by_modality.tolist(),
        "deep_loo_sum_scores": None if loo_sum is None else loo_sum.tolist(),
        "secondary_surface_score": secondary_score,
        "secondary_modality_scores": secondary_modalities,
        "h0_innovation_indices": (None if h0_innovation_indices is None
                                  else h0_innovation_indices.tolist()),
        "surface_component_dle_mm": surface_component_dle,
        "surface_patch_support_hits": surface_patch_hits,
        "surface_selection_solver": surface_solver,
        "surface_final_solver": final_surface_solver,
        "elapsed_seconds": time.perf_counter() - tick,
    }
    np.savez_compressed(output / (name + ".npz"),
        truth=observation["truth"].astype(np.float32),
        selected=selected.astype(np.float32), vertices=vertices.astype(np.float32),
        times=np.asarray(shared["times"]), active=active_samples, baseline=baseline)
    (output / (name + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
    print(name, f"gate={gate_score:.6g}", selected_family,
          f"AUC={values['auc_tie_corrected']:.3f}", flush=True)


# %% 3. 开发集汇总。阈值和二级零界都必须在全新校准/验证中重跑完整流水线。
h0_rows = [row for row in rows if not row["deep_present_decision"]]
h1_rows = [row for row in rows if row["deep_present_decision"]]
surface_rows = [row for row in rows if row["scenario"] != "deep_only"]
deep_rows = [row for row in rows if row["scenario"] != "surface_only"]
global_auc = np.array([row["auc_tie_corrected"] for row in rows], float)
surface_auc = np.array([row["surface_auc_tie_corrected"] for row in surface_rows], float)
surface_dle = np.array([row["surface_dle_mm"] for row in surface_rows], float)
surface_component_dle = np.array(
    [row["surface_component_dle_max_mm"] for row in surface_rows], float)
surface_sd = np.array([row["surface_sd_mm"] for row in surface_rows], float)
deep_distance = np.array([row["deep_peak_distance_mm"] for row in deep_rows], float)
requested_auc = np.array([row["auc"] for row in rows], float)
deep_only_surface_false_case_ids = [row["case_id"] for row in rows
                                    if row["scenario"] == "deep_only"
                                    and row["surface_active_count"] > 0]
family_error_case_ids = [row["case_id"] for row in rows
                         if row["deep_present_decision"] != int(row["scenario"] != "surface_only")]
nonfinite_global_auc_case_ids = [row["case_id"] for row in rows
                                 if not np.isfinite(row["auc_tie_corrected"])]
nonfinite_surface_auc_case_ids = [row["case_id"] for row in surface_rows
                                  if not np.isfinite(row["surface_auc_tie_corrected"])]
nonfinite_surface_dle_case_ids = [row["case_id"] for row in surface_rows
                                  if not np.isfinite(row["surface_dle_mm"])]
nonfinite_surface_sd_case_ids = [row["case_id"] for row in surface_rows
                                 if not np.isfinite(row["surface_sd_mm"])]
nonfinite_deep_distance_case_ids = [row["case_id"] for row in deep_rows
                                    if not np.isfinite(row["deep_peak_distance_mm"])]
nonfinite_requested_auc_case_ids = [row["case_id"] for row in rows
                                    if not np.isfinite(row["auc"])]
surface_patch_miss_case_ids = [row["case_id"] for row in surface_rows
                               if row["surface_patch_hit_count"] != row["surface_patch_count"]]
finite_global_auc = global_auc[np.isfinite(global_auc)]
finite_surface_auc = surface_auc[np.isfinite(surface_auc)]
finite_surface_dle = surface_dle[np.isfinite(surface_dle)]
finite_surface_component_dle = surface_component_dle[np.isfinite(surface_component_dle)]
finite_surface_sd = surface_sd[np.isfinite(surface_sd)]
finite_deep_distance = deep_distance[np.isfinite(deep_distance)]
finite_requested_auc = requested_auc[np.isfinite(requested_auc)]
summary = {
    "complete": len(rows) == len(cases), "development_only": True,
    "formal_acceptance_allowed": False,
    "case_count": len(rows), "gate_h0_decision_count": len(h0_rows),
    "gate_h1_decision_count": len(h1_rows),
    "family_correct_count_posthoc": len(rows) - len(family_error_case_ids),
    "family_error_case_ids": family_error_case_ids,
    "nonfinite_global_AUC_case_ids": nonfinite_global_auc_case_ids,
    "minimum_global_AUC": None if not finite_global_auc.size else float(finite_global_auc.min()),
    "nonfinite_surface_AUC_case_ids": nonfinite_surface_auc_case_ids,
    "minimum_surface_AUC": None if not finite_surface_auc.size else float(finite_surface_auc.min()),
    "surface_DLE_finite_count": int(np.isfinite(surface_dle).sum()),
    "surface_case_count": len(surface_rows),
    "nonfinite_surface_DLE_case_ids": nonfinite_surface_dle_case_ids,
    "maximum_surface_DLE_mm": None if not finite_surface_dle.size else float(finite_surface_dle.max()),
    "nonfinite_surface_component_DLE_case_ids": [
        row["case_id"] for row in surface_rows
        if not np.isfinite(row["surface_component_dle_max_mm"])],
    "maximum_surface_component_DLE_mm": (
        None if not finite_surface_component_dle.size
        else float(finite_surface_component_dle.max())),
    "auc_screen_metric": "An_auc tie-corrected AUC",
    "nonfinite_requested_parcel_AUC_case_ids_supplemental_not_screened": (
        nonfinite_requested_auc_case_ids),
    "minimum_requested_parcel_AUC_supplemental_not_screened": (
        None if not finite_requested_auc.size else float(finite_requested_auc.min())),
    "exact_patch_miss_case_ids_supplemental_not_screened": surface_patch_miss_case_ids,
    "nonfinite_surface_SD_case_ids": nonfinite_surface_sd_case_ids,
    "maximum_surface_SD_mm": None if not finite_surface_sd.size else float(finite_surface_sd.max()),
    "deep_distance_finite_count": int(np.isfinite(deep_distance).sum()),
    "deep_case_count": len(deep_rows),
    "nonfinite_deep_distance_case_ids": nonfinite_deep_distance_case_ids,
    "maximum_deep_peak_distance_mm": (None if not finite_deep_distance.size
                                       else float(finite_deep_distance.max())),
    "deep_false_positive_count": int(sum(row["deep_false_positive"] for row in rows)),
    "deep_only_surface_false_case_ids": deep_only_surface_false_case_ids,
    "deep_only_surface_false_count": len(deep_only_surface_false_case_ids),
    "surface_solver_failure_count": int(sum(not row["surface_solver_ok"] for row in h1_rows)),
    "wall_seconds": time.perf_counter() - started,
    "deep_presence_rule": "development threshold on fixed (3 * EEG + MEG) / 4 held-out improvement",
    "deep_location_rule": "argmax sum of modality-normalized leave-one-deep-out improvements",
    "surface_rule": "H0: combined40 BIC dipoles in sparse MRF innovation coordinates, then R^-1 physical map; H1: train20 fixed D then W=1 bounded-MM surface for held-out presence, freeze model and refit amplitudes on combined40",
    "selection_bias_note": "development only; formal selection requires independent pure-surface conformal calibration plus untouched H0/H1 validation",
}
summary["development_screen_passed"] = bool(
    not family_error_case_ids and not nonfinite_global_auc_case_ids
    and not nonfinite_surface_auc_case_ids and not nonfinite_surface_dle_case_ids
    and not nonfinite_surface_sd_case_ids
    and not nonfinite_deep_distance_case_ids
    and summary["minimum_global_AUC"] is not None and summary["minimum_global_AUC"] >= .9
    and summary["minimum_surface_AUC"] is not None and summary["minimum_surface_AUC"] >= .9
    and summary["maximum_surface_DLE_mm"] is not None
    and summary["maximum_surface_DLE_mm"] <= 15 + 1e-6
    and not summary["nonfinite_surface_component_DLE_case_ids"]
    and summary["maximum_surface_component_DLE_mm"] is not None
    and summary["maximum_surface_component_DLE_mm"] <= 15 + 1e-6
    and summary["maximum_surface_SD_mm"] is not None
    and summary["maximum_surface_SD_mm"] <= 20 + 1e-6
    and summary["maximum_deep_peak_distance_mm"] is not None
    and summary["maximum_deep_peak_distance_mm"] <= 10 + 1e-6
    and summary["deep_false_positive_count"] == 0
    and summary["deep_only_surface_false_count"] == 0
    and summary["surface_solver_failure_count"] == 0)
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(output / "completion.json").write_text(
    json.dumps({"complete": summary["complete"], "case_count": len(rows),
                "development_screen_passed": summary["development_screen_passed"],
                "wall_seconds": summary["wall_seconds"]}, indent=2,
               allow_nan=False) + "\n", encoding="utf-8")


# %% 4. 一张图同时检查门控、AUC、表层DLE和深源距离。
figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
x = np.arange(len(rows))
colors = {"surface_only": "#0072B2", "deep_only": "#D55E00",
          "deep_plus_surface": "#009E73", "deep_plus_two_surface": "#6A51A3"}
case_colors = [colors[row["scenario"]] for row in rows]
axes[0, 0].scatter(x, [row["gate_score"] for row in rows], c=case_colors, s=48)
axes[0, 0].axhline(gate_threshold, color="#333333", linestyle="--", linewidth=1.2)
axes[0, 0].set(title="Deep-presence gate", ylabel="fixed (3×EEG + MEG) / 4 gain")
axes[0, 1].scatter(x, [row["auc_tie_corrected"] for row in rows], c=case_colors, s=48)
axes[0, 1].axhline(.9, color="#333333", linestyle="--", linewidth=1.2)
axes[0, 1].set(title="Whole-grid localization", ylabel="Tie-corrected AUC", ylim=(0, 1.03))
surface_x = [index for index, row in enumerate(rows) if row["scenario"] != "deep_only"]
axes[1, 0].scatter(surface_x, [rows[index]["surface_dle_mm"] for index in surface_x],
                   c=[case_colors[index] for index in surface_x], s=48)
axes[1, 0].axhline(15, color="#333333", linestyle="--", linewidth=1.2)
axes[1, 0].set(title="Surface localization error", ylabel="DLE (mm)")
deep_x = [index for index, row in enumerate(rows) if row["scenario"] != "surface_only"]
axes[1, 1].scatter(deep_x, [rows[index]["deep_peak_distance_mm"] for index in deep_x],
                   c=[case_colors[index] for index in deep_x], s=48)
axes[1, 1].axhline(10, color="#333333", linestyle="--", linewidth=1.2)
axes[1, 1].set(title="Deep peak localization error", ylabel="Distance (mm)")
for axis in axes.ravel():
    axis.set_xlabel("Development case number")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{row['case_number']:02d}" for row in rows], fontsize=7)
    axis.grid(axis="y", color="#DDDDDD", linewidth=.7)
figure.suptitle("Hierarchical OASTER localization — development only", fontweight="bold")
figure.savefig(output / "hierarchical_localization.png", dpi=220, facecolor="white")
plt.close(figure)

lines = ["# 层级浅深盲定位开发结果", "",
    "一级用开发集冻结的 (3×EEG+MEG)/4 held-out 改善判深源；H1 内用双模态 LOO 改善之和定位深源，固定深源后再拟合表层。", "",
    "| 病例 | 场景 | EEG/MEG SNR | gate | 盲选 | An_auc | requested AUC | 表层DLE/逐源最大 mm | 精确patch命中 | 深峰距离 mm |",
    "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|"]
for row in rows:
    surface_dle = "—" if not np.isfinite(row["surface_dle_mm"]) else f"{row['surface_dle_mm']:.2f}"
    component_dle = ("—" if not np.isfinite(row["surface_component_dle_max_mm"])
                     else f"{row['surface_component_dle_max_mm']:.2f}")
    patch_hits = ("—" if not row["surface_patch_count"] else
                  f"{row['surface_patch_hit_count']}/{row['surface_patch_count']}")
    deep_distance = "—" if not np.isfinite(row["deep_peak_distance_mm"]) else f"{row['deep_peak_distance_mm']:.2f}"
    lines.append(f"| {row['case_number']:02d} | {row['scenario']} | {row['eeg_snr_db']:+d}/{row['meg_snr_db']:+d} | "
                 f"{row['gate_score']:.6f} | {row['selected_family']} | {row['auc_tie_corrected']:.3f} | "
                 f"{row['auc']:.3f} | {surface_dle}/{component_dle} | {patch_hits} | {deep_distance} |")
lines += ["", f"Development screen passed：`{summary['development_screen_passed']}`。",
          f"一级路径错误病例：`{family_error_case_ids or '无'}`。",
          f"缺失深源距离病例：`{nonfinite_deep_distance_case_ids or '无'}`。", "",
          f"纯深源表层误报病例：`{deep_only_surface_false_case_ids or '无'}`。",
          f"补充警告——精确真值 patch 未命中病例：`{surface_patch_miss_case_ids or '无'}`。",
          f"补充 requested AUC 最低值（不替代 An_auc 验收）：`{summary['minimum_requested_parcel_AUC_supplemental_not_screened']}`。", "",
          "这是看过失败模式后的 development 候选，不是正式校准或 validation。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
