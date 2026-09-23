"""只读已保存开发解：分解浅深预测误差，不重新反演、不改变检出阈值。"""

# %% 复用既有双观测生成器；只读取两个混合开发配置的已保存源矩阵。
import hashlib
import json
import os
from pathlib import Path
import sys
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from scipy import sparse
from benchmark import protocol, metrics
from benchmark.erp_replicates import prepare_replicated_case
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
import run_erp_whole_head_matrix as original

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
cases = [case for case in json.loads(manifest.read_text(encoding="utf-8"))
         if case["eeg_snr_db"] == case["meg_snr_db"] == 5 and case["configuration_number"] in (3, 4)]
n_surf = shared["n_surf"]
upper = sparse.triu(sparse.csr_matrix(shared["adjacency"])[:n_surf, :n_surf], k=1).tocoo()
neighbors = sparse.csr_matrix((np.ones(2 * upper.nnz),
    (np.r_[upper.row, upper.col], np.r_[upper.col, upper.row])), shape=(len(shared["vertices"]),) * 2)
transition = sparse.diags(1 / np.maximum(np.asarray(neighbors.sum(axis=1)).ravel(), 1)) @ neighbors
mrf = sparse.eye(neighbors.shape[0]) - .5 * transition

# %% 真值仅用于事后归因，不送入 inverse；此脚本不调用任何求解器。
for directory, prepare in (("dev_trial_covariance_irls_5_5", prepare_trial_covariance_case),
                           ("dev_predictive_admm_5_5", prepare_replicated_case)):
    source_dir = root / "results/erp_whole_head/adaptive_v6" / directory
    for case in cases:
        saved = source_dir / (case["case_id"] + ".npz")
        if not saved.exists():
            print(json.dumps({"directory": directory, "case_id": case["case_id"], "status": "not_saved"}))
            continue
        observation = prepare(shared, case, seed_root=2026092206)
        with np.load(saved) as arrays:
            null, full = arrays["null"].astype(float), arrays["full"].astype(float)
            assert np.allclose(arrays["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
        recorded = json.loads(saved.with_suffix(".json").read_text(encoding="utf-8"))
        baseline, active = observation["baseline"], observation["active_windows"][0]
        basis, _ = _smooth_temporal_basis(observation["training"], baseline, active)
        weights = observation["channel_weights"]
        gain = weights[:, None] * observation["gain"]
        true = observation["truth"] @ basis.T
        x0, x1 = null @ basis.T, full @ basis.T
        f0, f1 = gain @ x0, gain @ x1
        true_surface, true_deep = gain[:, :n_surf] @ true[:n_surf], gain[:, n_surf:] @ true[n_surf:]
        predicted_surface, predicted_deep = gain[:, :n_surf] @ x1[:n_surf], gain[:, n_surf:] @ x1[n_surf:]
        deep_energy = float(np.sum(true_deep ** 2))
        scale = recorded["evidence"]["expected_response_noise_energy"]
        report = {"directory": directory, "case_id": case["case_id"], "status": "read_only_saved_solution",
                  "npz_sha256": hashlib.sha256(saved.read_bytes()).hexdigest()}
        for label, data in (("train", observation["training"]), ("confirmation", observation["confirmation"])):
            response = weights[:, None] * (data - data[:, baseline].mean(axis=1, keepdims=True)) @ basis.T
            loss0, loss1 = float(np.sum((response - f0) ** 2)), float(np.sum((response - f1) ** 2))
            report[label + "_loss_null"] = loss0
            report[label + "_loss_full"] = loss1
            report[label + "_normalized_improvement"] = (loss0 - loss1) / scale
        clean = true_surface + true_deep
        report.update(clean_loss_null=float(np.sum((clean - f0) ** 2)),
            clean_loss_full=float(np.sum((clean - f1) ** 2)), true_deep_sensor_energy=deep_energy,
            h0_surface_deep_projection=float(np.sum((f0 - true_surface) * true_deep) / deep_energy),
            h1_surface_deep_projection=float(np.sum((predicted_surface - true_surface) * true_deep) / deep_energy),
            deep_prediction_projection=float(np.sum(predicted_deep * true_deep) / deep_energy),
            deep_prediction_relative_error=float(np.linalg.norm(predicted_deep - true_deep) / np.sqrt(deep_energy)),
            deep_prediction_cosine=float(np.sum(predicted_deep * true_deep) / max(np.linalg.norm(predicted_deep) * np.sqrt(deep_energy), 1e-30)))
        amplitude = metrics.source_amplitude(full, observation["active"], baseline)
        peak = n_surf + int(np.argmax(amplitude[n_surf:]))
        target = int(case["deep_index"])
        report.update(true_deep_index=target, predicted_deep_index=peak,
            peak_distance_mm=float(np.linalg.norm(shared["vertices"][target] - shared["vertices"][peak]) * 1000),
            deep_leadfield_cosine=float(gain[:, target] @ gain[:, peak] / (np.linalg.norm(gain[:, target]) * np.linalg.norm(gain[:, peak]))),
            true_point_recovered_norm_ratio=float(np.linalg.norm(x1[target]) / np.linalg.norm(true[target])),
            peak_point_norm_ratio=float(np.linalg.norm(x1[peak]) / np.linalg.norm(true[target])))
        for label in ("null", "full"):
            detail = recorded["fitting"][label + "_model"]["windows"][0]
            report[label + "_converged"] = detail["solver"]["converged"]
            report[label + "_surface_lambda"] = detail["source_lambda_surface"]
            report[label + "_deep_lambda"] = detail["source_lambda_deep"]
        if directory == "dev_trial_covariance_irls_5_5":
            detail = recorded["fitting"]["full_model"]["windows"][0]
            solver = detail["solver"]
            sensitivity = np.linalg.norm(gain, axis=0)
            gain_scale = float(np.median(sensitivity[sensitivity > 0]))
            penalty = np.maximum(sensitivity / gain_scale, .1) ** .8
            penalty[:n_surf] *= detail["source_lambda_surface"]
            penalty[n_surf:] *= detail["source_lambda_deep"]
            for label, source in (("estimated", x1), ("true", true)):
                z = mrf @ source * gain_scale
                radius = np.sqrt(np.sum(z * z, axis=1) + solver["amplitude_delta"] ** 2)
                edge = z[upper.row] - z[upper.col]
                radius_edge = np.sqrt(np.sum(edge * edge, axis=1) + solver["edge_delta"] ** 2)
                source_cost = penalty * solver["amplitude_epsilon"] * (
                    np.log1p(radius / solver["amplitude_epsilon"]) - np.log1p(solver["amplitude_delta"] / solver["amplitude_epsilon"]))
                edge_cost = detail["edge_lambda"] * solver["edge_epsilon"] * np.sum(
                    np.log1p(radius_edge / solver["edge_epsilon"]) - np.log1p(solver["edge_delta"] / solver["edge_epsilon"]))
                train = weights[:, None] * (observation["training"] - observation["training"][:, baseline].mean(axis=1, keepdims=True)) @ basis.T
                report[label + "_data_cost"] = float(.5 * np.sum((train - gain @ source) ** 2))
                report[label + "_surface_penalty_cost"] = float(source_cost[:n_surf].sum())
                report[label + "_deep_penalty_cost"] = float(source_cost[n_surf:].sum())
                report[label + "_edge_penalty_cost"] = float(edge_cost)
                report[label + "_total_objective"] = report[label + "_data_cost"] + float(source_cost.sum()) + float(edge_cost)
                if label == "estimated":
                    row_norms = np.linalg.norm(z, axis=1)
                    report["max_row_norm_over_frozen_epsilon"] = float(row_norms.max() / solver["amplitude_epsilon"])
                    report["true_deep_row_norm_over_frozen_epsilon"] = float(row_norms[target] / solver["amplitude_epsilon"])
                    report["peak_deep_row_norm_over_frozen_epsilon"] = float(row_norms[peak] / solver["amplitude_epsilon"])
                    final_weights = penalty * solver["amplitude_epsilon"] / (radius * (solver["amplitude_epsilon"] + radius))
                    report["true_deep_final_quadratic_weight"] = float(final_weights[target])
                    report["peak_deep_final_quadratic_weight"] = float(final_weights[peak])
            report["recorded_final_objective"] = solver["final_log_objective"]
            assert np.isclose(report["estimated_total_objective"], solver["final_log_objective"], rtol=2e-5)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
