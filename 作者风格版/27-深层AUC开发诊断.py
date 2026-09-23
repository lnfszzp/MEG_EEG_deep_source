"""只读诊断五个已见开发病例；不读取校准集或验证集，不重新求逆。"""

# %% 固定输入：四组已经保存的 5/5 dB 开发结果。
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
from benchmark import metrics, protocol
from benchmark.erp_replicates import prepare_replicated_case
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
import run_erp_whole_head_matrix as original

result_root = root / "results/erp_whole_head/adaptive_v6"
output = result_root / "development_diagnosis/deep_auc_root_cause.json"
variants = {
    "mean_irls": ("dev_predictive_wide_5_5", prepare_replicated_case),
    "trial_irls": ("dev_trial_covariance_irls_5_5", prepare_trial_covariance_case),
    "mean_admm": ("dev_predictive_admm_5_5", prepare_replicated_case),
    "trial_admm": ("dev_trial_covariance_admm_5_5", prepare_trial_covariance_case),
}
manifest = json.loads((result_root / "protocol/development_manifest.json").read_text(encoding="utf-8"))
cases = [case for case in manifest if case["eeg_snr_db"] == case["meg_snr_db"] == 5]
assert len(cases) == 5 and all(case["panel"] == "erp_v6_development" for case in cases)
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = int(shared["n_surf"])
assert int(shared["n_deep"]) == 16

# %% 同一真值下读取已保存解，重建同一开发观测，只计算诊断量。
rows = []
deep_tables = []
for variant, (directory, prepare) in variants.items():
    for case in cases:
        saved = result_root / directory / (case["case_id"] + ".npz")
        details = result_root / directory / (case["case_id"] + ".json")
        assert saved.is_file() and details.is_file()
        arrays = np.load(saved)
        record = json.loads(details.read_text(encoding="utf-8"))
        observation = prepare(shared, case, seed_root=2026092206)
        assert np.array_equal(arrays["truth"], observation["truth"].astype(np.float32))

        full = np.asarray(arrays["full"], dtype=float)
        null = np.asarray(arrays["null"], dtype=float)
        active = np.asarray(observation["active"], dtype=int)
        baseline = np.asarray(observation["baseline"], dtype=bool)
        gain = np.asarray(observation["gain"], dtype=float)
        weights = np.asarray(observation["channel_weights"], dtype=float)
        confirmation = np.asarray(observation["confirmation"], dtype=float)
        basis, _ = _smooth_temporal_basis(confirmation, baseline, observation["active_windows"][0])
        centered = confirmation - confirmation[:, baseline].mean(axis=1, keepdims=True)
        weighted = weights[:, None] * centered
        weighted_gain = weights[:, None] * gain
        response = weighted @ basis.T
        null_prediction = weighted_gain @ (null @ basis.T)
        full_prediction = weighted_gain @ (full @ basis.T)
        residual0 = response - null_prediction
        surface_delta = weighted_gain[:, :n_surf] @ ((full[:n_surf] - null[:n_surf]) @ basis.T)
        deep_delta = weighted_gain[:, n_surf:] @ (full[n_surf:] @ basis.T)
        full_delta = full_prediction - null_prediction

        # 训练解自己选择深点；确认份只估计这一固定时空方向的幅度和符号。
        energy = np.sum(full[:, active] ** 2, axis=1)
        deep_energy = energy[n_surf:]
        selected_local = int(np.argmax(deep_energy))
        selected = n_surf + selected_local
        selected_delta = np.outer(weighted_gain[:, selected], full[selected] @ basis.T)
        control = surface_delta.ravel()
        selected_flat = selected_delta.ravel()
        if np.dot(control, control) > 0:
            selected_partial = selected_delta - (np.dot(control, selected_flat) / np.dot(control, control)) * surface_delta
            deep_partial = deep_delta - (np.dot(control, deep_delta.ravel()) / np.dot(control, control)) * surface_delta
        else:
            selected_partial = selected_delta.copy()
            deep_partial = deep_delta.copy()

        baseline_covariance = weighted[:, baseline] @ weighted[:, baseline].T / (baseline.sum() - 1)
        basis_sum = basis[:, active].sum(axis=1)
        direction_stats = {}
        for name, direction in {
            "full_h1_minus_h0": full_delta,
            "all_deep_component": deep_delta,
            "selected_deep_component": selected_delta,
            "all_deep_partial_surface_change": deep_partial,
            "selected_deep_partial_surface_change": selected_partial,
        }.items():
            cross = float(np.sum(residual0 * direction))
            squared_norm = float(np.sum(direction ** 2))
            projected = direction @ basis_sum
            variance = float(np.sum(direction * (baseline_covariance @ direction))
                             + projected @ baseline_covariance @ projected / baseline.sum())
            direction_stats[name] = {
                "confirmation_cross": cross,
                "training_direction_squared_norm": squared_norm,
                "confirmation_amplitude": cross / squared_norm if squared_norm > 0 else None,
                "one_sided_directional_z": cross / np.sqrt(variance) if variance > 0 else None,
                "optimally_scaled_loss_improvement": max(cross, 0.0) ** 2 / squared_norm if squared_norm > 0 else None,
            }

        amplitude = metrics.source_amplitude(full, active, baseline)
        deep_amplitude = amplitude[n_surf:]
        true_index = case.get("deep_index")
        true_local = int(true_index - n_surf) if true_index is not None else None
        positions = np.asarray(shared["vertices"], dtype=float)
        peak_distance = (float(np.linalg.norm(positions[selected] - positions[true_index]) * 1000)
                         if true_index is not None else None)

        # Lead-field相似度只使用该病例训练预处理后的 gain；不使用确认标签选列。
        norms = np.linalg.norm(weighted_gain, axis=0)
        normalized_gain = weighted_gain / np.maximum(norms, np.finfo(float).tiny)
        lead = {}
        if true_index is not None:
            correlations = np.abs(normalized_gain.T @ normalized_gain[:, true_index])
            correlations[true_index] = -1
            surface_match = int(np.argmax(correlations[:n_surf]))
            deep_match = n_surf + int(np.argmax(correlations[n_surf:]))
            lead = {
                "true_deep_weighted_gain_norm": float(norms[true_index]),
                "selected_to_true_abs_cosine": float(abs(normalized_gain[:, selected] @ normalized_gain[:, true_index])),
                "closest_surface_index_by_abs_cosine": surface_match,
                "closest_surface_abs_cosine": float(correlations[surface_match]),
                "closest_other_deep_index_by_abs_cosine": deep_match,
                "closest_other_deep_abs_cosine": float(correlations[deep_match]),
            }

        converged = {
            name: bool(record["fitting"][name + "_model"]["windows"][0]["solver"]["converged"])
            for name in ("null", "full")
        }
        evidence = record["evidence"]
        expected = float(evidence["expected_response_noise_energy"])
        calculated_score = (float(evidence["null_loss"]) - float(evidence["full_loss"])) / expected
        assert np.isclose(calculated_score, float(record["evidence"]["loss_improvement"]) / expected)
        assert np.isclose(calculated_score,
                          (2 * direction_stats["full_h1_minus_h0"]["confirmation_cross"]
                           - direction_stats["full_h1_minus_h0"]["training_direction_squared_norm"]) / expected,
                          rtol=2e-5, atol=2e-7)
        rows.append({
            "variant": variant,
            "case_id": case["case_id"],
            "scenario": case["scenario"],
            "has_deep_true": true_index is not None,
            "null_converged": converged["null"],
            "full_converged": converged["full"],
            "predictive_score": calculated_score,
            "null_loss_noise_units": float(evidence["null_loss"]) / expected,
            "full_loss_noise_units": float(evidence["full_loss"]) / expected,
            "selected_deep_index": selected,
            "true_deep_index": true_index,
            "true_deep_active_energy_rank_of_16": (1 + int(np.count_nonzero(deep_energy > deep_energy[true_local]))
                                                   if true_local is not None else None),
            "true_deep_baseline_adjusted_rank_of_16": (1 + int(np.count_nonzero(deep_amplitude > deep_amplitude[true_local]))
                                                        if true_local is not None else None),
            "true_deep_energy_fraction_of_deep_peak": (float(deep_energy[true_local] / deep_energy.max())
                                                        if true_local is not None and deep_energy.max() > 0 else None),
            "selected_deep_distance_to_truth_mm": peak_distance,
            "deep_active_energy_auc_from_rank": (float(np.mean(deep_energy[true_local] > np.delete(deep_energy, true_local))
                                                          + .5 * np.mean(deep_energy[true_local] == np.delete(deep_energy, true_local)))
                                                 if true_local is not None else None),
            "deep_to_global_baseline_adjusted_peak_ratio": float(deep_amplitude.max() / amplitude.max()) if amplitude.max() > 0 else 0.0,
            "directional_confirmation": direction_stats,
            "lead_field": lead,
        })
        for local in range(16):
            index = n_surf + local
            deep_tables.append({
                "variant": variant,
                "case_id": case["case_id"],
                "deep_index": index,
                "is_true": bool(index == true_index),
                "distance_to_true_mm": (float(np.linalg.norm(positions[index] - positions[true_index]) * 1000)
                                        if true_index is not None else None),
                "active_energy": float(deep_energy[local]),
                "active_energy_fraction_of_deep_peak": float(deep_energy[local] / deep_energy.max()) if deep_energy.max() > 0 else 0.0,
                "baseline_adjusted_amplitude": float(deep_amplitude[local]),
                "active_energy_rank": 1 + int(np.count_nonzero(deep_energy > deep_energy[local])),
            })

# %% 保存单一 JSON；其中所有阈值外判断均留给后续独立校准，开发诊断不作验收结论。
payload = {
    "scope": "five previously seen 5/5 dB development cases only; no calibration or validation files read; no inverse refit",
    "interpretation": {
        "predictive_score": "fixed-amplitude H1 minus H0 held-out loss improvement in baseline-noise units",
        "confirmation_amplitude": "least-squares scalar on a training-selected fixed prediction direction; >0 is sign replication, 1 reproduces training amplitude",
        "one_sided_directional_z": "development diagnostic using confirmation baseline covariance; not a calibrated p-value",
        "partial_surface_change": "deep direction residualized against the one observed H1-vs-H0 cortical reallocation direction",
    },
    "rows": rows,
    "deep_point_tables": deep_tables,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(output)
print(f"rows={len(rows)}, deep_points={len(deep_tables)}")
