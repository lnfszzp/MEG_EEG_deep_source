"""开发消融：固定比较训练证据权重与白化后逐通道等权重。"""

# %% 参数和只读开发清单。
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_partial_confirmation import score_partial_deep_presence
from candidates.oaster_predictive import fit_predictive_models, score_predictive_models
import run_erp_whole_head_matrix as original

manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_sensor_balanced_m10_m10_cases02_04.json"
source = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04"
output = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_trial_equal_channels_admm_mrf080_m10_m10_cases03_04"
seed_root = 2026092206
solver_settings = {"solver_kind": "admm", "mrf_strength": .8}
minimum_clear_auc_gain = .02
if output.exists():
    raise FileExistsError(f"不覆盖已有消融：{output}")
all_cases = json.loads(manifest.read_text(encoding="utf-8"))
cases = [case for case in all_cases if case["configuration_number"] in (3, 4)]
assert [case["configuration_number"] for case in cases] == [3, 4]
assert all(case["panel"] == "erp_v6_development" for case in cases)
assert all(case["eeg_snr_db"] == case["meg_snr_db"] == -10 for case in cases)

# %% 固定代码和数据指纹；此脚本没有校准或验证入口。
paths = [Path(__file__), *[root / name for name in (
    "benchmark/erp_replicates.py", "benchmark/erp_protocol.py", "benchmark/protocol.py",
    "benchmark/metrics.py", "candidates/oaster_predictive.py",
    "candidates/oaster_partial_confirmation.py", "candidates/oaster_balanced.py",
    "candidates/graph_reweight_solver.py", "candidates/oaster_rebuilt.py",
    "algorithms/spatial_fused_fusion.py", "run_erp_whole_head_matrix.py",
    "run_strict_oaster.py", "protected_multilayer.py")],
    *sorted((root / "metrics/user_metrics").glob("*.py"))]
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
assert source_metadata["phase"] == "development" and source_metadata["covariance"] == "trial"
assert source_metadata["solver_settings"] == solver_settings
assert source_metadata["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
for case in cases:
    assert (source / (case["case_id"] + ".json")).is_file()
    assert (source / (case["case_id"] + ".npz")).is_file()
output.mkdir(parents=True)
metadata = {
    "complete": False, "phase": "development", "ablation": "whitened_channel_equal_weights",
    "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "case_ids": [case["case_id"] for case in cases], "seed_root": seed_root,
    "preparation": "prepare_trial_covariance_case", "solver_settings": solver_settings,
    "source": str(source.resolve()),
    "source_metadata_sha256": hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest(),
    "paired_difference": "channel_weights only; original fits loaded from the completed source run and equal-weight fits reuse the same deterministic prepared arrays",
    "weight_modes": ["training_evidence", "whitened_channel_equal"],
    "minimum_clear_auc_gain": minimum_clear_auc_gain,
    "calibration_or_validation_read": False, "truth_used_by_fit_or_confirmation_scores": False,
    "code_sha256": code_hashes, "shared_fingerprint": original._shared_fingerprint(shared)}
(output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 每例只准备一次；两种拟合唯一差别是 channel_weights。
metric_rows, evidence_rows = [], []
started = time.perf_counter()
for case in cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    tick = time.perf_counter()
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    original_weights = np.asarray(observation["channel_weights"], float)
    equal_weights = np.ones_like(original_weights)
    assert original_weights.shape == equal_weights.shape == (observation["training"].shape[0],)
    original_case = json.loads((source / (case["case_id"] + ".json")).read_text(encoding="utf-8"))
    with np.load(source / (case["case_id"] + ".npz")) as arrays:
        original_null = arrays["null"].astype(float)
        original_full = arrays["full"].astype(float)
        assert np.allclose(arrays["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    estimates = {}
    for mode, weights in (("training_evidence", original_weights),
                          ("whitened_channel_equal", equal_weights)):
        print(f"{case['case_id']} {mode} starting", flush=True)
        if mode == "training_evidence":
            null, full, fitting = original_null, original_full, original_case["fitting"]
        else:
            null, full, fitting = fit_predictive_models(
                observation["training"], observation["gain"], shared["n_surf"],
                adjacency=shared["adjacency"], baseline=observation["baseline"],
                active=observation["active_windows"][0], channel_weights=weights,
                solver_settings=solver_settings)
        prediction, prediction_detail = score_predictive_models(
            observation["confirmation"], observation["gain"], null, full,
            baseline=observation["baseline"], active=observation["active_windows"][0],
            channel_weights=weights)
        partial, partial_detail = score_partial_deep_presence(
            observation["confirmation"], observation["gain"], null, full, shared["n_surf"],
            baseline=observation["baseline"], active=observation["active_windows"][0],
            channel_weights=weights)
        convergence = {name: bool(fitting[name + "_model"]["windows"][0]["solver"]["converged"])
                       for name in ("null", "full")}
        identity = {key: case[key] for key in
                    ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
        evidence_rows.append({**identity, "weight_mode": mode,
            "prediction_score": prediction, "partial_confirmation_score": partial,
            "partial_selected_deep_index": partial_detail["selected_deep_index"],
            "partial_conditional_gain_fraction": partial_detail["conditional_gain_fraction"],
            "null_converged": int(convergence["null"]), "full_converged": int(convergence["full"]),
            "all_converged": int(all(convergence.values())),
            "eeg_weight": float(weights[0]), "meg_weight": float(weights[-1]),
            "null_confirmation_loss": prediction_detail["null_loss"],
            "full_confirmation_loss": prediction_detail["full_loss"]})
        for hypothesis, estimate in (("H0_surface_only", null), ("H1_surface_deep", full)):
            values = metrics.evaluate_estimate(
                estimate, observation["truth"], shared["vertices"], observation["groups"],
                shared["n_surf"], observation["active"], shared["auc_cortex"],
                baseline=observation["baseline"])
            metric_rows.append({**identity, "weight_mode": mode, "hypothesis": hypothesis,
                "An_cal_AUC": values["auc"], "An_auc": values["auc_tie_corrected"],
                "surface_An_auc": values["surface_auc_tie_corrected"],
                "deep_An_auc": values["deep_auc_tie_corrected"],
                "surface_SD_mm": values["surface_sd_mm"], "deep_SD_mm": values["deep_sd_mm"],
                "surface_DLE_mm": values["surface_dle_mm"], "deep_DLE_mm": values["deep_dle_mm"],
                "deep_amplitude_ratio": values["deep_score"],
                "deep_peak_distance_mm": values["deep_peak_distance_mm"]})
        estimates[mode + "_null"] = null.astype(np.float32)
        estimates[mode + "_full"] = full.astype(np.float32)
        print(f"  T={prediction:.6g}, partial={partial:.6g}, converged={convergence}", flush=True)
    np.savez_compressed(output / (case["case_id"] + ".npz"),
        truth=observation["truth"].astype(np.float32), **estimates)
    for name, rows in (("metrics.csv", metric_rows), ("evidence.csv", evidence_rows)):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  elapsed={time.perf_counter() - tick:.1f}s", flush=True)

# %% 固定验收：两例都收敛且 H1 的 An_cal AUC 均至少提高 0.02，否则拒绝。
equal_evidence = [row for row in evidence_rows if row["weight_mode"] == "whitened_channel_equal"]
deltas = []
for case in cases:
    h1 = {row["weight_mode"]: row for row in metric_rows
          if row["case_id"] == case["case_id"] and row["hypothesis"] == "H1_surface_deep"}
    deltas.append({"case_id": case["case_id"],
                   "An_cal_AUC_gain": h1["whitened_channel_equal"]["An_cal_AUC"] - h1["training_evidence"]["An_cal_AUC"],
                   "An_auc_gain": h1["whitened_channel_equal"]["An_auc"] - h1["training_evidence"]["An_auc"]})
all_converged = all(row["all_converged"] for row in equal_evidence)
clear_auc_gain = all(row["An_cal_AUC_gain"] >= minimum_clear_auc_gain for row in deltas)
accepted = all_converged and clear_auc_gain
completion = {"complete": True, "case_count": len(cases), "all_equal_fits_converged": all_converged,
    "clear_An_cal_AUC_gain_in_both_cases": clear_auc_gain, "accepted": accepted,
    "decision": "accept" if accepted else "reject", "deltas": deltas,
    "wall_seconds": time.perf_counter() - started}
(output / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(output / "metadata.json").write_text(json.dumps({**metadata, **completion}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = ["# 白化后逐通道等权重开发消融", "",
    "只使用 sensor-balanced 开发病例 03/04；未读取 calibration 或 validation。原权重解读取已完成 trial-covariance 结果，全 1 权重解复用同一份 `prepare_trial_covariance_case` 输出，唯一变化是 `channel_weights`。",
    "", "| 病例 | 权重 | 假设 | 收敛 H0/H1 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) | prediction | partial |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for case in cases:
    for mode in ("training_evidence", "whitened_channel_equal"):
        evidence = next(row for row in evidence_rows if row["case_id"] == case["case_id"] and row["weight_mode"] == mode)
        for hypothesis in ("H0_surface_only", "H1_surface_deep"):
            row = next(row for row in metric_rows if row["case_id"] == case["case_id"] and row["weight_mode"] == mode and row["hypothesis"] == hypothesis)
            show = lambda value: "NA" if not np.isfinite(value) else f"{value:.4f}"
            lines.append(f"| {case['configuration_id']} | {mode} | {hypothesis} | {evidence['null_converged']}/{evidence['full_converged']} | "
                f"{show(row['An_cal_AUC'])} | {show(row['An_auc'])} | {show(row['surface_An_auc'])}/{show(row['deep_An_auc'])} | "
                f"{show(row['surface_SD_mm'])}/{show(row['deep_SD_mm'])} | {show(row['surface_DLE_mm'])}/{show(row['deep_DLE_mm'])} | "
                f"{show(evidence['prediction_score'])} | {show(evidence['partial_confirmation_score'])} |")
lines.extend(["", "## 结论", "", f"**{completion['decision'].upper()}**。预先固定的明显改善标准为：两例 H1 的 An_cal AUC 均提高至少 {minimum_clear_auc_gain:.2f}，且等权重 H0/H1 全部收敛。",
    "", *[f"- {row['case_id']}: An_cal AUC {row['An_cal_AUC_gain']:+.4f}，An_auc {row['An_auc_gain']:+.4f}" for row in deltas],
    "", "该结论仅是开发消融，不是验证结果；confirmation 分数未校准，不能解释为假阳性率。"])
(output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
