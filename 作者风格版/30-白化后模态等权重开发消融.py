"""开发集消融：试次协方差白化后，比较现行模态证据权重与逐通道等权重。"""

# %% 参数。只读取已经完成的开发病例，不接触校准集和验证集。
import argparse
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
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True,
    help="已有 trial-covariance ADMM 开发结果；用于现行权重对照，不重新计算")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--case-number", type=int, nargs="+", required=True,
    help="只允许开发配置号 0..4；建议至少包含一个纯表层和一个含深源病例")
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"不覆盖旧消融结果：{args.output}")
case_numbers = list(dict.fromkeys(args.case_number))
if not case_numbers or any(number not in range(5) for number in case_numbers):
    raise ValueError("--case-number 只能是 0..4")

# %% 锁定已有开发源、清单和代码指纹。缺失病例直接报错，不等待正在运行的任务。
source_metadata = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))
assert source_metadata["phase"] == "development"
assert source_metadata["covariance"] == "trial"
assert source_metadata["solver_settings"] == {"solver_kind": "admm"}
for relative, expected in source_metadata["code_sha256"].items():
    path = root / relative
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, \
        f"已有结果的代码指纹已变化：{relative}"
manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
assert source_metadata["manifest_sha256"] == manifest_hash
all_cases = json.loads(manifest.read_text(encoding="utf-8"))
case_by_id = {case["case_id"]: case for case in all_cases}
source_cases = [case_by_id[case_id] for case_id in source_metadata["case_ids"]]
snr_pairs = {(case["eeg_snr_db"], case["meg_snr_db"]) for case in source_cases}
assert len(source_cases) == 5 and len(snr_pairs) == 1
cases = [case for case in source_cases if case["configuration_number"] in case_numbers]
assert len(cases) == len(case_numbers)
for case in cases:
    for suffix in (".json", ".npz"):
        if not (args.source / (case["case_id"] + suffix)).is_file():
            raise FileNotFoundError(f"已有病例尚未完成：{case['case_id']}{suffix}")

paths = [Path(__file__), *[root / name for name in (
    "candidates/oaster_predictive.py", "candidates/oaster_partial_confirmation.py",
    "candidates/oaster_balanced.py", "candidates/graph_reweight_solver.py",
    "candidates/oaster_rebuilt.py", "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_trial_covariance.py", "benchmark/erp_replicates.py",
    "benchmark/erp_protocol.py", "benchmark/protocol.py", "benchmark/metrics.py",
    "run_strict_oaster.py", "run_erp_whole_head_matrix.py", "protected_multilayer.py")],
    *sorted((root / "metrics/user_metrics").glob("*.py"))]
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
assert source_metadata["shared_fingerprint"] == original._shared_fingerprint(shared)
seed_root = int(source_metadata["seed_root"])
args.output.mkdir(parents=True)
metadata = {"complete": False, "phase": "development", "ablation": "whitened_channel_equal_weights",
    "source": str(args.source), "source_metadata_sha256": hashlib.sha256(
        (args.source / "metadata.json").read_bytes()).hexdigest(),
    "manifest_sha256": manifest_hash, "case_ids": [case["case_id"] for case in cases],
    "case_numbers": case_numbers, "eeg_snr_db": list(snr_pairs)[0][0],
    "meg_snr_db": list(snr_pairs)[0][1], "seed_root": seed_root,
    "solver_settings": {"solver_kind": "admm"}, "code_sha256": code_hashes,
    "shared_fingerprint": original._shared_fingerprint(shared),
    "statistical_basis": "Each retained channel has unit training-mean noise variance after separate EEG/MEG whitening; the Gaussian joint likelihood therefore gives every whitened channel weight one.",
    "comparison": "Only channel_weights changes. Raw train/confirmation means, empirical trial covariance whiteners, gains, source model and ADMM settings are identical.",
    "calibration_or_validation_read": False, "decision_threshold": None}
(args.output / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 每例先核对并读取现行解，再只用全 1 权重重算 H0/H1。
metric_rows, evidence_rows = [], []
started = time.perf_counter()
for case in cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    tick = time.perf_counter()
    case_id = case["case_id"]
    print(f"equal weights: {case_id} starting", flush=True)
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    original_case = json.loads((args.source / (case_id + ".json")).read_text(encoding="utf-8"))
    assert original_case["case_id"] == case_id
    if "raw_mean_sha256" in original_case:
        for key in ("eeg_train", "meg_train", "eeg_confirmation", "meg_confirmation"):
            actual = hashlib.sha256(np.ascontiguousarray(observation[key]).tobytes()).hexdigest()
            assert original_case["raw_mean_sha256"][key] == actual, f"原始均值不一致：{case_id} {key}"
    with np.load(args.source / (case_id + ".npz")) as arrays:
        current_null, current_full = arrays["null"].astype(float), arrays["full"].astype(float)
        assert np.allclose(arrays["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    current_weights = np.asarray(observation["channel_weights"], dtype=float)
    equal_weights = np.ones_like(current_weights)
    assert current_weights.shape == equal_weights.shape == (observation["training"].shape[0],)
    null, full, fitting = fit_predictive_models(
        observation["training"], observation["gain"], shared["n_surf"],
        adjacency=shared["adjacency"], baseline=observation["baseline"],
        active=observation["active_windows"][0], channel_weights=equal_weights,
        solver_settings={"solver_kind": "admm"})

    identity = {key: case[key] for key in
        ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
    for weight_mode, used_null, used_full, used_weights, used_fitting in (
            ("training_evidence", current_null, current_full, current_weights, original_case["fitting"]),
            ("whitened_channel_equal", null, full, equal_weights, fitting)):
        prediction, prediction_detail = score_predictive_models(
            observation["confirmation"], observation["gain"], used_null, used_full,
            baseline=observation["baseline"], active=observation["active_windows"][0],
            channel_weights=used_weights)
        partial, partial_detail = score_partial_deep_presence(
            observation["confirmation"], observation["gain"], used_null, used_full,
            shared["n_surf"], baseline=observation["baseline"],
            active=observation["active_windows"][0], channel_weights=used_weights)
        convergence = {name: bool(used_fitting[name + "_model"]["windows"][0]["solver"]["converged"])
                       for name in ("null", "full")}
        evidence_rows.append({**identity, "weight_mode": weight_mode,
            "prediction_score": prediction, "partial_confirmation_score": partial,
            "partial_selected_deep_index": partial_detail["selected_deep_index"],
            "partial_conditional_gain_fraction": partial_detail["conditional_gain_fraction"],
            "null_converged": int(convergence["null"]), "full_converged": int(convergence["full"]),
            "all_converged": int(all(convergence.values())),
            "eeg_weight": float(used_weights[0]), "meg_weight": float(used_weights[-1]),
            "elapsed_equal_fit_seconds": time.perf_counter() - tick})
        for hypothesis, estimate in (("H0_surface_only", used_null), ("H1_surface_deep", used_full)):
            values = metrics.evaluate_estimate(
                estimate, observation["truth"], shared["vertices"], observation["groups"],
                shared["n_surf"], observation["active"], shared["auc_cortex"],
                baseline=observation["baseline"])
            metric_rows.append({**identity, "weight_mode": weight_mode, "hypothesis": hypothesis,
                "An_cal_AUC": values["auc"], "An_auc": values["auc_tie_corrected"],
                "surface_An_auc": values["surface_auc_tie_corrected"],
                "deep_An_auc": values["deep_auc_tie_corrected"],
                "surface_SD_mm": values["surface_sd_mm"], "deep_SD_mm": values["deep_sd_mm"],
                "surface_DLE_mm": values["surface_dle_mm"], "deep_DLE_mm": values["deep_dle_mm"],
                "deep_amplitude_ratio": values["deep_score"],
                "legacy_amplitude_TP": values["deep_detected"],
                "legacy_amplitude_FP": values["deep_false_positive"],
                "deep_peak_distance_mm": values["deep_peak_distance_mm"],
                "active_count": values["active_count"],
                "surface_active_count": values["surface_active_count"],
                "deep_active_count": values["deep_active_count"]})

    equal_file = args.output / (case_id + "-equal.npz")
    np.savez_compressed(equal_file, truth=observation["truth"].astype(np.float32),
        null=null.astype(np.float32), full=full.astype(np.float32), vertices=shared["vertices"],
        times=shared["times"], active=observation["active"], baseline=observation["baseline"])
    detail = {"case_id": case_id, "complete": True,
        "original_modality_weights": observation["metadata"]["modality_weights"],
        "equal_channel_weights": True, "retained_channels": observation["metadata"]["retained_channels"],
        "source_npz_sha256": hashlib.sha256((args.source / (case_id + ".npz")).read_bytes()).hexdigest(),
        "equal_npz_sha256": hashlib.sha256(equal_file.read_bytes()).hexdigest(),
        "fitting": fitting, "truth_used_by_inverse_or_scores": False,
        "truth_used_only_for_posthoc_metrics": True}
    (args.output / (case_id + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    archive._atomic_csv(args.output / "metrics.csv", metric_rows, metric_rows[0].keys())
    archive._atomic_csv(args.output / "evidence.csv", evidence_rows, evidence_rows[0].keys())
    print(f"  current/equal T={evidence_rows[-2]['prediction_score']:.6g}/"
          f"{evidence_rows[-1]['prediction_score']:.6g}, partial="
          f"{evidence_rows[-2]['partial_confirmation_score']:.6g}/"
          f"{evidence_rows[-1]['partial_confirmation_score']:.6g}, "
          f"converged={evidence_rows[-1]['all_converged']}, seconds={time.perf_counter() - tick:.1f}", flush=True)

# %% 只报告配对开发结果；没有阈值、假阳性率或验证结论。
assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
completion = {"complete": len(evidence_rows) == 2 * len(cases) and len(metric_rows) == 4 * len(cases),
    "case_count": len(cases), "all_equal_fits_converged": all(
        row["all_converged"] for row in evidence_rows if row["weight_mode"] == "whitened_channel_equal"),
    "wall_seconds": time.perf_counter() - started,
    "interpretation": "Development ablation only. Scores are uncalibrated; no FPR or held-out validation claim."}
(args.output / "completion.json").write_text(
    json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(args.output / "metadata.json").write_text(
    json.dumps({**metadata, **completion}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
